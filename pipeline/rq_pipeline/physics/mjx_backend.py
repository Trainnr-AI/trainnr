"""The second engine: MJX with the Warp implementation, batched.

docs/e2e-research/49's adapter, kept to the shape the review promised —
a loop plus two sizing knobs. It consumes the SAME compiled `mjModel`
the CPU backend loads (no conversion layer: keyframes, sensors, and the
sysid-identified parameters all survive, which is exactly why this path
won over the Newton engine), and returns FULLPHYSICS-layout state rows
`[time, qpos, qvel, act]` so every referee and state slice reads them
unchanged.

Instrument doctrine (docs/49): GPU rollouts are float32 and not
bit-repeatable, so results from this backend are a DIFFERENT instrument
from CPU MuJoCo — `instrument` says so, certificates bind to it, and
CPU MuJoCo stays the metrology reference. The acceptance gauntlet
(tests/test_mjx_backend.py) is the door: census parity and a measured
divergence bound against the reference before anything downstream
trusts a trajectory.

`MJXBatchedStepper` gives the same `reset`/`step` shape as the CPU
`Stepper` over a batch of worlds at once — the seam the gymnasium env
would vectorize over once it's measured on the GPU card.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from rq_pipeline.physics.backend import (
    ROLLOUT_STATE_RANK,
    FullPhysicsLayout,
    ModelCounts,
    census,
    check_rollout_shapes,
    instrument_stamp,
    no_model_message,
)
from rq_pipeline.physics.registry import GPU_ENGINE, engine

if TYPE_CHECKING:  # pragma: no cover
    import numpy


def _require_mjx() -> tuple[Any, Any, Any]:
    try:
        # Lazy by design — the module must import cleanly without the
        # extra, precisely so it can raise this helpful error.
        import jax  # noqa: PLC0415
        import mujoco  # noqa: PLC0415
        from mujoco import mjx  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "the MJX backend needs the 'mjx' extra: uv sync --extra mjx"
        ) from error
    return mujoco, mjx, jax


@engine(GPU_ENGINE, doc="MJX-Warp: batched worlds on the device, float32")
class MJXWarpBackend:
    """Batched rollouts over the identified model, on MJX's Warp path.

    Every `Engine` door: `instrument`, `counts`, `default_initial_state`
    and `keyframe_state` (state rows are engine-independent, so those
    come from the compiled model), the batched `rollout` contract, and
    `stepper` — BATCHED, every world in lockstep: controls are
    `(nbatch, nu)`, states `(nbatch, steps, width)`; a single world is a
    batch of one. Plus `naconmax`/`njmax`, the per-world contact and
    constraint capacities MJX cannot infer for busy scenes (the kitting
    bundle dumps core on the GPU without them).
    """

    observables: frozenset[str] = frozenset()  # the row and the sensors

    NAME_PREFIX = "mjx"
    DEFAULT_IMPL = "warp"
    # Above this many geoms MJX's default contact/constraint capacities are
    # not trusted: the ALOHA kitting scene (105 geoms) dumped core on the
    # RTX 3090 Ti with the defaults and ran with naconmax=4096, njmax=8192
    # (2026-08-27). A Python refusal that names the knobs beats a core dump.
    SIZING_REQUIRED_ABOVE_GEOMS = 32
    # The companion library each MJX implementation runs on, for the stamp.
    COMPANIONS: ClassVar[dict[str, str]] = {"warp": "warp", "jax": "jax"}

    def __init__(
        self,
        *,
        impl: str = DEFAULT_IMPL,
        naconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        self._mujoco, self._mjx, self._jax = _require_mjx()
        self._impl = impl
        self.name = f"{self.NAME_PREFIX}-{impl}"
        self._sizing = {
            key: value
            for key, value in (("naconmax", naconmax), ("njmax", njmax))
            if value is not None
        }
        self._model: Any = None
        # (mx, template, layout, run-or-None): the device model once per
        # loaded model; the whole-episode program when first asked for.
        self._program: tuple[Any, Any, Any, Any] | None = None

    @property
    def instrument(self) -> str:
        """Engine + versions + device: float32 physics on a device with
        no bit-repeatability is a different instrument from CPU MuJoCo,
        and every certificate must say which one produced it."""
        import importlib  # noqa: PLC0415

        qualifiers = []
        companion = self.COMPANIONS.get(self._impl)
        if companion is not None:
            module = importlib.import_module(companion)
            qualifiers.append(f"{companion}-{module.__version__}")
        qualifiers.append(self._jax.devices()[0].platform)
        return instrument_stamp(self.name, self._mujoco.__version__, *qualifiers)

    @property
    def model(self) -> Any:
        return self._require_model()

    def load_spec(self, spec: Any) -> None:
        """Compile here — the backend stays the one door models come
        alive through, and the census gate the door they enter by."""
        self._model = spec.compile()
        self._program = None

    def load_model(self, model: Any) -> None:
        """Adopt a model the CPU backend already compiled — the paired
        use: one compile, two instruments, divergence measured."""
        self._model = model
        self._program = None

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError(no_model_message())
        return self._model

    def default_initial_state(self) -> Any:
        from rq_pipeline.physics.mujoco_backend import reset_state  # noqa: PLC0415

        return reset_state(self._require_model())

    def keyframe_state(self, name: str) -> Any:
        from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: PLC0415

        return keyframe_state(self._require_model(), name)

    def counts(self) -> ModelCounts:
        return census(self._require_model())

    def rollout(
        self,
        initial_states: numpy.ndarray,
        controls: numpy.ndarray,
    ) -> numpy.ndarray:
        """Batched trajectories, same contract as the CPU backend:
        (nbatch, nstate) x (nbatch, nstep, nu) -> (nbatch, nstep,
        nstate), FULLPHYSICS rows. One `jax.lax.scan` per batch, vmapped
        — every world steps in lockstep on the device.

        Deterministic ON A CPU DEVICE only; on GPU, kernel ordering
        makes runs statistically — not bitwise — repeatable (docs/49).
        """
        import jax.numpy as jnp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        model = self._require_model()
        initial, control = check_rollout_shapes(
            initial_states, controls, state_width=FullPhysicsLayout(model).width
        )
        _, _, run = self._compiled()
        return np.asarray(run(jnp.asarray(initial), jnp.asarray(control)))

    def stepper(self, initial_states: Any, steps: int) -> MJXBatchedStepper:
        """The batched stepping loop over the loaded model: the door the
        vectorized env and a batched harness drive (see the class)."""
        return MJXBatchedStepper(self, initial_states, steps)

    def _device(self) -> tuple[Any, Any, FullPhysicsLayout]:
        """The device model and the data template, built once per loaded
        model; refuses a busy scene without explicit sizing."""
        if self._program is not None:
            return self._program[:3]
        mjx = self._mjx
        model = self._require_model()
        if model.ngeom > self.SIZING_REQUIRED_ABOVE_GEOMS and not self._sizing:
            raise ValueError(
                f"{model.ngeom} geoms: pass naconmax and njmax explicitly - the "
                "GPU path dumps core, not a warning, when a busy scene overflows "
                "MJX's default capacities (the kitting bundle ran with "
                "naconmax=4096, njmax=8192)"
            )
        layout = FullPhysicsLayout(model)
        mx = mjx.put_model(model, impl=self._impl)
        template = mjx.make_data(model, impl=self._impl, **self._sizing)
        self._program = (mx, template, layout, None)
        return mx, template, layout

    def _compiled(self) -> tuple[Any, Any, Any]:
        """The device model, the data template and the jitted batched
        episode — built once per loaded model and reused. Measured
        before this cache (RTX 3090 Ti, 3 worlds x 50 steps): every
        `rollout` call re-traced and recompiled, 0.8 s per call after
        an 8.6 s first; a cached program re-specialises only when the
        batch or step shape changes."""
        mx, template, layout = self._device()
        if self._program[3] is not None:
            return mx, template, self._program[3]
        mjx, jax = self._mjx, self._jax
        import jax.numpy as jnp  # noqa: PLC0415

        seat = _seat(template, layout)

        def tick(data: Any, ctrl: Any) -> tuple[Any, Any]:
            data = mjx.step(mx, data.replace(ctrl=ctrl))
            return data, _row(data, layout, jnp)

        def episode(row: Any, ctrls: Any) -> Any:
            _, states = jax.lax.scan(tick, seat(row), ctrls)
            return states

        run = jax.jit(jax.vmap(episode))
        self._program = (mx, template, layout, run)
        return mx, template, run


def _seat(template: Any, layout: FullPhysicsLayout) -> Any:
    """A FULLPHYSICS row into the data template (one world)."""

    def seat(row: Any) -> Any:
        data = template.replace(
            time=row[layout.TIME], qpos=row[layout.qpos], qvel=row[layout.qvel]
        )
        if layout.na:
            data = data.replace(act=row[layout.act])
        return data

    return seat


def _row(data: Any, layout: FullPhysicsLayout, jnp: Any) -> Any:
    """One world's data as a FULLPHYSICS row."""
    parts = [jnp.reshape(data.time, (1,)), data.qpos, data.qvel]
    if layout.na:
        parts.append(data.act)
    return jnp.concatenate(parts)


class MJXBatchedStepper:
    """The GPU engine's stepping loop, batched: every world advances in
    lockstep under its own control. The same contract as the CPU
    `Stepper` with a leading batch axis — `advance((nbatch, nu), k)`,
    `sensordata (nbatch, nsensordata)`, `states (nbatch, steps, width)`.
    After each physics step the forward pass is recomputed so sensors,
    poses and the row describe ONE instant (the CPU stepper's R7 rule);
    the whole-episode `rollout` skips that, so it is the faster path
    when nobody observes mid-episode. One program per substep count,
    cached; a busy scene is refused without explicit sizing."""

    def __init__(
        self, backend: MJXWarpBackend, initial_states: Any, steps: int
    ) -> None:
        import jax.numpy as jnp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        mx, template, layout = backend._device()
        initial = np.asarray(initial_states, dtype=float)
        if initial.ndim != ROLLOUT_STATE_RANK or initial.shape[1] != layout.width:
            raise ValueError(
                f"initial_states must be (nbatch, {layout.width}), got {initial.shape}"
            )
        self._mjx, self._jax, self._jnp = backend._mjx, backend._jax, jnp
        self._mx, self._layout = mx, layout
        self.model = backend.model
        self.nbatch, self.steps, self.step = int(initial.shape[0]), steps, 0
        jax, mjx = self._jax, self._mjx
        seat = _seat(template, layout)
        self._data = jax.jit(jax.vmap(lambda row: mjx.forward(mx, seat(row))))(
            jnp.asarray(initial)
        )
        self._programs: dict[int, Any] = {}
        self._rows: list[Any] = []
        self._sensor_rows: list[Any] = []
        self.extras: Mapping[str, Any] = {}

    @property
    def done(self) -> bool:
        return self.step >= self.steps

    @property
    def sensordata(self) -> numpy.ndarray:
        """(nbatch, nsensordata) at the current instant."""
        import numpy as np  # noqa: PLC0415

        return np.asarray(self._data.sensordata)

    @property
    def states(self) -> numpy.ndarray:
        """(nbatch, step, width): every physics step so far."""
        return self._stack(self._rows, self._layout.width)

    @property
    def sensors(self) -> numpy.ndarray:
        """(nbatch, step, nsensordata): every physics step so far."""
        return self._stack(self._sensor_rows, int(self.model.nsensordata))

    def _stack(self, chunks: list[Any], width: int) -> numpy.ndarray:
        import numpy as np  # noqa: PLC0415

        if not chunks:
            return np.empty((self.nbatch, 0, width))
        return np.asarray(self._jnp.concatenate(chunks, axis=1))

    def advance(self, controls: Any, substeps: int) -> None:
        """Hold `controls` (nbatch, nu) for `substeps` physics steps —
        fewer at the end of the budget, never more."""
        import numpy as np  # noqa: PLC0415

        if substeps <= 0:
            raise ValueError(f"substeps must be positive, got {substeps}")
        control = np.asarray(controls, dtype=float)
        if control.shape != (self.nbatch, self.model.nu):
            raise ValueError(
                f"controls must be ({self.nbatch}, {self.model.nu}), "
                f"got {control.shape}"
            )
        if self.done:
            return
        count = min(substeps, self.steps - self.step)
        self._data, rows, sensor_rows = self._program(count)(
            self._data, self._jnp.asarray(control)
        )
        self._rows.append(rows)
        self._sensor_rows.append(sensor_rows)
        self.step += count

    def _program(self, count: int) -> Any:
        if count in self._programs:
            return self._programs[count]
        mjx, jax, jnp, mx, layout = (
            self._mjx,
            self._jax,
            self._jnp,
            self._mx,
            self._layout,
        )

        def tick(data: Any, _: Any) -> tuple[Any, tuple[Any, Any]]:
            data = mjx.forward(mx, mjx.step(mx, data))  # R7: one instant
            return data, (_row(data, layout, jnp), data.sensordata)

        def hold(data: Any, ctrl: Any) -> tuple[Any, Any, Any]:
            data, (rows, sensor_rows) = jax.lax.scan(
                tick, data.replace(ctrl=ctrl), None, length=count
            )
            return data, rows, sensor_rows

        self._programs[count] = jax.jit(jax.vmap(hold))
        return self._programs[count]
