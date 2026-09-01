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

Sizing, stated the way MJX means it: `naconmax` is the contact capacity
for the WHOLE batch (every world together) and `njmax` the constraint
capacity per world. On overflow mujoco_warp skips the remaining
contacts silently — wrong physics and no error (its collision driver:
"the remaining contacts will be skipped") — so every program here also
reports the contact count, and the backend refuses to hand back a
trajectory that touched the ceiling.

`MJXBatchedStepper` is the CPU `Stepper`'s contract over a batch of
worlds at once — the seam the vectorized gymnasium env drives (docs/49).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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


@dataclass
class _DevicePrograms:
    """The device model, the data template and the row layout of one
    loaded model, and the jitted programs built over them — held on the
    backend, once per loaded model, so a new episode or a new stepper
    never recompiles. JAX keys its cache on the function object: a
    fresh closure per call cost 0.8 s per call, measured (docs/07
    2026-08-27), before this cache."""

    mx: Any
    template: Any
    layout: FullPhysicsLayout
    rollout: Callable[..., Any] | None = None
    seat_forward: Callable[..., Any] | None = None
    holds: dict[int, Any] = field(default_factory=dict)


@engine(GPU_ENGINE, doc="MJX-Warp: batched worlds on the device, float32")
class MJXWarpBackend:
    """Batched rollouts over the identified model, on MJX's Warp path.

    Every `Engine` door: `instrument`, `counts`, `default_initial_state`
    and `keyframe_state` (state rows are engine-independent, so those
    come from the compiled model), the batched `rollout` contract, and
    `stepper` — BATCHED, every world in lockstep: controls are
    `(nbatch, nu)`, states `(nbatch, steps, width)`; a single world is a
    batch of one (`row[None, :]`). The harness's single-world loop is
    the CPU engine's; the vectorized env is this one's.
    """

    NAME_PREFIX = "mjx"
    DEFAULT_IMPL = "warp"
    # The companion library each MJX implementation runs on, for the stamp.
    COMPANIONS: ClassVar[dict[str, str]] = {"warp": "warp", "jax": "jax"}
    # Above this many geoms MJX's default capacities are not trusted: a
    # HEURISTIC, set well below the one measurement — the ALOHA kitting
    # scene (105 geoms) dumped core on the RTX 3090 Ti with the defaults
    # and ran with naconmax=4096, njmax=8192 (2026-08-27). A Python
    # refusal that names the knobs beats a core dump.
    SIZING_REQUIRED_ABOVE_GEOMS = 32
    SIZING_KNOBS = ("naconmax", "njmax")
    LOAD_DOORS = ("load_spec", "load_model")
    observables: frozenset[str] = frozenset()  # the row and the sensors

    def __init__(
        self,
        *,
        impl: str = DEFAULT_IMPL,
        naconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        if impl not in self.COMPANIONS:
            raise ValueError(
                f"unknown MJX implementation {impl!r}; known: {sorted(self.COMPANIONS)}"
            )
        self._mujoco, self._mjx, self._jax = _require_mjx()
        self._impl = impl
        self.name = f"{self.NAME_PREFIX}-{impl}"
        self._sizing = {
            key: value
            for key, value in zip(self.SIZING_KNOBS, (naconmax, njmax), strict=True)
            if value is not None
        }
        if len(self._sizing) == 1:
            raise ValueError(
                "pass naconmax and njmax together: naconmax is the contact capacity "
                "for the whole batch, njmax the constraint capacity per world"
            )
        self._model: Any = None
        self._programs: _DevicePrograms | None = None

    @property
    def instrument(self) -> str:
        """Engine + versions + device + architecture: float32 physics on
        a device with no bit-repeatability is a different instrument
        from CPU MuJoCo, and every certificate must say which one
        produced it (`physics/backend.py::instrument_stamp`)."""
        import importlib  # noqa: PLC0415

        module = importlib.import_module(self.COMPANIONS[self._impl])
        return instrument_stamp(
            self.name,
            self._mujoco.__version__,
            f"{self.COMPANIONS[self._impl]}-{module.__version__}",
            self._jax.devices()[0].platform,
        )

    @property
    def model(self) -> Any:
        return self._require_model()

    def load_spec(self, spec: Any) -> None:
        """Compile here — the backend stays the one door models come
        alive through, and the census gate the door they enter by."""
        self._adopt(spec.compile())

    def load_model(self, model: Any) -> None:
        """Adopt a model the CPU backend already compiled — the paired
        use: one compile, two instruments, divergence measured."""
        self._adopt(model)

    def _adopt(self, model: Any) -> None:
        if model.ngeom > self.SIZING_REQUIRED_ABOVE_GEOMS and not self._sizing:
            raise ValueError(
                f"{model.ngeom} geoms: pass naconmax and njmax explicitly - the "
                "GPU path dumps core, not a warning, when a busy scene overflows "
                "MJX's default capacities (the kitting bundle ran with "
                "naconmax=4096, njmax=8192 for two worlds)"
            )
        self._model = model
        self._programs = None

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError(no_model_message(self.LOAD_DOORS))
        return self._model

    def default_initial_state(self) -> Any:
        from rq_pipeline.physics.mujoco_backend import reset_state  # noqa: PLC0415

        return reset_state(self._require_model())

    def keyframe_state(self, name: str) -> Any:
        from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: PLC0415

        return keyframe_state(self._require_model(), name)

    def validate_start(
        self, state: Any, placements: Sequence[Any], *, trial: int
    ) -> Any:
        """Same compiled model, same check as the CPU engine."""
        from rq_pipeline.physics.placement import require_start  # noqa: PLC0415

        return require_start(self._require_model(), state, placements, trial=trial)

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
        makes runs statistically — not bitwise — repeatable (docs/49,
        measured docs/52 §5).
        """
        import jax.numpy as jnp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        model = self._require_model()
        initial, control = check_rollout_shapes(
            initial_states, controls, state_width=FullPhysicsLayout(model).width
        )
        states, contacts = self._rollout_program()(
            jnp.asarray(initial), jnp.asarray(control)
        )
        self._check_capacity(contacts)
        return np.asarray(states)

    def stepper(self, initial_state: Any, steps: int) -> MJXBatchedStepper:
        """The batched stepping loop over the loaded model: the door the
        vectorized env and a batched harness drive (see the class).

        `initial_state` is (nbatch, nstate) here — the parameter name is
        the `Engine` protocol's, which runtime_checkable isinstance can
        never verify (it checks attribute presence, not signatures)."""
        return MJXBatchedStepper(self, initial_state, steps)

    # ---- the programs, once per loaded model ---------------------------

    def _device(self) -> _DevicePrograms:
        if self._programs is not None:
            return self._programs
        mjx = self._mjx
        model = self._require_model()
        layout = FullPhysicsLayout(model)
        mx = mjx.put_model(model, impl=self._impl)
        template = mjx.make_data(model, impl=self._impl, **self._sizing)
        self._programs = _DevicePrograms(mx=mx, template=template, layout=layout)
        return self._programs

    def _rollout_program(self) -> Any:
        programs = self._device()
        if programs.rollout is not None:
            return programs.rollout
        mjx, jax = self._mjx, self._jax
        import jax.numpy as jnp  # noqa: PLC0415

        mx, layout = programs.mx, programs.layout
        seat = _seat(programs.template, layout)

        def tick(data: Any, ctrl: Any) -> tuple[Any, tuple[Any, Any]]:
            data = mjx.step(mx, data.replace(ctrl=ctrl))
            return data, (_row(data, layout, jnp), contact_count(data, jnp))

        def episode(row: Any, ctrls: Any) -> tuple[Any, Any]:
            _, (states, contacts) = jax.lax.scan(tick, seat(row), ctrls)
            return states, jnp.max(contacts)

        programs.rollout = jax.jit(jax.vmap(episode))
        return programs.rollout

    def _seat_forward_program(self) -> Any:
        programs = self._device()
        if programs.seat_forward is not None:
            return programs.seat_forward
        mjx, jax = self._mjx, self._jax
        mx = programs.mx
        seat = _seat(programs.template, programs.layout)
        programs.seat_forward = jax.jit(
            jax.vmap(lambda row: mjx.forward(mx, seat(row)))
        )
        return programs.seat_forward

    def _hold_program(self, count: int) -> Any:
        """Advance `count` physics steps under one control, per world:
        the stepper's tick, specialised per substep count and cached."""
        programs = self._device()
        if count in programs.holds:
            return programs.holds[count]
        mjx, jax = self._mjx, self._jax
        import jax.numpy as jnp  # noqa: PLC0415

        mx, layout = programs.mx, programs.layout

        def tick(data: Any, _: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            # R7: the forward pass after the step, so sensors, poses and
            # the row describe ONE instant — the CPU stepper's rule.
            data = mjx.forward(mx, mjx.step(mx, data))
            return data, (
                _row(data, layout, jnp),
                data.sensordata,
                contact_count(data, jnp),
            )

        def hold(data: Any, ctrl: Any) -> tuple[Any, Any, Any, Any]:
            data, (rows, sensor_rows, contacts) = jax.lax.scan(
                tick, data.replace(ctrl=ctrl), None, length=count
            )
            return data, rows, sensor_rows, jnp.max(contacts)

        programs.holds[count] = jax.jit(jax.vmap(hold))
        return programs.holds[count]

    def _check_capacity(self, contacts: Any) -> None:
        """Refuse a trajectory that touched the contact ceiling: past it
        mujoco_warp skips contacts silently, and the rows would be
        physics nobody asked for."""
        import numpy as np  # noqa: PLC0415

        programs = self._device()
        capacity = getattr(programs.template, "naconmax", None)
        if capacity is None:
            return
        peak = int(np.max(np.asarray(contacts)))
        if peak >= int(capacity):
            raise RuntimeError(
                f"contact capacity reached: naconmax={int(capacity)} is the total "
                f"across the whole batch and a step generated {peak} contacts - "
                "mujoco_warp skips the rest silently; raise naconmax (the kitting "
                "bundle ran at 4096 for two worlds)"
            )


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


def contact_count(data: Any, jnp: Any) -> Any:
    """The live contact count of one world: Warp's `nacon` (the capacity
    the backend refuses past), or the JAX implementation's `ncon` — the
    two implementations name it differently (measured 2026-08-29 on the
    law branch; the fix ported here 2026-08-31)."""
    for name in ("nacon", "ncon"):
        if hasattr(data, name):
            return jnp.max(jnp.asarray(getattr(data, name)))
    raise AttributeError("mjx.Data exposes neither nacon nor ncon")


class MJXBatchedStepper:
    """The GPU engine's stepping loop, batched: every world advances in
    lockstep under its own control. The same contract as the CPU
    `Stepper` with a leading batch axis — `advance((nbatch, nu), k)`,
    `sensordata (nbatch, nsensordata)`, `states (nbatch, steps, width)`
    — and, like the CPU stepper, host arrays preallocated for the whole
    budget and filled as it advances (one device-to-host copy per
    `advance`). After each physics step the forward pass is recomputed
    so sensors, poses and the row describe ONE instant (the CPU
    stepper's R7 rule); the whole-episode `rollout` skips that and is
    the faster path when nobody observes mid-episode. The programs live
    on the backend, cached per substep count: a new stepper never
    recompiles. A step that touches the contact ceiling is refused."""

    def __init__(
        self, backend: MJXWarpBackend, initial_state: Any, steps: int
    ) -> None:
        import jax.numpy as jnp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        programs = backend._device()
        layout = programs.layout
        initial = np.asarray(initial_state, dtype=float)
        if initial.ndim != ROLLOUT_STATE_RANK or initial.shape[1] != layout.width:
            raise ValueError(
                "MJX is a batched engine: initial_state must be "
                f"(nbatch, {layout.width}), got {initial.shape} - a single world "
                "is row[None, :]"
            )
        self._backend = backend
        self._jnp = jnp
        self.model = backend.model
        self.nbatch, self.steps, self.step = int(initial.shape[0]), steps, 0
        self._data = backend._seat_forward_program()(jnp.asarray(initial))
        self.states = np.empty((self.nbatch, steps, layout.width))
        self.sensors = np.empty((self.nbatch, steps, int(self.model.nsensordata)))
        self.extras: Mapping[str, Any] = {}  # nothing beyond the row and the sensors

    @property
    def done(self) -> bool:
        return self.step >= self.steps

    @property
    def sensordata(self) -> numpy.ndarray:
        """(nbatch, nsensordata) at the current instant."""
        import numpy as np  # noqa: PLC0415

        return np.asarray(self._data.sensordata)

    def advance(self, control: Any, substeps: int) -> None:
        """Hold `control` (nbatch, nu) for `substeps` physics steps —
        fewer at the end of the budget, never more. The parameter name
        is the `Stepper` protocol's (see `stepper` above)."""
        import numpy as np  # noqa: PLC0415

        if substeps <= 0:
            raise ValueError(f"substeps must be positive, got {substeps}")
        control = np.asarray(control, dtype=float)
        if control.shape != (self.nbatch, self.model.nu):
            raise ValueError(
                f"control must be ({self.nbatch}, {self.model.nu}), "
                f"got {control.shape}"
            )
        if self.done:
            return
        count = min(substeps, self.steps - self.step)
        self._data, rows, sensor_rows, contacts = self._backend._hold_program(count)(
            self._data, self._jnp.asarray(control)
        )
        self._backend._check_capacity(contacts)
        stop = self.step + count
        self.states[:, self.step : stop] = np.asarray(rows)
        self.sensors[:, self.step : stop] = np.asarray(sensor_rows)
        self.step = stop
