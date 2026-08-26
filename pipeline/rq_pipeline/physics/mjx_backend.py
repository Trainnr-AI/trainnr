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

The gymnasium-env wrapper (a vectorized env over the same tasks) lands
with the GPU card where it can be measured; this module is the engine
seam it will drive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rq_pipeline.physics.backend import (
    FullPhysicsLayout,
    ModelCounts,
    check_rollout_shapes,
    instrument_stamp,
    no_model_message,
)

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


class MJXWarpBackend:
    """Batched rollouts over the identified model, on MJX's Warp path.

    Of the harness's `Engine` doors it offers `instrument`, `counts`,
    `default_initial_state` and `keyframe_state` (state rows are
    engine-independent, so those come from the compiled model), and
    the batched `rollout` contract — plus `naconmax`/`njmax`, the
    per-world contact and constraint capacities MJX cannot infer for
    busy scenes (the ALOHA bundle overflows the defaults; the warning
    names the numbers to pass). `stepper` is the door it does not have
    yet: the vectorized env over this backend (docs/32 §10).
    """

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
        self._program: tuple[Any, Any, Any] | None = None  # (mx, template, run)

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
        model = self._require_model()
        return ModelCounts(
            actuators=model.nu,
            sensors=model.nsensor,
            geoms=model.ngeom,
            cameras=model.ncam,
        )

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

    def _compiled(self) -> tuple[Any, Any, Any]:
        """The device model, the data template and the jitted batched
        episode — built once per loaded model and reused. Measured
        before this cache (RTX 3090 Ti, 3 worlds x 50 steps): every
        `rollout` call re-traced and recompiled, 0.8 s per call after
        an 8.6 s first; a cached program re-specialises only when the
        batch or step shape changes."""
        if self._program is not None:
            return self._program
        mjx, jax = self._mjx, self._jax
        import jax.numpy as jnp  # noqa: PLC0415

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

        def seat(row: Any) -> Any:
            data = template.replace(
                time=row[layout.TIME], qpos=row[layout.qpos], qvel=row[layout.qvel]
            )
            if layout.na:
                data = data.replace(act=row[layout.act])
            return data

        def tick(data: Any, ctrl: Any) -> tuple[Any, Any]:
            data = mjx.step(mx, data.replace(ctrl=ctrl))
            parts = [jnp.reshape(data.time, (1,)), data.qpos, data.qvel]
            if layout.na:
                parts.append(data.act)
            return data, jnp.concatenate(parts)

        def episode(row: Any, ctrls: Any) -> Any:
            _, states = jax.lax.scan(tick, seat(row), ctrls)
            return states

        self._program = (mx, template, jax.jit(jax.vmap(episode)))
        return self._program
