"""The canonical physics backend: CPU MuJoCo, batched via `mujoco.rollout`.

`mujoco.rollout` is the same batched API the sysid toolbox is built on
(docs/e2e-research/30-the-pipeline.md stage ③ adopts it as the evaluation
loop core: threaded on CPU, divergence detection built in, and it accepts
homogeneous model sequences — which is how per-unit variance gets swept).
This adapter is deliberately thin: load, census, roll out. Anything
smarter belongs to a stage, not to the backend.

Heavy imports are lazy behind the `sim` extra, same pattern as the
LeRobot export: the module must import cleanly when MuJoCo is absent,
precisely so it can raise the helpful error.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rq_pipeline.physics.backend import ModelCounts

if TYPE_CHECKING:  # pragma: no cover
    import numpy

# Array rank expectations for rollout inputs, named for the shape checks.
_STATE_BATCH_RANK = 2  # (nbatch, nstate)
_CONTROL_RANK = 3  # (nbatch, nstep, nu)


def _require_mujoco() -> Any:
    try:
        # Lazy by design — see module docstring.
        import mujoco  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "the MuJoCo backend needs the 'sim' extra: uv sync --extra sim"
        ) from error
    return mujoco


class MuJoCoBackend:
    """`PhysicsBackend` implementation over CPU MuJoCo."""

    name = "mujoco"

    def __init__(self) -> None:
        self._mujoco = _require_mujoco()
        self._model: Any = None

    def load_mjcf(self, path: Path) -> None:
        self._model = self._mujoco.MjModel.from_xml_path(str(path))

    def load_mjcf_string(self, xml: str) -> None:
        """For tests and generated models; same census rules apply."""
        self._model = self._mujoco.MjModel.from_xml_string(xml)

    def load_spec(self, spec: Any) -> None:
        """For MjSpec-composed scenes (task builders); same census rules.

        Compiling here rather than accepting a compiled model keeps the
        backend the single place models come alive — and the census gate
        the single door they enter through.
        """
        self._model = spec.compile()

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("no model loaded — call load_mjcf first")
        return self._model

    def counts(self) -> ModelCounts:
        model = self._require_model()
        return ModelCounts(actuators=model.nu, sensors=model.nsensor, geoms=model.ngeom)

    def default_initial_state(self) -> numpy.ndarray:
        """The model's home state as a full-physics state vector —
        the row format `rollout` expects for `initial_states`."""
        mujoco = self._mujoco
        model = self._require_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        import numpy as np  # noqa: PLC0415

        state = np.empty(size)
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return state

    def rollout(
        self,
        initial_states: numpy.ndarray,
        controls: numpy.ndarray,
    ) -> numpy.ndarray:
        """Batched trajectories: (nbatch, nstate) x (nbatch, nstep, nu)
        → (nbatch, nstep, nstate). Deterministic given inputs."""
        mujoco = self._mujoco
        import numpy as np  # noqa: PLC0415
        from mujoco import rollout as mj_rollout  # noqa: PLC0415

        model = self._require_model()
        initial = np.asarray(initial_states, dtype=float)
        control = np.asarray(controls, dtype=float)
        if initial.ndim != _STATE_BATCH_RANK:
            raise ValueError(
                f"initial_states must be (nbatch, nstate), got {initial.shape}"
            )
        if control.ndim != _CONTROL_RANK or control.shape[0] != initial.shape[0]:
            raise ValueError(
                "controls must be (nbatch, nstep, nu) with the same nbatch "
                f"as initial_states, got {control.shape}"
            )
        data = mujoco.MjData(model)
        state, _sensordata = mj_rollout.rollout(model, data, initial, control)
        return state

    def closed_loop_rollout(
        self,
        initial_state: numpy.ndarray,
        policy: Any,
        steps: int,
        control_interval: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """One policy-in-the-loop episode; see the protocol docstring.

        The policy observes `data.sensordata` (a copy — it cannot write
        into the simulator) and its control is clamped to nothing: what
        it commands is what the actuators get, exactly like the wire.
        """
        mujoco = self._mujoco
        import numpy as np  # noqa: PLC0415

        model = self._require_model()
        if steps <= 0 or control_interval <= 0:
            raise ValueError(
                f"steps and control_interval must be positive, got "
                f"{steps} and {control_interval}"
            )
        size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != (size,):
            raise ValueError(
                f"initial_state must have shape ({size},), got {initial.shape}"
            )
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # Populate sensordata for the policy's first observation.
        mujoco.mj_forward(model, data)

        states = np.empty((steps, size))
        sensors = np.empty((steps, model.nsensordata))
        for step in range(steps):
            if step % control_interval == 0:
                control = np.asarray(policy(step, data.sensordata.copy()), dtype=float)
                if control.shape != (model.nu,):
                    raise ValueError(
                        f"policy returned control of shape {control.shape}, "
                        f"model has {model.nu} actuators"
                    )
                data.ctrl[:] = control
            mujoco.mj_step(model, data)
            # mj_step leaves sensordata evaluated at the PRE-integration
            # state; recompute so sensors[k] and states[k] describe the
            # same instant and the policy's next observation is fresh.
            # Measured by this suite's fourth review (R7): without this,
            # every recording carried an accidental one-physics-tick
            # sensor lag nobody chose. Latency, when we model it, will be
            # an explicit fitted parameter (mujoco.sysid fits sensor
            # delays), never a side effect of the stepping loop.
            mujoco.mj_forward(model, data)
            sensors[step] = data.sensordata
            mujoco.mj_getState(
                model, data, states[step], mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
        return states, sensors

    def closed_loop_vision_rollout(
        self,
        initial_state: numpy.ndarray,
        policy: Any,
        steps: int,
        control_interval: int,
        cameras: Sequence[Any],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """The vision episode: the policy sees RENDERED PIXELS + state.

        Added 2026-08-25 for Paper 2's released-checkpoint evaluation:
        ArmnetBench policies observe three cameras and six joint
        positions, never the simulator's internals. `cameras` is a
        sequence of specs with (key, camera_name, width, height); the
        policy receives a LeRobot-shaped dict per control step:
        {"observation.images.<key>": uint8 (H, W, 3), ...,
         "observation.state": float32 (6,)} and returns nu controls.
        The state is the first six SENSOR values (the so101 wrapper's
        jointpos block) — sensors, not qpos: same instrument rule as
        the sensor-only rollout above.

        One renderer per unique resolution, shared across cameras.
        """
        mujoco = self._mujoco
        import numpy as np  # noqa: PLC0415

        model = self._require_model()
        if steps <= 0 or control_interval <= 0:
            raise ValueError(
                f"steps and control_interval must be positive, got "
                f"{steps} and {control_interval}"
            )
        size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != (size,):
            raise ValueError(
                f"initial_state must have shape ({size},), got {initial.shape}"
            )
        renderers: dict[tuple[int, int], Any] = {}
        for camera in cameras:
            key = (camera.height, camera.width)
            if key not in renderers:
                renderers[key] = mujoco.Renderer(
                    model, height=camera.height, width=camera.width
                )
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)

        states = np.empty((steps, size))
        sensors = np.empty((steps, model.nsensordata))
        try:
            for step in range(steps):
                if step % control_interval == 0:
                    observation: dict[str, Any] = {
                        "observation.state": np.asarray(
                            data.sensordata[:6], dtype=np.float32
                        ).copy()
                    }
                    for camera in cameras:
                        renderer = renderers[(camera.height, camera.width)]
                        renderer.update_scene(data, camera=camera.camera_name)
                        observation[f"observation.images.{camera.key}"] = (
                            renderer.render()
                        )
                    control = np.asarray(policy(step, observation), dtype=float)
                    if control.shape != (model.nu,):
                        raise ValueError(
                            f"policy returned shape {control.shape}, "
                            f"expected ({model.nu},)"
                        )
                    data.ctrl[:] = control
                mujoco.mj_step(model, data)
                # Same R7 rule as the sensor-only rollout above: recompute
                # so sensors[k], states[k] and the next render all describe
                # ONE instant — no accidental one-tick lag.
                mujoco.mj_forward(model, data)
                sensors[step] = data.sensordata
                mujoco.mj_getState(
                    model, data, states[step], mujoco.mjtState.mjSTATE_FULLPHYSICS
                )
        finally:
            for renderer in renderers.values():
                renderer.close()
        return states, sensors
