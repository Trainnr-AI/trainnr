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
