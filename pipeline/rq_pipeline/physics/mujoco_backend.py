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


def keyframe_state(model: Any, name: str) -> numpy.ndarray:
    """The named keyframe of a COMPILED model as a FULLPHYSICS state row.

    Free function on purpose: the backend method below wraps it for
    harness callers, but demo generators and tools hold a bare model and
    were each re-typing the reset-forward-getState ritual.
    """
    mujoco = _require_mujoco()
    import numpy as np  # noqa: PLC0415

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key < 0:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            for i in range(model.nkey)
        ]
        raise KeyError(f"no keyframe {name!r} in model; it has {names}")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state = np.empty(size)
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return state


class Stepper:
    """One seated model, advanced a control tick at a time — the stepping
    discipline in ONE place.

    Until 2026-08-26 this loop existed three times by hand (the harness
    rollouts here, the kitting choreographer, train-watch's playback),
    each carrying the same-instant rule separately. Now: seat a
    FULLPHYSICS state, hold a control for `substeps` physics steps, and
    record the per-physics-step `states`/`sensors` rows every referee
    reads. The gymnasium env (rq_pipeline.envs) is a `Stepper` behind
    `reset`/`step`; the harness's episode is `advance` in a loop.
    """

    def __init__(self, model: Any, initial_state: Any, steps: int) -> None:
        mujoco = _require_mujoco()
        import numpy as np  # noqa: PLC0415

        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        initial = np.asarray(initial_state, dtype=float)
        if initial.shape != (size,):
            raise ValueError(
                f"initial_state must have shape ({size},), got {initial.shape}"
            )
        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        full = mujoco.mjtState.mjSTATE_FULLPHYSICS
        mujoco.mj_setState(model, self.data, initial, full)
        # Populate sensordata (and poses, for renders) for the first
        # observation.
        mujoco.mj_forward(model, self.data)
        self.steps = steps
        self.step = 0
        self.states = np.empty((steps, size))
        self.sensors = np.empty((steps, model.nsensordata))

    @property
    def done(self) -> bool:
        return self.step >= self.steps

    def advance(self, control: Any, substeps: int) -> None:
        """Hold `control` for `substeps` physics steps — fewer at the end
        of the budget, never more. `control` must be exactly nu wide."""
        mujoco = self._mujoco
        model = self.model
        import numpy as np  # noqa: PLC0415

        if substeps <= 0:
            raise ValueError(f"substeps must be positive, got {substeps}")
        control = np.asarray(control, dtype=float)
        if control.shape != (model.nu,):
            raise ValueError(
                f"policy returned control of shape {control.shape}, "
                f"model has {model.nu} actuators"
            )
        self.data.ctrl[:] = control
        for _ in range(substeps):
            if self.done:
                return
            mujoco.mj_step(model, self.data)
            # mj_step leaves sensordata evaluated at the PRE-integration
            # state; recompute so sensors[k], states[k] and any render all
            # describe ONE instant and the next observation is fresh.
            # Measured by this suite's fourth review (R7): without this,
            # every recording carried an accidental one-physics-tick
            # sensor lag nobody chose. Latency, when we model it, will be
            # an explicit fitted parameter (mujoco.sysid fits sensor
            # delays), never a side effect of the stepping loop.
            mujoco.mj_forward(model, self.data)
            self.sensors[self.step] = self.data.sensordata
            full = mujoco.mjtState.mjSTATE_FULLPHYSICS
            mujoco.mj_getState(model, self.data, self.states[self.step], full)
            self.step += 1


class MuJoCoBackend:
    """`PhysicsBackend` implementation over CPU MuJoCo."""

    name = "mujoco"

    def __init__(self) -> None:
        self._mujoco = _require_mujoco()
        self._model: Any = None

    @property
    def model(self) -> Any:
        """The compiled model, for callers that step it themselves
        (`Stepper`, renderers). Loading stays the backend's job."""
        return self._require_model()

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
        return ModelCounts(
            actuators=model.nu,
            sensors=model.nsensor,
            geoms=model.ngeom,
            cameras=model.ncam,
        )

    def default_initial_state(self) -> numpy.ndarray:
        """The model's RESET state (qpos0, zero velocity) as a
        full-physics state vector — the row format `rollout` expects.

        This is the model's zero, not necessarily a pose the robot can
        occupy: for a two-arm rig it may not be. Task protocols that
        need a real pose name a keyframe (`EpisodeProtocol.home`) and
        get it through `keyframe_state`.
        """
        mujoco = self._mujoco
        model = self._require_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        return self._state_of(data)

    def keyframe_state(self, name: str) -> numpy.ndarray:
        """The named keyframe as a full-physics state vector.

        A bundle that ships keyframes has declared where it lives;
        starting anywhere else can measure the wrong thing. Measured
        reason (aloha2-nominal, 2026-08-26): from qpos=0 both ALOHA
        arms point straight up, and driving them to `neutral_pose`
        folds the elbows inward before the shoulders lean back — the
        two grippers meet at the top centre and jam at over 1 kN. The
        real rig never traverses that pose; a harness that starts there
        measures a collision, not a policy.
        """
        return keyframe_state(self._require_model(), name)

    def _state_of(self, data: Any) -> numpy.ndarray:
        mujoco = self._mujoco
        model = self._require_model()
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

    def _closed_loop(
        self,
        initial_state: numpy.ndarray,
        steps: int,
        control_interval: int,
        control_for: Any,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """The one stepping loop both closed-loop rollouts share.

        `control_for(step, data)` builds the observation (its business)
        and returns nu controls (checked here). Everything else —
        validation, seating the state, the R7 same-instant rule — is
        physics discipline and must not fork between the sensor and
        vision paths.
        """
        if steps <= 0 or control_interval <= 0:
            raise ValueError(
                f"steps and control_interval must be positive, got "
                f"{steps} and {control_interval}"
            )
        stepper = Stepper(self._require_model(), initial_state, steps)
        while not stepper.done:
            stepper.advance(control_for(stepper.step, stepper.data), control_interval)
        return stepper.states, stepper.sensors

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

        def control_for(step: int, data: Any) -> Any:
            return policy(step, data.sensordata.copy())

        return self._closed_loop(initial_state, steps, control_interval, control_for)

    def closed_loop_vision_rollout(  # noqa: PLR0913 - state_width is keyword-only and a scalar
        self,
        initial_state: numpy.ndarray,
        policy: Any,
        steps: int,
        control_interval: int,
        cameras: Sequence[Any],
        *,
        state_width: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """The vision episode: the policy sees RENDERED PIXELS + state.

        Added 2026-08-25 for Paper 2's released-checkpoint evaluation:
        ArmnetBench policies observe three cameras and six joint
        positions, never the simulator's internals. `cameras` is a
        sequence of specs with (key, camera_name, width, height); the
        policy receives a LeRobot-shaped dict per control step:
        {"observation.images.<key>": uint8 (H, W, 3), ...,
         "observation.state": float32 (state_width,)} and returns nu
        controls. The state is the first `state_width` SENSOR values
        (the bundle wrapper's jointpos block: six for the SO-101,
        fourteen for ALOHA 2) — sensors, not qpos: same instrument
        rule as the sensor-only rollout above.

        One renderer per unique resolution, shared across cameras.
        """
        mujoco = self._mujoco
        import numpy as np  # noqa: PLC0415

        model = self._require_model()
        renderers: dict[tuple[int, int], Any] = {}
        for camera in cameras:
            key = (camera.height, camera.width)
            if key not in renderers:
                renderers[key] = mujoco.Renderer(
                    model, height=camera.height, width=camera.width
                )

        def control_for(step: int, data: Any) -> Any:
            observation: dict[str, Any] = {
                "observation.state": np.asarray(
                    data.sensordata[:state_width], dtype=np.float32
                ).copy()
            }
            for camera in cameras:
                renderer = renderers[(camera.height, camera.width)]
                renderer.update_scene(data, camera=camera.camera_name)
                observation[f"observation.images.{camera.key}"] = renderer.render()
            return policy(step, observation)

        try:
            return self._closed_loop(
                initial_state, steps, control_interval, control_for
            )
        finally:
            for renderer in renderers.values():
                renderer.close()
