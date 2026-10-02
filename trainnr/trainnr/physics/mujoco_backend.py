"""The physics backend: CPU MuJoCo — load, census, keyframes, batched
rollouts for sysid, and the one stepping loop (`Stepper`).

`mujoco.rollout` is the same batched API the sysid toolbox is built on
(docs/e2e-research/30-the-pipeline.md stage ③ adopts it as the evaluation
loop core: threaded on CPU, divergence detection built in, and it accepts
homogeneous model sequences — which is how per-unit variance gets swept).
Closed-loop episodes — a policy in the loop, sensors or pixels — are the
gymnasium env's (trainnr.envs) and the harness's, both over
`Stepper`; since docs/32 step 5 this module carries no episode loop of
its own. Anything smarter belongs to a stage, not to the backend.

Heavy imports are lazy behind the `sim` extra, same pattern as the
LeRobot export: the module must import cleanly when MuJoCo is absent,
precisely so it can raise the helpful error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trainnr.physics.backend import (
    FullPhysicsLayout,
    ModelCounts,
    census,
    check_rollout_shapes,
    instrument_stamp,
    no_model_message,
)
from trainnr.physics.registry import CPU_ENGINE, engine

if TYPE_CHECKING:  # pragma: no cover
    import numpy


def _require_mujoco() -> Any:
    try:
        # Lazy by design — see module docstring.
        import mujoco  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "the MuJoCo backend needs the 'sim' extra: uv sync --extra sim"
        ) from error
    return mujoco


def seat_at_keyframe(model: Any, data: Any, name: str) -> None:
    """Reset `data` to the named keyframe and forward — the ritual
    `keyframe_state` packs into a row and a caller who needs the live
    `MjData` (a probe reading `sensordata`, say) runs directly.

    Refuses by name, listing what the model actually has: a keyframe
    typo should not read as "the pose is at the origin."
    """
    mujoco = _require_mujoco()

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key < 0:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            for i in range(model.nkey)
        ]
        raise KeyError(f"no keyframe {name!r} in model; it has {names}")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


def keyframe_state(model: Any, name: str) -> numpy.ndarray:
    """The named keyframe of a COMPILED model as a FULLPHYSICS state row.

    Free function on purpose: the backend method below wraps it for
    harness callers, but demo generators and tools hold a bare model and
    were each re-typing the reset-forward-getState ritual.
    """
    mujoco = _require_mujoco()

    data = mujoco.MjData(model)
    seat_at_keyframe(model, data, name)
    return full_state(model, data)


def full_state(
    model: Any, data: Any, out: numpy.ndarray | None = None
) -> numpy.ndarray:
    """`data`'s FULLPHYSICS row — the getState ritual, spelled once.
    Writes into `out` when given (the Stepper's preallocated rows)."""
    mujoco = _require_mujoco()
    import numpy as np  # noqa: PLC0415

    if out is None:
        out = np.empty(FullPhysicsLayout(model).width)
    mujoco.mj_getState(model, data, out, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return out


def reset_state(model: Any) -> numpy.ndarray:
    """The model's own reset (`mj_resetData`, forward) as a state row —
    the start every protocol without a keyframe uses."""
    mujoco = _require_mujoco()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return full_state(model, data)


class Stepper:
    """One seated model, advanced a control tick at a time — the stepping
    discipline in ONE place.

    Until 2026-08-26 this loop existed three times by hand (the harness
    rollouts here, the kitting choreographer, train-watch's playback),
    each carrying the same-instant rule separately. Now: seat a
    FULLPHYSICS state, hold a control for `substeps` physics steps, and
    record the per-physics-step `states`/`sensors` rows every referee
    reads. The gymnasium env (trainnr.envs) is a `Stepper` behind
    `reset`/`step`; the harness's episode is `advance` in a loop.
    """

    def __init__(
        self, model: Any, initial_state: Any, steps: int, *, data: Any = None
    ) -> None:
        """`data` lets a caller seat into an MjData it already holds — a
        passive viewer binds to one object for its lifetime, so the env
        keeps one across resets. Default: a fresh MjData."""
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
        self.data = mujoco.MjData(model) if data is None else data
        full = mujoco.mjtState.mjSTATE_FULLPHYSICS
        mujoco.mj_setState(model, self.data, initial, full)
        # Populate sensordata (and poses, for renders) for the first
        # observation.
        mujoco.mj_forward(model, self.data)
        self.steps = steps
        self.step = 0
        self.states = np.empty((steps, size))
        self.sensors = np.empty((steps, model.nsensordata))
        # Nothing beyond the row and the sensors: a rigid engine.
        self.extras: Mapping[str, Any] = {}

    @property
    def sensordata(self) -> numpy.ndarray:
        """What a policy observes now — a copy, so it cannot write back."""
        return self.data.sensordata.copy()

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
            full_state(model, self.data, out=self.states[self.step])
            self.step += 1


@engine(CPU_ENGINE, doc="CPU MuJoCo: the metrology instrument, deterministic")
class MuJoCoBackend:
    """CPU MuJoCo: the one door models come alive through (load + census),
    keyframes, and the batched rollout sysid uses."""

    name = CPU_ENGINE
    LOAD_DOORS = ("load_spec", "load_model", "load_mjcf", "load_mjcf_string")
    observables: frozenset[str] = frozenset()  # the row and the sensors, nothing more

    def __init__(self) -> None:
        self._mujoco = _require_mujoco()
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        """Whether `load_*` has run — the question a caller may ask without
        reaching into the backend's state."""
        return self._model is not None

    @property
    def model(self) -> Any:
        """The compiled model, for callers that step it themselves
        (`Stepper`, renderers). Loading stays the backend's job."""
        return self._require_model()

    @property
    def instrument(self) -> str:
        """Backend, version and CPU architecture, e.g.
        "mujoco-3.11.0+x86_64": a certificate produced by a different
        version — or, measured, a different architecture — is a
        different instrument (`instrument_stamp`)."""
        return instrument_stamp(self.name, self._mujoco.__version__)

    def stepper(self, initial_state: Any, steps: int, *, data: Any = None) -> Stepper:
        """One episode's stepping loop over the loaded model — the seam the
        harness and the env drive; a second engine offers the same method."""
        return Stepper(self._require_model(), initial_state, steps, data=data)

    def load_model(self, model: Any) -> None:
        """Adopt an already compiled model — the same door the GPU engine
        offers, so a registry caller can hand one compile to both."""
        self._model = model

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
            raise RuntimeError(no_model_message(self.LOAD_DOORS))
        return self._model

    def validate_start(
        self, state: Any, placements: Sequence[Any], *, trial: int
    ) -> Any:
        """A trial's declared start, checked on this model before the
        trial is spent (`physics/placement.py`); refuses with the culprit."""
        from trainnr.physics.placement import require_start  # noqa: PLC0415

        return require_start(self._require_model(), state, placements, trial=trial)

    def counts(self) -> ModelCounts:
        return census(self._require_model())

    def default_initial_state(self) -> numpy.ndarray:
        """The model's RESET state (qpos0, zero velocity) as a
        full-physics state vector — the row format `rollout` expects.

        This is the model's zero, not necessarily a pose the robot can
        occupy: for a two-arm rig it may not be. Task protocols that
        need a real pose name a keyframe (`EpisodeProtocol.home`) and
        get it through `keyframe_state`.
        """
        return reset_state(self._require_model())

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

    def rollout(
        self,
        initial_states: numpy.ndarray,
        controls: numpy.ndarray,
    ) -> numpy.ndarray:
        """Batched trajectories: (nbatch, nstate) x (nbatch, nstep, nu)
        → (nbatch, nstep, nstate). Deterministic given inputs."""
        mujoco = self._mujoco
        from mujoco import rollout as mj_rollout  # noqa: PLC0415

        model = self._require_model()
        initial, control = check_rollout_shapes(
            initial_states, controls, state_width=FullPhysicsLayout(model).width
        )
        data = mujoco.MjData(model)
        state, _sensordata = mj_rollout.rollout(model, data, initial, control)
        return state
