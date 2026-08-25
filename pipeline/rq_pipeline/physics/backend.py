"""The backend protocol every simulation stage codes against."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class PhysicsBackend(Protocol):
    """What a physics engine must provide to host pipeline stages.

    Deliberately small: load a model, report what came alive, roll out
    batches. Rendering and identification are stage-level concerns that
    bind to a concrete backend, not to this protocol.
    """

    name: str

    def load_mjcf(self, path: Path) -> None:
        """Load the canonical robot/scene model. MJCF, never USD — the
        USD scene layer is flattened into the model at bundle-build time."""

    def counts(self) -> ModelCounts:
        """Post-load census. Callers gate on this — see
        rq_pipeline.robot.model_checks.assert_model_alive."""

    def default_initial_state(self) -> Sequence[float]:
        """The model's reset state (qpos0, at rest) as a state row."""

    def keyframe_state(self, name: str) -> Sequence[float]:
        """A named keyframe as a state row — the bundle's declared
        poses. Raises KeyError if the model has no such keyframe."""

    def rollout(
        self,
        initial_states: Sequence[Sequence[float]],
        controls: Sequence[Sequence[Sequence[float]]],
    ) -> Sequence[Sequence[Sequence[float]]]:
        """Batched state trajectories; one entry per initial state."""

    def load_spec(self, spec: object) -> None:
        """Load an MjSpec-composed scene (task builders); same census
        rules as `load_mjcf` — one door for every model."""

    def closed_loop_rollout(
        self,
        initial_state: Sequence[float],
        policy: object,
        steps: int,
        control_interval: int,
    ) -> tuple[object, object]:
        """One episode driven by a policy observing the model's SENSORS.

        `policy(step_index, sensordata) -> control vector`, held for
        `control_interval` physics steps — policies run slower than
        physics, and they see what the instrument's sensors report, never
        privileged state. Returns (states, sensor_history)."""

    def closed_loop_vision_rollout(  # noqa: PLR0913 - mirrors the implementation
        self,
        initial_state: Sequence[float],
        policy: object,
        steps: int,
        control_interval: int,
        cameras: Sequence[object],
        *,
        state_width: int,
    ) -> tuple[object, object]:
        """One episode driven by a policy observing RENDERED PIXELS plus
        the first `state_width` sensor values — the vision harness's
        loop. `state_width` has no default here on purpose: six is the
        SO-101's jointpos block and fourteen is ALOHA 2's, and a silent
        six on a fourteen-servo rig would feed a policy half its state
        with no error. Callers say which rig they mean."""


class ModelCounts:
    """What the loaded model actually contains, for fail-loudly gates."""

    __slots__ = ("actuators", "cameras", "geoms", "sensors")

    def __init__(
        self, actuators: int, sensors: int, geoms: int, cameras: int = 0
    ) -> None:
        self.actuators = actuators
        self.sensors = sensors
        self.geoms = geoms
        # Cameras are counted because a vision policy on a camera-less
        # model fails silently (black frames score 0% with no error);
        # the bundle contract pins how many the rig carries.
        self.cameras = cameras
