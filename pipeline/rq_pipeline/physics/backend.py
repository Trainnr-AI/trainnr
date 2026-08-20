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

    def rollout(
        self,
        initial_states: Sequence[Sequence[float]],
        controls: Sequence[Sequence[Sequence[float]]],
    ) -> Sequence[Sequence[Sequence[float]]]:
        """Batched state trajectories; one entry per initial state."""

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


class ModelCounts:
    """What the loaded model actually contains, for fail-loudly gates."""

    __slots__ = ("actuators", "geoms", "sensors")

    def __init__(self, actuators: int, sensors: int, geoms: int) -> None:
        self.actuators = actuators
        self.sensors = sensors
        self.geoms = geoms
