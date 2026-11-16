"""The tracking family: a humanoid follows a reference motion made from
video (docs/86 §5.2, `tasks/tracking`), built by trainnr_mjlab on mjlab's
tracking task.

A tracking task names its reference (a `retarget_video` recording, by
version) and what success is: the clip played to its end with no
termination. mjlab's terminations are the tracking bounds themselves (the
torso's height and tilt, the hands' and feet's height against the
reference), so success needs no second threshold. The spec also carries
the evaluation's conditions: its paired trials, pushes as in training,
and a command lag. Acceptance (`tools/accept-task.py`) checks the
reference is one, then runs a few training iterations on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from trainnr.bundles.hashing import content_stamp
from trainnr.tasks.registry import register

FAMILY = "g1-tracking"
ROBOT = "unitree_g1"
# mjlab's G1 tracking task, as the trainer builds it and a run names it.
MJLAB_TASK = "Mjlab-Tracking-Flat-Unitree-G1"
DEFAULT_TRIALS = 40
MAX_LAG_MS = 200.0
SUCCESS = "the clip played to its end with no termination (mjlab's tracking bounds)"


@dataclass(frozen=True)
class TrackingSpec:
    """A tracking task's knobs. `reference` is a `retarget_video`
    recording by version; `trials` the paired trials an evaluation runs;
    `pushes` whether the evaluation pushes the robot as training did;
    `lag_ms` the evaluation's command lag, drawn per trial in [0, lag_ms]."""

    reference: str = ""
    trials: int = DEFAULT_TRIALS
    pushes: bool = True
    lag_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError(f"trials must be at least 1, got {self.trials}")
        if not 0.0 <= self.lag_ms <= MAX_LAG_MS:
            raise ValueError(f"lag_ms is in [0, {MAX_LAG_MS:g}], got {self.lag_ms}")


@dataclass(frozen=True)
class TrackingTask:
    """What a declared tracking task is, for the tools and the readers."""

    name: str
    robot: str
    task_spec: TrackingSpec
    instruction: str = "track the reference motion to its end without falling"
    control_hz: int = 50
    success: str = SUCCESS
    spec: None = None  # no scene of its own: mjlab's flat G1 scene

    @property
    def stamp(self) -> str:
        return content_stamp(self.name, {"robot": self.robot, **asdict(self.task_spec)})


def is_tracking(task_id: str) -> bool:
    """Whether a task id names the tracking family."""
    return task_id.split("/", 1)[-1] == FAMILY


@register(FAMILY, rig="g1")
def build_g1_tracking(*, spec: TrackingSpec = TrackingSpec()) -> TrackingTask:  # noqa: B008
    """The Unitree G1 tracking a reference made from video. Refuses a task
    that names no reference."""
    if not spec.reference:
        raise ValueError(
            f"{FAMILY} needs a reference: the recording retarget_video made, by "
            "version (settings={'reference': '<name>@<version>'})"
        )
    return TrackingTask(name=FAMILY, robot=ROBOT, task_spec=spec)
