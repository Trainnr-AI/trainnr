"""One task shape for every rig (docs/32 §6).

Until 2026-08-26 the SO-101 and ALOHA 2 builders returned two
dataclasses that agreed on three fields and differed on the rest, and
the rig's state width and the task's instruction lived in callers'
keyword arguments — "no silent default where a wrong value is
possible" enforced by making every caller repeat the value. Now the
task carries them: a builder knows its rig's jointpos block and the
sentence its referee judges, and the env, the exporter and the
certificate read them from one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rq_pipeline.bundles.hashing import content_stamp
from rq_pipeline.protocol import CameraSpec, EpisodeProtocol

# Episode design shared by every rig's tasks: 500 Hz physics (the
# bundles' timestep) with a control tick every ten steps — the 50 Hz
# the public ALOHA data and the real buses run at — and four paired
# starts (the spawn box's corners), so trials pair across policies by
# index. A task that needs another rate says so on its own protocol.
CONTROL_INTERVAL = 10
PAIRED_TRIALS = 4


@runtime_checkable
class TaskSpec(Protocol):
    """A task's DATA: a frozen dataclass whose fields an agent may write
    and a hash names; `trials` is the one field every spec declares, so an
    evaluation can be sized to its starts (`envs.robotiq.make_env`)."""

    trials: int


@dataclass(frozen=True)
class Task:
    """A composed scene, the protocol that scores episodes in it, and
    what a policy is told and shown.

    `state_width` is the bundle's jointpos block — the `agent_pos` a
    policy sees (six for the SO-101, fourteen for ALOHA 2) — and
    `instruction` the language the task is judged under, exposed as
    `task_description` by the env and stamped into every record.
    `bundle_dir` is the robot bundle the scene was composed from — its
    hash stamps every record. `target` is the reach task's goal point
    (a scene fact its referee and viewer share); None elsewhere.
    """

    name: str
    spec: Any
    protocol: EpisodeProtocol
    cameras: tuple[CameraSpec, ...]
    state_width: int
    instruction: str
    bundle_dir: Path
    target: tuple[float, float, float] | None = None
    # The task's DATA — what an agent may write and a hash names (a
    # frozen dataclass per task kind, e.g. `aloha2.KittingSpec`); None for
    # a task that is code only. `stamp` is its `name@hash`.
    task_spec: TaskSpec | None = None

    def __post_init__(self) -> None:
        if self.state_width <= 0:
            raise ValueError(f"state_width must be positive, got {self.state_width}")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.cameras:
            raise ValueError(f"task {self.name!r} declares no cameras")

    @property
    def stamp(self) -> str | None:
        """`name@hash` over the task spec's fields — the identity a
        certificate cites beside the bundle's; None without a spec."""
        if self.task_spec is None:
            return None
        return content_stamp(self.name, asdict(self.task_spec))

    @property
    def control_hz(self) -> int:
        """The policy's rate: one control tick per `control_interval`
        physics steps of the spec's timestep — readable before compile,
        so the LeRobot plugin, the env's `render_fps` and the tools all
        ask the task instead of restating a number."""
        timestep = float(self.spec.option.timestep)
        return round(1.0 / (timestep * self.protocol.control_interval))
