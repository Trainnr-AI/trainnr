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

from dataclasses import dataclass
from typing import Any

from rq_pipeline.evaluate.harness import EpisodeProtocol
from rq_pipeline.evaluate.vision import CameraSpec


@dataclass(frozen=True)
class Task:
    """A composed scene, the protocol that scores episodes in it, and
    what a policy is told and shown.

    `state_width` is the bundle's jointpos block — the `agent_pos` a
    policy sees (six for the SO-101, fourteen for ALOHA 2) — and
    `instruction` the language the task is judged under, exposed as
    `task_description` by the env and stamped into every record.
    `target` is the reach task's goal point (a scene fact its referee
    and viewer share); None elsewhere.
    """

    name: str
    spec: Any
    protocol: EpisodeProtocol
    cameras: tuple[CameraSpec, ...]
    state_width: int
    instruction: str
    target: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.state_width <= 0:
            raise ValueError(f"state_width must be positive, got {self.state_width}")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.cameras:
            raise ValueError(f"task {self.name!r} declares no cameras")
