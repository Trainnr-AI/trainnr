"""Task scenes: robot + scene + episode protocol, composed as programs.

Scenes are BUILT, not hand-authored XML: `MjSpec` attach composes the
robot bundle into a scene the way stage ⓪'s design already promised
("assemblies are programs over graded components"). A builder returns a
`Task` — the spec and the `EpisodeProtocol` together, because a success
predicate divorced from the scene it judges is a bug waiting for a
rename — and registers itself (`registry.register`), so the env and the
tools find it by id instead of importing it by name.
"""

from rq_pipeline.tasks.components import add_car, attach_arm, compose
from rq_pipeline.tasks.registry import TaskEntry, register, resolve, tasks
from rq_pipeline.tasks.task import CONTROL_INTERVAL, PAIRED_TRIALS, Task

__all__ = [
    "CONTROL_INTERVAL",
    "PAIRED_TRIALS",
    "Task",
    "TaskEntry",
    "add_car",
    "attach_arm",
    "compose",
    "register",
    "resolve",
    "tasks",
]
