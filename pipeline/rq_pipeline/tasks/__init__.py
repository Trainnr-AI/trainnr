"""Task scenes: robot + scene + episode protocol, composed as programs.

Scenes are BUILT, not hand-authored XML: `MjSpec` attach composes the
robot bundle into a scene the way stage ⓪'s design already promised
("assemblies are programs over graded components"). A builder returns
the spec and the `EpisodeProtocol` together, because a success predicate
divorced from the scene it judges is a bug waiting for a rename.
"""

from rq_pipeline.tasks.components import add_car, attach_arm, compose
from rq_pipeline.tasks.so101 import (
    SO101Task,
    build_lift,
    build_reach,
    build_stack,
    scripted_no_close,
    scripted_pick,
    scripted_stack,
    scripted_stack_no_release,
)

__all__ = [
    "SO101Task",
    "add_car",
    "attach_arm",
    "build_lift",
    "build_reach",
    "build_stack",
    "compose",
    "scripted_no_close",
    "scripted_pick",
    "scripted_stack",
    "scripted_stack_no_release",
]
