"""Task scenes: robot + scene + episode protocol, composed as programs.

Scenes are BUILT, not hand-authored XML: `MjSpec` attach composes the
robot bundle into a scene the way stage ⓪'s design already promised
("assemblies are programs over graded components"). A builder returns
the spec and the `EpisodeProtocol` together, because a success predicate
divorced from the scene it judges is a bug waiting for a rename.
"""

from rq_pipeline.tasks.so101 import SO101Task, build_reach

__all__ = ["SO101Task", "build_reach"]
