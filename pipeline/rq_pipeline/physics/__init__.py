"""Physics: CPU MuJoCo, the only engine the identification stage supports
(`mujoco.sysid` is CPU MuJoCo, and MJCF stays the canonical robot
format). GPU engines — MJX-Warp, Newton — would arrive as another
gymnasium env over the same tasks (rq_pipeline.envs), which is the seam
that keeps every stage independent of the engine underneath.
"""

from rq_pipeline.physics.backend import ModelCounts
from rq_pipeline.physics.mujoco_backend import MuJoCoBackend, Stepper

__all__ = ["ModelCounts", "MuJoCoBackend", "Stepper"]
