"""Physics backends behind one protocol.

MuJoCo is the default and the only backend the identification stage
supports — `mujoco.sysid` is CPU MuJoCo, and MJCF stays the canonical
robot format. Newton (Apache-2.0, MuJoCo-Warp backed, NVIDIA GPU) slots
in as an optional rollout/render backend through the same protocol; the
ecosystem is converging on it, so nothing here may depend on which
backend is underneath.
"""

from rq_pipeline.physics.backend import ModelCounts, PhysicsBackend
from rq_pipeline.physics.mujoco_backend import MuJoCoBackend

__all__ = ["ModelCounts", "MuJoCoBackend", "PhysicsBackend"]
