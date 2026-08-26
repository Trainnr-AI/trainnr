"""What a loaded model reports about itself, for fail-loudly gates.

Until 2026-08-26 this module also declared a `PhysicsBackend` Protocol
with one implementation ever; docs/32 step 5 removed it — the gymnasium
env (rq_pipeline.envs) is the backend-agnostic seam now, and a second
engine would arrive as another env, not another Protocol.
"""

from __future__ import annotations


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
