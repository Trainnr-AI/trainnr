"""Stage ② — the robot as a measured artifact, not a downloaded file."""

from rq_pipeline.robot.model_checks import DeadModelError, assert_model_alive

__all__ = ["DeadModelError", "assert_model_alive"]
