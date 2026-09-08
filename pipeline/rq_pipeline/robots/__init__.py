"""The robot seam: how a real robot's telemetry enters the pipeline.

Until 2026-09-09 a robot entered only as a MuJoCo model directory, and
telemetry entered only as this repo's own wire format or a thin CSV
reader shaped like a two-wheel drivetrain. This package is the third
registry seam, the same shape as plugin tasks (`tasks/registry.py`) and
rented-GPU providers (`cloud/provider.py`): a Protocol, a registry, an
entry-point group, and built-in adapters.

- `recording`: the seam's unit. A recording is named CHANNELS — each a
  timestamped array with a unit — plus a census of what the robot
  reported. Nothing downstream knows what a rosbag or a wire file is.
- `adapter`: the `RobotAdapter` Protocol and the registry.
- `adapters/`: `wire` (this repo's Pico rig), `lerobot` (a recorded
  dataset directory), `mcap` (a ROS 2 bag — MCAP is rosbag2's default
  storage since Iron, 2023-05-23, and carries its schemas inside the
  file, so no ROS installation is needed to read one).
- `ingest`: read a source through its adapter, write it into a project
  as a stamped `recording` artifact with a `recording.json` manifest.
"""

from rq_pipeline.robots.adapter import (
    ENTRY_POINT_GROUP,
    AdapterEntry,
    RobotAdapter,
    adapter,
    adapters,
    resolve,
)
from rq_pipeline.robots.recording import Channel, Recording, RecordingManifest

__all__ = [
    "ENTRY_POINT_GROUP",
    "AdapterEntry",
    "Channel",
    "Recording",
    "RecordingManifest",
    "RobotAdapter",
    "adapter",
    "adapters",
    "resolve",
]
