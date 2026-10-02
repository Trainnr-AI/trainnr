"""The standard ROS 2 messages as CDR layouts, defined once and read by
every adapter that meets them — the rosbag2 store and the MCAP bag
alike (review 2026-09-24: the MCAP adapter decoded `JointState` and
`Imu` field by field while the bag adapter declared `Imu` as data).

Copied from the message definitions of ROS 2 (`common_interfaces`:
`std_msgs`, `builtin_interfaces`, `geometry_msgs`, `sensor_msgs`). A
vendor's own messages (Unitree's, DFKI's) stay with the adapter that
reads them; these are the shared vocabulary they nest.

The clock rules, stated once. A message carries a header stamp (the
publisher's clock) and a bag records when it arrived (the recorder's
clock). Which one a channel is timed by is the adapter's declared rule,
said in the recording's notes:

- `CLOCK_RECEIVE`: the bag's receive time for every message (rosbag2;
  Unitree's `LowState` carries no header at all).
- `CLOCK_HEADER_ELSE_LOG`: the header stamp where the publisher set one,
  else the log time (MCAP's `/joint_states` and `/imu` from ROS drivers).
"""

from __future__ import annotations

from typing import Any

from trainnr.robots.cdr import NS_PER_S, SEQUENCE, STRING, Field, Layout

TIME = "builtin_interfaces/msg/Time"
HEADER = "std_msgs/msg/Header"
POINT = "geometry_msgs/msg/Point"
VECTOR3 = "geometry_msgs/msg/Vector3"
QUATERNION = "geometry_msgs/msg/Quaternion"
POSE = "geometry_msgs/msg/Pose"
POSE_WITH_COVARIANCE = "geometry_msgs/msg/PoseWithCovariance"
TWIST = "geometry_msgs/msg/Twist"
TWIST_WITH_COVARIANCE = "geometry_msgs/msg/TwistWithCovariance"
ACCEL = "geometry_msgs/msg/Accel"
ROS_IMU = "sensor_msgs/msg/Imu"
ROS_JOINT_STATE = "sensor_msgs/msg/JointState"

_XYZ = (Field("x", "float64"), Field("y", "float64"), Field("z", "float64"))

STANDARD_LAYOUTS: dict[str, Layout] = {
    TIME: (Field("sec", "int32"), Field("nanosec", "uint32")),
    HEADER: (Field("stamp", TIME), Field("frame_id", STRING)),
    POINT: _XYZ,
    VECTOR3: _XYZ,
    QUATERNION: (*_XYZ, Field("w", "float64")),
    POSE: (Field("position", POINT), Field("orientation", QUATERNION)),
    POSE_WITH_COVARIANCE: (
        Field("pose", POSE),
        Field("covariance", "float64", 36),
    ),
    TWIST: (Field("linear", VECTOR3), Field("angular", VECTOR3)),
    TWIST_WITH_COVARIANCE: (
        Field("twist", TWIST),
        Field("covariance", "float64", 36),
    ),
    ACCEL: (Field("linear", VECTOR3), Field("angular", VECTOR3)),
    ROS_IMU: (
        Field("header", HEADER),
        Field("orientation", QUATERNION),
        Field("orientation_covariance", "float64", 9),
        Field("angular_velocity", VECTOR3),
        Field("angular_velocity_covariance", "float64", 9),
        Field("linear_acceleration", VECTOR3),
        Field("linear_acceleration_covariance", "float64", 9),
    ),
    ROS_JOINT_STATE: (
        Field("header", HEADER),
        Field("name", STRING, SEQUENCE),
        Field("position", "float64", SEQUENCE),
        Field("velocity", "float64", SEQUENCE),
        Field("effort", "float64", SEQUENCE),
    ),
}

CLOCK_RECEIVE = "time is the bag's receive timestamp for every message"
CLOCK_HEADER_ELSE_LOG = (
    "time is the header stamp where the publisher set one, else the log time"
)


def header_seconds(decoded: dict[str, Any]) -> float:
    """A decoded message's header stamp in seconds (0.0 when unset)."""
    stamp = decoded["header"]["stamp"]
    return float(stamp["sec"]) + float(stamp["nanosec"]) / NS_PER_S


__all__ = [
    "ACCEL",
    "CLOCK_HEADER_ELSE_LOG",
    "CLOCK_RECEIVE",
    "HEADER",
    "POINT",
    "POSE",
    "POSE_WITH_COVARIANCE",
    "QUATERNION",
    "ROS_IMU",
    "ROS_JOINT_STATE",
    "STANDARD_LAYOUTS",
    "TIME",
    "TWIST",
    "TWIST_WITH_COVARIANCE",
    "VECTOR3",
    "header_seconds",
]
