"""A rosbag2 SQLite bag (`.db3`) as a recording, read with no ROS installed.

rosbag2's original storage is one SQLite file: a `topics` table (name,
type, serialization format) and a `messages` table (topic, timestamp,
CDR bytes). Unlike MCAP, the file carries no message definitions, so a
type is decoded by a LAYOUT this module declares as data: a tuple of
fields, each a CDR primitive, a fixed array, a string, or a nested
layout. Known layouts and the channel names they map to live in
`LAYOUTS` and `PROFILES`; a bag whose types have no layout is refused
by name, with the types it carries, so the next profile is a data entry
and not a code change.

The first profile is the DFKI Go2 state-estimation set (Zenodo record
19336009, CC-BY-4.0): `interfaces/msg/JointState` (position, velocity,
effort, acceleration, 12 each), `interfaces/msg/JointCmd` (position,
velocity, effort, kp, kd), `interfaces/msg/QuadState` (base pose, twist,
acceleration, the joint state again, foot contacts) and
`interfaces/msg/ContactState`. Their joint order is fl, fr, bl, br with
abad, shoulder, knee per leg (their `mit_controller_node.yaml`), mapped
onto Menagerie's names here.

Standard library plus numpy. NOTE (2026-09-24): a rosbags-based ROS 2
adapter is landing on the sibling branch `go2-telemetry-2026-09-24`;
this reader is the minimal one the identification needed first and
should collapse into that adapter at merge, keeping the layouts as data.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.recording import (
    BASE_POSE,
    BASE_TWIST,
    BASIS_PUBLIC,
    COLLECTION_ROBOT_OP,
    FOOT_CONTACT,
    IMU_ANGULAR_VELOCITY,
    IMU_LINEAR_ACCELERATION,
    IMU_ORIENTATION,
    JOINT_ACCELERATION,
    JOINT_COMMAND,
    JOINT_COMMAND_VELOCITY,
    JOINT_EFFORT,
    JOINT_FEEDFORWARD,
    JOINT_KD,
    JOINT_KP,
    JOINT_POSITION,
    JOINT_VELOCITY,
    Channel,
    Recording,
    monotone,
)

NAME = "db3"
SUFFIX = ".db3"
NS_PER_S = 1e9
CDR_HEADER = 4  # the encapsulation header every ROS 2 CDR message starts with
LITTLE_ENDIAN_CDR = (0, 1)  # representation identifier bytes for CDR_LE

# --- the layout language ------------------------------------------------------

# A field: (name, kind, count). kind is a struct code for a primitive
# ("d", "f", "i", "I", "q", "Q", "h", "H", "b", "B", "?"), "s" for a
# string, or the name of another layout for a nested message. count > 1
# is a fixed-size array; count 0 is a variable-length sequence.
Field = tuple[str, str, int]
Layout = tuple[Field, ...]

PRIMITIVE_SIZES = {
    "d": 8,
    "q": 8,
    "Q": 8,
    "f": 4,
    "i": 4,
    "I": 4,
    "h": 2,
    "H": 2,
    "b": 1,
    "B": 1,
    "?": 1,
}

LAYOUTS: dict[str, Layout] = {
    "builtin_interfaces/msg/Time": (("sec", "i", 1), ("nanosec", "I", 1)),
    "std_msgs/msg/Header": (
        ("stamp", "builtin_interfaces/msg/Time", 1),
        ("frame_id", "s", 1),
    ),
    "geometry_msgs/msg/Point": (("x", "d", 1), ("y", "d", 1), ("z", "d", 1)),
    "geometry_msgs/msg/Vector3": (("x", "d", 1), ("y", "d", 1), ("z", "d", 1)),
    "geometry_msgs/msg/Quaternion": (
        ("x", "d", 1),
        ("y", "d", 1),
        ("z", "d", 1),
        ("w", "d", 1),
    ),
    "geometry_msgs/msg/Pose": (
        ("position", "geometry_msgs/msg/Point", 1),
        ("orientation", "geometry_msgs/msg/Quaternion", 1),
    ),
    "geometry_msgs/msg/PoseWithCovariance": (
        ("pose", "geometry_msgs/msg/Pose", 1),
        ("covariance", "d", 36),
    ),
    "geometry_msgs/msg/Twist": (
        ("linear", "geometry_msgs/msg/Vector3", 1),
        ("angular", "geometry_msgs/msg/Vector3", 1),
    ),
    "geometry_msgs/msg/TwistWithCovariance": (
        ("twist", "geometry_msgs/msg/Twist", 1),
        ("covariance", "d", 36),
    ),
    "geometry_msgs/msg/Accel": (
        ("linear", "geometry_msgs/msg/Vector3", 1),
        ("angular", "geometry_msgs/msg/Vector3", 1),
    ),
    "sensor_msgs/msg/Imu": (
        ("header", "std_msgs/msg/Header", 1),
        ("orientation", "geometry_msgs/msg/Quaternion", 1),
        ("orientation_covariance", "d", 9),
        ("angular_velocity", "geometry_msgs/msg/Vector3", 1),
        ("angular_velocity_covariance", "d", 9),
        ("linear_acceleration", "geometry_msgs/msg/Vector3", 1),
        ("linear_acceleration_covariance", "d", 9),
    ),
    # DFKI's dfki-quad interfaces (ws/src/interfaces/msg, read 2026-09-24).
    "interfaces/msg/JointState": (
        ("header", "std_msgs/msg/Header", 1),
        ("position", "d", 12),
        ("velocity", "d", 12),
        ("effort", "d", 12),
        ("acceleration", "d", 12),
    ),
    "interfaces/msg/JointCmd": (
        ("header", "std_msgs/msg/Header", 1),
        ("position", "d", 12),
        ("velocity", "d", 12),
        ("effort", "d", 12),
        ("kp", "d", 12),
        ("kd", "d", 12),
    ),
    "interfaces/msg/ContactState": (
        ("header", "std_msgs/msg/Header", 1),
        ("ground_contact_force", "d", 4),
    ),
    "interfaces/msg/QuadState": (
        ("header", "std_msgs/msg/Header", 1),
        ("pose", "geometry_msgs/msg/PoseWithCovariance", 1),
        ("twist", "geometry_msgs/msg/TwistWithCovariance", 1),
        ("acceleration", "geometry_msgs/msg/Accel", 1),
        ("joint_state", "interfaces/msg/JointState", 1),
        ("foot_contact", "?", 4),
        ("ground_contact_force", "d", 12),
        ("belly_contact", "?", 1),
    ),
}


def _align(offset: int, size: int) -> int:
    """CDR aligns each primitive to its size, counted from the start of
    the encapsulated body — the four header bytes do not count."""
    body = offset - CDR_HEADER
    return CDR_HEADER + (body + size - 1) // size * size


def decode(layout_name: str, blob: bytes) -> dict[str, Any]:
    """One CDR message by its layout: nested dicts, arrays as lists."""
    if tuple(blob[:2]) != LITTLE_ENDIAN_CDR:
        raise ValueError(f"only little-endian CDR is read; header {blob[:4]!r}")
    value, _end = _decode_at(LAYOUTS[layout_name], blob, CDR_HEADER)
    return value


def _decode_at(layout: Layout, blob: bytes, offset: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    for name, kind, declared in layout:
        count = declared
        if kind == "s":
            offset = _align(offset, 4)
            (length,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            out[name] = blob[offset : offset + length - 1].decode("utf-8", "replace")
            offset += length
        elif kind in PRIMITIVE_SIZES:
            size = PRIMITIVE_SIZES[kind]
            if count == 0:
                offset = _align(offset, 4)
                (count,) = struct.unpack_from("<I", blob, offset)
                offset += 4
            offset = _align(offset, size)
            values = struct.unpack_from(f"<{count}{kind}", blob, offset)
            offset += size * count
            out[name] = values[0] if count == 1 else list(values)
        else:
            nested = LAYOUTS[kind]
            if count == 1:
                out[name], offset = _decode_at(nested, blob, offset)
            else:
                items = []
                for _ in range(count):
                    item, offset = _decode_at(nested, blob, offset)
                    items.append(item)
                out[name] = items
    return out, offset


# --- the bag ------------------------------------------------------------------


@dataclass(frozen=True)
class Topic:
    id: int
    name: str
    type: str
    count: int


def topics(path: Path) -> tuple[Topic, ...]:
    """The bag's topics with their message counts."""
    with sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT t.id, t.name, t.type, COUNT(m.id) FROM topics t "
            "LEFT JOIN messages m ON m.topic_id = t.id GROUP BY t.id ORDER BY t.id"
        ).fetchall()
    return tuple(Topic(int(i), n, t, int(c)) for i, n, t, c in rows)


def messages(path: Path, topic: Topic) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Every message on `topic`, decoded by its type's layout; times are
    the bag's receive stamps in seconds."""
    with sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? "
            "ORDER BY timestamp",
            (topic.id,),
        ).fetchall()
    times = np.asarray([row[0] for row in rows], dtype=np.float64) / NS_PER_S
    return times, [decode(topic.type, row[1]) for row in rows]


# --- profiles: which topics become which channels ------------------------------

# DFKI's joint order (mit_controller_node.yaml) → Menagerie Go2 names.
DFKI_JOINTS = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
DFKI_FEET = ("FL", "FR", "RL", "RR")
POSE_COMPONENTS = ("x", "y", "z", "qw", "qx", "qy", "qz")  # MuJoCo's free-joint order
TWIST_COMPONENTS = ("vx", "vy", "vz", "wx", "wy", "wz")


@dataclass(frozen=True)
class Profile:
    """A named set of topics and how their fields become channels."""

    name: str
    robot: str
    joints: tuple[str, ...]
    feet: tuple[str, ...]
    joint_state_topic: str
    joint_cmd_topic: str | None
    quad_state_topic: str | None
    imu_topic: str | None
    notes: tuple[str, ...]
    basis: str


PROFILES: dict[str, Profile] = {
    "dfki-go2": Profile(
        name="dfki-go2",
        robot="Unitree Go2",
        basis=BASIS_PUBLIC,
        joints=DFKI_JOINTS,
        feet=DFKI_FEET,
        joint_state_topic="/joint_states",
        joint_cmd_topic="/joint_cmd",
        quad_state_topic="/quad_state",
        imu_topic="/imu_measurement",
        notes=(
            "public log: DFKI Bremen's Go2 under their own MPC/WBC controller, "
            "not Unitree's (Zenodo record 19336009, CC-BY-4.0)",
            "joint effort is the motor's current-derived estimate, not a torque sensor",
            "base pose and twist are the state estimator's, not ground truth",
        ),
    ),
}


def profile_for(found: Mapping[str, str]) -> Profile:
    """The profile whose topics the bag carries (name → type)."""
    for profile in PROFILES.values():
        needed = [profile.joint_state_topic]
        if all(topic in found for topic in needed):
            return profile
    carried = ", ".join(f"{name} ({kind})" for name, kind in sorted(found.items()))
    raise ValueError(
        f"no rosbag2 profile matches this bag's topics; it carries {carried}; "
        f"known profiles: {sorted(PROFILES)} (add a layout and a profile as data)"
    )


def _stack(rows: Sequence[dict[str, Any]], *keys: str) -> np.ndarray:
    def pick(row: dict[str, Any]) -> Any:
        for key in keys:
            row = row[key]
        return row

    return np.asarray([pick(row) for row in rows], dtype=np.float64)


def _channel(
    name: str, times: np.ndarray, values: np.ndarray, unit: str, parts: tuple[str, ...]
) -> Channel:
    kept, mask = monotone(times)
    return Channel(name, kept, values[mask], unit, parts)


def read_bag(path: Path) -> Recording:
    """The bag as a recording, through the profile its topics match."""
    path = Path(path)
    found = {t.name: t.type for t in topics(path)}
    by_name = {t.name: t for t in topics(path)}
    profile = profile_for(found)
    joints = profile.joints
    channels: dict[str, Channel] = {}
    times, rows = messages(path, by_name[profile.joint_state_topic])
    for key, channel, unit in (
        ("position", JOINT_POSITION, "rad"),
        ("velocity", JOINT_VELOCITY, "rad/s"),
        ("effort", JOINT_EFFORT, "N*m"),
        ("acceleration", JOINT_ACCELERATION, "rad/s^2"),
    ):
        channels[channel] = _channel(channel, times, _stack(rows, key), unit, joints)
    if profile.joint_cmd_topic in found:
        times, rows = messages(path, by_name[profile.joint_cmd_topic])
        for key, channel, unit in (
            ("position", JOINT_COMMAND, "rad"),
            ("velocity", JOINT_COMMAND_VELOCITY, "rad/s"),
            ("effort", JOINT_FEEDFORWARD, "N*m"),
            ("kp", JOINT_KP, "N*m/rad"),
            ("kd", JOINT_KD, "N*m*s/rad"),
        ):
            channels[channel] = _channel(
                channel, times, _stack(rows, key), unit, joints
            )
    if profile.quad_state_topic in found:
        times, rows = messages(path, by_name[profile.quad_state_topic])
        pose = np.column_stack(
            [_stack(rows, "pose", "pose", "position", axis) for axis in ("x", "y", "z")]
            + [
                _stack(rows, "pose", "pose", "orientation", axis)
                for axis in ("w", "x", "y", "z")
            ]
        )
        twist = np.column_stack(
            [_stack(rows, "twist", "twist", "linear", a) for a in ("x", "y", "z")]
            + [_stack(rows, "twist", "twist", "angular", a) for a in ("x", "y", "z")]
        )
        # QuadState's `acceleration` field is not carried: in the field201
        # bag it reads thousands of m/s^2 (an unfiltered difference of the
        # estimator's velocity), so the fit derives the base's acceleration
        # from the IMU instead.
        contact = _stack(rows, "foot_contact")
        channels[BASE_POSE] = _channel(
            BASE_POSE, times, pose, "m, unit quaternion", POSE_COMPONENTS
        )
        channels[BASE_TWIST] = _channel(
            BASE_TWIST, times, twist, "m/s, rad/s", TWIST_COMPONENTS
        )
        channels[FOOT_CONTACT] = _channel(
            FOOT_CONTACT, times, contact, "bool", profile.feet
        )
    if profile.imu_topic in found:
        times, rows = messages(path, by_name[profile.imu_topic])
        quat = np.column_stack(
            [_stack(rows, "orientation", a) for a in ("w", "x", "y", "z")]
        )
        gyro = np.column_stack(
            [_stack(rows, "angular_velocity", a) for a in ("x", "y", "z")]
        )
        acc = np.column_stack(
            [_stack(rows, "linear_acceleration", a) for a in ("x", "y", "z")]
        )
        channels[IMU_ORIENTATION] = _channel(
            IMU_ORIENTATION, times, quat, "unit quaternion", ("qw", "qx", "qy", "qz")
        )
        channels[IMU_ANGULAR_VELOCITY] = _channel(
            IMU_ANGULAR_VELOCITY, times, gyro, "rad/s", ("x", "y", "z")
        )
        channels[IMU_LINEAR_ACCELERATION] = _channel(
            IMU_LINEAR_ACCELERATION, times, acc, "m/s^2", ("x", "y", "z")
        )
    census = {
        "robot": profile.robot,
        "profile": profile.name,
        "joints": list(joints),
        "topics": {t.name: {"type": t.type, "messages": t.count} for t in topics(path)},
    }
    return Recording(
        source=path.name,
        adapter=NAME,
        channels=channels,
        census=census,
        notes=list(profile.notes),
        collection=COLLECTION_ROBOT_OP,
        basis=profile.basis,
    )


@adapter(NAME, doc="A rosbag2 SQLite bag (.db3) decoded by declared layouts")
class Rosbag2Sqlite:
    name = NAME

    def accepts(self, source: Path) -> bool:
        """A rosbag2 store whose topics match one of `PROFILES`. Content,
        not suffix: the Unitree adapter (`robots/adapters/rosbag2.py`)
        reads the same container for its own message types."""
        source = Path(source)
        if source.suffix != SUFFIX or not source.is_file():
            return False
        try:
            found = {t.name: t.type for t in topics(source)}
            profile_for(found)
        except (ValueError, OSError, sqlite3.DatabaseError):
            return False
        return True

    def read(self, source: Path) -> Recording:
        return read_bag(Path(source))
