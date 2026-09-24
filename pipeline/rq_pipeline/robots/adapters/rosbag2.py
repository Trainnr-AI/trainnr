"""A ROS 2 bag in rosbag2's sqlite3 form as a recording — any robot whose
messages are tables here, read with no ROS installed.

rosbag2's original storage (the default up to Humble, 2022, and what
Unitree's `unitree_ros2` tutorials and DFKI's field bags record) is a
directory: `metadata.yaml` naming the topics and their types, and one or
more `<name>_<n>.db3` SQLite files with a `topics` table and a `messages`
table of CDR bytes. The standard library reads both (`sqlite3`, and the
four YAML lines this needs, read as lines); every message is decoded by
the one CDR reader (`robots.cdr`) from a LAYOUT copied from the vendor's
`.msg` file.

Three tables carry everything that differs between robots, and none of
them is code:

- `LAYOUTS`: a message type's fields (Unitree's `LowState`, `LowCmd`,
  `SportModeState`; DFKI's `JointState`, `JointCmd`, `QuadState`; the
  standard `sensor_msgs/Imu` and the geometry messages they nest).
- `CHANNELS`: what each message type yields, in the pipeline's channel
  names where one exists (the identifier's contract) and the source's
  own where not, with its unit and its components.
- `PROFILES`: whose bag it is — which message types anchor it, which
  it decodes, the robot, the basis ("public log" for DFKI's controller,
  unknown for a Unitree bag until the caller says whose), the joint
  order, the notes a reader must see. A bag is read through the ONE
  profile its types match; two matches or none are refused by name,
  with the types it carries, so the next robot is a data entry.

Clock: rosbag2 stamps a message when it was received (`timestamp`, ns
since the epoch). That receive time is every channel's clock — the
same rule for every profile, and the census reports the rate and jitter
MEASURED from it, never the rate a vendor promises.

2026-09-24: this module absorbed `robot/rosbag_sqlite.py` (DFKI's
profile, written first for the identification) — two readers of one
container had both claimed every `.db3` and refused every bag at the
merge. One decoder, one adapter, profiles as data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.robots import quality
from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.cdr import NS_PER_S, Field, Layout, Reader
from rq_pipeline.robots.joint_orders import (
    GO2_MENAGERIE_FEET,
    GO2_MENAGERIE_JOINTS,
    GO2_MOTORS,
)
from rq_pipeline.robots.recording import (
    BASE_POSE,
    BASE_TWIST,
    BASIS_PUBLIC,
    BASIS_UNKNOWN,
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
    UNKNOWN_UNIT,
    Channel,
    Recording,
    monotone,
)

NAME = "rosbag2"
SUFFIX = ".db3"
METADATA_FILE = "metadata.yaml"
CDR_FORMAT = "cdr"

# -- message layouts, as data ---------------------------------------------------

# Unitree's (unitree_ros2, cyclonedds_ws/src/unitree/unitree_go/msg, read
# 2026-09-24).
LOW_STATE = "unitree_go/msg/LowState"
LOW_CMD = "unitree_go/msg/LowCmd"
SPORT_MODE_STATE = "unitree_go/msg/SportModeState"
# The DDS topics Unitree's stack publishes them on (their SDK and their
# simulator's bridge alike); the live capture subscribes to these.
TOPIC_LOW_STATE = "rt/lowstate"
TOPIC_LOW_CMD = "rt/lowcmd"
TOPIC_SPORT_MODE_STATE = "rt/sportmodestate"
TOPIC_TYPES: dict[str, str] = {
    TOPIC_LOW_STATE: LOW_STATE,
    TOPIC_LOW_CMD: LOW_CMD,
    TOPIC_SPORT_MODE_STATE: SPORT_MODE_STATE,
}
# DFKI's (dfki-ric-underactuated-lab/dfki-quad, ws/src/interfaces/msg, read
# 2026-09-24) and the standard messages they nest.
DFKI_JOINT_STATE = "interfaces/msg/JointState"
DFKI_JOINT_CMD = "interfaces/msg/JointCmd"
DFKI_QUAD_STATE = "interfaces/msg/QuadState"
DFKI_CONTACT_STATE = "interfaces/msg/ContactState"
ROS_IMU = "sensor_msgs/msg/Imu"
HEADER = "std_msgs/msg/Header"
TIME = "builtin_interfaces/msg/Time"
VECTOR3 = "geometry_msgs/msg/Vector3"
QUATERNION = "geometry_msgs/msg/Quaternion"

LAYOUTS: dict[str, Layout] = {
    "IMUState": (
        Field("quaternion", "float32", 4),
        Field("gyroscope", "float32", 3),
        Field("accelerometer", "float32", 3),
        Field("rpy", "float32", 3),
        Field("temperature", "int8"),
    ),
    "MotorState": (
        Field("mode", "uint8"),
        Field("q", "float32"),
        Field("dq", "float32"),
        Field("ddq", "float32"),
        Field("tau_est", "float32"),
        Field("q_raw", "float32"),
        Field("dq_raw", "float32"),
        Field("ddq_raw", "float32"),
        Field("temperature", "int8"),
        Field("lost", "uint32"),
        Field("reserve", "uint32", 2),
    ),
    "MotorCmd": (
        Field("mode", "uint8"),
        Field("q", "float32"),
        Field("dq", "float32"),
        Field("tau", "float32"),
        Field("kp", "float32"),
        Field("kd", "float32"),
        Field("reserve", "uint32", 3),
    ),
    "BmsState": (
        Field("version_high", "uint8"),
        Field("version_low", "uint8"),
        Field("status", "uint8"),
        Field("soc", "uint8"),
        Field("current", "int32"),
        Field("cycle", "uint16"),
        Field("bq_ntc", "int8", 2),
        Field("mcu_ntc", "int8", 2),
        Field("cell_vol", "uint16", 15),
    ),
    "BmsCmd": (Field("off", "uint8"), Field("reserve", "uint8", 3)),
    "TimeSpec": (Field("sec", "int32"), Field("nanosec", "uint32")),
    LOW_CMD: (
        Field("head", "uint8", 2),
        Field("level_flag", "uint8"),
        Field("frame_reserve", "uint8"),
        Field("sn", "uint32", 2),
        Field("version", "uint32", 2),
        Field("bandwidth", "uint16"),
        Field("motor_cmd", "MotorCmd", 20),
        Field("bms_cmd", "BmsCmd"),
        Field("wireless_remote", "uint8", 40),
        Field("led", "uint8", 12),
        Field("fan", "uint8", 2),
        Field("gpio", "uint8"),
        Field("reserve", "uint32"),
        Field("crc", "uint32"),
    ),
    LOW_STATE: (
        Field("head", "uint8", 2),
        Field("level_flag", "uint8"),
        Field("frame_reserve", "uint8"),
        Field("sn", "uint32", 2),
        Field("version", "uint32", 2),
        Field("bandwidth", "uint16"),
        Field("imu_state", "IMUState"),
        Field("motor_state", "MotorState", 20),
        Field("bms_state", "BmsState"),
        Field("foot_force", "int16", 4),
        Field("foot_force_est", "int16", 4),
        Field("tick", "uint32"),
        Field("wireless_remote", "uint8", 40),
        Field("bit_flag", "uint8"),
        Field("adc_reel", "float32"),
        Field("temperature_ntc1", "int8"),
        Field("temperature_ntc2", "int8"),
        Field("power_v", "float32"),
        Field("power_a", "float32"),
        Field("fan_frequency", "uint16", 4),
        Field("reserve", "uint32"),
        Field("crc", "uint32"),
    ),
    SPORT_MODE_STATE: (
        Field("stamp", "TimeSpec"),
        Field("error_code", "uint32"),
        Field("imu_state", "IMUState"),
        Field("mode", "uint8"),
        Field("progress", "float32"),
        Field("gait_type", "uint8"),
        Field("foot_raise_height", "float32"),
        Field("position", "float32", 3),
        Field("body_height", "float32"),
        Field("velocity", "float32", 3),
        Field("yaw_speed", "float32"),
        Field("range_obstacle", "float32", 4),
        Field("foot_force", "int16", 4),
        Field("foot_position_body", "float32", 12),
        Field("foot_speed_body", "float32", 12),
    ),
    TIME: (Field("sec", "int32"), Field("nanosec", "uint32")),
    HEADER: (Field("stamp", TIME), Field("frame_id", "string")),
    "geometry_msgs/msg/Point": (
        Field("x", "float64"),
        Field("y", "float64"),
        Field("z", "float64"),
    ),
    VECTOR3: (Field("x", "float64"), Field("y", "float64"), Field("z", "float64")),
    QUATERNION: (
        Field("x", "float64"),
        Field("y", "float64"),
        Field("z", "float64"),
        Field("w", "float64"),
    ),
    "geometry_msgs/msg/Pose": (
        Field("position", "geometry_msgs/msg/Point"),
        Field("orientation", QUATERNION),
    ),
    "geometry_msgs/msg/PoseWithCovariance": (
        Field("pose", "geometry_msgs/msg/Pose"),
        Field("covariance", "float64", 36),
    ),
    "geometry_msgs/msg/Twist": (Field("linear", VECTOR3), Field("angular", VECTOR3)),
    "geometry_msgs/msg/TwistWithCovariance": (
        Field("twist", "geometry_msgs/msg/Twist"),
        Field("covariance", "float64", 36),
    ),
    "geometry_msgs/msg/Accel": (Field("linear", VECTOR3), Field("angular", VECTOR3)),
    ROS_IMU: (
        Field("header", HEADER),
        Field("orientation", QUATERNION),
        Field("orientation_covariance", "float64", 9),
        Field("angular_velocity", VECTOR3),
        Field("angular_velocity_covariance", "float64", 9),
        Field("linear_acceleration", VECTOR3),
        Field("linear_acceleration_covariance", "float64", 9),
    ),
    DFKI_JOINT_STATE: (
        Field("header", HEADER),
        Field("position", "float64", 12),
        Field("velocity", "float64", 12),
        Field("effort", "float64", 12),
        Field("acceleration", "float64", 12),
    ),
    DFKI_JOINT_CMD: (
        Field("header", HEADER),
        Field("position", "float64", 12),
        Field("velocity", "float64", 12),
        Field("effort", "float64", 12),
        Field("kp", "float64", 12),
        Field("kd", "float64", 12),
    ),
    DFKI_CONTACT_STATE: (
        Field("header", HEADER),
        Field("ground_contact_force", "float64", 4),
    ),
    DFKI_QUAD_STATE: (
        Field("header", HEADER),
        Field("pose", "geometry_msgs/msg/PoseWithCovariance"),
        Field("twist", "geometry_msgs/msg/TwistWithCovariance"),
        Field("acceleration", "geometry_msgs/msg/Accel"),
        Field("joint_state", DFKI_JOINT_STATE),
        Field("foot_contact", "bool", 4),
        Field("ground_contact_force", "float64", 12),
        Field("belly_contact", "bool"),
    ),
}

MOTOR_SLOTS = 20  # every LowState and LowCmd carries 20; a Go2 fills 12
FEET = ("FR", "FL", "RR", "RL")  # Unitree's foot order in LowState
XYZ = ("x", "y", "z")
QUATERNION_WXYZ = ("w", "x", "y", "z")  # Unitree's order, unlike sensor_msgs
POSE_COMPONENTS = ("x", "y", "z", "qw", "qx", "qy", "qz")  # MuJoCo's free-joint
TWIST_COMPONENTS = ("vx", "vy", "vz", "wx", "wy", "wz")
NOTE_TAU_EST = (
    "joint.effort is Unitree's tau_est: estimated from motor current by the "
    "motor driver, not measured by a torque sensor"
)
NOTE_CLOCK = "time is rosbag2's receive timestamp; LowState carries no header stamp"
NOTE_FOOT_FORCE = "foot.force is the sensor's raw count; Unitree publishes no unit"
NOTE_LOW_CMD = (
    "joint.command, command_velocity, feedforward, kp, kd are LowCmd: what the "
    "controller SENT, not what the motors did; a live capture pairs it with "
    "rt/lowstate by receive time"
)


@dataclass(frozen=True)
class Extract:
    """One channel from a decoded message: where its columns come from."""

    channel: str
    unit: str
    components: tuple[str, ...]
    take: Any  # decoded message dict -> list[float]


def _motors(field_name: str, slot: str = "motor_state") -> Any:
    return lambda m: [m[slot][i][field_name] for i in range(len(GO2_MOTORS))]


def _imu(field_name: str) -> Any:
    return lambda m: list(m["imu_state"][field_name])


def _top(field_name: str) -> Any:
    return lambda m: list(m[field_name])


def _scalar(*path: str) -> Any:
    def take(m: dict[str, Any]) -> list[float]:
        value: Any = m
        for key in path:
            value = value[key]
        return [float(value)]

    return take


def _xyz(*path: str) -> Any:
    def take(m: dict[str, Any]) -> list[float]:
        value: Any = m
        for key in path:
            value = value[key]
        return [value["x"], value["y"], value["z"]]

    return take


def _wxyz(*path: str) -> Any:
    def take(m: dict[str, Any]) -> list[float]:
        value: Any = m
        for key in path:
            value = value[key]
        return [value["w"], value["x"], value["y"], value["z"]]

    return take


def _pose(m: dict[str, Any]) -> list[float]:
    pose = m["pose"]["pose"]
    return _xyz("position")(pose) + _wxyz("orientation")(pose)


def _twist(m: dict[str, Any]) -> list[float]:
    twist = m["twist"]["twist"]
    return _xyz("linear")(twist) + _xyz("angular")(twist)


def _bools(field_name: str) -> Any:
    return lambda m: [float(v) for v in m[field_name]]


# What each message type yields. A type decoded by several profiles yields
# the same channels in every one of them.
CHANNELS: dict[str, tuple[Extract, ...]] = {
    LOW_STATE: (
        Extract(JOINT_POSITION, "rad", GO2_MOTORS, _motors("q")),
        Extract(JOINT_VELOCITY, "rad/s", GO2_MOTORS, _motors("dq")),
        Extract(JOINT_EFFORT, "N*m", GO2_MOTORS, _motors("tau_est")),
        Extract(JOINT_ACCELERATION, "rad/s^2", GO2_MOTORS, _motors("ddq")),
        Extract("motor.temperature", "degC", GO2_MOTORS, _motors("temperature")),
        Extract("motor.lost", "count", GO2_MOTORS, _motors("lost")),
        Extract(IMU_ORIENTATION, "quaternion", QUATERNION_WXYZ, _imu("quaternion")),
        Extract(IMU_ANGULAR_VELOCITY, "rad/s", XYZ, _imu("gyroscope")),
        Extract(IMU_LINEAR_ACCELERATION, "m/s^2", XYZ, _imu("accelerometer")),
        Extract("imu.rpy", "rad", ("roll", "pitch", "yaw"), _imu("rpy")),
        Extract("imu.temperature", "degC", ("imu",), _scalar("imu_state", "temperature")),  # noqa: E501
        Extract("foot.force", UNKNOWN_UNIT, FEET, _top("foot_force")),
        Extract("foot.force_est", UNKNOWN_UNIT, FEET, _top("foot_force_est")),
        Extract("battery.soc", "percent", ("soc",), _scalar("bms_state", "soc")),
        Extract("battery.current", UNKNOWN_UNIT, ("current",), _scalar("bms_state", "current")),  # noqa: E501
        Extract("power.voltage", "V", ("v",), _scalar("power_v")),
        Extract("power.current", "A", ("a",), _scalar("power_a")),
    ),
    LOW_CMD: (
        Extract(JOINT_COMMAND, "rad", GO2_MOTORS, _motors("q", "motor_cmd")),
        Extract(JOINT_COMMAND_VELOCITY, "rad/s", GO2_MOTORS, _motors("dq", "motor_cmd")),  # noqa: E501
        Extract(JOINT_FEEDFORWARD, "N*m", GO2_MOTORS, _motors("tau", "motor_cmd")),
        Extract(JOINT_KP, "N*m/rad", GO2_MOTORS, _motors("kp", "motor_cmd")),
        Extract(JOINT_KD, "N*m*s/rad", GO2_MOTORS, _motors("kd", "motor_cmd")),
        Extract("motor.command_mode", "enum", GO2_MOTORS, _motors("mode", "motor_cmd")),
    ),
    SPORT_MODE_STATE: (
        Extract("sport.position", "m", XYZ, _top("position")),
        Extract("sport.velocity", "m/s", XYZ, _top("velocity")),
        Extract("sport.yaw_speed", "rad/s", ("yaw",), _scalar("yaw_speed")),
        Extract("sport.body_height", "m", ("height",), _scalar("body_height")),
        Extract("sport.mode", "enum", ("mode",), _scalar("mode")),
        Extract("sport.gait", "enum", ("gait",), _scalar("gait_type")),
    ),
    DFKI_JOINT_STATE: (
        Extract(JOINT_POSITION, "rad", GO2_MENAGERIE_JOINTS, _top("position")),
        Extract(JOINT_VELOCITY, "rad/s", GO2_MENAGERIE_JOINTS, _top("velocity")),
        Extract(JOINT_EFFORT, "N*m", GO2_MENAGERIE_JOINTS, _top("effort")),
        Extract(JOINT_ACCELERATION, "rad/s^2", GO2_MENAGERIE_JOINTS, _top("acceleration")),  # noqa: E501
    ),
    DFKI_JOINT_CMD: (
        Extract(JOINT_COMMAND, "rad", GO2_MENAGERIE_JOINTS, _top("position")),
        Extract(JOINT_COMMAND_VELOCITY, "rad/s", GO2_MENAGERIE_JOINTS, _top("velocity")),  # noqa: E501
        Extract(JOINT_FEEDFORWARD, "N*m", GO2_MENAGERIE_JOINTS, _top("effort")),
        Extract(JOINT_KP, "N*m/rad", GO2_MENAGERIE_JOINTS, _top("kp")),
        Extract(JOINT_KD, "N*m*s/rad", GO2_MENAGERIE_JOINTS, _top("kd")),
    ),
    # QuadState's `acceleration` field is not carried: in the field201 bag
    # it reads thousands of m/s^2 (an unfiltered difference of the
    # estimator's velocity), so the fit derives the base's acceleration
    # from the IMU instead.
    DFKI_QUAD_STATE: (
        Extract(BASE_POSE, "m, unit quaternion", POSE_COMPONENTS, _pose),
        Extract(BASE_TWIST, "m/s, rad/s", TWIST_COMPONENTS, _twist),
        Extract(FOOT_CONTACT, "bool", GO2_MENAGERIE_FEET, _bools("foot_contact")),
    ),
    ROS_IMU: (
        Extract(IMU_ORIENTATION, "unit quaternion", ("qw", "qx", "qy", "qz"), _wxyz("orientation")),  # noqa: E501
        Extract(IMU_ANGULAR_VELOCITY, "rad/s", XYZ, _xyz("angular_velocity")),
        Extract(IMU_LINEAR_ACCELERATION, "m/s^2", XYZ, _xyz("linear_acceleration")),
    ),
}  # fmt: skip
NOTES: dict[str, tuple[str, ...]] = {
    LOW_STATE: (NOTE_CLOCK, NOTE_TAU_EST, NOTE_FOOT_FORCE),
    LOW_CMD: (NOTE_LOW_CMD,),
    SPORT_MODE_STATE: (),
    DFKI_JOINT_STATE: (),
    DFKI_JOINT_CMD: (),
    DFKI_QUAD_STATE: (),
    ROS_IMU: (),
}


# -- profiles: whose bag, and what of it becomes channels ------------------------


@dataclass(frozen=True)
class Profile:
    """A family of bags: the types that say "this is one of mine" (any of
    `anchors` present), the types it decodes, and what a reader must know
    about the result."""

    name: str
    robot: str
    basis: str
    anchors: tuple[str, ...]
    decodes: tuple[str, ...]
    joints: tuple[str, ...]
    notes: tuple[str, ...] = ()
    census: Mapping[str, Any] = field(default_factory=dict)


PROFILES: dict[str, Profile] = {
    "unitree-go2": Profile(
        name="unitree-go2",
        robot="Unitree Go2",
        # A Unitree bag may be the operator's own robot: the caller says whose.
        basis=BASIS_UNKNOWN,
        anchors=(LOW_STATE, LOW_CMD, SPORT_MODE_STATE),
        decodes=(LOW_STATE, LOW_CMD, SPORT_MODE_STATE),
        joints=GO2_MOTORS,
        census={"motor_slots": MOTOR_SLOTS, "motors_read": len(GO2_MOTORS)},
    ),
    "dfki-go2": Profile(
        name="dfki-go2",
        robot="Unitree Go2",
        basis=BASIS_PUBLIC,
        anchors=(DFKI_JOINT_STATE,),
        decodes=(DFKI_JOINT_STATE, DFKI_JOINT_CMD, DFKI_QUAD_STATE, ROS_IMU),
        joints=GO2_MENAGERIE_JOINTS,
        notes=(
            "public log: DFKI Bremen's Go2 under their own MPC/WBC controller, "
            "not Unitree's (Zenodo record 19336009, CC-BY-4.0)",
            "joint effort is the motor's current-derived estimate, not a torque sensor",
            "base pose and twist are the state estimator's, not ground truth",
            "time is rosbag2's receive timestamp, not the messages' header stamps",
        ),
    ),
}


def profile_for(types: Mapping[str, str]) -> Profile:
    """The one profile whose anchor types the bag carries (topic → type);
    none or two refused by name with the types it carries."""
    carried = set(types.values())
    matches = [p for p in PROFILES.values() if carried & set(p.anchors)]
    if len(matches) == 1:
        return matches[0]
    listed = ", ".join(f"{name} ({kind})" for name, kind in sorted(types.items()))
    if not matches:
        raise ValueError(
            f"no rosbag2 profile matches this bag's topics; it carries {listed}; "
            f"known profiles: {sorted(PROFILES)} (add a layout and a profile as data)"
        )
    raise ValueError(
        f"this bag matches the rosbag2 profiles {[p.name for p in matches]}; one "
        f"bag is one robot's — it carries {listed}"
    )


DOC = "A ROS 2 bag (rosbag2 sqlite3) by declared layouts and profiles, no ROS needed"


@adapter(NAME, doc=DOC)
class Rosbag2Adapter:
    """A ROS 2 bag (rosbag2 sqlite3) by declared layouts and profiles, no ROS needed."""

    name = NAME

    def accepts(self, source: Path) -> bool:
        """A rosbag2 store that carries an anchor type of some profile.
        Content, not suffix: a `.db3` of another robot's topics is not
        claimed, and neither is a file that is no database."""
        files = _files(Path(source))
        if not files:
            return False
        carried = _topic_types(files)
        return any(carried & set(p.anchors) for p in PROFILES.values())

    def read(self, source: Path) -> Recording:
        source = Path(source)
        files = _files(source)
        if not files:
            raise ValueError(f"{source}: no {SUFFIX} file")
        topics, messages = _read_files(files)
        census_topics = {
            topic: {"type": kind, "messages": len(messages.get(topic, []))}
            for topic, (kind, _fmt) in topics.items()
        }
        try:
            profile = profile_for({t: kind for t, (kind, _fmt) in topics.items()})
        except ValueError as why:
            raise ValueError(f"{source}: {why}") from None
        channels: dict[str, Channel] = {}
        notes: list[str] = list(profile.notes)
        for kind in profile.decodes:
            carrying = [t for t, (k, fmt) in topics.items() if k == kind]
            if len(carrying) > 1:
                raise ValueError(
                    f"{source}: {kind} on {len(carrying)} topics {carrying}; the "
                    f"profile {profile.name!r} reads one — which is the robot's?"
                )
            if not carrying or topics[carrying[0]][1] != CDR_FORMAT:
                continue
            msgs = messages.get(carrying[0], [])
            if not msgs:
                continue
            channels.update(_channels(kind, carrying[0], msgs))
            notes.extend(n for n in NOTES.get(kind, ()) if n not in notes)
        if not channels:
            raise ValueError(
                f"{source}: no message the {profile.name!r} profile decodes "
                f"({list(profile.decodes)}) holds any message; topics: "
                f"{ {t: v['type'] for t, v in census_topics.items()} }"
            )
        recording = Recording(
            source=source.name,
            adapter=NAME,
            collection=COLLECTION_ROBOT_OP,
            channels=channels,
            census={
                "profile": profile.name,
                "robot": profile.robot,
                "joints": list(profile.joints),
                "topics": census_topics,
                "files": [f.name for f in files],
                "recorded": _recorded_at(source),
                **profile.census,
            },
            notes=notes,
            basis=profile.basis,
        )
        recording.census[quality.QUALITY_KEY] = quality.describe(recording)
        return recording


# -- the container: sqlite3 ----------------------------------------------------

STORE_FILE = "capture_0.db3"  # the one file a live capture writes
CREATE_TOPICS = (
    "CREATE TABLE IF NOT EXISTS topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
    "type TEXT NOT NULL, serialization_format TEXT NOT NULL, "
    "offered_qos_profiles TEXT NOT NULL)"
)
CREATE_MESSAGES = (
    "CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, topic_id INTEGER "
    "NOT NULL, timestamp INTEGER NOT NULL, data BLOB NOT NULL)"
)
METADATA_TEXT = (
    "rosbag2_bagfile_information:\n"
    "  version: 5\n"
    "  storage_identifier: sqlite3\n"
    "  starting_time:\n"
    "    nanoseconds_since_epoch: {start}\n"
)


class Store:
    """A rosbag2 sqlite3 store being WRITTEN: what a live capture appends
    CDR messages to, so the bag adapter above reads a capture exactly as
    it reads a bag (one decoder, the layouts as data). Not thread-safe:
    one writer thread owns it. `close` writes the `metadata.yaml` a bag
    carries, with the first message's receive time as the start."""

    def __init__(self, root: Path, types: dict[str, str] | None = None) -> None:
        self.root = Path(root)
        self.types = dict(TOPIC_TYPES if types is None else types)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / STORE_FILE)
        self._db.execute(CREATE_TOPICS)
        self._db.execute(CREATE_MESSAGES)
        self._ids: dict[str, int] = {}
        self._start_ns: int | None = None
        self.count = 0

    def topic(self, name: str) -> int:
        """The topic's id, declared on first use; a topic the store was not
        told the type of is refused by name (a bag names every type)."""
        if name not in self.types:
            raise ValueError(f"no message type declared for topic {name!r}")
        if name not in self._ids:
            kind = self.types[name]
            cur = self._db.execute(
                "INSERT INTO topics(name, type, serialization_format, "
                "offered_qos_profiles) VALUES (?, ?, ?, '')",
                (name, kind, CDR_FORMAT),
            )
            self._ids[name] = int(cur.lastrowid or 0)
        return self._ids[name]

    def append(self, rows: list[tuple[str, int, bytes]]) -> None:
        """(topic name, receive time ns since the epoch, CDR bytes) rows."""
        if not rows:
            return
        self._db.executemany(
            "INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)",
            [(self.topic(t), ns, data) for t, ns, data in rows],
        )
        self._db.commit()
        self.count += len(rows)
        if self._start_ns is None:
            self._start_ns = min(ns for _, ns, _ in rows)

    def close(self) -> Path:
        self._db.commit()
        self._db.close()
        (self.root / METADATA_FILE).write_text(
            METADATA_TEXT.format(start=self._start_ns or 0), encoding="utf-8"
        )
        return self.root


def _files(source: Path) -> list[Path]:
    """The `.db3` files of a bag directory (with its `metadata.yaml`) or
    the one file named; empty for anything else."""
    if source.is_dir():
        if not (source / METADATA_FILE).is_file():
            return []
        return sorted(source.glob(f"*{SUFFIX}"))
    if source.is_file() and source.suffix == SUFFIX:
        return [source]
    return []


def _topic_types(files: list[Path]) -> set[str]:
    """The message types the bag's `topics` table declares (a glance for
    `accepts`, never the messages themselves); empty for a file that is
    not a rosbag2 store."""
    kinds: set[str] = set()
    for path in files:
        uri = f"file:{path.as_posix()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as db:
                kinds.update(kind for (kind,) in db.execute("SELECT type FROM topics"))
        except sqlite3.DatabaseError:
            return set()
    return kinds


def _read_files(
    files: list[Path],
) -> tuple[dict[str, tuple[str, str]], dict[str, list[tuple[int, bytes]]]]:
    topics: dict[str, tuple[str, str]] = {}
    messages: dict[str, list[tuple[int, bytes]]] = {}
    for path in files:
        uri = f"file:{path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            by_id: dict[int, str] = {}
            for tid, name, kind, fmt in db.execute(
                "SELECT id, name, type, serialization_format FROM topics"
            ):
                by_id[tid] = name
                topics[name] = (kind, fmt)
            for tid, stamp, data in db.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
            ):
                messages.setdefault(by_id[tid], []).append((int(stamp), bytes(data)))
    return topics, messages


def _recorded_at(source: Path) -> str | None:
    """The bag's own start time from `metadata.yaml` (ISO 8601, UTC), read
    as lines — the one fact the messages do not carry themselves."""
    meta = source / METADATA_FILE if source.is_dir() else source.parent / METADATA_FILE
    if not meta.is_file():
        return None
    lines = meta.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "starting_time:" and i + 1 < len(lines):
            key, _, value = lines[i + 1].strip().partition(":")
            if key == "nanoseconds_since_epoch" and value.strip().isdigit():
                from datetime import datetime, timezone  # noqa: PLC0415

                seconds = int(value) / NS_PER_S
                stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
                return stamp.replace(microsecond=0).isoformat()
    return None


# -- messages to channels -----------------------------------------------------


def _channels(
    kind: str, topic: str, msgs: list[tuple[int, bytes]]
) -> dict[str, Channel]:
    layout = LAYOUTS[kind]
    extracts = CHANNELS[kind]
    times: list[float] = []
    columns: list[list[list[float]]] = [[] for _ in extracts]
    for stamp_ns, data in msgs:
        decoded = Reader(data).message(layout, LAYOUTS)
        times.append(stamp_ns / NS_PER_S)
        for rows, extract in zip(columns, extracts, strict=True):
            rows.append(extract.take(decoded))
    times_arr, keep = monotone(np.asarray(times, dtype=np.float64))
    out: dict[str, Channel] = {}
    for rows, extract in zip(columns, extracts, strict=True):
        values = np.asarray(rows, dtype=np.float64)[keep]
        out[extract.channel] = Channel(
            extract.channel,
            times_arr,
            values,
            unit=extract.unit,
            components=extract.components,
        )
    return out
