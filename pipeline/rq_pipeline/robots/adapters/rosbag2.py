"""A ROS 2 bag in rosbag2's sqlite3 form as a recording — Unitree's own
low-level state, read with no ROS installed.

rosbag2's original storage (the default up to Humble, 2022, and what
Unitree's `unitree_ros2` tutorials record) is a directory: `metadata.yaml`
naming the topics and their types, and one or more `<name>_<n>.db3`
SQLite files with a `topics` table and a `messages` table of CDR bytes.
The standard library reads both (`sqlite3`, and the four YAML lines this
needs, read as lines); the messages are decoded through `robots.cdr` by
LAYOUTS copied from Unitree's `.msg` files (`unitree_go/msg/LowState`,
`SportModeState` and the messages they nest). A third vendor's message
is another layout entry and another `CHANNELS` table, never a decoder.

What becomes a channel, and in what unit, is the `CHANNELS` table: the
12 leg motors of the 20 slots a `LowState` carries (Unitree's leg order,
`GO2_MOTORS`), the IMU, the foot-force sensors, the battery. `tau_est`
is what Unitree calls it — an estimate from motor current, not a
measurement — and the channel's note says so. Clock: rosbag2 stamps a
message when it was received (`timestamp`, ns since the epoch); a
`LowState` carries no header stamp, so that receive time is the
recording's clock, and the census reports the rate and jitter MEASURED
from it, never the 500 Hz the SDK promises.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.robots import quality
from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.cdr import NS_PER_S, Field, Layout, Reader
from rq_pipeline.robots.recording import (
    COLLECTION_ROBOT_OP,
    IMU_ANGULAR_VELOCITY,
    IMU_LINEAR_ACCELERATION,
    IMU_ORIENTATION,
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

# -- Unitree's messages, as data (unitree_ros2, cyclonedds_ws/src/unitree/
# unitree_go/msg, read 2026-09-24) --------------------------------------------

LOW_STATE = "unitree_go/msg/LowState"
SPORT_MODE_STATE = "unitree_go/msg/SportModeState"
LOW_CMD = "unitree_go/msg/LowCmd"
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
    "TimeSpec": (Field("sec", "int32"), Field("nanosec", "uint32")),
    "MotorCmd": (
        Field("mode", "uint8"),
        Field("q", "float32"),
        Field("dq", "float32"),
        Field("tau", "float32"),
        Field("kp", "float32"),
        Field("kd", "float32"),
        Field("reserve", "uint32", 3),
    ),
    "BmsCmd": (Field("off", "uint8"), Field("reserve", "uint8", 3)),
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
}

MOTOR_SLOTS = 20  # every LowState carries 20; a Go2 fills 12
# Unitree's leg order: front-right, front-left, rear-right, rear-left;
# hip (abduction), thigh, calf — the order `unitree_go` and the SDK's
# `LegID` use.
GO2_MOTORS = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)  # fmt: skip
FEET = ("FR", "FL", "RR", "RL")
XYZ = ("x", "y", "z")
QUATERNION_WXYZ = ("w", "x", "y", "z")  # Unitree's order, unlike sensor_msgs
NOTE_TAU_EST = (
    "joint.effort is Unitree's tau_est: estimated from motor current by the "
    "motor driver, not measured by a torque sensor"
)
NOTE_CLOCK = "time is rosbag2's receive timestamp; LowState carries no header stamp"
NOTE_FOOT_FORCE = "foot.force is the sensor's raw count; Unitree publishes no unit"
NOTE_LOW_CMD = (
    "joint.command is what the controller sent on rt/lowcmd: the target, the "
    "feed-forward torque and the motor-side kp/kd the motor's own loop ran; "
    "paired with rt/lowstate by receive time"
)


@dataclass(frozen=True)
class Extract:
    """One channel from a decoded message: where its columns come from."""

    channel: str
    unit: str
    components: tuple[str, ...]
    take: Any  # decoded message dict -> list[float]


def _motors(field: str) -> Any:
    return lambda m: [m["motor_state"][i][field] for i in range(len(GO2_MOTORS))]


def _motor_cmds(field: str) -> Any:
    return lambda m: [m["motor_cmd"][i][field] for i in range(len(GO2_MOTORS))]


def _imu(field: str) -> Any:
    return lambda m: list(m["imu_state"][field])


def _top(field: str) -> Any:
    return lambda m: list(m[field])


def _scalar(*path: str) -> Any:
    def take(m: dict[str, Any]) -> list[float]:
        value: Any = m
        for key in path:
            value = value[key]
        return [float(value)]

    return take


# What each message type yields, in the pipeline's channel names where a
# name exists (the identifier's contract) and the source's own where not.
CHANNELS: dict[str, tuple[Extract, ...]] = {
    LOW_STATE: (
        Extract(JOINT_POSITION, "rad", GO2_MOTORS, _motors("q")),
        Extract(JOINT_VELOCITY, "rad/s", GO2_MOTORS, _motors("dq")),
        Extract(JOINT_EFFORT, "N*m", GO2_MOTORS, _motors("tau_est")),
        Extract("joint.acceleration", "rad/s^2", GO2_MOTORS, _motors("ddq")),
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
        Extract(JOINT_COMMAND, "rad", GO2_MOTORS, _motor_cmds("q")),
        Extract(JOINT_COMMAND_VELOCITY, "rad/s", GO2_MOTORS, _motor_cmds("dq")),
        Extract(JOINT_FEEDFORWARD, "N*m", GO2_MOTORS, _motor_cmds("tau")),
        Extract(JOINT_KP, "N*m/rad", GO2_MOTORS, _motor_cmds("kp")),
        Extract(JOINT_KD, "N*m*s/rad", GO2_MOTORS, _motor_cmds("kd")),
        Extract("motor.command_mode", "enum", GO2_MOTORS, _motor_cmds("mode")),
    ),
    SPORT_MODE_STATE: (
        Extract("sport.position", "m", XYZ, _top("position")),
        Extract("sport.velocity", "m/s", XYZ, _top("velocity")),
        Extract("sport.yaw_speed", "rad/s", ("yaw",), _scalar("yaw_speed")),
        Extract("sport.body_height", "m", ("height",), _scalar("body_height")),
        Extract("sport.mode", "enum", ("mode",), _scalar("mode")),
        Extract("sport.gait", "enum", ("gait",), _scalar("gait_type")),
    ),
}  # fmt: skip
NOTES: dict[str, tuple[str, ...]] = {
    LOW_STATE: (NOTE_CLOCK, NOTE_TAU_EST, NOTE_FOOT_FORCE),
    LOW_CMD: (NOTE_LOW_CMD,),
    SPORT_MODE_STATE: (),
}


DOC = "A ROS 2 bag (rosbag2 sqlite3): Unitree LowState as channels, no ROS needed"


@adapter(NAME, doc=DOC)
class Rosbag2Adapter:
    """A ROS 2 bag (rosbag2 sqlite3): Unitree LowState as channels, no ROS needed."""

    name = NAME

    def accepts(self, source: Path) -> bool:
        """A rosbag2 store that carries a message type this adapter has a
        layout for. Content, not suffix: another adapter reads the same
        container for other robots' topics (`robot/rosbag_sqlite.py`),
        and two adapters claiming every `.db3` would refuse them all."""
        source = Path(source)
        if source.is_dir():
            if not (source / METADATA_FILE).is_file():
                return False
            files = sorted(source.glob(f"*{SUFFIX}"))
        elif source.is_file() and source.suffix == SUFFIX:
            files = [source]
        else:
            return False
        return bool(files) and any(kind in LAYOUTS for kind in _topic_types(files))

    def read(self, source: Path) -> Recording:
        source = Path(source)
        files = sorted(source.glob(f"*{SUFFIX}")) if source.is_dir() else [source]
        if not files:
            raise ValueError(f"{source}: no {SUFFIX} file")
        topics, messages = _read_files(files)
        channels: dict[str, Channel] = {}
        notes: list[str] = []
        census_topics: dict[str, dict[str, Any]] = {}
        for topic, (kind, fmt) in topics.items():
            msgs = messages.get(topic, [])
            census_topics[topic] = {"type": kind, "messages": len(msgs)}
            if not msgs or fmt != CDR_FORMAT or kind not in LAYOUTS:
                continue
            channels.update(_channels(kind, topic, msgs))
            notes.extend(n for n in NOTES[kind] if n not in notes)
        if not channels:
            raise ValueError(
                f"{source}: no message this adapter decodes ({sorted(CHANNELS)}); "
                f"topics: { {t: v['type'] for t, v in census_topics.items()} }"
            )
        recording = Recording(
            source=source.name,
            adapter=NAME,
            collection=COLLECTION_ROBOT_OP,
            channels=channels,
            census={
                "topics": census_topics,
                "files": [f.name for f in files],
                "motor_slots": MOTOR_SLOTS,
                "motors_read": len(GO2_MOTORS),
                "recorded": _recorded_at(source),
            },
            notes=notes,
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
