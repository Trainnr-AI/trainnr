"""A ROS 2 bag in MCAP form as a recording — read with no ROS installed.

MCAP is rosbag2's default storage since the Iron release (2023-05-23):
a self-describing container whose Schema records carry the `.msg`
definitions, so the file explains its own messages. This module reads
the container (magic, records by opcode, chunks) and decodes the two
messages the pipeline needs from their CDR bytes: `sensor_msgs/msg/
JointState` (name[], position[], velocity[], effort[] — effort "in Nm or
N" per the message definition) and `sensor_msgs/msg/Imu` (orientation,
angular_velocity, linear_acceleration). Anything else on the bag is
counted in the census and left alone.

Standard library only, by design: the point of the seam is that a
recording enters with nothing installed. The one limit that follows:
rosbag2 compresses chunks with zstd (its default) or lz4, and
decompressing needs a library. Uncompressed chunks and unchunked bags
read directly; a compressed bag is refused BY NAME with the one-line
recipe (`ros2 bag convert`, or record with `--compression-mode none`).
`zstandard` and `lz4` are used when importable, so installing either
lifts the limit without a code change.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.recording import (
    COLLECTION_ROBOT_OP,
    IMU_ANGULAR_VELOCITY,
    IMU_LINEAR_ACCELERATION,
    IMU_ORIENTATION,
    JOINT_EFFORT,
    JOINT_POSITION,
    JOINT_VELOCITY,
    Channel,
    Recording,
    monotone,
)

NAME = "mcap"
SUFFIX = ".mcap"
MAGIC = b"\x89MCAP0\r\n"

# Record opcodes (mcap.dev/spec).
OP_HEADER = 0x01
OP_FOOTER = 0x02
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_DATA_END = 0x0F

JOINT_STATE = "sensor_msgs/msg/JointState"
IMU = "sensor_msgs/msg/Imu"
CDR_ENCODING = "cdr"
NS_PER_S = 1e9


class UnsupportedBagError(ValueError):
    """The bag needs something this reader does not have — named."""


@dataclass
class _Schema:
    id: int
    name: str
    encoding: str


@dataclass
class _Channel:
    id: int
    schema_id: int
    topic: str
    encoding: str


@dataclass
class _Bag:
    schemas: dict[int, _Schema] = field(default_factory=dict)
    channels: dict[int, _Channel] = field(default_factory=dict)
    # channel id -> list of (log_time_ns, data)
    messages: dict[int, list[tuple[int, bytes]]] = field(default_factory=dict)
    compressed_chunks: set[str] = field(default_factory=set)


@adapter(NAME, doc="A ROS 2 bag (MCAP): JointState and Imu as channels, no ROS needed")
class McapAdapter:
    """A ROS 2 bag (MCAP): JointState and Imu as channels, no ROS needed."""

    name = NAME

    def accepts(self, source: Path) -> bool:
        source = Path(source)
        if source.is_dir():
            return any(source.glob(f"*{SUFFIX}"))
        return source.is_file() and source.suffix == SUFFIX

    def read(self, source: Path) -> Recording:
        source = Path(source)
        files = sorted(source.glob(f"*{SUFFIX}")) if source.is_dir() else [source]
        if not files:
            raise ValueError(f"{source}: no {SUFFIX} file")
        bag = _Bag()
        for path in files:
            _read_file(path, bag)
        if bag.compressed_chunks and not bag.messages:
            codecs = sorted(bag.compressed_chunks)
            raise UnsupportedBagError(
                f"{source}: chunks compressed with {codecs} and no decompressor is "
                "installed — `pip install zstandard` (or lz4), or convert the bag "
                "with `ros2 bag convert` / record with `--compression-mode none`"
            )
        channels: dict[str, Channel] = {}
        topics: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        for channel in bag.channels.values():
            schema = bag.schemas.get(channel.schema_id)
            kind = schema.name if schema else "unknown"
            msgs = bag.messages.get(channel.id, [])
            topics[channel.topic] = {"type": kind, "messages": len(msgs)}
            if not msgs or channel.encoding != CDR_ENCODING:
                continue
            if kind == JOINT_STATE:
                channels.update(_joint_state_channels(channel.topic, msgs, notes))
            elif kind == IMU:
                channels.update(_imu_channels(channel.topic, msgs))
        if not channels:
            raise ValueError(
                f"{source}: no {JOINT_STATE} or {IMU} messages decoded; topics: "
                f"{ {t: v['type'] for t, v in topics.items()} }"
            )
        if bag.compressed_chunks:
            notes.append(
                f"some chunks were compressed with {sorted(bag.compressed_chunks)} and "
                "skipped (no decompressor installed)"
            )
        return Recording(
            source=source.name,
            adapter=NAME,
            collection=COLLECTION_ROBOT_OP,
            channels=channels,
            census={"topics": topics, "files": [f.name for f in files]},
            notes=notes,
        )


# -- the container ----------------------------------------------------------


def _read_file(path: Path, bag: _Bag) -> None:
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        raise ValueError(f"{path}: not an MCAP file (bad magic)")
    _read_records(memoryview(data)[len(MAGIC) :], bag, top_level=True)


def _read_records(buf: memoryview, bag: _Bag, *, top_level: bool) -> None:
    pos = 0
    end = len(buf)
    while pos + 9 <= end:
        op = buf[pos]
        (length,) = struct.unpack_from("<Q", buf, pos + 1)
        body = buf[pos + 9 : pos + 9 + length]
        pos += 9 + length
        if op == OP_SCHEMA:
            sid, name, encoding, _ = _schema(body)
            bag.schemas[sid] = _Schema(sid, name, encoding)
        elif op == OP_CHANNEL:
            cid, sid, topic, encoding = _channel(body)
            bag.channels[cid] = _Channel(cid, sid, topic, encoding)
        elif op == OP_MESSAGE:
            cid, log_time, payload = _message(body)
            bag.messages.setdefault(cid, []).append((log_time, payload))
        elif op == OP_CHUNK and top_level:
            _chunk(body, bag)
        elif op in (OP_DATA_END, OP_FOOTER) and top_level:
            # The summary section that follows repeats schemas/channels
            # and adds indexes; nothing new for a reader that streamed.
            break


def _string(buf: memoryview, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, pos)
    return bytes(buf[pos + 4 : pos + 4 + n]).decode("utf-8"), pos + 4 + n


def _schema(body: memoryview) -> tuple[int, str, str, bytes]:
    (sid,) = struct.unpack_from("<H", body, 0)
    name, pos = _string(body, 2)
    encoding, pos = _string(body, pos)
    (n,) = struct.unpack_from("<I", body, pos)
    return sid, name, encoding, bytes(body[pos + 4 : pos + 4 + n])


def _channel(body: memoryview) -> tuple[int, int, str, str]:
    cid, sid = struct.unpack_from("<HH", body, 0)
    topic, pos = _string(body, 4)
    encoding, _ = _string(body, pos)
    return cid, sid, topic, encoding


def _message(body: memoryview) -> tuple[int, int, bytes]:
    cid, _seq, log_time, _pub = struct.unpack_from("<HIQQ", body, 0)
    return cid, log_time, bytes(body[22:])


def _chunk(body: memoryview, bag: _Bag) -> None:
    # message_start_time u64, message_end_time u64, uncompressed_size u64,
    # uncompressed_crc u32, compression string, records_size u64, records
    pos = 8 + 8 + 8 + 4
    compression, pos = _string(body, pos)
    (size,) = struct.unpack_from("<Q", body, pos)
    records = body[pos + 8 : pos + 8 + size]
    if compression == "":
        _read_records(records, bag, top_level=False)
        return
    raw = _decompress(compression, bytes(records))
    if raw is None:
        bag.compressed_chunks.add(compression)
        return
    _read_records(memoryview(raw), bag, top_level=False)


def _decompress(compression: str, data: bytes) -> bytes | None:
    try:
        if compression == "zstd":
            import zstandard  # noqa: PLC0415

            return zstandard.ZstdDecompressor().decompressobj().decompress(data)
        if compression == "lz4":
            import lz4.frame  # noqa: PLC0415

            return bytes(lz4.frame.decompress(data))
    except ImportError:
        return None
    return None


# -- CDR decoding of the two messages ---------------------------------------


class _Cdr:
    """A little-endian CDR reader over one message's bytes."""

    def __init__(self, data: bytes) -> None:
        # 4-byte encapsulation header: representation id + options.
        self.data = data
        self.pos = 4

    def align(self, n: int) -> None:
        # CDR aligns relative to the start of the body (after the header).
        rel = self.pos - 4
        self.pos += (-rel) % n

    def u32(self) -> int:
        self.align(4)
        (v,) = struct.unpack_from("<I", self.data, self.pos)
        self.pos += 4
        return v

    def i32(self) -> int:
        self.align(4)
        (v,) = struct.unpack_from("<i", self.data, self.pos)
        self.pos += 4
        return v

    def f64(self) -> float:
        self.align(8)
        (v,) = struct.unpack_from("<d", self.data, self.pos)
        self.pos += 8
        return v

    def string(self) -> str:
        n = self.u32()
        s = self.data[self.pos : self.pos + n - 1].decode("utf-8") if n else ""
        self.pos += n
        return s

    def f64_seq(self) -> list[float]:
        n = self.u32()
        return [self.f64() for _ in range(n)]

    def string_seq(self) -> list[str]:
        n = self.u32()
        return [self.string() for _ in range(n)]

    def header(self) -> float:
        """std_msgs/Header: stamp (sec i32, nanosec u32), frame_id."""
        sec = self.i32()
        nsec = self.u32()
        self.string()
        return sec + nsec / NS_PER_S


def _joint_state_channels(
    topic: str, msgs: list[tuple[int, bytes]], notes: list[str]
) -> dict[str, Channel]:
    times: list[float] = []
    names: tuple[str, ...] | None = None
    pos_rows: list[list[float]] = []
    vel_rows: list[list[float]] = []
    eff_rows: list[list[float]] = []
    for log_time, data in msgs:
        r = _Cdr(data)
        stamp = r.header()
        t = stamp if stamp > 0 else log_time / NS_PER_S
        joint_names = tuple(r.string_seq())
        position = r.f64_seq()
        velocity = r.f64_seq()
        effort = r.f64_seq()
        if names is None:
            names = joint_names
        elif joint_names != names:
            # Joint order can differ between publishers; align by name.
            order = [joint_names.index(n) if n in joint_names else -1 for n in names]
            position = [
                position[i] if i >= 0 and i < len(position) else np.nan for i in order
            ]
            velocity = [
                velocity[i] if i >= 0 and i < len(velocity) else np.nan for i in order
            ]
            effort = [
                effort[i] if i >= 0 and i < len(effort) else np.nan for i in order
            ]
        width = len(names)
        times.append(t)
        pos_rows.append(_pad(position, width))
        vel_rows.append(_pad(velocity, width))
        eff_rows.append(_pad(effort, width))
    if names is None:
        return {}
    times_arr, keep = monotone(np.asarray(times, dtype=np.float64))
    prefix = (
        "" if topic in ("/joint_states", "joint_states") else f"{topic.strip('/')}."
    )
    out: dict[str, Channel] = {
        f"{prefix}{JOINT_POSITION}": Channel(
            f"{prefix}{JOINT_POSITION}",
            times_arr,
            np.asarray(pos_rows, dtype=np.float64)[keep],
            unit="rad or m (JointState: rad for revolute, m for prismatic)",
            components=names,
        )
    }
    vel = np.asarray(vel_rows, dtype=np.float64)[keep]
    if not np.all(np.isnan(vel)):
        out[f"{prefix}{JOINT_VELOCITY}"] = Channel(
            f"{prefix}{JOINT_VELOCITY}",
            times_arr,
            vel,
            unit="rad/s or m/s",
            components=names,
        )
    eff = np.asarray(eff_rows, dtype=np.float64)[keep]
    if not np.all(np.isnan(eff)):
        out[f"{prefix}{JOINT_EFFORT}"] = Channel(
            f"{prefix}{JOINT_EFFORT}", times_arr, eff, unit="N*m or N", components=names
        )
    else:
        notes.append(f"{topic}: effort array empty — no torque in this bag")
    return out


def _imu_channels(topic: str, msgs: list[tuple[int, bytes]]) -> dict[str, Channel]:
    times: list[float] = []
    quat: list[list[float]] = []
    gyro: list[list[float]] = []
    accel: list[list[float]] = []
    for log_time, data in msgs:
        r = _Cdr(data)
        stamp = r.header()
        times.append(stamp if stamp > 0 else log_time / NS_PER_S)
        quat.append([r.f64() for _ in range(4)])  # x y z w
        for _ in range(9):
            r.f64()  # orientation covariance
        gyro.append([r.f64() for _ in range(3)])
        for _ in range(9):
            r.f64()
        accel.append([r.f64() for _ in range(3)])
    times_arr, keep = monotone(np.asarray(times, dtype=np.float64))
    prefix = "" if topic in ("/imu", "imu", "/imu/data") else f"{topic.strip('/')}."
    return {
        f"{prefix}{IMU_ORIENTATION}": Channel(
            f"{prefix}{IMU_ORIENTATION}",
            times_arr,
            np.asarray(quat)[keep],
            unit="quaternion",
            components=("x", "y", "z", "w"),
        ),
        f"{prefix}{IMU_ANGULAR_VELOCITY}": Channel(
            f"{prefix}{IMU_ANGULAR_VELOCITY}",
            times_arr,
            np.asarray(gyro)[keep],
            unit="rad/s",
            components=("x", "y", "z"),
        ),
        f"{prefix}{IMU_LINEAR_ACCELERATION}": Channel(
            f"{prefix}{IMU_LINEAR_ACCELERATION}",
            times_arr,
            np.asarray(accel)[keep],
            unit="m/s^2",
            components=("x", "y", "z"),
        ),
    }


def _pad(values: list[float], width: int) -> list[float]:
    return list(values[:width]) + [np.nan] * max(0, width - len(values))
