"""A recording: named channels of timestamped values, and their census.

This is the one shape every adapter produces and every consumer reads.
The identifier wants row-aligned arrays (`robot/identify.ExcitationData`);
the Studio wants something to plot; the stamp wants files. A recording is
all three: channels in memory, and on disk one `.npz` of arrays beside a
`recording.json` manifest that names the source, the adapter, the
robot's census, every channel with its unit and rate, and the duration.

Units are strings the way the fit records spell them (`rad`, `rad/s`,
`N*m`, `A`, `degC`, `m/s^2`), never inferred: an adapter that does not
know a channel's unit says `"unknown"`, and the identifier refuses to fit
a channel whose unit it cannot place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.bundles.basis import (  # noqa: F401 - re-exported: the adapters' home
    BASES,
    BASIS_OWN,
    BASIS_PUBLIC,
    BASIS_SIMULATION,
    BASIS_UNKNOWN,
)
from rq_pipeline.bundles.json_record import JsonRecord


def monotone(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only samples whose time is finite and exceeds every earlier
    kept time — a duplicate, out-of-order or missing stamp is dropped,
    never re-invented. Returns the kept times and the keep mask, for
    every adapter that reads a clock."""
    keep = np.zeros(len(times), dtype=bool)
    last = -np.inf
    for i, t in enumerate(times):
        if np.isfinite(t) and t > last:
            keep[i] = True
            last = t
    return times[keep], keep


MANIFEST_FILE = "recording.json"
SIGNALS_FILE = "signals.npz"
SCHEMA = "trainnr-recording/1"
UNKNOWN_UNIT = "unknown"

# The channel names an identifier looks for, spelled once. An adapter maps
# its source's names onto these where it can; anything else keeps the
# source's own name and is carried, not dropped.
JOINT_POSITION = "joint.position"
JOINT_VELOCITY = "joint.velocity"
JOINT_ACCELERATION = "joint.acceleration"
JOINT_EFFORT = "joint.effort"
JOINT_COMMAND = "joint.command"  # the commanded joint position
JOINT_COMMAND_VELOCITY = "joint.command_velocity"
JOINT_FEEDFORWARD = "joint.feedforward"  # the commanded feed-forward torque
JOINT_KP = "joint.kp"  # the motor-side PD gains the command ran under
JOINT_KD = "joint.kd"
IMU_ANGULAR_VELOCITY = "imu.angular_velocity"
IMU_LINEAR_ACCELERATION = "imu.linear_acceleration"
IMU_ORIENTATION = "imu.orientation"
# A floating base, in MuJoCo's free-joint conventions: pose = xyz + wxyz
# quaternion; twist and acceleration = linear (world frame) + angular
# (body frame). An adapter whose source uses another convention
# converts before it names these.
BASE_POSE = "base.pose"
BASE_TWIST = "base.twist"
BASE_ACCELERATION = "base.acceleration"
FOOT_CONTACT = "foot.contact"  # one column per foot, 1.0 in contact


@dataclass(frozen=True)
class Channel:
    """One named signal: times (seconds, strictly increasing) and values
    (one row per time; columns are the signal's components, named)."""

    name: str
    times: np.ndarray
    values: np.ndarray
    unit: str = UNKNOWN_UNIT
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.times.ndim != 1:
            raise ValueError(f"channel {self.name!r}: times must be 1-D")
        if len(self.times) != len(self.values):
            raise ValueError(
                f"channel {self.name!r}: {len(self.times)} times vs "
                f"{len(self.values)} values"
            )
        if len(self.times) > 1 and not np.all(np.diff(self.times) > 0):
            raise ValueError(
                f"channel {self.name!r}: times must be strictly increasing"
            )
        width = self.values.shape[1] if self.values.ndim > 1 else 1
        if self.components and len(self.components) != width:
            raise ValueError(
                f"channel {self.name!r}: {len(self.components)} component names "
                f"for {width} columns"
            )

    @property
    def rate_hz(self) -> float | None:
        """The mean sample rate, or None for fewer than two samples."""
        if len(self.times) < 2:  # noqa: PLR2004 - a rate needs two samples
            return None
        span = float(self.times[-1] - self.times[0])
        return (len(self.times) - 1) / span if span > 0 else None

    @property
    def width(self) -> int:
        return self.values.shape[1] if self.values.ndim > 1 else 1


@dataclass(frozen=True)
class ChannelInfo(JsonRecord):
    """What the manifest says about one channel."""

    name: str
    unit: str
    samples: int
    width: int
    rate_hz: float | None
    components: list[str] = field(default_factory=list)


# How a recording was COLLECTED — the data-collection source (docs/76 §5).
# An adapter sets the one it knows; a caller may override (a LeRobot
# dataset pressed in sim is "scripted", not "teleop").
COLLECTION_TELEOP = "teleop"  # a human drove a leader device
COLLECTION_ROBOT_OP = "robot-operation"  # the robot ran (operator, script or policy)
COLLECTION_MOCAP = "mocap"  # optical or inertial motion capture of a human
COLLECTION_WEARABLE = "wearable"  # IMUs, gloves, suits worn by a human (designed)
COLLECTION_SCRIPTED = "scripted"  # a scripted expert in sim
COLLECTION_UNKNOWN = "unknown"
COLLECTIONS = (
    COLLECTION_TELEOP,
    COLLECTION_ROBOT_OP,
    COLLECTION_MOCAP,
    COLLECTION_WEARABLE,
    COLLECTION_SCRIPTED,
    COLLECTION_UNKNOWN,
)


@dataclass(frozen=True)
class RecordingManifest(JsonRecord):
    """`recording.json`: where this came from and what it holds."""

    schema: str
    source: str  # the source's NAME (a file's name, a topic set), never a path
    adapter: str
    duration_s: float
    channels: list[ChannelInfo]
    census: dict[str, Any] = field(default_factory=dict)  # what the robot reported
    notes: list[str] = field(default_factory=list)  # honest caveats, per adapter
    collection: str = COLLECTION_UNKNOWN  # one of COLLECTIONS
    basis: str = BASIS_UNKNOWN  # one of BASES: whose robot


@dataclass(frozen=True)
class Recording:
    """The in-memory recording an adapter returns."""

    source: str
    adapter: str
    channels: dict[str, Channel]
    census: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    collection: str = COLLECTION_UNKNOWN
    basis: str = BASIS_UNKNOWN

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, got {self.basis!r}")

    @property
    def duration_s(self) -> float:
        starts = [float(c.times[0]) for c in self.channels.values() if len(c.times)]
        ends = [float(c.times[-1]) for c in self.channels.values() if len(c.times)]
        return (max(ends) - min(starts)) if starts else 0.0

    def manifest(self) -> RecordingManifest:
        return RecordingManifest(
            schema=SCHEMA,
            source=self.source,
            adapter=self.adapter,
            duration_s=self.duration_s,
            channels=[
                ChannelInfo(
                    name=c.name,
                    unit=c.unit,
                    samples=len(c.times),
                    width=c.width,
                    rate_hz=c.rate_hz,
                    components=list(c.components),
                )
                for c in self.channels.values()
            ],
            census=dict(self.census),
            notes=list(self.notes),
            collection=self.collection,
            basis=self.basis,
        )

    def write(self, out: Path) -> Path:
        """The recording as an artifact directory: manifest + one `.npz`
        with `<channel>.times` and `<channel>.values` arrays."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        # Any: numpy's stub types **kwargs against savez's own keywords.
        arrays: dict[str, Any] = {}
        for name, channel in self.channels.items():
            arrays[f"{name}.times"] = channel.times
            arrays[f"{name}.values"] = channel.values
        np.savez_compressed(out / SIGNALS_FILE, **arrays)
        self.manifest().write(out / MANIFEST_FILE)
        return out

    @classmethod
    def read(cls, root: Path) -> Recording:
        root = Path(root)
        manifest = RecordingManifest.read(root / MANIFEST_FILE)
        if manifest.schema != SCHEMA:
            raise ValueError(
                f"{root}: schema {manifest.schema!r}, this code speaks {SCHEMA!r}"
            )
        # JsonRecord.read rebuilds the outer record only; the nested
        # channel infos arrive as dicts and are rebuilt here.
        infos = [
            c if isinstance(c, ChannelInfo) else ChannelInfo(**c)
            for c in manifest.channels
        ]
        with np.load(root / SIGNALS_FILE) as arrays:
            channels = {
                info.name: Channel(
                    name=info.name,
                    times=arrays[f"{info.name}.times"],
                    values=arrays[f"{info.name}.values"],
                    unit=info.unit,
                    components=tuple(info.components),
                )
                for info in infos
            }
        return cls(
            source=manifest.source,
            adapter=manifest.adapter,
            channels=channels,
            census=dict(manifest.census),
            notes=list(manifest.notes),
            collection=manifest.collection,
            basis=manifest.basis,
        )

    def excitation(
        self, *, controls: str, measurements: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two channels as the identifier's row-aligned arrays: the
        measurement channel's clock, with the control channel held at
        its last value at each measurement time (zero-order hold — the
        way a controller's command is actually applied)."""
        control = self.channels[controls]
        measured = self.channels[measurements]
        idx = np.searchsorted(control.times, measured.times, side="right") - 1
        idx = np.clip(idx, 0, len(control.times) - 1)
        held = control.values[idx]
        return measured.times, held, measured.values
