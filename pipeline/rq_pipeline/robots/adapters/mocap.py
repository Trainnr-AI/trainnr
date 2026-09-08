"""Motion capture as a recording: marker or joint CSV, BVH skeletons.

The data-collection source that starts from a HUMAN, not a robot: an
optical mocap export (a headed CSV of marker or joint positions over
time, the shape OptiTrack, Vicon and Rokoko all export) or a BVH file (a
skeleton hierarchy plus per-frame channel values, the format every
animation tool speaks). Both become the recording shape — channels named
by marker or joint, positions in metres, rotations in radians — with the
capture rate and the marker set in the census.

What this adapter does NOT do, said plainly: retarget to a robot. A human
skeleton is not a robot's kinematics; the mapping is designed (docs/76
§5, the mocap lane from the Unitree review, docs/e2e-research/65 §4) and
built when a robot needs it. The honesty check that review named — replay
the retargeted motion through the robot's MuJoCo model and assert it
reproduces the commanded motion — belongs to the retargeting step, so it
is not claimed here.

Units: BVH positions are in the file's own units (usually centimetres
from Blender and Maya, sometimes metres) and rotations in degrees; both
are converted and the manifest says what was assumed. CSV columns are
taken in metres unless the header says `mm`.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.recording import COLLECTION_MOCAP, Channel, Recording

NAME = "mocap"
CSV_SUFFIX = ".csv"
BVH_SUFFIX = ".bvh"
# Column-name patterns a mocap CSV uses for the clock.
TIME_COLUMNS = ("time", "time (seconds)", "timestamp", "t", "frame_time")
FRAME_COLUMNS = ("frame", "frame_index", "#")
# Marker columns come as `<name>.x` / `<name>_X` / `<name> X`.
COMPONENT_RE = re.compile(r"^(?P<name>.+?)[._ ](?P<axis>[xyzXYZ]|rx|ry|rz|RX|RY|RZ)$")
MM_HINTS = ("mm", "millimet")
DEFAULT_RATE_HZ = 120.0
MIN_MARKER_COLUMNS = 3  # x, y, z of at least one marker
MIN_FRAMES = 2  # motion needs two frames
FRAME_TABLE_NDIM = 2  # frames x channels
CM_TO_M = 0.01
BVH_DEFAULT_UNIT = "cm (assumed; BVH does not state units)"


@adapter(NAME, doc="Motion capture: marker/joint CSV or BVH skeleton, human motion in")
class MocapAdapter:
    """Motion capture: marker/joint CSV or BVH skeleton, human motion in."""

    name = NAME

    def accepts(self, source: Path) -> bool:
        source = Path(source)
        if not source.is_file():
            return False
        if source.suffix.lower() == BVH_SUFFIX:
            return True
        if source.suffix.lower() == CSV_SUFFIX:
            # A CSV is mocap when its header has marker-shaped columns;
            # a robot log CSV (time, controls, measurements) is not ours.
            with source.open(newline="", errors="replace") as handle:
                header = handle.readline()
            marker_columns = sum(
                1 for c in header.split(",") if COMPONENT_RE.match(c.strip())
            )
            return marker_columns >= MIN_MARKER_COLUMNS
        return False

    def read(self, source: Path) -> Recording:
        source = Path(source)
        if not self.accepts(source):
            raise ValueError(f"{source} is not a mocap CSV or BVH file")
        if source.suffix.lower() == BVH_SUFFIX:
            return _read_bvh(source)
        return _read_csv(source)


# -- CSV -----------------------------------------------------------------------


def _read_csv(source: Path) -> Recording:
    with source.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: no header row")
        fields = [f.strip() for f in reader.fieldnames]
        rows = list(reader)
    if len(rows) < MIN_FRAMES:
        raise ValueError(f"{source}: need at least two frames")
    lower = {f.lower(): f for f in fields}
    time_col = next((lower[c] for c in TIME_COLUMNS if c in lower), None)
    frame_col = next((lower[c] for c in FRAME_COLUMNS if c in lower), None)
    notes: list[str] = []
    if time_col is not None:
        times = _column(rows, time_col, source)
    elif frame_col is not None:
        times = _column(rows, frame_col, source) / DEFAULT_RATE_HZ
        notes.append(
            f"no time column; frames taken at {DEFAULT_RATE_HZ:g} Hz (assumed)"
        )
    else:
        times = np.arange(len(rows), dtype=np.float64) / DEFAULT_RATE_HZ
        notes.append(
            f"no time or frame column; rows taken at {DEFAULT_RATE_HZ:g} Hz (assumed)"
        )
    times, keep = _monotone(times)
    scale = (
        CM_TO_M * 10 if any(h in " ".join(fields).lower() for h in MM_HINTS) else 1.0
    )
    if scale != 1.0:
        notes.append("header names millimetres; positions scaled to metres")
    # Group `<name>.<axis>` columns into one channel per marker/joint.
    groups: dict[str, dict[str, str]] = {}
    for field in fields:
        m = COMPONENT_RE.match(field)
        if m:
            groups.setdefault(m.group("name"), {})[m.group("axis").lower()] = field
    channels: dict[str, Channel] = {}
    for name, axes in groups.items():
        pos_axes = [a for a in ("x", "y", "z") if a in axes]
        rot_axes = [a for a in ("rx", "ry", "rz") if a in axes]
        if pos_axes:
            values = (
                np.column_stack([_column(rows, axes[a], source) for a in pos_axes])[
                    keep
                ]
                * scale
            )
            channels[f"marker.{name}.position"] = Channel(
                f"marker.{name}.position",
                times,
                values,
                unit="m",
                components=tuple(pos_axes),
            )
        if rot_axes:
            values = np.deg2rad(
                np.column_stack([_column(rows, axes[a], source) for a in rot_axes])[
                    keep
                ]
            )
            channels[f"marker.{name}.rotation"] = Channel(
                f"marker.{name}.rotation",
                times,
                values,
                unit="rad",
                components=tuple(rot_axes),
            )
    if not channels:
        raise ValueError(f"{source}: no marker columns (`<name>.x` and friends) found")
    return Recording(
        source=source.name,
        adapter=NAME,
        collection=COLLECTION_MOCAP,
        channels=channels,
        census={
            "markers": sorted(groups),
            "frames": int(keep.sum()),
            "rate_hz": _rate(times),
            "format": "csv",
        },
        notes=[*notes, "human motion; not retargeted to any robot (docs/76 §5)"],
    )


def _column(rows: list[dict[str, str]], name: str, source: Path) -> np.ndarray:
    try:
        return np.array(
            [float(r[name]) if r[name] not in ("", None) else np.nan for r in rows]
        )
    except (ValueError, KeyError) as error:
        raise ValueError(f"{source}: bad value in column {name!r}") from error


# -- BVH -----------------------------------------------------------------------


def _read_bvh(source: Path) -> Recording:
    text = source.read_text(errors="replace")
    if "HIERARCHY" not in text or "MOTION" not in text:
        raise ValueError(f"{source}: not a BVH file (no HIERARCHY/MOTION)")
    hierarchy, motion = text.split("MOTION", 1)
    joints = _bvh_joints(hierarchy)
    values, dt = _bvh_frames(source, motion, joints)
    times = np.arange(len(values), dtype=np.float64) * dt
    channels: dict[str, Channel] = {}
    col = 0
    for joint, chans in joints:
        channels.update(_bvh_joint_channels(joint, chans, values, col, times))
        col += len(chans)
    return Recording(
        source=source.name,
        adapter=NAME,
        collection=COLLECTION_MOCAP,
        channels=channels,
        census={
            "joints": [j for j, _ in joints],
            "frames": len(values),
            "rate_hz": round(1.0 / dt, 3) if dt > 0 else None,
            "format": "bvh",
        },
        notes=[
            "BVH positions assumed centimetres (scaled to metres); rotations "
            "converted from degrees; Euler order is the file's channel order",
            "human motion; not retargeted to any robot (docs/76 §5)",
        ],
    )


def _bvh_joints(hierarchy: str) -> list[tuple[str, list[str]]]:
    """(joint name, its CHANNELS) in file order; End Sites have none."""
    joints: list[tuple[str, list[str]]] = []
    name: str | None = None
    for raw in hierarchy.splitlines():
        line = raw.strip()
        if line.startswith(("ROOT", "JOINT")):
            name = line.split(None, 1)[1].strip()
        elif line.startswith("CHANNELS") and name is not None:
            parts = line.split()
            joints.append((name, parts[2 : 2 + int(parts[1])]))
            name = None
        elif line.startswith("End Site"):
            name = None
    return joints


def _bvh_frames(
    source: Path, motion: str, joints: list[tuple[str, list[str]]]
) -> tuple[np.ndarray, float]:
    """The frame table (frames x channels) and the frame period."""
    frames_line = re.search(r"Frames:\s*(\d+)", motion)
    dt_line = re.search(r"Frame Time:\s*([0-9.eE+-]+)", motion)
    if not frames_line or not dt_line:
        raise ValueError(f"{source}: MOTION lacks Frames/Frame Time")
    n_frames = int(frames_line.group(1))
    dt = float(dt_line.group(1))
    # Frame rows follow the "Frame Time:" line; anything else in MOTION
    # (blank lines, a stray header) is not a row.
    after = motion.split(dt_line.group(0), 1)[1]
    rows: list[list[float]] = []
    for ln in after.splitlines():
        parts = ln.split()
        if not parts:
            continue
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
        if len(rows) >= n_frames:
            break
    width = sum(len(ch) for _, ch in joints)
    widths = sorted({len(r) for r in rows})
    if not rows or widths != [width]:
        raise ValueError(
            f"{source}: frame rows have {widths} channels per frame, hierarchy "
            f"declares {width}"
        )
    return np.array(rows, dtype=np.float64), dt


def _bvh_joint_channels(
    joint: str, chans: list[str], values: np.ndarray, col: int, times: np.ndarray
) -> dict[str, Channel]:
    out: dict[str, Channel] = {}
    pos_idx = [(i, c) for i, c in enumerate(chans) if c.endswith("position")]
    rot_idx = [(i, c) for i, c in enumerate(chans) if c.endswith("rotation")]
    if pos_idx:
        block = values[:, [col + i for i, _ in pos_idx]] * CM_TO_M
        out[f"joint.{joint}.position"] = Channel(
            f"joint.{joint}.position",
            times,
            block,
            unit=BVH_DEFAULT_UNIT.replace("cm", "m"),
            components=tuple(c[0].lower() for _, c in pos_idx),
        )
    if rot_idx:
        block = np.deg2rad(values[:, [col + i for i, _ in rot_idx]])
        out[f"joint.{joint}.rotation"] = Channel(
            f"joint.{joint}.rotation",
            times,
            block,
            unit="rad",
            components=tuple(c[0].lower() for _, c in rot_idx),
        )
    return out


def _rate(times: np.ndarray) -> float | None:
    if len(times) < MIN_FRAMES:
        return None
    span = float(times[-1] - times[0])
    return round((len(times) - 1) / span, 3) if span > 0 else None


def _monotone(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.zeros(len(times), dtype=bool)
    last = -np.inf
    for i, t in enumerate(times):
        if np.isfinite(t) and t > last:
            keep[i] = True
            last = t
    return times[keep], keep
