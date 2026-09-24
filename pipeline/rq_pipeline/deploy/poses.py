"""The robot's pose, tick by tick, saved beside a deployment's run so the
Studio's own MuJoCo viewport can replay it (`deploy/viewport_source.py`).

A gate on Unitree's stack runs in THEIR simulator: our physics cannot
re-run it, but every runtime says where the robot is each tick
(`pose()`: base position, base quaternion w x y z, joints in the
manifest's policy order), so the pictures can be replayed exactly as they
happened, no physics. The gate's Rerun stream holds geom transforms, not
qpos; this file holds what a replay needs and nothing else.

One file per run: `<deployment>/.viewer/<stream>-poses.npz` (the hidden
folder, so a replay file never moves the deployment's identity). Frames
are grouped into named segments ("trial 3", "stop · zeroed"), in the
order they ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POSES_SUFFIX = "-poses.npz"
# The npz's arrays, named once: the writer and the reader share them.
SEGMENTS = "segments"  # the segment names, in order
SEGMENT = "segment"  # per frame: the index of its segment
POSITION = "position"  # per frame: base position (3)
QUATERNION = "quat"  # per frame: base orientation, w x y z (4)
JOINTS = "joints"  # per frame: joints in the manifest's policy order
DT = "dt"  # seconds between two frames (the control tick)
JOINT_NAMES = "joint_names"  # the policy order the joints columns follow
QUAT_WIDTH = 4
POSITION_WIDTH = 3
# A gate trial's segment name (the gate's and the course's trials both).
TRIAL_SEGMENT = "trial {index}"


def trial_segment(index: int) -> str:
    """A trial's segment name in a gate's pose file."""
    return TRIAL_SEGMENT.format(index=index)


def poses_file(viewer_stream: Path) -> Path:
    """The pose file beside a saved stream (`.viewer/gate-dds.rrd` ->
    `.viewer/gate-dds-poses.npz`)."""
    viewer_stream = Path(viewer_stream)
    return viewer_stream.with_name(viewer_stream.stem + POSES_SUFFIX)


@dataclass
class PoseTrack:
    """Poses as a run produces them, one segment at a time."""

    dt: float
    joint_names: tuple[str, ...]
    _segments: list[str] = field(default_factory=list)
    _rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )

    def begin(self, name: str) -> None:
        """Frames from here on belong to `name` (a trial, a stop)."""
        self._segments.append(str(name))

    def add(self, pose: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """One tick's pose: (base position, base quaternion, joints)."""
        if not self._segments:
            raise ValueError("a pose before any segment: call begin(name) first")
        position, quat, joints = pose
        self._rows.append(
            (
                len(self._segments) - 1,
                np.asarray(position, dtype=np.float32),
                np.asarray(quat, dtype=np.float32),
                np.asarray(joints, dtype=np.float32),
            )
        )

    def __len__(self) -> int:
        return len(self._rows)

    def save(self, path: Path) -> Path:
        """Written atomically; an empty track writes nothing and says so."""
        if not self._rows:
            raise ValueError("no poses were recorded: nothing to replay")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            staging,
            **{
                SEGMENTS: np.array(self._segments),
                SEGMENT: np.array([r[0] for r in self._rows], dtype=np.int32),
                POSITION: np.stack([r[1] for r in self._rows]),
                QUATERNION: np.stack([r[2] for r in self._rows]),
                JOINTS: np.stack([r[3] for r in self._rows]),
                DT: np.float64(self.dt),
                JOINT_NAMES: np.array(self.joint_names),
            },
        )
        staging.replace(path)
        return path


@dataclass(frozen=True)
class PoseFile:
    """A saved track, read back: its segments and each one's frames."""

    segments: tuple[str, ...]
    segment: np.ndarray
    position: np.ndarray
    quat: np.ndarray
    joints: np.ndarray
    dt: float
    joint_names: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> PoseFile:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"no poses saved at {path}: the run was made before its poses "
                "were recorded; run it again to replay it"
            )
        with np.load(path, allow_pickle=False) as raw:
            missing = [
                k
                for k in (
                    SEGMENTS,
                    SEGMENT,
                    POSITION,
                    QUATERNION,
                    JOINTS,
                    DT,
                    JOINT_NAMES,
                )
                if k not in raw
            ]
            if missing:
                raise ValueError(f"{path.name}: not a pose file, lacks {missing}")
            out = cls(
                segments=tuple(str(s) for s in raw[SEGMENTS]),
                segment=raw[SEGMENT].astype(np.int64),
                position=raw[POSITION].astype(np.float64),
                quat=raw[QUATERNION].astype(np.float64),
                joints=raw[JOINTS].astype(np.float64),
                dt=float(raw[DT]),
                joint_names=tuple(str(s) for s in raw[JOINT_NAMES]),
            )
        out.check()
        return out

    def check(self) -> None:
        """Refuse, by name, a file whose arrays do not line up."""
        n = len(self.segment)
        if not (len(self.position) == len(self.quat) == len(self.joints) == n):
            raise ValueError("pose file: its per-frame arrays differ in length")
        if self.position.shape[1:] != (POSITION_WIDTH,):
            raise ValueError(f"pose file: position is {self.position.shape[1:]}")
        if self.quat.shape[1:] != (QUAT_WIDTH,):
            raise ValueError(f"pose file: quaternion is {self.quat.shape[1:]}")
        if self.joints.shape[1:] != (len(self.joint_names),):
            raise ValueError(
                f"pose file: {self.joints.shape[1:]} joint columns for "
                f"{len(self.joint_names)} joint names"
            )
        if n and (self.segment.min() < 0 or self.segment.max() >= len(self.segments)):
            raise ValueError("pose file: a frame names a segment that does not exist")
        if self.dt <= 0:
            raise ValueError(f"pose file: dt {self.dt} is not a time step")

    def frames(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment `index`'s frames: (positions, quaternions, joints)."""
        if not 0 <= index < len(self.segments):
            raise IndexError(
                f"no segment {index}: this file has {len(self.segments)} "
                f"({', '.join(self.segments)})"
            )
        rows = self.segment == index
        return self.position[rows], self.quat[rows], self.joints[rows]
