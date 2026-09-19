"""A LeRobot dataset directory (format v3) as a recording.

LeRobot's `meta/info.json` names every feature with its shape and its
per-component names; the frames live in parquet under `data/`. The
joint channels come from `observation.state` (measured) and `action`
(commanded), with the feature's own component names carried across. The
episode index is kept as a channel so a consumer can cut the recording
at episode boundaries. Camera frames are in video files and are NOT
carried into the recording — a recording is signals; the dataset itself
stays the picture's home.

Reads parquet through pyarrow (in the `sim` extra's tree via mujoco's
dependencies); no LeRobot import, so the reader works in any venv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.robots.adapter import adapter
from rq_pipeline.robots.recording import (
    COLLECTION_SCRIPTED,
    COLLECTION_TELEOP,
    JOINT_COMMAND,
    JOINT_POSITION,
    UNKNOWN_UNIT,
    Channel,
    Recording,
)

NAME = "lerobot"
INFO_FILE = "meta/info.json"
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
TIMESTAMP_FEATURE = "timestamp"
EPISODE_FEATURE = "episode_index"
# LeRobot records joint positions and commands in the robot's own units;
# v3 does not state them. Radians for revolute joints is the convention
# every exporter in this repo and LeRobot's own SO-101 code follow, but
# the manifest says it is assumed.
ASSUMED_JOINT_UNIT = "rad (assumed; LeRobot v3 does not state units)"


@adapter(NAME, doc="A LeRobot dataset directory (v3): state and action as channels")
class LeRobotAdapter:
    """A LeRobot dataset directory (v3): state and action as channels."""

    name = NAME

    def accepts(self, source: Path) -> bool:
        return (Path(source) / INFO_FILE).is_file()

    def read(self, source: Path) -> Recording:
        source = Path(source)
        if not self.accepts(source):
            raise ValueError(f"{source} has no {INFO_FILE}; not a LeRobot dataset")
        info = json.loads((source / INFO_FILE).read_text())
        features: dict[str, Any] = info.get("features", {})
        table = _read_parquet(source, info)
        if TIMESTAMP_FEATURE not in table:
            raise ValueError(f"{source}: no {TIMESTAMP_FEATURE!r} column")
        times = np.asarray(table[TIMESTAMP_FEATURE], dtype=np.float64)
        episodes = np.asarray(
            table.get(EPISODE_FEATURE, np.zeros(len(times))), dtype=np.float64
        )
        # Episodes restart the clock; make one strictly increasing timeline
        # by offsetting each episode past the previous one's end.
        times = _stitch_episodes(times, episodes)
        channels: dict[str, Channel] = {}
        for feature, target, unit in (
            (STATE_FEATURE, JOINT_POSITION, ASSUMED_JOINT_UNIT),
            (ACTION_FEATURE, JOINT_COMMAND, ASSUMED_JOINT_UNIT),
        ):
            if feature in table:
                names = tuple(features.get(feature, {}).get("names") or ())
                values = np.asarray(table[feature], dtype=np.float64)
                channels[target] = Channel(
                    target, times, values, unit=unit, components=names
                )
        channels["episode"] = Channel(
            "episode",
            times,
            episodes.reshape(-1, 1),
            unit="index",
            components=("episode",),
        )
        # Every other numeric, non-image feature travels under its own name.
        for feature, spec in features.items():
            if (
                feature in table
                and feature
                not in (
                    STATE_FEATURE,
                    ACTION_FEATURE,
                    TIMESTAMP_FEATURE,
                    EPISODE_FEATURE,
                )
                and spec.get("dtype") not in ("video", "image")
            ):
                values = np.asarray(table[feature], dtype=np.float64)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                channels[feature] = Channel(
                    feature,
                    times,
                    values,
                    unit=UNKNOWN_UNIT,
                    components=tuple(spec.get("names") or ()),
                )
        # A dataset this repo exported from a sim press carries its provenance
        # sidecar with an expert stamp: that is scripted collection, not teleop.
        collection = COLLECTION_TELEOP
        sidecar = source / "provenance.json"
        if sidecar.is_file():
            try:
                if json.loads(sidecar.read_text()).get("expert"):
                    collection = COLLECTION_SCRIPTED
            except ValueError:
                pass
        cameras = [
            k for k, v in features.items() if v.get("dtype") in ("video", "image")
        ]
        return Recording(
            source=source.name,
            adapter=NAME,
            collection=collection,
            channels=channels,
            census={
                "codebase_version": info.get("codebase_version"),
                "fps": info.get("fps"),
                "episodes": int(info.get("total_episodes", len(np.unique(episodes)))),
                "frames": len(times),
                "joints": len(features.get(STATE_FEATURE, {}).get("names") or []),
                "cameras": cameras,
                "robot_type": info.get("robot_type"),
            },
            notes=[
                "camera frames stay in the dataset's video files; a recording is "
                "signals",
                "episodes are stitched onto one clock, each offset past the previous "
                "end",
            ],
        )


def _read_parquet(source: Path, info: dict[str, Any]) -> dict[str, np.ndarray]:
    """Every parquet file under data/, concatenated, as columns."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    files = sorted((source / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"{source}: no parquet files under data/")
    tables = [pq.read_table(f) for f in files]
    columns: dict[str, np.ndarray] = {}
    for name in tables[0].column_names:
        parts = [t.column(name).to_pylist() for t in tables if name in t.column_names]
        flat = [row for part in parts for row in part]
        columns[name] = np.asarray(flat)
    return columns


def _stitch_episodes(times: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    out = times.astype(np.float64).copy()
    offset = 0.0
    last_end = None
    for episode in np.unique(episodes):
        mask = episodes == episode
        if last_end is not None:
            step = float(np.median(np.diff(out[mask]))) if mask.sum() > 1 else 0.02
            offset = last_end + step - float(times[mask][0])
        out[mask] = times[mask] + offset
        last_end = float(out[mask][-1])
    return out
