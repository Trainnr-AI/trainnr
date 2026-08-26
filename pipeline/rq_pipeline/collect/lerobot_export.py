"""Export an aligned episode as a LeRobot dataset.

The heavy dependencies live behind the `train` extra (`uv sync --extra
train`); this module refuses helpfully without them, so the
dependency-free core stays dependency-free.

One honest compromise, documented rather than hidden: **LeRobot validates
frame timestamps against a uniform fps grid, and the wire clock is not
uniform** (camera bursts with gaps between them). Exported timestamps are
therefore the synthetic grid `index / fps`, and the real 50 Hz-counter
time survives per frame in the extra feature `wire_timestamp_s` — nothing
is lost, and nothing pretends the camera ticked like a metronome.
"""

from __future__ import annotations

import json
from pathlib import Path

from rq_pipeline.bundles.profile import RobotProfile
from rq_pipeline.collect.frames import AlignedEpisode
from rq_pipeline.collect.provenance import IMAGE_AXES, PROVENANCE_FILE
from rq_pipeline.protocol import RGB_CHANNELS

STATE_NAMES = ["x", "y", "heading", "ticks_left", "ticks_right"]
ACTION_NAMES = ["pan_us", "tilt_us", "grip_us", "duty_percent"]


def export_episode(
    episode: AlignedEpisode,
    root: Path,
    profile: RobotProfile,
    repo_id: str = "rq-pipeline/rig",
    task: str = "chase the blob",
) -> Path:
    """Write one episode as a fresh LeRobot dataset rooted at `root`.

    The fps grid comes from `profile.camera_fps` — the robot's number,
    from the robot's bundle. A `provenance.json` sidecar records the
    source recording's `name@hash` stamp, which robot profile shaped the
    dataset, and the dropped-frame count — the dataset must be traceable
    to the exact bytes it came from, same rule as the certificate.
    """
    try:
        # Lazy by design: these are the 'train' extra, and the module must
        # import cleanly (to raise THIS error) when they are absent.
        import numpy as np  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import (  # noqa: PLC0415
            LeRobotDataset,
        )
        from lerobot.utils.constants import (  # noqa: PLC0415
            ACTION,
            OBS_IMAGES,
            OBS_STATE,
        )
    except ImportError as error:
        raise ImportError(
            "LeRobot export needs the 'train' extra: uv sync --extra train"
        ) from error

    if not episode.frames:
        raise ValueError("episode has no frames — nothing to export")

    first = episode.frames[0]
    features = {
        f"{OBS_IMAGES}.front": {
            "dtype": "image",
            "shape": (first.height, first.width, RGB_CHANNELS),
            "names": list(IMAGE_AXES),
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        ACTION: {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
        "wire_timestamp_s": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["seconds"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=profile.camera_fps,
        root=root,
        features=features,
        use_videos=False,
    )
    for frame in episode.frames:
        image = np.frombuffer(frame.rgb888, dtype=np.uint8).reshape(
            frame.height, frame.width, RGB_CHANNELS
        )
        dataset.add_frame(
            {
                f"{OBS_IMAGES}.front": image,
                OBS_STATE: np.array(
                    [
                        frame.state.x,
                        frame.state.y,
                        frame.state.heading,
                        float(frame.state.ticks_left),
                        float(frame.state.ticks_right),
                    ],
                    dtype=np.float32,
                ),
                ACTION: np.array(frame.action, dtype=np.float32),
                "wire_timestamp_s": np.array([frame.timestamp_s], dtype=np.float32),
                "task": task,
            }
        )
    dataset.save_episode()

    provenance = {
        "source": episode.source,
        "robot_profile": profile.name,
        "frames": len(episode.frames),
        "dropped_incomplete": episode.dropped_incomplete,
        "timestamp_note": (
            "dataset timestamps are a synthetic uniform grid at "
            f"{profile.camera_fps} fps; real seq-clock time is wire_timestamp_s"
        ),
    }
    (Path(root) / PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return Path(root)
