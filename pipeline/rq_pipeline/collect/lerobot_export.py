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

from rq_pipeline.collect.frames import AlignedEpisode

# Nominal camera burst rate on the rig; the grid is synthetic (see above).
EXPORT_FPS = 4

STATE_NAMES = ["x", "y", "heading", "ticks_left", "ticks_right"]
ACTION_NAMES = ["pan_us", "tilt_us", "grip_us", "duty_percent"]
PROVENANCE_FILE = "provenance.json"


def export_episode(
    episode: AlignedEpisode,
    root: Path,
    repo_id: str = "rq-pipeline/rig",
    task: str = "chase the blob",
) -> Path:
    """Write one episode as a fresh LeRobot dataset rooted at `root`.

    A `provenance.json` sidecar records the source recording's
    `name@hash` stamp and the dropped-frame count — the dataset must be
    traceable to the exact bytes it came from, same rule as the
    certificate.
    """
    try:
        # Lazy by design: these are the 'train' extra, and the module must
        # import cleanly (to raise THIS error) when they are absent.
        import numpy as np  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import (  # noqa: PLC0415
            LeRobotDataset,
        )
    except ImportError as error:
        raise ImportError(
            "LeRobot export needs the 'train' extra: uv sync --extra train"
        ) from error

    if not episode.frames:
        raise ValueError("episode has no frames — nothing to export")

    first = episode.frames[0]
    features = {
        "observation.images.front": {
            "dtype": "image",
            "shape": (first.height, first.width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
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
        fps=EXPORT_FPS,
        root=root,
        features=features,
        use_videos=False,
    )
    for frame in episode.frames:
        image = np.frombuffer(frame.rgb888, dtype=np.uint8).reshape(
            frame.height, frame.width, 3
        )
        dataset.add_frame(
            {
                "observation.images.front": image,
                "observation.state": np.array(
                    [
                        frame.state.x,
                        frame.state.y,
                        frame.state.heading,
                        float(frame.state.ticks_left),
                        float(frame.state.ticks_right),
                    ],
                    dtype=np.float32,
                ),
                "action": np.array(frame.action, dtype=np.float32),
                "wire_timestamp_s": np.array([frame.timestamp_s], dtype=np.float32),
                "task": task,
            }
        )
    dataset.save_episode()

    provenance = {
        "source": episode.source,
        "frames": len(episode.frames),
        "dropped_incomplete": episode.dropped_incomplete,
        "timestamp_note": (
            "dataset timestamps are a synthetic uniform grid at "
            f"{EXPORT_FPS} fps; real seq-clock time is wire_timestamp_s"
        ),
    }
    (Path(root) / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2))
    return Path(root)
