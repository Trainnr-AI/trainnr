"""Kitting demonstrations → a LeRobot dataset: T5's training data.

`tools/kitting-demos.py` writes one directory per referee-approved
episode — `trajectory.npz` (per-physics-step states, sensors, actions),
`frames/<tick>.jpg` from the `top` camera, and a manifest with the
draw, the DR scales and the frame rate. This module turns a batch of
those into the dataset a `lerobot-train` run consumes.

Two contracts, both inherited from the harness so training data and
evaluation observe the same thing:

- `observation.state` is the bundle's raw fourteen jointpos — exactly
  `sensordata[0:14]`, what the gymnasium env's `agent_pos` hands a
  policy with `state_width=14`. No gym-aloha remap: the policy trains
  and is judged on OUR rig.
- `action` is the fourteen commanded actuator positions, in ctrl order.

The heavy dependencies live behind the `train` extra; the module
imports cleanly without them so it can raise the helpful error.
"""

from __future__ import annotations

import json
from pathlib import Path

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.tasks.aloha2 import KITTING_INSTRUCTION

PROVENANCE_FILE = "provenance.json"
SERVO_NAMES = [
    f"{arm}/{joint}"
    for arm in ("left", "right")
    for joint in (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    )
]
STATE_WIDTH = len(SERVO_NAMES)  # 14
DEFAULT_BUNDLE = Path(__file__).resolve().parents[3] / "robots" / "aloha2-nominal"


def episode_dirs(demos_dir: Path) -> list[Path]:
    """The generator's episodes, in order; refuses an empty batch."""
    episodes = sorted(p for p in demos_dir.glob("episode_*") if p.is_dir())
    if not episodes:
        raise ValueError(f"no episode_* directories under {demos_dir}")
    return episodes


def export_kitting_demos(  # noqa: PLR0913 - four keyword-only knobs, each a named default
    demos_dir: Path,
    root: Path,
    *,
    repo_id: str = "rq-pipeline/aloha2-kitting",
    task: str = KITTING_INSTRUCTION,
    bundle_dir: Path = DEFAULT_BUNDLE,
    use_videos: bool = True,
) -> Path:
    """Write every episode under `demos_dir` as one LeRobot dataset at `root`.

    fps comes from the manifest (control_hz / frame_every_control_ticks),
    never assumed. The provenance sidecar records the bundle stamp the
    demos were simulated on and every episode's manifest — the dataset
    is traceable to the exact dynamics and draws it came from, the same
    rule as the certificate.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "kitting export needs the 'train' extra (use .venv-train)"
        ) from error

    episodes = episode_dirs(demos_dir)
    manifests = [json.loads((ep / "manifest.json").read_text()) for ep in episodes]
    rates = {(m["control_hz"], m["frame_every_control_ticks"]) for m in manifests}
    if len(rates) != 1:
        raise ValueError(f"episodes disagree on frame rate: {sorted(rates)}")
    (control_hz, frame_every), *_ = rates
    if control_hz % frame_every:
        raise ValueError(
            f"{control_hz} Hz is not divisible by frame_every={frame_every}"
        )
    fps = control_hz // frame_every

    first_frame = next(iter(sorted((episodes[0] / "frames").glob("*.jpg"))))
    height, width = np.asarray(Image.open(first_frame)).shape[:2]
    features = {
        "observation.images.top": {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_WIDTH,),
            "names": SERVO_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_WIDTH,),
            "names": SERVO_NAMES,
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root, features=features, use_videos=use_videos
    )
    frame_counts = []
    for episode in episodes:
        trajectory = np.load(episode / "trajectory.npz")
        sensors, actions = trajectory["sensors"], trajectory["actions"]
        # Sensors are per PHYSICS step, actions per CONTROL tick (measured:
        # 14000 rows against 1400); frames are named by physics step.
        steps_per_control = len(sensors) // len(actions)
        if steps_per_control * len(actions) != len(sensors):
            raise ValueError(
                f"{episode}: {len(sensors)} sensor rows are not a whole "
                f"multiple of {len(actions)} action rows"
            )
        frames = sorted((episode / "frames").glob("*.jpg"))
        if not frames:
            raise ValueError(f"{episode} has no frames")
        for frame_path in frames:
            tick = int(frame_path.stem)  # the physics step the frame was taken at
            dataset.add_frame(
                {
                    "observation.images.top": np.asarray(Image.open(frame_path)),
                    "observation.state": np.asarray(
                        sensors[tick, :STATE_WIDTH], dtype=np.float32
                    ),
                    "action": np.asarray(
                        actions[tick // steps_per_control], dtype=np.float32
                    ),
                    "task": task,
                }
            )
        dataset.save_episode()
        frame_counts.append(len(frames))
    if hasattr(dataset, "finalize"):
        dataset.finalize()

    provenance = {
        "bundle": stamp(bundle_dir.name, bundle_dir),
        "source": str(demos_dir),
        "episodes": len(episodes),
        "frames": frame_counts,
        "fps": fps,
        "state_semantics": "sensordata[0:14]: the bundle's fourteen jointpos "
        "(radians; grippers in metres) — what the harness's vision rollout "
        "observes with state_width=14",
        "action_semantics": "fourteen commanded actuator positions, ctrl order",
        "manifests": manifests,
    }
    Path(root, PROVENANCE_FILE).write_text(json.dumps(provenance, indent=1))
    return Path(root)
