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

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.bundles.json_record import JsonRecord
from rq_pipeline.collect.provenance import IMAGE_AXES, PROVENANCE_FILE
from rq_pipeline.protocol import RGB_CHANNELS
from rq_pipeline.tasks.aloha2 import (
    ALOHA_TOP_CAMERAS,
    ARM_NAMES,
    BUNDLE_XML,
    KITTING_INSTRUCTION,
    SERVOS,
)


class DemoLayout:
    """The demo batch on disk — one spelling for the writer (the demo
    generator), the reader (this exporter) and the tests. Frames are
    named by the PHYSICS step they were rendered at."""

    EPISODE_DIR = "episode_{index:04d}"
    EPISODE_GLOB = "episode_*"
    FRAMES_DIR = "frames"
    FRAME_FILE = "{tick:06d}.jpg"
    FRAME_GLOB = "*.jpg"
    TRAJECTORY_FILE = "trajectory.npz"
    MANIFEST_FILE = "manifest.json"
    STATES, SENSORS, ACTIONS = "states", "sensors", "actions"  # the npz keys


UNSTAMPED_EXPERT = "unstamped (batch generated before 2026-08-27)"


@dataclass(frozen=True)
class Manifest(JsonRecord):
    """What a kept episode records about itself: the draw, the dynamics
    it ran under, the referee's verdict, and the rates the exporter
    needs. Written by the generator, read here; one shape."""

    seed: int
    attempt: int
    draws: dict[str, tuple[float, float]]
    dr_span: float
    damping_scale: float
    gain_scale: float
    retries: list
    control_hz: int
    frame_every_control_ticks: int
    action_semantics: str = "commanded actuator positions, ctrl order"
    verdict: str = "success (task referee)"
    # The scripted expert's own stamp (`tasks.aloha2.expert_stamp`); the
    # default reads batches written before 2026-08-27, which carried none.
    expert: str = UNSTAMPED_EXPERT

    def write_to(self, episode_dir: Path) -> Path:
        return self.write(Path(episode_dir) / DemoLayout.MANIFEST_FILE)

    @classmethod
    def read_from(cls, episode_dir: Path) -> Manifest:
        return cls.read(Path(episode_dir) / DemoLayout.MANIFEST_FILE)


@dataclass(frozen=True)
class DatasetProvenance(JsonRecord):
    """The dataset's sidecar: the bundle the demos were simulated on, the
    expert that produced them, every episode's manifest — traceable to
    the exact dynamics and draws, the same rule as the certificate."""

    bundle: str
    source: str  # the batch's NAME, never its absolute path
    expert: str
    episodes: int
    frames: list[int]
    fps: int
    manifests: list[dict[str, Any]]
    state_semantics: str = (
        "sensordata[0:14]: the bundle's fourteen jointpos (radians; grippers in "
        "metres) — what the harness's vision rollout observes with state_width=14"
    )
    action_semantics: str = "fourteen commanded actuator positions, ctrl order"


def write_episode(  # noqa: PLR0913 - the whole episode, every part named
    demos_dir: Path,
    index: int,
    *,
    states: Any,
    sensors: Any,
    actions: Any,
    frames: Iterable[tuple[int, Any]],
    manifest: Any,
) -> Path:
    """One kept episode onto disk in `DemoLayout`; returns its directory.

    `manifest` is anything with `write_to(episode_dir)` — this task's
    legacy `Manifest` or the press's go-forward `EpisodeManifest`; the
    layout does not care which sidecar schema rides in it."""
    import numpy as np  # noqa: PLC0415 - sim extra
    from PIL import Image  # noqa: PLC0415

    episode_dir = demos_dir / DemoLayout.EPISODE_DIR.format(index=index)
    frames_dir = episode_dir / DemoLayout.FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_dir / DemoLayout.TRAJECTORY_FILE,
        **{
            DemoLayout.STATES: np.asarray(states, dtype=np.float32),
            DemoLayout.SENSORS: np.asarray(sensors, dtype=np.float32),
            DemoLayout.ACTIONS: np.asarray(actions, dtype=np.float32),
        },
    )
    for tick, frame in frames:
        Image.fromarray(frame).save(
            frames_dir / DemoLayout.FRAME_FILE.format(tick=tick), quality=JPEG_QUALITY
        )
    manifest.write_to(episode_dir)
    return episode_dir


JPEG_QUALITY = 85
SERVO_NAMES = [
    f"{arm}/{joint}"
    for arm in ARM_NAMES
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
STATE_WIDTH = SERVOS
if len(SERVO_NAMES) != STATE_WIDTH:  # the name list and the rig must agree
    raise AssertionError(f"{len(SERVO_NAMES)} servo names for {STATE_WIDTH} servos")
DEFAULT_BUNDLE = BUNDLE_XML.parent
TOP_CAMERA = ALOHA_TOP_CAMERAS[0]


def episode_dirs(demos_dir: Path) -> list[Path]:
    """The generator's episodes, in order; refuses an empty batch."""
    episodes = sorted(p for p in demos_dir.glob(DemoLayout.EPISODE_GLOB) if p.is_dir())
    if not episodes:
        raise ValueError(f"no {DemoLayout.EPISODE_GLOB} directories under {demos_dir}")
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
        from lerobot.utils.constants import (  # noqa: PLC0415
            ACTION,
            OBS_IMAGES,
            OBS_STATE,
        )
        from PIL import Image  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "kitting export needs the 'train' extra on Python >= 3.12 "
            "(uv sync --python 3.12 --extra train)"
        ) from error

    episodes = episode_dirs(demos_dir)
    manifests = [Manifest.read_from(ep) for ep in episodes]
    rates = {(m.control_hz, m.frame_every_control_ticks) for m in manifests}
    if len(rates) != 1:
        raise ValueError(f"episodes disagree on frame rate: {sorted(rates)}")
    (control_hz, frame_every), *_ = rates
    experts = {m.expert for m in manifests}
    if len(experts) != 1:
        # Two choreographies in one batch is two datasets; a stamp that
        # names the expert exists so this cannot pass unnoticed.
        raise ValueError(f"episodes disagree on the expert: {sorted(experts)}")
    (expert,) = experts
    if control_hz % frame_every:
        raise ValueError(
            f"{control_hz} Hz is not divisible by frame_every={frame_every}"
        )
    fps = control_hz // frame_every

    first_frame = next(
        iter(sorted((episodes[0] / DemoLayout.FRAMES_DIR).glob(DemoLayout.FRAME_GLOB)))
    )
    height, width = np.asarray(Image.open(first_frame)).shape[:2]
    # The keys the env's plugin derives (envs/lerobot_plugin.py): training
    # and evaluation observe the same thing under the same name.
    image_key = f"{OBS_IMAGES}.{TOP_CAMERA.key}"
    features = {
        image_key: {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, RGB_CHANNELS),
            "names": list(IMAGE_AXES),
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (STATE_WIDTH,),
            "names": SERVO_NAMES,
        },
        ACTION: {
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
        trajectory = np.load(episode / DemoLayout.TRAJECTORY_FILE)
        sensors = trajectory[DemoLayout.SENSORS]
        actions = trajectory[DemoLayout.ACTIONS]
        # Sensors are per PHYSICS step, actions per CONTROL tick (measured:
        # 14000 rows against 1400); frames are named by physics step.
        steps_per_control = len(sensors) // len(actions)
        if steps_per_control * len(actions) != len(sensors):
            raise ValueError(
                f"{episode}: {len(sensors)} sensor rows are not a whole "
                f"multiple of {len(actions)} action rows"
            )
        frames = sorted((episode / DemoLayout.FRAMES_DIR).glob(DemoLayout.FRAME_GLOB))
        if not frames:
            raise ValueError(f"{episode} has no frames")
        for frame_path in frames:
            tick = int(frame_path.stem)  # the physics step the frame was taken at
            dataset.add_frame(
                {
                    image_key: np.asarray(Image.open(frame_path)),
                    OBS_STATE: np.asarray(
                        sensors[tick, :STATE_WIDTH], dtype=np.float32
                    ),
                    ACTION: np.asarray(
                        actions[tick // steps_per_control], dtype=np.float32
                    ),
                    "task": task,
                }
            )
        dataset.save_episode()
        frame_counts.append(len(frames))
    if hasattr(dataset, "finalize"):
        dataset.finalize()

    DatasetProvenance(
        bundle=stamp(bundle_dir.name, bundle_dir),
        source=Path(demos_dir).name,
        expert=expert,
        episodes=len(episodes),
        frames=frame_counts,
        fps=fps,
        manifests=[asdict(m) for m in manifests],
    ).write(Path(root) / PROVENANCE_FILE)
    return Path(root)
