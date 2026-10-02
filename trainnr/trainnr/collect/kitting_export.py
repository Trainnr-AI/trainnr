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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trainnr.collect.press import EpisodeSidecar

from trainnr.bundles.json_record import JsonRecord

# Re-export: the tests and tools read the sidecar's name from this
# module, where the provenance classes live.
from trainnr.collect.provenance import PROVENANCE_FILE  # noqa: F401
from trainnr.tasks.aloha2 import (
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
    EXPORT_FILE = "export.json"  # what the batch needs to become a dataset (ExportSpec)
    SHARD_FILE = "shard-{index:02d}.json"  # one press run's accounting (ShardRecord)
    SHARD_GLOB = "shard-*.json"
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
    # The batch's content stamp (`name@hash`, the hashed-files rule of
    # bundles/hashing). `source` stays the NAME for the union check;
    # this is the lineage a project index reads. Empty on datasets
    # exported before 2026-09-09 — reported as unrecorded, never invented.
    source_stamp: str = ""


def write_episode(  # noqa: PLR0913 - the whole episode, every part named
    demos_dir: Path,
    index: int,
    *,
    states: Any,
    sensors: Any,
    actions: Any,
    frames: Iterable[tuple[int, Any]],
    manifest: EpisodeSidecar,
    camera_frames: Mapping[str, Iterable[tuple[int, Any]]] | None = None,
) -> Path:
    """One kept episode onto disk in `DemoLayout`; returns its directory.

    `manifest` is any `press.EpisodeSidecar` — this task's legacy
    `Manifest` or the press's go-forward `EpisodeManifest`; the layout
    does not care which sidecar schema rides in it."""
    import numpy as np  # noqa: PLC0415 - sim extra
    from PIL import Image  # noqa: PLC0415

    episode_dir = demos_dir / DemoLayout.EPISODE_DIR.format(index=index)
    frames_dir = episode_dir / DemoLayout.FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Any: numpy's stub types **kwargs against savez's own keywords.
    arrays: dict[str, Any] = {
        DemoLayout.STATES: np.asarray(states, dtype=np.float32),
        DemoLayout.SENSORS: np.asarray(sensors, dtype=np.float32),
        DemoLayout.ACTIONS: np.asarray(actions, dtype=np.float32),
    }
    np.savez_compressed(episode_dir / DemoLayout.TRAJECTORY_FILE, **arrays)
    flat = list(frames)
    if flat and camera_frames:
        raise ValueError("an episode stores flat frames OR per-camera frames, not both")
    for tick, frame in flat:
        Image.fromarray(frame).save(
            frames_dir / DemoLayout.FRAME_FILE.format(tick=tick), quality=JPEG_QUALITY
        )
    # The multi-camera layout: frames/<camera_key>/<tick>.jpg — one
    # subdirectory per declared camera (docs/66 §4; the flat layout
    # stays what every committed single-camera batch reads as).
    for camera, cam_frames in (camera_frames or {}).items():
        camera_dir = frames_dir / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        for tick, frame in cam_frames:
            Image.fromarray(frame).save(
                camera_dir / DemoLayout.FRAME_FILE.format(tick=tick),
                quality=JPEG_QUALITY,
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
    """The T5 front of the one exporter (`demo_export.export_episodes`):
    the kitting constants — servo names, top camera, legacy `Manifest`
    sidecar — over the shared loop, refusals and provenance write.
    `clamp_constant_dims` off: every kitting joint moves, and the T5
    datasets predate the guard (their stats stay byte-stable)."""
    # Function-local: demo_export imports this module (the layout and
    # provenance classes live here), so the top level would be a cycle.
    from trainnr.collect.demo_export import export_episodes  # noqa: PLC0415

    episodes = episode_dirs(demos_dir)
    return export_episodes(
        demos_dir,
        root,
        repo_id=repo_id,
        use_videos=use_videos,
        manifests=[Manifest.read_from(ep) for ep in episodes],
        state_width=STATE_WIDTH,
        state_names=SERVO_NAMES,
        camera_key=TOP_CAMERA.key,
        instruction=task,
        bundle_dir=bundle_dir,
        state_semantics=None,  # DatasetProvenance's defaults ARE the kitting text
        action_semantics=None,
        clamp_constant_dims=False,
    )
