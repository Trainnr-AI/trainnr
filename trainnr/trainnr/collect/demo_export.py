"""Pressed demonstrations → a LeRobot dataset: the ONE exporter.

`kitting_export.export_kitting_demos` proved the shape on T5, then
`export_demos` generalised it — as a line-for-line copy. Both fronts
now share `export_episodes` (folded 2026-09-01): the task-specific
constants (state names, camera, instruction, bundle, sidecar class)
are parameters; the loop, the refusals and the provenance write exist
once.

One refusal the kitting front never had the schema for: episodes whose
`dynamics_basis` strings disagree do not become one dataset — mixing
"we declared this span" with "we measured this interval" is exactly
the confusion the paired study exists to measure
(docs/e2e-research/58 §8, 62 §1). Kitting's legacy `Manifest` carries
no basis, so the check is vacuous there by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trainnr.bundles.hashing import stamp
from trainnr.bundles.json_record import JsonRecord
from trainnr.bundles.locate import robots_dir
from trainnr.collect.kitting_export import (
    DatasetProvenance,
    DemoLayout,
    episode_dirs,
)
from trainnr.collect.press import EpisodeManifest
from trainnr.collect.provenance import IMAGE_AXES, PROVENANCE_FILE
from trainnr.protocol import RGB_CHANNELS

if TYPE_CHECKING:
    from trainnr.tasks.task import Task

__all__ = [
    "episode_camera_keys",
    "episode_dirs",
    "export_demos",
    "guard_constant_dims",
]


# LeRobot's own guidance for its asynchronous image writer: four
# threads per camera, no extra processes. Without it every frame is
# encoded and written on the export's one thread — campaign 3
# (2026-09-03) spent 12 minutes writing 240,000 frames the press had
# rendered in 4.5; the writer is the export's wall, not the
# reads.
IMAGE_WRITER_THREADS_PER_CAMERA = 4


def episode_camera_keys(episode_dir: Path) -> tuple[str, ...]:
    """The cameras a pressed episode stored: subdirectory names under
    `frames/` (the multi-camera layout, a note), or `()` for the
    legacy flat single-camera layout every committed batch uses."""
    frames_dir = Path(episode_dir) / DemoLayout.FRAMES_DIR
    if not frames_dir.is_dir():
        return ()
    return tuple(sorted(p.name for p in frames_dir.iterdir() if p.is_dir()))


def guard_constant_dims(root: Path, *, floor: float = 1e-6) -> list[str]:
    """Clamp zero-variance feature dimensions in the dataset's stats to
    std = 1. A dimension the expert holds CONSTANT (lift's base and
    wrist) gets std = 0 from the stats pass, and LeRobot's
    `(x - mean) / (std + 1e-8)` then turns the mean's own float32
    rounding error into a hundreds-of-sigma training target — measured
    2026-08-31: the wrist normalized to ±453 and ACT's L1 pinned at
    ~73 for a whole run. With std = 1 a constant dimension
    normalizes to ~0 and unnormalizes to its constant, which is the
    only honest reading of "no variance". Returns the patched keys."""
    import json  # noqa: PLC0415

    stats_path = Path(root) / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    patched: list[str] = []

    def clamp(node: Any, label: str) -> Any:
        if isinstance(node, list):
            return [clamp(item, f"{label}[{i}]") for i, item in enumerate(node)]
        if isinstance(node, (int, float)) and abs(node) < floor:
            patched.append(label)
            return 1.0
        return node

    for key, feature in stats.items():
        if isinstance(feature, dict) and "std" in feature:
            feature["std"] = clamp(feature["std"], f"{key}.std")
    if patched:
        stats_path.write_text(json.dumps(stats, indent=1))
    return patched


def _one(values: set[Any], what: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"episodes disagree on {what}: {sorted(map(str, values))}")
    (value,) = values
    return value


def export_episodes(  # noqa: PLR0913 - every fact of one dataset, named
    demos_dir: Path,
    root: Path,
    *,
    repo_id: str,
    use_videos: bool,
    manifests: list[Any],
    state_width: int,
    state_names: list[str],
    camera_key: str,
    instruction: str,
    bundle_dir: Path,
    state_semantics: str | None,
    action_semantics: str | None,
    clamp_constant_dims: bool,
    action_width: int | None = None,
    action_names: list[str] | None = None,
    image_writer_threads: int | None = None,
) -> Path:
    """The engine every exporter front drives: every episode under
    `demos_dir` as one LeRobot dataset at `root`. fps comes from the
    manifests (control_hz / frame_every), never assumed; the provenance
    sidecar carries the bundle stamp and every manifest verbatim."""
    try:
        import numpy as np  # noqa: PLC0415

        # The availability probe for the helpful error below; the
        # dataset itself is opened in `_open_dataset`.
        from lerobot.datasets.lerobot_dataset import (  # noqa: F401, PLC0415
            LeRobotDataset,
        )
        from lerobot.utils.constants import (  # noqa: PLC0415
            ACTION,
            OBS_IMAGES,
            OBS_STATE,
        )
        from PIL import Image  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "demo export needs the 'train' extra on Python >= 3.12 "
            "(uv sync --python 3.12 --extra train)"
        ) from error

    episodes = episode_dirs(demos_dir)
    control_hz, frame_every = _one(
        {(m.control_hz, m.frame_every_control_ticks) for m in manifests},
        "frame rate",
    )
    expert = _one({m.expert for m in manifests}, "the expert")
    # 58 §8: a dataset is one basis or it is two datasets. Legacy
    # sidecars carry no basis; a batch of them agrees on None.
    _one(
        {getattr(m, "dynamics_basis", None) for m in manifests},
        "the dynamics basis",
    )
    # Same rule for the visual ranges (a note): the DRAWS should
    # differ per episode — that is the point — but where the ranges
    # came from must be one story per dataset.
    _one(
        {getattr(m, "visual_basis", "") or None for m in manifests},
        "the visual basis",
    )
    if control_hz % frame_every:
        raise ValueError(
            f"{control_hz} Hz is not divisible by frame_every={frame_every}"
        )
    fps = control_hz // frame_every

    # One layout per batch: every episode stores the same cameras
    # (multi-camera subdirs) or all are flat single-camera batches,
    # where the caller's `camera_key` names the one camera.
    layout = _one({episode_camera_keys(ep) for ep in episodes}, "the camera layout")
    cameras: list[str | None] = list(layout) or [None]

    def frame_paths(episode: Path, camera: str | None) -> list[Path]:
        base = episode / DemoLayout.FRAMES_DIR
        if camera is not None:
            base = base / camera
        return sorted(base.glob(DemoLayout.FRAME_GLOB))

    def image_key(camera: str | None) -> str:
        # The keys the env's plugin derives (envs/lerobot_plugin.py):
        # training and evaluation observe the same thing under the same
        # name — the env renders EVERY declared camera by its key.
        return f"{OBS_IMAGES}.{camera if camera is not None else camera_key}"

    features: dict[str, Any] = {}
    for camera in cameras:
        sample = frame_paths(episodes[0], camera)
        if not sample:
            raise ValueError(f"{episodes[0]} has no frames for camera {camera!r}")
        height, width = np.asarray(Image.open(sample[0])).shape[:2]
        features[image_key(camera)] = {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, RGB_CHANNELS),
            "names": list(IMAGE_AXES),
        }
    features |= {
        OBS_STATE: {
            "dtype": "float32",
            "shape": (state_width,),
            "names": state_names,
        },
        ACTION: {
            # An arm's action IS its state's width (targets per joint);
            # an RL teacher's is not (a 51-D observation in, 14 targets
            # out) - the width is declared, never assumed (D2, 2026-09-02).
            "dtype": "float32",
            "shape": (state_width if action_width is None else action_width,),
            "names": state_names if action_names is None else action_names,
        },
    }
    dataset = _open_dataset(
        repo_id,
        fps,
        root,
        features,
        use_videos=use_videos,
        image_writer_threads=(
            IMAGE_WRITER_THREADS_PER_CAMERA * len(cameras)
            if image_writer_threads is None
            else image_writer_threads
        ),
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
        ticks = _add_episode_frames(
            dataset,
            episode,
            cameras=cameras,
            frame_paths=frame_paths,
            image_key=image_key,
            sensors=sensors,
            actions=actions,
            state_width=state_width,
            steps_per_control=steps_per_control,
            instruction=instruction,
        )
        dataset.save_episode()
        frame_counts.append(ticks)
    _close_dataset(dataset)
    if clamp_constant_dims:
        guard_constant_dims(Path(root))

    provenance = DatasetProvenance(
        bundle=stamp(bundle_dir.name, bundle_dir),
        source=Path(demos_dir).name,
        source_stamp=stamp(Path(demos_dir).name, Path(demos_dir)),
        expert=expert,
        episodes=len(episodes),
        frames=frame_counts,
        fps=fps,
        manifests=[asdict(m) for m in manifests],
        **({"state_semantics": state_semantics} if state_semantics else {}),
        **({"action_semantics": action_semantics} if action_semantics else {}),
    )
    provenance.write(Path(root) / PROVENANCE_FILE)
    return Path(root)


def _open_dataset(  # noqa: PLR0913 - the dataset's facts, each named
    repo_id: str,
    fps: int,
    root: Path,
    features: dict[str, Any],
    *,
    use_videos: bool,
    image_writer_threads: int,
) -> Any:
    """A LeRobot dataset with its asynchronous image writer running."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        features=features,
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
    )


def _close_dataset(dataset: Any) -> None:
    """Finalize, then release the writer's threads: `save_episode`
    already waited for every frame of every episode (LeRobot's own
    contract), so nothing is left in flight here."""
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
    elif hasattr(getattr(dataset, "writer", None), "stop_image_writer"):
        dataset.writer.stop_image_writer()


def _add_episode_frames(  # noqa: PLR0913 - one episode's facts, each named
    dataset: Any,
    episode: Path,
    *,
    cameras: list[str | None],
    frame_paths: Any,
    image_key: Any,
    sensors: Any,
    actions: Any,
    state_width: int,
    steps_per_control: int,
    instruction: str,
) -> int:
    """One episode's rows into the dataset; returns the frame count.
    Every camera captured at the same physics steps, by construction in
    the press — a disagreement is a broken batch."""
    import numpy as np  # noqa: PLC0415 - the caller imported the train extra
    from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    camera_paths = {camera: frame_paths(episode, camera) for camera in cameras}
    if not all(camera_paths.values()):
        raise ValueError(f"{episode} has no frames")
    ticks = _one(
        {tuple(int(p.stem) for p in paths) for paths in camera_paths.values()},
        f"{episode.name}'s frame ticks across cameras",
    )
    for i, tick in enumerate(ticks):
        row: dict[str, Any] = {
            image_key(camera): np.asarray(Image.open(camera_paths[camera][i]))
            for camera in cameras
        }
        row |= {
            OBS_STATE: np.asarray(sensors[tick, :state_width], dtype=np.float32),
            ACTION: np.asarray(actions[tick // steps_per_control], dtype=np.float32),
            "task": instruction,
        }
        dataset.add_frame(row)
    return len(ticks)


def export_demos(
    demos_dir: Path,
    root: Path,
    *,
    task: Task,
    repo_id: str,
    use_videos: bool = True,
) -> Path:
    """Write every episode under `demos_dir` as one LeRobot dataset at
    `root`, features shaped by `task`: the state width, the camera key,
    the instruction and the bundle stamp all come from the task the
    demos were pressed on."""
    episodes = episode_dirs(demos_dir)
    return export_episodes(
        demos_dir,
        root,
        repo_id=repo_id,
        use_videos=use_videos,
        manifests=[EpisodeManifest.read_from(ep) for ep in episodes],
        state_width=task.state_width,
        state_names=[f"q{i}" for i in range(task.state_width)],
        camera_key=task.cameras[0].key,  # the camera the press rendered
        instruction=task.instruction,
        bundle_dir=task.bundle_dir,
        state_semantics=(
            f"sensordata[0:{task.state_width}]: the task's jointpos block "
            f"(radians) - what the harness observes with "
            f"state_width={task.state_width}"
        ),
        action_semantics=(
            f"{task.state_width} commanded actuator positions, ctrl order"
        ),
        clamp_constant_dims=True,
    )


@dataclass(frozen=True)
class ExportSpec(JsonRecord):
    """What a batch needs to become a dataset, written by the press that
    made it (a note: the dataset carries the rig): the state
    vector's width and names, the camera keys, the instruction, the
    bundle the robot came from. A press with no `Task` object (the RL
    rollout press, D2) exports through the same engine as one with.
    `bundle` is the robot bundle's NAME under robots/, never a path."""

    state_width: int
    state_names: list[str]
    cameras: list[str]
    instruction: str
    bundle: str
    state_semantics: str = ""
    action_semantics: str = ""
    clamp_constant_dims: bool = True
    # Empty = the arm case (actions are per-state-column targets).
    action_names: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def write_to(self, demos_dir: Path) -> Path:
        return self.write(Path(demos_dir) / DemoLayout.EXPORT_FILE)

    @classmethod
    def read_from(cls, demos_dir: Path) -> ExportSpec:
        path = Path(demos_dir) / DemoLayout.EXPORT_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"{demos_dir} carries no {DemoLayout.EXPORT_FILE} - the press that "
                "made it must write an ExportSpec, or export through a Task"
            )
        return cls.read(path)


def export_batch(
    demos_dir: Path,
    root: Path,
    *,
    repo_id: str,
    use_videos: bool = True,
    image_writer_threads: int | None = None,
) -> Path:
    """The third front: a batch that describes itself (`export.json`)
    becomes a LeRobot dataset with no Task in hand - the state names,
    cameras and bundle come from the batch, the loop and refusals from
    the one engine."""
    spec = ExportSpec.read_from(demos_dir)
    if spec.state_width != len(spec.state_names):
        raise ValueError(
            f"{DemoLayout.EXPORT_FILE}: state_width {spec.state_width} but "
            f"{len(spec.state_names)} state names"
        )
    return export_episodes(
        demos_dir,
        root,
        repo_id=repo_id,
        use_videos=use_videos,
        manifests=[EpisodeManifest.read_from(ep) for ep in episode_dirs(demos_dir)],
        state_width=spec.state_width,
        state_names=spec.state_names,
        camera_key=spec.cameras[0] if spec.cameras else "camera",
        instruction=spec.instruction,
        bundle_dir=robots_dir() / spec.bundle,
        state_semantics=spec.state_semantics or None,
        action_semantics=spec.action_semantics or None,
        clamp_constant_dims=spec.clamp_constant_dims,
        action_width=len(spec.action_names) if spec.action_names else None,
        action_names=spec.action_names or None,
        image_writer_threads=image_writer_threads,
    )
