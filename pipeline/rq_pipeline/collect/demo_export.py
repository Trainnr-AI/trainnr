"""Pressed demonstrations → a LeRobot dataset: the ONE exporter.

`kitting_export.export_kitting_demos` proved the shape on T5, then
`export_demos` generalised it — as a line-for-line copy. Both fronts
now share `_export_episodes` (folded 2026-09-01): the task-specific
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

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.collect.kitting_export import (
    DatasetProvenance,
    DemoLayout,
    episode_dirs,
)
from rq_pipeline.collect.press import EpisodeManifest
from rq_pipeline.collect.provenance import IMAGE_AXES, PROVENANCE_FILE
from rq_pipeline.protocol import RGB_CHANNELS

if TYPE_CHECKING:
    from rq_pipeline.tasks.task import Task

__all__ = ["episode_dirs", "export_demos", "guard_constant_dims"]


def guard_constant_dims(root: Path, *, floor: float = 1e-6) -> list[str]:
    """Clamp zero-variance feature dimensions in the dataset's stats to
    std = 1. A dimension the expert holds CONSTANT (lift's base and
    wrist) gets std = 0 from the stats pass, and LeRobot's
    `(x - mean) / (std + 1e-8)` then turns the mean's own float32
    rounding error into a hundreds-of-sigma training target — measured
    2026-08-31: the wrist normalized to ±453 and ACT's L1 pinned at
    ~73 for a whole run (docs/07). With std = 1 a constant dimension
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


def _export_episodes(  # noqa: PLR0913 - every fact of one dataset, named
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
) -> Path:
    """The engine both exporter fronts drive: every episode under
    `demos_dir` as one LeRobot dataset at `root`. fps comes from the
    manifests (control_hz / frame_every), never assumed; the provenance
    sidecar carries the bundle stamp and every manifest verbatim."""
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
    image_key = f"{OBS_IMAGES}.{camera_key}"
    features = {
        image_key: {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, RGB_CHANNELS),
            "names": list(IMAGE_AXES),
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (state_width,),
            "names": state_names,
        },
        ACTION: {
            "dtype": "float32",
            "shape": (state_width,),
            "names": state_names,
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
                        sensors[tick, :state_width], dtype=np.float32
                    ),
                    ACTION: np.asarray(
                        actions[tick // steps_per_control], dtype=np.float32
                    ),
                    "task": instruction,
                }
            )
        dataset.save_episode()
        frame_counts.append(len(frames))
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    if clamp_constant_dims:
        guard_constant_dims(Path(root))

    provenance = DatasetProvenance(
        bundle=stamp(bundle_dir.name, bundle_dir),
        source=Path(demos_dir).name,
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
    return _export_episodes(
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
