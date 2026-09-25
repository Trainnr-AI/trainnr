"""Capture to scene (docs/78 §3, E1's second half): a phone video or a
folder of frames becomes a scene artifact through the chain the
research chose - ffmpeg for the frames, COLMAP for the poses, Brush for
the splat - with the alignment, the proxy and the gap the record asks
for. Every tool is a subprocess named with its version and licence;
one that is missing refuses by name before anything runs.

What this module decides and says so on the record:

- **Alignment.** COLMAP's frame is arbitrary and unscaled. The floor is
  the largest plane through the splat's visible centres (RANSAC), its
  normal turned to +z with the side most of the scene lies on as up,
  its centroid moved to the origin. Scale is metres per COLMAP unit
  when the caller declares one (a measured length in the capture) and
  1.0, recorded as unrecorded, when not: the scene's metres are then
  COLMAP's units, and the record says so.
- **The proxy.** No dense reconstruction runs here (COLMAP's needs
  CUDA; the 2DGS chain waits on the box), so the proxy is the visible
  surface itself: the visible centres' top surface on a 2 cm grid, the
  surface the stage's heightfield would sample anyway (Poisson was
  tried first and aborted the interpreter from C++ on a flat room). The
  gap then measures the proxy's fidelity to the surface it was built
  from, not to the world - the caveat on every record this makes.
- **Physics.** Declared by the caller or absent; nothing measured.

The chain is resumable by artifact presence: frames, poses, splat,
each skipped when its output already exists, so a failed Brush run
does not re-run COLMAP.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.robot.fit_record import code_version
from rq_pipeline.scenes import gap as gap_audit
from rq_pipeline.scenes.colmap_model import read_text_model
from rq_pipeline.scenes.gap import DEFAULT_TOLERANCE_M
from rq_pipeline.scenes.obj import write_obj
from rq_pipeline.scenes.proxy import (
    OVERHANG_PARAMS,
    OVERHANG_PARTS,
    PARTS_FILE,
    ensure_parts,
)
from rq_pipeline.scenes.record import (
    CAPTURE_COMMAND_PREFIX,
    CAPTURE_LOG_FILE,
    CAPTURE_STAGE_PREFIX,
    DECLARED,
    DECLARED_FRICTION_SPAN,
    FLOOR_FRICTION,
    GROUND_FILE,
    OVERHANG_FILE,
    PROXY_FILE,
    PROXY_FROM_SPLAT,
    PROXY_MJCF,
    SCENE_FILE,
    SPLAT_FILE,
    UNRECORDED,
    Alignment,
    Capture,
    Physics,
    SceneRecord,
    Tool,
    proxy_mjcf,
)
from rq_pipeline.scenes.splat import (
    VISIBLE_OPACITY,
    Splats,
    describe,
    euler_xyz_matrix,
    read_ply,
    write_ply,
)
from rq_pipeline.scenes.splatters import (
    DEFAULT_SPLATTER,
    DEFAULT_STEPS,
    RENDERS_DIR,
    SPLAT_EXPORT,
    SPLATTERS,
    Splatter,
    choose_splatter,
)
from rq_pipeline.scenes.tooling import MissingToolError, install_hint, tool_version
from rq_pipeline.scenes.volume import (
    OVERHANG_CLEARANCE_M,
    VOXEL_M,
    joined,
    overhang_components,
    split_by_clearance,
)
from rq_pipeline.viz import STUDIO_ADDRESS, studio_listening

# The work folders inside the scene, kept: the capture's provenance.
FRAMES_DIR = "frames"
COLMAP_DIR = "colmap"
SPARSE_DIR = "sparse/0"
DATASET_DIR = "dataset"  # the trainer's view: images/ and sparse/0/ as links
DATABASE = "database.db"
LOG_FILE = CAPTURE_LOG_FILE
CAPTURE_SOURCE = "capture"  # the record's source word: capture/<video or folder name>
FRAMES_PER_SECOND = 2.0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
COLMAP_CAMERA = "OPENCV"  # one phone, one lens: a single camera with distortion
SEQUENTIAL_OVERLAP = 20
LOOP_DETECTION_PERIOD = 10
PROXY_CELL_M = DEFAULT_TOLERANCE_M  # the proxy's grid: the audit's tolerance
PROXY_HOLE_CELLS = 3  # holes up to this many cells across are filled
# A cell's height is this percentile of its centres, not their maximum: the
# top few percent of a real capture's centres are floaters (grass tips,
# specks), and the maximum picked every one (a lawn as a bed of spikes
# the Go2 hung on, 2026-09-23).
PROXY_CELL_PERCENTILE = 90.0
# Every cell then takes the median of its window: a splat's centres scatter
# a few centimetres about a flat surface, and at a 2 cm pitch that scatter
# is a sawtooth of 40° walls to a foot the size of a cell (the paving's
# neighbour steps p90 1.7 cm raw, 0.35 cm after the median, 2026-09-23). A
# median keeps edges: a step's two levels stay, only its rim can move.
PROXY_SPIKE_WINDOW = 5  # cells, the neighbourhood whose median a cell takes
PROXY_SPIKE_M = 0.05  # a cell moved farther than this by the median counts as a spike
PROXY_FILL_NEIGHBOURS = (
    3  # of the ring of 8: fewer and the cell is a fringe, not a hole
)
FLOOR_DISTANCE_M = 0.02
FLOOR_RANSAC_N = 3
FLOOR_ITERATIONS = 2000
PARALLEL_EPS = 1e-12  # a normal already along z: no axis to turn about
GIMBAL_EPS = 1e-9  # pitch at ±90°: roll and yaw share an axis
MIN_FRAMES = 3  # fewer cannot triangulate
FFMPEG_VERSION_WORD = 2  # 'ffmpeg version N.N.N ...'
SCALE_UNRECORDED = "unrecorded: the scene's metres are COLMAP's units"
ALIGNMENT_SOURCE = (
    "the largest plane through the visible centres (RANSAC) as the floor, its "
    "normal to +z, most of the scene above it, its centroid at the origin"
)
PROXY_METHOD = (
    "below {clearance} m, the visible gaussian centres' top surface on a {cell} m "
    "grid (each cell the {percentile:g}th percentile of its centres, then the "
    "median of its {window}x{window} neighbours, small holes filled from their "
    "neighbours): the ground; at or above it, the centres as an occupancy volume "
    "on {voxel} m voxels, the exposed faces of the occupied ones: what stands, "
    "with air beneath a table top; not a dense reconstruction"
)
CAPTURE_NOTES = (
    "the proxy is the visible surface itself (the ground's top on a grid, what "
    "stands above it as voxels): the gap measures its fidelity to the splat, "
    "not to the world",
    "no dense reconstruction: COLMAP's patch-match needs CUDA and the 2DGS chain "
    "waits on the box (docs/78 §3)",
    "the colour is the zeroth harmonic; the higher harmonics stay in the file",
)


Runner = Callable[[Sequence[str | Path], Path, Path], None]


def run_logged(argv: Sequence[str | Path], cwd: Path, log: Path) -> None:
    """The default runner: a subprocess, its output appended to the log,
    its failure raised with the command's name and exit status."""
    line = f"{CAPTURE_COMMAND_PREFIX}{' '.join(str(a) for a in argv)}"
    print(
        line, flush=True
    )  # the job's log sees the stages; the tools' floods stay here
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n{line}\n")
        out.flush()
        done = subprocess.run(
            [str(a) for a in argv],
            cwd=cwd,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if done.returncode != 0:
        raise RuntimeError(
            f"{Path(str(argv[0])).name} exited {done.returncode}; see {log}"
        )


@dataclass(frozen=True)
class Tools:
    """The chain's binaries, each found by name or the door refuses; the
    splat trainer by the registry (`splatters`), `auto` taking the best
    one this machine has."""

    ffmpeg: Path | None
    ffprobe: Path | None
    colmap: Path
    splatter: Splatter
    trainer: Path  # what the splatter runs from: its binary, or an interpreter

    @staticmethod
    def find(
        *,
        brush: Path | None = None,
        video: bool = True,
        splatter: str = DEFAULT_SPLATTER,
    ) -> Tools:
        missing = []
        found = {n: shutil.which(n) for n in ("ffmpeg", "ffprobe", "colmap")}
        if video:
            missing += [
                f"{n} ({install_hint(n)})"
                for n in ("ffmpeg", "ffprobe")
                if found[n] is None
            ]
        if found["colmap"] is None:
            missing.append(f"colmap ({install_hint('colmap')})")
        spec: Splatter | None = None
        trainer: Path | None = None
        try:
            spec, trainer = choose_splatter(splatter, brush)
        except MissingToolError as why:
            missing.append(str(why))
        if missing or spec is None or trainer is None:
            raise MissingToolError("the capture chain needs " + ", ".join(missing))
        return Tools(
            ffmpeg=Path(found["ffmpeg"]) if found["ffmpeg"] else None,
            ffprobe=Path(found["ffprobe"]) if found["ffprobe"] else None,
            colmap=Path(found["colmap"]),  # type: ignore[arg-type]
            splatter=spec,
            trainer=trainer,
        )


def colmap_option(binary: Path, command: str, option: str) -> list[str]:
    """`[option, "0"]` when this COLMAP build lists it for the command,
    else nothing: 4.2 names it `FeatureExtraction.use_gpu` (default on,
    even in Homebrew's CPU-only build, which then fails at run time),
    older builds `SiftExtraction.use_gpu`; a flag a build does not list
    is refused at parse time."""
    try:
        out = subprocess.run(
            [str(binary), command, "-h"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [option, "0"] if option in (out.stdout + out.stderr) else []


def colmap_version(binary: Path) -> str:
    m = re.search(r"COLMAP\s+([\d.]+)", tool_version(binary, "-h"))
    return m.group(1) if m else UNRECORDED


# -- frames ---------------------------------------------------------------


def probe_video(ffprobe: Path, video: Path) -> Capture:
    """What the video says about itself; unrecorded where it says nothing."""
    try:
        out = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration:stream=width,height,nb_frames",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        info = json.loads(out.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return Capture(device=UNRECORDED, app=UNRECORDED)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    w, h = stream.get("width"), stream.get("height")
    duration = fmt.get("duration")
    frames = stream.get("nb_frames")
    return Capture(
        frames=int(frames) if str(frames or "").isdigit() else UNRECORDED,
        resolution=f"{w}x{h}" if w and h else UNRECORDED,
        duration_s=round(float(duration), 2) if duration else UNRECORDED,
    )


def extract_frames(  # noqa: PLR0913 - the stage's own knobs, each named
    tools: Tools, video: Path, out: Path, *, fps: float, run: Runner, log: Path
) -> list[Path]:
    """Frames at `fps` from the video, PNG, numbered; skipped when present."""
    if out.is_dir() and any(out.glob("frame_*.png")):
        return sorted(out.glob("frame_*.png"))
    out.mkdir(parents=True, exist_ok=True)
    assert tools.ffmpeg is not None
    run(
        [
            tools.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video,
            "-vf",
            f"fps={fps:g}",
            out / "frame_%04d.png",
        ],
        out,
        log,
    )
    return sorted(out.glob("frame_*.png"))


def copy_frames(folder: Path, out: Path) -> list[Path]:
    """A folder of images becomes the scene's frames, renamed in order."""
    kept_before = (
        sorted(p for p in out.glob("frame_*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if out.is_dir()
        else []
    )
    if kept_before:
        return kept_before
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"{folder}: no {', '.join(IMAGE_SUFFIXES)} images")
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    for i, src in enumerate(images):
        dst = out / f"frame_{i + 1:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        kept.append(dst)
    return kept


# -- poses ----------------------------------------------------------------


@dataclass(frozen=True)
class Poses:
    """What COLMAP registered."""

    images_registered: int
    images_given: int
    points: int
    camera_model: str
    # the mapper may split a walk into several models where the chain of
    # matches breaks; every model's frame count, and which one was used
    models: tuple[int, ...] = ()
    chosen: str = "0"  # the mapper's folder name of the model Brush trains on


def read_sparse(sparse: Path) -> Poses:
    """Counts from a COLMAP text model (images.txt, points3D.txt,
    cameras.txt): registered images, sparse points, the camera model."""
    model = read_text_model(sparse)
    return Poses(
        images_registered=len(model.images),
        images_given=0,
        points=int(model.points.shape[0]),
        camera_model=model.camera_model or UNRECORDED,
    )


def choose_model(work: Path) -> Poses:
    """The largest of the mapper's models, every model's size recorded;
    the chosen one is what Brush trains on (`DATASET_DIR`)."""
    models = sorted(p for p in (work / "sparse").iterdir() if p.is_dir())
    counted = [read_sparse(m) for m in models]
    sizes = tuple(p.images_registered for p in counted)
    best = max(range(len(counted)), key=lambda i: sizes[i])
    chosen = counted[best]
    return Poses(
        images_registered=chosen.images_registered,
        images_given=0,
        points=chosen.points,
        camera_model=chosen.camera_model,
        models=sizes,
        chosen=models[best].name,
    )


def _link_or_copy(link: Path, target: Path) -> None:
    """A directory link, or a copy where links need rights the user may
    lack (Windows without Developer Mode: WinError 1314)."""
    try:
        link.symlink_to(target.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(target, link)


def dataset_for_trainer(work: Path, frames: Path, chosen: str) -> Path:
    """The trainers' layout - `images/` beside `sparse/0/` - as links to
    the frames and the chosen model (the mapper's folder name), under
    `DATASET_DIR`."""
    dataset = work / DATASET_DIR
    (dataset / "sparse").mkdir(parents=True, exist_ok=True)
    images = dataset / "images"
    if not images.exists():
        _link_or_copy(images, frames)
    model = dataset / "sparse" / "0"
    if not model.exists():
        _link_or_copy(model, work / "sparse" / chosen)
    return dataset


def solve_poses(  # noqa: PLR0913 - the stage's own knobs, each named
    tools: Tools, frames: Path, work: Path, *, run: Runner, log: Path, sequential: bool
) -> Poses:
    """COLMAP's sparse chain on the frames: one camera, sequential
    matching with loop detection for a video (exhaustive for a folder of
    stills), the mapper, the model exported as text beside the binary.
    Skipped when the text model is present."""
    sparse = work / SPARSE_DIR
    if not (sparse / "images.txt").is_file():
        work.mkdir(parents=True, exist_ok=True)
        (work / "sparse").mkdir(exist_ok=True)
        db = work / DATABASE
        run(
            [
                tools.colmap,
                "feature_extractor",
                "--database_path",
                db,
                "--image_path",
                frames,
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                COLMAP_CAMERA,
                *colmap_option(
                    tools.colmap, "feature_extractor", "--FeatureExtraction.use_gpu"
                ),
            ],
            work,
            log,
        )
        if sequential:
            run(
                [
                    tools.colmap,
                    "sequential_matcher",
                    "--database_path",
                    db,
                    "--SequentialMatching.overlap",
                    str(SEQUENTIAL_OVERLAP),
                    "--SequentialMatching.loop_detection",
                    "1",
                    "--SequentialMatching.loop_detection_period",
                    str(LOOP_DETECTION_PERIOD),
                    *colmap_option(
                        tools.colmap, "sequential_matcher", "--FeatureMatching.use_gpu"
                    ),
                ],
                work,
                log,
            )
        else:
            run(
                [
                    tools.colmap,
                    "exhaustive_matcher",
                    "--database_path",
                    db,
                    *colmap_option(
                        tools.colmap, "exhaustive_matcher", "--FeatureMatching.use_gpu"
                    ),
                ],
                work,
                log,
            )
        run(
            [
                tools.colmap,
                "mapper",
                "--database_path",
                db,
                "--image_path",
                frames,
                "--output_path",
                work / "sparse",
            ],
            work,
            log,
        )
        models = sorted(p for p in (work / "sparse").iterdir() if p.is_dir())
        if not models:
            raise RuntimeError(f"COLMAP registered nothing: no model under {work}")
        for model in models:  # every model as text, so each can be counted
            run(
                [
                    tools.colmap,
                    "model_converter",
                    "--input_path",
                    model,
                    "--output_path",
                    model,
                    "--output_type",
                    "TXT",
                ],
                work,
                log,
            )
    poses = choose_model(work)
    given = len([p for p in frames.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    return Poses(
        images_registered=poses.images_registered,
        images_given=given,
        points=poses.points,
        camera_model=poses.camera_model,
        models=poses.models,
        chosen=poses.chosen,
    )


# -- the splat ------------------------------------------------------------


def train_splat(  # noqa: PLR0913 - the stage's own knobs, each named
    tools: Tools,
    dataset: Path,
    out: Path,
    *,
    steps: int,
    run: Runner,
    log: Path,
    narrate: bool,
) -> Path:
    """The trainer on the COLMAP dataset (`images/` beside `sparse/0/`),
    headless, exporting once at the end; into the Studio when one is
    listening (law 0). Skipped when the export is present."""
    export = out / SPLAT_EXPORT
    # a resume takes the export whichever trainer left it (a scene first
    # trained by Brush, resumed under `auto` once gsplat is installed)
    for spec in SPLATTERS.values():
        earlier = out.parent / spec.folder / SPLAT_EXPORT
        if earlier.is_file():
            return earlier
    out.mkdir(parents=True, exist_ok=True)
    argv = tools.splatter.argv(
        tools.trainer, dataset, out, steps=steps, narrate=narrate
    )
    run(argv, out, log)
    if not export.is_file():
        raise RuntimeError(f"{tools.splatter.title} exported nothing: no {export}")
    return export


# -- alignment ------------------------------------------------------------


@dataclass(frozen=True)
class Floor:
    """The plane the alignment stands on and how well it fit."""

    normal: np.ndarray
    centroid: np.ndarray
    inliers: int
    of: int


def find_floor(centres: np.ndarray, *, seed: int = 0) -> Floor:
    """The largest plane through the centres by RANSAC (Open3D), the
    normal turned toward the side most of the scene lies on."""
    import open3d as o3d  # noqa: PLC0415

    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(centres.astype(np.float64))
    )
    o3d.utility.random.seed(seed)
    (a, b, c, _d), inliers = cloud.segment_plane(
        distance_threshold=FLOOR_DISTANCE_M,
        ransac_n=FLOOR_RANSAC_N,
        num_iterations=FLOOR_ITERATIONS,
    )
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    inlier = centres[np.asarray(inliers)]
    centroid = inlier.mean(0)
    # up is the side with more of the scene on it, the floor itself
    # (within the plane's own threshold) not voting
    side = (centres - centroid) @ normal
    off = side[np.abs(side) > FLOOR_DISTANCE_M]
    if (off > 0).sum() < (off < 0).sum():
        normal = -normal
    return Floor(
        normal=normal, centroid=centroid, inliers=len(inliers), of=len(centres)
    )


def rotation_to_z(normal: np.ndarray) -> np.ndarray:
    """The rotation taking `normal` to +z (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z)
    s = np.linalg.norm(v)
    c = float(normal @ z)
    if s < PARALLEL_EPS:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s))


def euler_xyz_of(matrix: np.ndarray) -> tuple[float, float, float]:
    """MuJoCo's intrinsic x-y-z euler (radians) of a rotation matrix,
    the inverse of `splat.euler_xyz_matrix` (R = Rx·Ry·Rz, so R[0,2] =
    sin y, R[1,2] = -sin x cos y, R[2,2] = cos x cos y, R[0,1] = -cos y
    sin z, R[0,0] = cos y cos z)."""
    sy = float(np.clip(matrix[0, 2], -1.0, 1.0))
    y = float(np.arcsin(sy))
    if abs(abs(sy) - 1.0) < GIMBAL_EPS:  # pitch at ±90°: roll and yaw share an axis
        return (float(np.arctan2(matrix[2, 1], matrix[1, 1])), y, 0.0)
    x = float(np.arctan2(-matrix[1, 2], matrix[2, 2]))
    z = float(np.arctan2(-matrix[0, 1], matrix[0, 0]))
    return (x, y, z)


def align(
    splats: Splats, *, scale: float | None, seed: int = 0
) -> tuple[Splats, Alignment, Floor]:
    """The splat into the world frame: floor to z=0 and +z up, centroid
    at the origin, scaled by `scale` metres per unit when declared."""
    visible = splats.visible(VISIBLE_OPACITY)
    floor = find_floor(visible.means.astype(np.float64), seed=seed)
    rotation = rotation_to_z(floor.normal)
    factor = float(scale) if scale is not None else 1.0
    translation = -factor * (rotation @ floor.centroid)
    euler = euler_xyz_of(rotation)
    # the record carries eulers; make sure they rebuild the matrix used
    rebuilt = euler_xyz_matrix(np.array(euler))
    if not np.allclose(rebuilt, rotation, atol=1e-6):
        raise ValueError("the floor rotation does not round-trip through eulers")
    aligned = splats.transformed(
        scale=factor, rotation=rotation, translation=translation
    )
    record = Alignment(
        scale=factor,
        rotation_euler_xyz=euler,
        translation=(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ),
        source=ALIGNMENT_SOURCE
        + (
            f"; scale {factor:g} m per COLMAP unit, declared by the caller"
            if scale is not None
            else f"; scale {SCALE_UNRECORDED}"
        ),
    )
    return aligned, record, floor


# -- the proxy ------------------------------------------------------------


def top_surface_mesh(
    centres: np.ndarray, *, cell: float = PROXY_CELL_M
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The visible centres' top surface on a grid as a triangle mesh:
    each cell's height is the `PROXY_CELL_PERCENTILE`th percentile of its
    centres (the maximum picked every floater), then the median of its
    `PROXY_SPIKE_WINDOW`-cell neighbourhood (the centres' scatter at the
    cell pitch is a sawtooth a foot cannot cross; a speck above the lawn
    or a centre below it is struck by the same median, counted when it
    moved a cell by more than `PROXY_SPIKE_M`); a cell with no centre but at least
    `PROXY_FILL_NEIGHBOURS` seen cells around it takes the lowest of them
    (a hole in the capture, counted), up to `PROXY_HOLE_CELLS` deep; the
    rest are left out, so a sparse splat yields a sparse surface rather
    than an invented one. Deterministic, never fails on a plane - the
    surface the stage's heightfield would sample anyway."""
    lo = centres[:, :2].min(0)
    hi = centres[:, :2].max(0)
    ncol = max(int(np.ceil((hi[0] - lo[0]) / cell)) + 1, 2)
    nrow = max(int(np.ceil((hi[1] - lo[1]) / cell)) + 1, 2)
    col = np.clip(((centres[:, 0] - lo[0]) / cell).astype(int), 0, ncol - 1)
    row = np.clip(((centres[:, 1] - lo[1]) / cell).astype(int), 0, nrow - 1)
    height = cell_percentiles(
        row * ncol + col, centres[:, 2], nrow * ncol, PROXY_CELL_PERCENTILE
    ).reshape(nrow, ncol)
    seen = np.isfinite(height)
    despiked = despike(height, PROXY_SPIKE_WINDOW, PROXY_SPIKE_M)
    filled = 0
    # holes: the lowest neighbour within the ring, a few passes
    for _ in range(PROXY_HOLE_CELLS):
        empty = ~np.isfinite(height)
        if not empty.any():
            break
        padded = np.pad(height, 1, constant_values=np.inf)
        ring = np.stack(
            [
                padded[1 + dr : 1 + dr + nrow, 1 + dc : 1 + dc + ncol]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            ]
        )
        known = np.isfinite(ring).sum(0)
        ring[~np.isfinite(ring)] = np.inf
        lowest = ring.min(0)
        # a hole has surface around it; an isolated cell must not grow
        fill = empty & (known >= PROXY_FILL_NEIGHBOURS)
        height[fill] = lowest[fill]
        filled += int(fill.sum())
    keep = np.isfinite(height)
    index = -np.ones((nrow, ncol), dtype=np.int64)
    index[keep] = np.arange(int(keep.sum()))
    rr, cc = np.nonzero(keep)
    vertices = np.column_stack([lo[0] + cc * cell, lo[1] + rr * cell, height[keep]])
    faces = []
    for r in range(nrow - 1):
        for c in range(ncol - 1):
            a, b, d, e = (
                index[r, c],
                index[r, c + 1],
                index[r + 1, c],
                index[r + 1, c + 1],
            )
            if min(a, b, d, e) < 0:
                continue
            faces.append([a, b, e])
            faces.append([a, e, d])
    if not faces:
        raise ValueError("no cell of the grid has a surface")
    facts = {
        "cell_m": cell,
        "cells": int(keep.sum()),
        "cells_seen": int(seen.sum()),
        "cells_filled": filled,
        "cells_despiked": despiked,
    }
    return vertices, np.asarray(faces, dtype=np.int64), facts


def cell_percentiles(
    cells: np.ndarray, values: np.ndarray, count: int, percentile: float
) -> np.ndarray:
    """Per cell, the given percentile of the values that fell in it
    (nearest rank); -inf for a cell with none. One sort, no Python loop."""
    order = np.lexsort((values, cells))
    cells, values = cells[order], values[order]
    counts = np.bincount(cells, minlength=count)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.floor(percentile / 100.0 * np.maximum(counts - 1, 0)).astype(np.int64)
    out = np.full(count, -np.inf)
    has = counts > 0
    out[has] = values[starts[has] + rank[has]]
    return out


def despike(height: np.ndarray, window: int, tolerance: float) -> int:
    """Every seen cell takes the median of its `window`x`window`
    neighbourhood (unseen cells lending their nearest seen height for the
    median's sake), in place; returns how many moved by more than
    `tolerance`, the spikes. A step keeps its two levels: the median at
    its edge is one of them, and only cells within half a window of the
    edge can move."""
    from scipy import ndimage  # noqa: PLC0415

    seen = np.isfinite(height)
    if not seen.any():
        return 0
    nearest = ndimage.distance_transform_edt(
        ~seen, return_distances=False, return_indices=True
    )
    filled = height[nearest[0], nearest[1]]
    median = ndimage.median_filter(filled, size=window, mode="nearest")
    spikes = seen & (np.abs(height - median) > tolerance)
    height[seen] = median[seen]
    return int(spikes.sum())


def build_proxy(
    splats: Splats, out_obj: Path
) -> tuple[dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]:
    """The proxy to an OBJ, in two halves beside it: the ground (the
    visible centres below the clearance, their top on a grid) and the
    overhangs (the centres above it as an occupancy volume, one mesh per
    thing, returned for their decomposition); the proxy is their union.
    The mesh's facts with the method. A centre that is not finite (a
    diverged gaussian) is dropped and counted; a splat with no ground is
    refused by name."""
    visible = splats.visible(VISIBLE_OPACITY)
    centres = visible.means.astype(np.float64)
    finite = np.isfinite(centres).all(axis=1)
    centres = centres[finite]
    ground, _ = split_by_clearance(centres)
    if ground.shape[0] == 0:
        raise ValueError(
            f"no visible centre below {OVERHANG_CLEARANCE_M} m: the splat has no "
            "ground to stand on (is the scene aligned?)"
        )
    g_vertices, g_faces, facts = top_surface_mesh(ground)
    components, overhang = overhang_components(centres)
    o_vertices, o_faces = joined(components)
    folder = out_obj.parent
    write_obj(folder / GROUND_FILE, g_vertices, g_faces)
    if o_faces.shape[0]:
        write_obj(folder / OVERHANG_FILE, o_vertices, o_faces)
    else:  # a resume must not keep an earlier run's overhangs
        (folder / OVERHANG_FILE).unlink(missing_ok=True)
    vertices = np.concatenate([g_vertices, o_vertices], 0)
    faces = np.concatenate([g_faces, o_faces + g_vertices.shape[0]], 0)
    write_obj(out_obj, vertices, faces)
    facts_out = {
        "file": out_obj.name,
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "watertight": False,
        "edge_manifold": True,
        "centres_not_finite": int((~finite).sum()),
        "ground": {"file": GROUND_FILE, "clearance_m": OVERHANG_CLEARANCE_M},
        "overhang": {"file": OVERHANG_FILE if o_faces.shape[0] else None, **overhang},
        "method": PROXY_METHOD.format(
            clearance=OVERHANG_CLEARANCE_M,
            cell=PROXY_CELL_M,
            percentile=PROXY_CELL_PERCENTILE,
            window=PROXY_SPIKE_WINDOW,
            voxel=VOXEL_M,
        ),
        "from": PROXY_FROM_SPLAT.format(opacity=VISIBLE_OPACITY),
        **facts,
    }
    return facts_out, components


# What the record says a proxy came from when no dense reconstruction ran
# (the card reads the flag, not this prose).


# -- the chain ------------------------------------------------------------


# Told each stage line as the chain reaches it - the Studio's Running now
# panel (`mcp_jobs.Tracker.stage`) without the chain knowing the panel.
StageTold = Callable[[str], None]


def _log_line(log: Path, text: str, on_stage: StageTold | None = None) -> None:
    """The chain's own note: into the scene's log, and to stdout for the
    job's log (`mcp_jobs`), which saw nothing of a capture before; and to
    `on_stage` when a caller listens."""
    line = f"{CAPTURE_STAGE_PREFIX}{text}"
    print(line, flush=True)
    with log.open("a", encoding="utf-8") as out:
        out.write(f"{line}\n")
    if on_stage is not None:
        on_stage(text)


def capture_scene(  # noqa: PLR0913 - the capture's own knobs, each named
    source: Path,
    out_dir: Path,
    *,
    name: str,
    tools: Tools | None = None,
    run: Runner = run_logged,
    fps: float = FRAMES_PER_SECOND,
    steps: int = DEFAULT_STEPS,
    scale: float | None = None,
    floor_friction: Sequence[float] | None = None,
    friction_span: float = DECLARED_FRICTION_SPAN,
    device: str = UNRECORDED,
    lighting: str = UNRECORDED,
    narrate: bool | None = None,
    on_stage: StageTold | None = None,
) -> Path:
    """A video file or a folder of frames into `out_dir` as a scene
    artifact; returns the record's path. Resumable: each stage's output,
    when present, is kept. Refuses by name a missing tool, a source that
    is neither, and a name that is not plain."""
    source, out_dir = Path(source), Path(out_dir)
    is_video = source.is_file()
    if not is_video and not source.is_dir():
        raise FileNotFoundError(f"no video file or frames folder at {source}")
    tools = tools or Tools.find(video=is_video)
    if (out_dir / SCENE_FILE).is_file():
        raise ValueError(
            f"{out_dir} already holds a scene; a scene is never overwritten"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / LOG_FILE
    narrate = studio_listening(STUDIO_ADDRESS) if narrate is None else narrate
    started = datetime.now(timezone.utc)

    frames_dir = out_dir / FRAMES_DIR
    if is_video:
        assert tools.ffprobe is not None
        capture = probe_video(tools.ffprobe, source)
        frames = extract_frames(tools, source, frames_dir, fps=fps, run=run, log=log)
    else:
        capture = Capture()
        frames = copy_frames(source, frames_dir)
    capture = Capture(
        device=device,
        app=capture.app,
        frames=capture.frames if capture.frames != UNRECORDED else len(frames),
        resolution=capture.resolution,
        duration_s=capture.duration_s,
        lighting=lighting,
    )
    _log_line(log, f"{len(frames)} frames", on_stage=on_stage)
    if len(frames) < MIN_FRAMES:
        raise ValueError(f"{len(frames)} frames: a capture needs at least {MIN_FRAMES}")

    colmap_dir = out_dir / COLMAP_DIR
    poses = solve_poses(
        tools, frames_dir, colmap_dir, run=run, log=log, sequential=is_video
    )
    _log_line(
        log,
        f"COLMAP registered {poses.images_registered}/{poses.images_given} "
        f"frames, {poses.points} points; models of {list(poses.models)} frames, "
        f"model {poses.chosen} used",
        on_stage=on_stage,
    )
    if poses.images_registered < MIN_FRAMES:
        raise RuntimeError(
            f"COLMAP registered {poses.images_registered} of {poses.images_given} "
            "frames: too few for a scene"
        )
    dataset = dataset_for_trainer(colmap_dir, frames_dir, poses.chosen)
    _log_line(log, f"splat by {tools.splatter.title}, {steps} steps", on_stage=on_stage)
    export = train_splat(
        tools,
        dataset,
        out_dir / tools.splatter.folder,
        steps=steps,
        run=run,
        log=log,
        narrate=narrate,
    )
    # the trainer's own renders of its splat, when it makes them (gsplat does)
    renders = [
        p.relative_to(out_dir).as_posix()
        for p in sorted((export.parent / RENDERS_DIR).glob("*.png"))
    ]
    raw = read_ply(export)
    aligned, alignment, floor = align(raw, scale=scale)
    write_ply(aligned, out_dir / SPLAT_FILE)
    friction = [float(v) for v in floor_friction] if floor_friction else None
    _log_line(
        log,
        f"aligned: floor {floor.inliers}/{floor.of} visible centres within "
        f"{FLOOR_DISTANCE_M} of the plane",
        on_stage=on_stage,
    )

    proxy_facts, measured, physics = _proxy_stage(
        aligned, out_dir, friction=friction, friction_span=friction_span
    )
    minutes = round((datetime.now(timezone.utc) - started).total_seconds() / 60, 1)
    record = SceneRecord(
        name=name,
        source=f"{CAPTURE_SOURCE}/{source.name}",
        capture=capture,
        tools=_tools_used(
            tools, is_video=is_video, fps=fps, steps=steps, minutes=minutes
        ),
        splat={
            "file": SPLAT_FILE,
            "from": export.relative_to(out_dir).as_posix(),
            "renders": renders,
            "poses": {
                "registered": poses.images_registered,
                "given": poses.images_given,
                "points": poses.points,
                "camera_model": poses.camera_model,
                "models": list(poses.models),
                "chosen": poses.chosen,
            },
            "floor_inliers": floor.inliers,
            "floor_of": floor.of,
            **describe(aligned),
        },
        proxy=proxy_facts,
        alignment=alignment,
        gap=measured,
        physics=tuple(physics),
        created_utc=datetime.now(timezone.utc).isoformat(),
        code=code_version(),
        notes=(f"captured from {source}", *CAPTURE_NOTES),
    )
    return record.write(out_dir / SCENE_FILE)


def _proxy_stage(
    aligned: Splats,
    out_dir: Path,
    *,
    friction: list[float] | None,
    friction_span: float,
) -> tuple[dict[str, Any], gap_audit.Gap, list[Physics]]:
    """The proxy from the splat - the ground below the clearance as a
    top surface, what stands above it as an occupancy volume, the two
    joined as the proxy the audit and the viewer see - its MJCF, the gap,
    the parts of each, the declared physics."""
    proxy_facts, components = build_proxy(aligned, out_dir / PROXY_FILE)
    (out_dir / PROXY_MJCF).write_text(
        proxy_mjcf(PROXY_FILE, friction), encoding="utf-8"
    )
    try:
        measured = gap_audit.measure(aligned, out_dir / PROXY_FILE)
    except ImportError as missing:
        measured = gap_audit.unmeasured(str(missing))
    # A decomposition that cannot run (no CoACD) or dies (a worker gone,
    # a mesh CoACD rejects) is recorded, not fatal: the splat, the proxy
    # and the gap above are the scene; the parts are a stage's need.
    try:
        parts = ensure_parts(out_dir)
        proxy_facts["parts"] = parts.parts
        proxy_facts["parts_file"] = PARTS_FILE
    except (ImportError, RuntimeError, OSError) as why:
        proxy_facts["parts"] = UNRECORDED
        proxy_facts["parts_note"] = str(why)
    if components:
        try:
            above = ensure_parts(
                out_dir, params=OVERHANG_PARAMS, files=OVERHANG_PARTS, meshes=components
            )
            proxy_facts["overhang"]["parts"] = above.parts
            proxy_facts["overhang"]["parts_file"] = OVERHANG_PARTS.record
        except (ImportError, RuntimeError, OSError) as why:
            proxy_facts["overhang"]["parts"] = UNRECORDED
            proxy_facts["overhang"]["parts_note"] = str(why)
    physics: list[Physics] = []
    if friction is not None:
        physics.append(
            Physics(
                name=FLOOR_FRICTION,
                value=friction,
                basis=DECLARED,
                span=friction_span,
                cites="the caller of capture_scene",
                unit="MuJoCo friction triple",
            )
        )
    return proxy_facts, measured, physics


def _tools_used(
    tools: Tools, *, is_video: bool, fps: float, steps: int, minutes: float
) -> tuple[Tool, ...]:
    """Every tool the chain ran, with its version and licence."""
    used: list[Tool] = []
    if is_video and tools.ffmpeg is not None:
        words = tool_version(tools.ffmpeg, "-version").split(" ")
        used.append(
            Tool(
                name="ffmpeg",
                version=words[FFMPEG_VERSION_WORD]
                if len(words) > FFMPEG_VERSION_WORD
                else UNRECORDED,
                license="LGPL-2.1-or-later",
                role=f"frames at {fps:g} a second",
            )
        )
    matching = (
        "sequential matching with loop detection" if is_video else "exhaustive matching"
    )
    used += [
        Tool(
            name="COLMAP",
            version=colmap_version(tools.colmap),
            license="BSD-3-Clause",
            role=f"poses: one {COLMAP_CAMERA} camera, {matching}, the mapper",
        ),
        Tool(
            name=tools.splatter.title,
            version=tools.splatter.version(tools.trainer),
            license=tools.splatter.license,
            role=f"the splat: {steps} steps, {minutes} min for the whole chain",
        ),
        Tool(
            name="Open3D",
            version=_open3d_version(),
            license="MIT",
            role="the floor plane, the gap",
        ),
        Tool(
            name="rq_pipeline.scenes",
            version=code_version(),
            license="Apache-2.0",
            role="the chain, the alignment, the record",
        ),
    ]
    return tuple(used)


def _open3d_version() -> str:
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError:
        return UNRECORDED
    return str(o3d.__version__)
