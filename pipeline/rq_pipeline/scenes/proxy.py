"""The proxy as the solver touches it.

MuJoCo collides a mesh geom as its convex hull (the SDF plugin is the
only exception, and Neverwhere's simulator uses it); a hurdle course as
one hull is a box the robot stands on top of. So the proxy is decomposed
into convex parts, each a mesh geom, and the decomposition carries its
own gap - hull surface to proxy surface, both ways - because what a task
sees is the scene's gap (docs/78 §3) plus this one (the audit's caveat,
finding `scene-gap-neverwhere-hurdle-2026-09-22`).

CoACD (MIT; Wei et al. 2022) does the decomposition, behind the `scene`
extra, when the scene is imported or captured; without it the record
says the parts are unrecorded and a stage asking for the hulls terrain
refuses by name (a stage never writes into a scene: its version is its
bytes). Parts land in `proxy-parts/` beside the proxy with a record,
written once and whole: a second call reads the record back.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.scenes.gap import GAP_DIGITS, SURFACE_SAMPLES
from rq_pipeline.scenes.obj import read_obj, write_obj
from rq_pipeline.scenes.record import PROXY_FILE, UNRECORDED

PARTS_DIR = "proxy-parts"
PARTS_FILE = "proxy-parts.json"
PARTS_SCHEMA = "trainnr-proxy-parts/1"
PART_NAME = "part-{index:03d}.obj"
NEEDS_COACD = "the scene extra (CoACD): uv sync --extra scene"
TOOL = "CoACD"
# The samples the decomposition's gap is judged on, each way.


@dataclass(frozen=True)
class DecompositionParams:
    """CoACD's knobs, recorded with every decomposition. `threshold` is
    its concavity bound in the mesh's own units (metres here): the hulls
    may leave the surface by about this much, which the gap then
    measures rather than assumes."""

    threshold: float = 0.05
    max_convex_hull: int = 64
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 150
    mcts_max_depth: int = 3
    preprocess_mode: str = "auto"
    merge: bool = True
    seed: int = 0


DEFAULT_PARAMS = DecompositionParams()


@dataclass(frozen=True)
class Decomposition:
    """The record: how many parts, how heavy, how long, and how far the
    hulls sit from the surface they stand for."""

    parts: int
    vertices: int
    vertices_max_per_part: int
    seconds: float
    tool: str
    params: dict[str, Any]
    gap: dict[str, Any]
    files: tuple[str, ...]
    source: str = PROXY_FILE
    schema: str = PARTS_SCHEMA
    notes: tuple[str, ...] = field(default_factory=tuple)

    def part_paths(self, scene_dir: Path) -> list[Path]:
        return [Path(scene_dir) / PARTS_DIR / f for f in self.files]

    def meshes(self, scene_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        return [read_obj(p) for p in self.part_paths(scene_dir)]


def _coacd() -> Any:
    try:
        import coacd  # noqa: PLC0415
    except ImportError as why:
        raise ImportError(NEEDS_COACD) from why
    return coacd


def _tool_version() -> str:
    try:
        return f"{TOOL} {version('coacd')}"
    except PackageNotFoundError:
        return f"{TOOL} {UNRECORDED}"


def decompose(
    scene_dir: Path, *, params: DecompositionParams = DEFAULT_PARAMS
) -> Decomposition:
    """The scene's proxy into convex parts under its `PARTS_DIR`, with the
    record beside them; written once (an artifact's parts never change),
    and whole: the parts land in a staging folder renamed into place
    before the record is written, so a crash midway leaves nothing a
    later call would mistake for parts."""
    coacd = _coacd()
    scene_dir = Path(scene_dir)
    out_dir = scene_dir / PARTS_DIR
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; parts are written once")
    vertices, faces = read_obj(scene_dir / PROXY_FILE)
    started = time.perf_counter()
    hulls = coacd.run_coacd(coacd.Mesh(vertices, faces), **asdict(params))
    seconds = time.perf_counter() - started
    staging = scene_dir / f".{PARTS_DIR}.{os.getpid()}.tmp"
    staging.mkdir(parents=True)
    files = []
    for i, (hv, hf) in enumerate(hulls):
        name = PART_NAME.format(index=i)
        write_obj(staging / name, np.asarray(hv), np.asarray(hf))
        files.append(name)
    staging.rename(out_dir)
    parts = [(np.asarray(hv, dtype=np.float64), np.asarray(hf)) for hv, hf in hulls]
    record = Decomposition(
        parts=len(parts),
        vertices=int(sum(len(v) for v, _ in parts)),
        vertices_max_per_part=int(max(len(v) for v, _ in parts)),
        seconds=round(seconds, 2),
        tool=_tool_version(),
        params=asdict(params),
        gap=decomposition_gap(vertices, faces, parts, seed=params.seed),
        files=tuple(files),
    )
    (scene_dir / PARTS_FILE).write_text(
        json.dumps(asdict(record), indent=1) + "\n", encoding="utf-8"
    )
    return record


def decomposition_gap(
    vertices: np.ndarray,
    faces: np.ndarray,
    parts: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """How far the hulls sit from the proxy, both ways, in metres:
    `hulls_to_proxy` is material the solver has that the proxy did not
    (a hull bridging a gap between two hurdles); `proxy_to_hulls` is
    proxy surface no hull reaches. Mean and 95th percentile of each,
    with the method; `None` values and a note when Open3D is absent."""
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError:
        return {
            "hulls_to_proxy_mean_m": None,
            "hulls_to_proxy_p95_m": None,
            "proxy_to_hulls_mean_m": None,
            "proxy_to_hulls_p95_m": None,
            "samples": 0,
            "note": "Open3D absent: the decomposition's gap is unmeasured",
        }

    def scene_of(meshes: list[tuple[np.ndarray, np.ndarray]]) -> Any:
        scene = o3d.t.geometry.RaycastingScene()
        for v, f in meshes:
            scene.add_triangles(
                o3d.core.Tensor.from_numpy(np.asarray(v, np.float32)),
                o3d.core.Tensor.from_numpy(np.asarray(f, np.uint32)),
            )
        return scene

    def samples_of(meshes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        merged = o3d.geometry.TriangleMesh()
        for v, f in meshes:
            merged += o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(v, np.float64)),
                o3d.utility.Vector3iVector(np.asarray(f, np.int32)),
            )
        o3d.utility.random.seed(seed)
        return np.asarray(merged.sample_points_uniformly(SURFACE_SAMPLES).points)

    def distances(scene: Any, points: np.ndarray) -> np.ndarray:
        return (
            scene.compute_distance(
                o3d.core.Tensor.from_numpy(points.astype(np.float32))
            )
            .numpy()
            .astype(np.float64)
        )

    proxy = [(vertices, faces)]
    hulls_to_proxy = distances(scene_of(proxy), samples_of(parts))
    proxy_to_hulls = distances(scene_of(parts), samples_of(proxy))
    return {
        "hulls_to_proxy_mean_m": round(float(hulls_to_proxy.mean()), GAP_DIGITS),
        "hulls_to_proxy_p95_m": round(
            float(np.percentile(hulls_to_proxy, 95)), GAP_DIGITS
        ),
        "proxy_to_hulls_mean_m": round(float(proxy_to_hulls.mean()), GAP_DIGITS),
        "proxy_to_hulls_p95_m": round(
            float(np.percentile(proxy_to_hulls, 95)), GAP_DIGITS
        ),
        "samples": SURFACE_SAMPLES,
        "method": "uniform surface samples of each side to the other by Open3D "
        "ray casting",
    }


def load_decomposition(scene_dir: Path) -> Decomposition | None:
    """The record back from a scene folder, or None when never written."""
    path = Path(scene_dir) / PARTS_FILE
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != PARTS_SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')!r}, not {PARTS_SCHEMA!r}")
    raw["files"] = tuple(raw["files"])
    raw["notes"] = tuple(raw.get("notes", ()))
    return Decomposition(**raw)


def ensure_parts(
    scene_dir: Path, *, params: DecompositionParams = DEFAULT_PARAMS
) -> Decomposition:
    """The scene's parts, decomposed on first call and read back after."""
    existing = load_decomposition(scene_dir)
    if existing is not None:
        return existing
    return decompose(scene_dir, params=params)
