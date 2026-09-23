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
from rq_pipeline.scenes.record import OVERHANG_FILE, PROXY_FILE, UNRECORDED

PARTS_DIR = "proxy-parts"
PARTS_FILE = "proxy-parts.json"
OVERHANG_PARTS_DIR = "overhang-parts"
OVERHANG_PARTS_FILE = "overhang-parts.json"
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
# The overhangs of a whole scene are many things (a table, bushes, walls);
# capped at the proxy's 64 parts CoACD would merge them and a merged hull
# fills the air under a table. Threshold and the rest as the proxy's.
OVERHANG_PARAMS = DecompositionParams(max_convex_hull=512)


@dataclass(frozen=True)
class PartsFiles:
    """Which mesh is decomposed and where its parts and record land."""

    source: str
    directory: str
    record: str


PROXY_PARTS = PartsFiles(source=PROXY_FILE, directory=PARTS_DIR, record=PARTS_FILE)
OVERHANG_PARTS = PartsFiles(
    source=OVERHANG_FILE, directory=OVERHANG_PARTS_DIR, record=OVERHANG_PARTS_FILE
)


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
    directory: str = PARTS_DIR

    def part_paths(self, scene_dir: Path) -> list[Path]:
        return [Path(scene_dir) / self.directory / f for f in self.files]

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
    scene_dir: Path,
    *,
    params: DecompositionParams = DEFAULT_PARAMS,
    files: PartsFiles = PROXY_PARTS,
    meshes: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> Decomposition:
    """The scene's proxy into convex parts under its `PARTS_DIR`, with the
    record beside them; written once (an artifact's parts never change),
    and whole: the parts land in a staging folder renamed into place
    before the record is written, so a crash midway leaves nothing a
    later call would mistake for parts."""
    coacd = _coacd()
    scene_dir = Path(scene_dir)
    out_dir = scene_dir / files.directory
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; parts are written once")
    vertices, faces = read_obj(scene_dir / files.source)
    started = time.perf_counter()
    # `meshes`: the source as separate things (the overhangs' components),
    # each decomposed on its own so CoACD's samples resolve each one, in
    # parallel (one component a minute on one core was an hour for a
    # garden, 2026-09-23); the gap is still measured against the whole source.
    hulls = decompose_meshes(
        meshes if meshes is not None else [(vertices, faces)], params, coacd=coacd
    )
    seconds = time.perf_counter() - started
    staging = scene_dir / f".{files.directory}.{os.getpid()}.tmp"
    staging.mkdir(parents=True)
    names = []
    for i, (hv, hf) in enumerate(hulls):
        name = PART_NAME.format(index=i)
        write_obj(staging / name, np.asarray(hv), np.asarray(hf))
        names.append(name)
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
        files=tuple(names),
        source=files.source,
        directory=files.directory,
    )
    (scene_dir / files.record).write_text(
        json.dumps(asdict(record), indent=1) + "\n", encoding="utf-8"
    )
    return record


def _hulls_of(
    mesh: tuple[np.ndarray, np.ndarray], params: dict[str, Any]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One mesh's convex parts (a worker: imports CoACD itself)."""
    coacd = _coacd()
    vertices, faces = mesh
    return [
        (np.asarray(v, dtype=np.float64), np.asarray(f))
        for v, f in coacd.run_coacd(coacd.Mesh(vertices, faces), **params)
    ]


def decompose_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray]],
    params: DecompositionParams,
    *,
    coacd: Any = None,
    workers: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every mesh's convex parts, in order; several meshes across a
    process pool (CoACD holds one core), one mesh inline."""
    knobs = asdict(params)
    if len(meshes) <= 1:
        return [h for mesh in meshes for h in _hulls_of(mesh, knobs)]
    import multiprocessing  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    # spawned, not forked: the parent holds Rerun's and Open3D's threads
    # by now, and a fork of a threaded parent is the deadlock Python 3.12
    # warns about (docs/32 chose spawn for its workers too)
    count = min(len(meshes), workers or max(1, (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(
        max_workers=count, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_hulls_of, meshes, [knobs] * len(meshes)))
    return [h for hulls in results for h in hulls]


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


def load_decomposition(
    scene_dir: Path, files: PartsFiles = PROXY_PARTS
) -> Decomposition | None:
    """The record back from a scene folder, or None when never written."""
    path = Path(scene_dir) / files.record
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != PARTS_SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')!r}, not {PARTS_SCHEMA!r}")
    raw["files"] = tuple(raw["files"])
    raw["notes"] = tuple(raw.get("notes", ()))
    return Decomposition(**raw)


def ensure_parts(
    scene_dir: Path,
    *,
    params: DecompositionParams = DEFAULT_PARAMS,
    files: PartsFiles = PROXY_PARTS,
    meshes: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> Decomposition:
    """The scene's parts, decomposed on first call and read back after."""
    existing = load_decomposition(scene_dir, files)
    if existing is not None:
        return existing
    return decompose(scene_dir, params=params, files=files, meshes=meshes)
