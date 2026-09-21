"""The proxy as terrain the solver can touch, two ways, each with its
own measured gap against the proxy it stands for (docs/78 §4.1: the
task sees the scene's gap plus this one).

- `heightfield`: the proxy's top surface sampled on a grid, MuJoCo's
  `hfield`. Exact to the cell for every surface a foot lands on from
  above; anything under an overhang is absent and the fraction of the
  proxy that is, is recorded. The field's own representation for
  legged terrain (every generator terrain in mjlab is one). The default.
- `hulls`: CoACD's convex parts (`scenes.proxy`), one mesh geom each.
  Keeps undersides and objects; roofs concavities by up to the
  decomposition's threshold, measured and recorded on the parts record.

A builder adds its geoms to the stage's terrain body and returns the
facts the manifest records. `TERRAINS` is the registry the stage and
the door read; a new representation is a new entry, not a branch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.scenes.neverwhere import COLLISION_GROUP
from rq_pipeline.scenes.obj import read_obj
from rq_pipeline.scenes.proxy import ensure_parts
from rq_pipeline.scenes.record import PROXY_FILE

HEIGHTFIELD = "heightfield"
HULLS = "hulls"
DEFAULT_TERRAIN = HEIGHTFIELD
PART_MESH = "scene_part_{index:03d}"
HFIELD_NAME = "scene_heightfield"
# The grid cell: the audit's tolerance (docs/78 §3, a paw's width).
CELL_M = 0.02
# The solid base MuJoCo puts under a heightfield, in metres.
HFIELD_BASE_M = 0.1
# Where the top-surface ray starts and how close a sample must be to the
# hit to count as the top surface.
RAY_FROM_M = 50.0
TOP_TOLERANCE_M = 0.001
GAP_SAMPLES = 50_000
GAP_DIGITS = 5
TERRAIN_RGBA = (0.4, 0.4, 0.4, 0.3)
NEEDS_SCENE = "the scene extra (Open3D): uv sync --extra scene"


@dataclass(frozen=True)
class TerrainFacts:
    """What the manifest records about a stage's terrain."""

    kind: str
    geoms: int
    gap: dict[str, Any]
    note: str = ""

    def facts(self) -> dict[str, Any]:
        return asdict(self)


TerrainBuilder = Callable[[Any, Any, Path, list[float] | None], TerrainFacts]


def _friction(geom: Any, friction: list[float] | None) -> None:
    geom.rgba[:] = TERRAIN_RGBA
    if friction is not None:
        geom.friction[:] = [*friction, 0.0, 0.0][:3]


def hulls(
    spec: Any, body: Any, scene_dir: Path, friction: list[float] | None
) -> TerrainFacts:
    """Every convex part as a mesh geom with its vertices embedded."""
    import mujoco  # noqa: PLC0415

    parts = ensure_parts(scene_dir)
    for i, (vertices, _faces) in enumerate(parts.meshes(scene_dir)):
        name = PART_MESH.format(index=i)
        mesh = spec.add_mesh(name=name)
        mesh.uservert = np.asarray(vertices, dtype=np.float64).reshape(-1)
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            group=COLLISION_GROUP,
        )
        _friction(geom, friction)
    return TerrainFacts(
        kind=HULLS,
        geoms=parts.parts,
        gap=dict(parts.gap),
        note=f"{parts.tool}; concavities roofed up to the recorded gap",
    )


def _raycaster(vertices: np.ndarray, faces: np.ndarray) -> Any:
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError as why:
        raise ImportError(NEEDS_SCENE) from why
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.uint32)),
    )
    return scene


def _heights_at(caster: Any, xy: np.ndarray) -> np.ndarray:
    """The top surface's height at each (x, y), NaN where the ray from
    above meets nothing."""
    import open3d as o3d  # noqa: PLC0415

    rays = np.zeros((len(xy), 6), dtype=np.float32)
    rays[:, :2] = xy
    rays[:, 2] = RAY_FROM_M
    rays[:, 5] = -1.0
    t_hit = caster.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
    heights = RAY_FROM_M - t_hit
    heights[~np.isfinite(t_hit)] = np.nan
    return heights


@dataclass(frozen=True)
class Grid:
    """A sampled top surface: heights at grid VERTICES (nrow along y,
    ncol along x), the way MuJoCo's hfield takes them; the surface is
    linear between vertices."""

    heights: np.ndarray
    x0: float
    y0: float
    cell: float

    @property
    def centre(self) -> tuple[float, float]:
        nrow, ncol = self.heights.shape
        return (
            self.x0 + 0.5 * (ncol - 1) * self.cell,
            self.y0 + 0.5 * (nrow - 1) * self.cell,
        )

    @property
    def half(self) -> tuple[float, float]:
        nrow, ncol = self.heights.shape
        return (0.5 * (ncol - 1) * self.cell, 0.5 * (nrow - 1) * self.cell)

    def at(self, xy: np.ndarray) -> np.ndarray:
        """The surface's height at each (x, y): bilinear between the
        four vertices around it (MuJoCo splits a cell into two triangles;
        the difference is inside the cell's curvature)."""
        nrow, ncol = self.heights.shape
        u = np.clip((xy[:, 0] - self.x0) / self.cell, 0, ncol - 1 - 1e-9)
        v = np.clip((xy[:, 1] - self.y0) / self.cell, 0, nrow - 1 - 1e-9)
        c0, r0 = u.astype(int), v.astype(int)
        fu, fv = u - c0, v - r0
        h = self.heights
        return (
            h[r0, c0] * (1 - fu) * (1 - fv)
            + h[r0, c0 + 1] * fu * (1 - fv)
            + h[r0 + 1, c0] * (1 - fu) * fv
            + h[r0 + 1, c0 + 1] * fu * fv
        )


def sample_grid(
    vertices: np.ndarray, faces: np.ndarray, *, cell: float = CELL_M
) -> tuple[Grid, float]:
    """The proxy's top surface at grid vertices `cell` apart over its
    footprint, holes (no surface under the ray) filled with the lowest
    height so a foot finds ground everywhere inside the footprint;
    returns the grid and the filled fraction."""
    lo, hi = vertices.min(0), vertices.max(0)
    ncol = max(int(np.ceil((hi[0] - lo[0]) / cell)) + 1, 2)
    nrow = max(int(np.ceil((hi[1] - lo[1]) / cell)) + 1, 2)
    xs = lo[0] + np.arange(ncol) * cell
    ys = lo[1] + np.arange(nrow) * cell
    gx, gy = np.meshgrid(xs, ys)
    heights = _heights_at(
        _raycaster(vertices, faces), np.column_stack([gx.ravel(), gy.ravel()])
    ).reshape(nrow, ncol)
    holes = ~np.isfinite(heights)
    heights[holes] = np.nanmin(heights) if (~holes).any() else float(lo[2])
    return Grid(heights, float(lo[0]), float(lo[1]), cell), float(holes.mean())


def heightfield_gap(
    vertices: np.ndarray, faces: np.ndarray, grid: Grid, *, seed: int = 0
) -> dict[str, Any]:
    """The top surface against the grid: proxy surface samples that are
    the topmost at their (x, y) against the grid's height there; the
    fraction that are not topmost - vertical faces and undersides - is
    what the heightfield carries only as cliffs between vertices."""
    import open3d as o3d  # noqa: PLC0415

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    o3d.utility.random.seed(seed)
    samples = np.asarray(mesh.sample_points_uniformly(GAP_SAMPLES).points)
    top = _heights_at(_raycaster(vertices, faces), samples[:, :2])
    topmost = np.abs(top - samples[:, 2]) < TOP_TOLERANCE_M
    error = np.abs(samples[topmost, 2] - grid.at(samples[topmost, :2]))
    return {
        "top_surface_mean_m": round(float(error.mean()), GAP_DIGITS)
        if error.size
        else None,
        "top_surface_median_m": round(float(np.median(error)), GAP_DIGITS)
        if error.size
        else None,
        "top_surface_p95_m": round(float(np.percentile(error, 95)), GAP_DIGITS)
        if error.size
        else None,
        "not_top_surface_fraction": round(float(1.0 - topmost.mean()), GAP_DIGITS),
        "cell_m": grid.cell,
        "samples": int(samples.shape[0]),
        "method": "proxy surface samples topmost at their (x, y) against the grid's "
        "bilinear height; the rest are vertical faces and undersides, which the "
        "heightfield carries only as cliffs between vertices",
    }


def heightfield(
    spec: Any, body: Any, scene_dir: Path, friction: list[float] | None
) -> TerrainFacts:
    """The proxy's top surface as one `hfield` geom, data inline."""
    import mujoco  # noqa: PLC0415

    vertices, faces = read_obj(Path(scene_dir) / PROXY_FILE)
    grid, filled = sample_grid(vertices, faces)
    z_min, z_max = float(grid.heights.min()), float(grid.heights.max())
    z_range = max(z_max - z_min, CELL_M)  # a flat proxy still needs a height scale
    field = spec.add_hfield(name=HFIELD_NAME)
    field.nrow, field.ncol = grid.heights.shape
    field.size = [*grid.half, z_range, HFIELD_BASE_M]
    field.userdata = ((grid.heights - z_min) / z_range).ravel()
    geom = body.add_geom(
        name=HFIELD_NAME,
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=HFIELD_NAME,
        group=COLLISION_GROUP,
    )
    geom.pos[:] = [*grid.centre, z_min]
    _friction(geom, friction)
    gap = heightfield_gap(vertices, faces, grid) | {
        "holes_filled_fraction": round(filled, GAP_DIGITS)
    }
    return TerrainFacts(
        kind=HEIGHTFIELD,
        geoms=1,
        gap=gap,
        note=f"{grid.heights.shape[0]}x{grid.heights.shape[1]} vertices {grid.cell} m "
        "apart; undersides absent, holes filled with the lowest height",
    )


TERRAINS: dict[str, TerrainBuilder] = {HEIGHTFIELD: heightfield, HULLS: hulls}


def terrain_names() -> tuple[str, ...]:
    return tuple(TERRAINS)


def terrain_builder(name: str) -> TerrainBuilder:
    try:
        return TERRAINS[name]
    except KeyError:
        raise ValueError(f"no terrain {name!r}; one of {', '.join(TERRAINS)}") from None
