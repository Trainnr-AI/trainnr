"""The gap: the surface the eye sees against the geometry the solver
touches (docs/78 §3; the audit the field skips, docs/e2e-research/75
§2 - altering only the collision geometry drops real success 61.7
points while the simulation looks unchanged).

Four numbers, always recorded, never thresholded here:

- chamfer: the mean distance from a visible gaussian's centre to the
  proxy surface, and from a point on the proxy surface to the nearest
  visible gaussian, averaged;
- p95: the 95th percentile of the visible-to-proxy distance - the
  worst of what the eye sees that the solver misses, minus outliers;
- beyond-tolerance fraction: how much of the visible surface sits
  farther than the tolerance from any collider;
- hidden fraction: how much of the proxy surface sits farther than the
  tolerance from any visible gaussian - geometry the solver touches
  that the eye never sees.

And one more number, coverage: the share of the visible surface that
lies inside the proxy's footprint at all. A capture sees the whole
room; a proxy covers the course. The three distance numbers are scoped
to the footprint, so a room beyond the course reads as coverage, not
as a gap (the first audit of a real scene, 2026-09-22, had mixed the
two and reported a 3.2 m gap on a well-aligned floor).

The proxy distances come from Open3D's ray-casting scene (the `scene`
extra); the gaussian-side distances from a k-d tree over the visible
centres. Without Open3D the audit refuses by name; a scene record then
carries `None` in every distance and the reason in `note`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.scenes.record import Gap
from rq_pipeline.scenes.splat import VISIBLE_OPACITY, Splats

DEFAULT_TOLERANCE_M = 0.02  # a paw's width: two centimetres
MAX_VISIBLE_SAMPLES = 200_000
PROXY_SAMPLES = 100_000
NEEDS_SCENE = "the scene extra (Open3D): uv sync --extra scene"
METHOD = (
    "visible gaussian centres (opacity >= {opacity}) inside the proxy's extent grown "
    "by 0.5 m, to the proxy surface by Open3D ray casting; proxy surface "
    "samples to the nearest visible centre anywhere by k-d tree; distances in metres"
)


def _open3d() -> Any:
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError as why:
        raise ImportError(NEEDS_SCENE) from why
    return o3d


# How far beyond the proxy's own extent the audit still looks: a floor
# proxy five centimetres above the visible floor must be inside its own
# footprint, a corridor wall three metres away must not.
FOOTPRINT_MARGIN_M = 0.5


def _footprint(surface: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """The proxy's axis-aligned extent, grown by `margin` on every side."""
    return surface.min(0) - margin, surface.max(0) + margin


def measure(
    splats: Splats,
    proxy_obj: Path,
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    seed: int = 0,
) -> Gap:
    """The four numbers for one scene, scoped to the proxy's FOOTPRINT:
    the visible gaussians inside the proxy's extent (grown by
    `FOOTPRINT_MARGIN_M`), since a capture sees the whole room and a proxy covers
    the course - the room beyond the course is coverage, reported as
    such, never a gap. The hidden fraction is judged against every
    visible gaussian, wherever it is."""
    o3d = _open3d()
    rng = np.random.default_rng(seed)
    visible = splats.visible(VISIBLE_OPACITY)
    centres = visible.means.astype(np.float32)
    mesh = o3d.io.read_triangle_mesh(str(proxy_obj))
    if not mesh.has_triangles():
        raise ValueError(f"{proxy_obj}: no triangles to measure against")
    o3d.utility.random.seed(seed)
    surface = np.asarray(mesh.sample_points_uniformly(PROXY_SAMPLES).points)
    lo, hi = _footprint(surface, FOOTPRINT_MARGIN_M)
    inside = np.all((centres >= lo) & (centres <= hi), axis=1)
    covered = float(inside.mean()) if centres.shape[0] else 0.0
    footprint = centres[inside]
    if footprint.shape[0] > MAX_VISIBLE_SAMPLES:
        footprint = footprint[
            rng.choice(footprint.shape[0], MAX_VISIBLE_SAMPLES, replace=False)
        ]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    to_proxy = (
        scene.compute_distance(o3d.core.Tensor(footprint, dtype=o3d.core.Dtype.Float32))
        .numpy()
        .astype(np.float64)
        if footprint.shape[0]
        else np.zeros(0)
    )
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres.astype(np.float64)))
    )
    to_visible = np.array(
        [tree.search_knn_vector_3d(p, 1)[2][0] ** 0.5 for p in surface]
    )
    empty = to_proxy.size == 0
    return Gap(
        chamfer_m=None
        if empty
        else round(float(0.5 * (to_proxy.mean() + to_visible.mean())), 5),
        p95_m=None if empty else round(float(np.percentile(to_proxy, 95)), 5),
        beyond_tolerance_fraction=None
        if empty
        else round(float((to_proxy > tolerance_m).mean()), 5),
        hidden_fraction=round(float((to_visible > tolerance_m).mean()), 5),
        tolerance_m=tolerance_m,
        visible_samples=int(footprint.shape[0]),
        proxy_samples=int(surface.shape[0]),
        footprint_fraction=round(covered, 5),
        method=METHOD.format(opacity=VISIBLE_OPACITY),
        note="" if not empty else "no visible gaussian inside the proxy's footprint",
    )


def unmeasured(reason: str, *, tolerance_m: float = DEFAULT_TOLERANCE_M) -> Gap:
    """The record's honest shape when the audit could not run."""
    return Gap(
        chamfer_m=None,
        p95_m=None,
        beyond_tolerance_fraction=None,
        hidden_fraction=None,
        tolerance_m=tolerance_m,
        visible_samples=0,
        proxy_samples=0,
        footprint_fraction=None,
        method=METHOD.format(opacity=VISIBLE_OPACITY),
        note=reason,
    )
