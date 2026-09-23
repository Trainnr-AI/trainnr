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

# The audit's tolerance: a foot's width, two centimetres (docs/78 §3). The
# proxy's grid cell and the heightfield's are this same number, imported.
DEFAULT_TOLERANCE_M = 0.02
MAX_VISIBLE_SAMPLES = 200_000
PROXY_SAMPLES = 100_000
# A proxy's own surface sampled for a gap (the parts', the heightfield's).
SURFACE_SAMPLES = 50_000
GAP_DIGITS = 5
NEEDS_SCENE = "the scene extra (Open3D): uv sync --extra scene"
METHOD = (
    "visible gaussian centres (opacity >= {opacity}) inside the proxy's extent grown "
    "by {margin:g} m, to the proxy surface by Open3D ray casting; proxy surface "
    "samples to the nearest visible centre anywhere by k-d tree; distances in metres"
)


def open3d() -> Any:
    """Open3D, or the refusal that names the extra that brings it: the
    one guard every scene module uses."""
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


# Contact sites: at most this many are kept (drawn with the seed), and
# the audit's scope is the ball of `radius_m` around each.
SITE_SAMPLES = 2000
# Points per distance block: (chunk, sites, 3) float64 is 100 MB at these
# sizes; the first draft's 20,000 made two 1 GB temporaries.
NEAR_CHUNK = 2_000
SITE_METHOD = (
    "scoped to the {sites} contact sites' {radius:g} m surroundings instead of the "
    "proxy's footprint (docs/78 §4.1: the gap where the task touches)"
)


def _near(points: np.ndarray, sites: np.ndarray, radius: float) -> np.ndarray:
    """Which points lie within `radius` of any site (chunked, numpy only)."""
    keep = np.zeros(points.shape[0], dtype=bool)
    r2 = radius * radius
    for start in range(0, points.shape[0], NEAR_CHUNK):
        chunk = points[start : start + NEAR_CHUNK].astype(np.float64)
        d2 = ((chunk[:, None, :] - sites[None, :, :]) ** 2).sum(-1)
        keep[start : start + NEAR_CHUNK] = d2.min(1) <= r2
    return keep


def measure(  # noqa: PLR0913 - the audit's own knobs, each named
    splats: Splats,
    proxy_obj: Path,
    *,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    seed: int = 0,
    sites: np.ndarray | None = None,
    radius_m: float = 0.1,
) -> Gap:
    """The four numbers for one scene, scoped to the proxy's FOOTPRINT:
    the visible gaussians inside the proxy's extent (grown by
    `FOOTPRINT_MARGIN_M`), since a capture sees the whole room and a proxy covers
    the course - the room beyond the course is coverage, reported as
    such, never a gap. The hidden fraction is judged against every
    visible gaussian, wherever it is. With `sites` (world points where a
    task touched), the scope is the ball of `radius_m` around them on
    both sides instead: the gap where the task touches."""
    o3d = open3d()
    rng = np.random.default_rng(seed)
    visible = splats.visible(VISIBLE_OPACITY)
    centres = visible.means.astype(np.float32)
    mesh = o3d.io.read_triangle_mesh(str(proxy_obj))
    if not mesh.has_triangles():
        raise ValueError(f"{proxy_obj}: no triangles to measure against")
    o3d.utility.random.seed(seed)
    surface = np.asarray(mesh.sample_points_uniformly(PROXY_SAMPLES).points)
    scope_note = ""
    if sites is not None:
        sites = np.asarray(sites, dtype=np.float64).reshape(-1, 3)
        if sites.shape[0] > SITE_SAMPLES:
            sites = sites[rng.choice(sites.shape[0], SITE_SAMPLES, replace=False)]
        inside = _near(centres, sites, radius_m)
        surface = surface[_near(surface, sites, radius_m)]
        scope_note = SITE_METHOD.format(sites=sites.shape[0], radius=radius_m)
    else:
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
        scene.compute_distance(o3d.core.Tensor.from_numpy(footprint.astype(np.float32)))
        .numpy()
        .astype(np.float64)
        if footprint.shape[0]
        else np.zeros(0)
    )
    if centres.shape[0] and surface.shape[0]:
        # Open3D's own nearest-neighbour distance, one call for the cloud
        # (the first draft looped 100,000 k-d tree queries from Python)
        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(surface.astype(np.float64))
        )
        visible_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(centres.astype(np.float64))
        )
        to_visible = np.asarray(cloud.compute_point_cloud_distance(visible_cloud))
    else:  # nothing visible, or no surface: the note below says so
        to_visible = np.zeros(0)
    empty = to_proxy.size == 0 or to_visible.size == 0
    return Gap(
        chamfer_m=None
        if empty
        else round(float(0.5 * (to_proxy.mean() + to_visible.mean())), 5),
        p95_m=None if empty else round(float(np.percentile(to_proxy, 95)), 5),
        beyond_tolerance_fraction=None
        if empty
        else round(float((to_proxy > tolerance_m).mean()), 5),
        hidden_fraction=None
        if empty
        else round(float((to_visible > tolerance_m).mean()), 5),
        tolerance_m=tolerance_m,
        visible_samples=int(footprint.shape[0]),
        proxy_samples=int(surface.shape[0]),
        footprint_fraction=round(covered, 5),
        method=METHOD.format(opacity=VISIBLE_OPACITY, margin=FOOTPRINT_MARGIN_M)
        + (f"; {scope_note}" if scope_note else ""),
        note="" if not empty else "no visible gaussian or proxy surface in the scope",
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
