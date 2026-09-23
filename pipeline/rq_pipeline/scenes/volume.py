"""What stands above the ground: the splat's visible centres above a
clearance height as an occupancy volume on a voxel grid, meshed as the
exposed faces of the occupied voxels - a table's top with air beneath
it, a bush, a wall - where the top-surface proxy could only roof them.
The ground the walker stands on stays the top surface of the centres
BELOW the clearance (`capture.top_surface_mesh`); the two together are
the scene's proxy, and a stage can carry them as a heightfield under
convex parts (`terrain.OVERHANGS`), so a walker goes under the table
instead of over it (2026-09-23, the garden).

Blocky by construction, at the voxel: a collision volume, not a
picture; CoACD's parts merge the blocks into hulls. numpy only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Below this height above the floor the world is ground the walker
# stands on (the heightfield); at or above it, things it goes under or
# around. The Go2 stands 0.32 m at the base; a 25 cm step is the most
# the ground may hold as a step, anything taller is a body.
OVERHANG_CLEARANCE_M = 0.25
VOXEL_M = 0.05
MIN_CENTRES = 2  # a voxel with fewer visible centres is a floater, not a thing
HEIGHT_CAP_M = 2.5  # above this the walker never reaches; the volume stops
MARGIN_M = 0.5  # the volume reaches this far past the ground's footprint
# A connected component of fewer voxels is a speck, not a thing the walker
# meets: 64 voxels at 5 cm is eight litres, a leaf cluster. The garden's
# volume was 7,142 components; 562 held eight voxels or more and took
# CoACD an hour, 78 held sixty-four or more with four fifths of the
# voxels and take minutes (2026-09-23).
MIN_COMPONENT_VOXELS = 64
# The six face directions of a voxel, and the corner offsets of each face
# wound outward (counter-clockwise seen from outside).
_FACES = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)


def occupancy(
    centres: np.ndarray, *, cell: float = VOXEL_M, min_count: int = MIN_CENTRES
) -> tuple[np.ndarray, np.ndarray]:
    """The voxels holding at least `min_count` centres, as a boolean grid
    (nx, ny, nz) and the world position of the grid's corner."""
    if centres.shape[0] == 0:
        return np.zeros((0, 0, 0), dtype=bool), np.zeros(3)
    origin = np.floor(centres.min(0) / cell) * cell
    index = np.floor((centres - origin) / cell).astype(np.int64)
    shape = index.max(0) + 1
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(counts, (index[:, 0], index[:, 1], index[:, 2]), 1)
    return counts >= min_count, origin


def voxel_mesh(
    grid: np.ndarray, *, cell: float, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The exposed faces of the occupied voxels as triangles, outward
    wound, vertices shared: a closed surface around every occupied
    component."""
    if grid.size == 0 or not grid.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    padded = np.pad(grid, 1)
    quads: list[np.ndarray] = []
    for (dx, dy, dz), corners in _FACES:
        # occupied here, empty on that side
        inner = padded[1:-1, 1:-1, 1:-1]
        beside = padded[
            1 + dx : padded.shape[0] - 1 + dx,
            1 + dy : padded.shape[1] - 1 + dy,
            1 + dz : padded.shape[2] - 1 + dz,
        ]
        where = np.argwhere(inner & ~beside)  # (m, 3) voxel indices
        if where.shape[0] == 0:
            continue
        offsets = np.asarray(corners, dtype=np.int64)  # (4, 3)
        quads.append(where[:, None, :] + offsets[None, :, :])  # (m, 4, 3)
    corners_all = np.concatenate(quads, 0)  # (q, 4, 3) integer corner coordinates
    flat = corners_all.reshape(-1, 3)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    idx = inverse.reshape(-1, 4)
    faces = np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]], 0)
    vertices = origin[None, :] + unique.astype(np.float64) * cell
    return vertices, faces.astype(np.int64)


Mesh = tuple[np.ndarray, np.ndarray]


def overhang_components(  # noqa: PLR0913 - the volume's own knobs, each named
    centres: np.ndarray,
    *,
    clearance: float = OVERHANG_CLEARANCE_M,
    cap: float = HEIGHT_CAP_M,
    cell: float = VOXEL_M,
    min_count: int = MIN_CENTRES,
    margin: float = MARGIN_M,
    min_voxels: int = MIN_COMPONENT_VOXELS,
) -> tuple[list[Mesh], dict[str, Any]]:
    """Everything between the clearance and the cap, over the ground's
    footprint grown by the margin (the far wall the walker never reaches
    is left out), as one voxel surface per connected thing - a table, a
    bush, a wall - each closed, specks dropped; with the facts. Each
    thing is decomposed on its own: as one mesh, the whole garden's
    overhangs left CoACD's samples too sparse to see the air under the
    table and it filled it (2026-09-23)."""
    from scipy import ndimage  # noqa: PLC0415

    ground, _ = split_by_clearance(centres, clearance)
    above = centres[(centres[:, 2] >= clearance) & (centres[:, 2] < cap)]
    if ground.shape[0]:
        lo, hi = ground[:, :2].min(0) - margin, ground[:, :2].max(0) + margin
        inside = np.all((above[:, :2] >= lo) & (above[:, :2] <= hi), axis=1)
        above = above[inside]
    grid, origin = occupancy(above, cell=cell, min_count=min_count)
    facts: dict[str, Any] = {
        "clearance_m": clearance,
        "cap_m": cap,
        "voxel_m": cell,
        "min_centres": min_count,
        "margin_m": margin,
        "min_component_voxels": min_voxels,
        "centres": int(above.shape[0]),
        "voxels": 0,
        "components": 0,
        "specks_dropped": 0,
        "faces": 0,
    }
    if grid.size == 0 or not grid.any():
        return [], facts
    labels, count = ndimage.label(grid)
    sizes = np.bincount(labels.ravel())[1:]
    meshes: list[Mesh] = []
    for label in np.flatnonzero(sizes >= min_voxels) + 1:
        vertices, faces = voxel_mesh(labels == label, cell=cell, origin=origin)
        meshes.append((vertices, faces))
    facts["voxels"] = int(sizes[sizes >= min_voxels].sum())
    facts["components"] = len(meshes)
    facts["specks_dropped"] = int(count - len(meshes))
    facts["faces"] = int(sum(f.shape[0] for _, f in meshes))
    return meshes, facts


def joined(meshes: list[Mesh]) -> Mesh:
    """Several meshes as one, faces re-indexed."""
    if not meshes:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    offsets = np.cumsum([0, *(v.shape[0] for v, _ in meshes[:-1])])
    return (
        np.concatenate([v for v, _ in meshes], 0),
        np.concatenate([f + o for (_, f), o in zip(meshes, offsets, strict=True)], 0),
    )


def overhang_mesh(
    centres: np.ndarray, **knobs: Any
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The components joined: one surface with the facts."""
    meshes, facts = overhang_components(centres, **knobs)
    vertices, faces = joined(meshes)
    return vertices, faces, facts


def split_by_clearance(
    centres: np.ndarray, clearance: float = OVERHANG_CLEARANCE_M
) -> tuple[np.ndarray, np.ndarray]:
    """(ground, above): the centres the walker stands on and the ones it
    goes under or around."""
    below = centres[:, 2] < clearance
    return centres[below], centres[~below]
