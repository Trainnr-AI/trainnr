"""The mesh-true Rerun 3D mirror: any MuJoCo model, drawn from its own
physics state.

Every geom becomes what the physics uses — mesh geoms as real Mesh3D
entities (geometry logged once, one Transform3D per frame), primitives
as oriented solids (cylinders, capsules and spheres by their bounding
box) — posed from geom_xpos/xmat. Live dashboards, replays and sim
predictions all draw the same shapes the solver collides, and none of
them invents a skeleton. Grown from the yellow rig's `tools/_rig3d.py`
(which now re-exports from here); promoted 2026-09-01 so the mjlab
recorder can mirror ANY entity the trainer runs — the Studio's 3D view
follows what is actually happening, not a canned scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

RIG_PATH = "world/rig"  # where every rig tool logs the mirror
# The Studio's Rerun ingest door — the ONE home for the address every
# feed and recorder connects to (the Rust shell binds the same port;
# crates/studio-shell/src/main.rs stays a documented mirror).
STUDIO_ADDRESS = "rerun+http://127.0.0.1:9876/proxy"


def mat_to_xyzw(flat: Any) -> Any:
    """Rotation matrix -> xyzw quaternion, via MuJoCo's own routine
    (mju_mat2Quat is wxyz; Rerun wants xyzw)."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(flat, dtype=float).reshape(9))
    return [quat[1], quat[2], quat[3], quat[0]]


def _bounding_half(mujoco: Any, kind: int, size: Any) -> list | None:
    """A primitive geom's bounding half-sizes; None for planes."""
    if kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return size.tolist()
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return [size[0], size[0], size[1]]
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return [size[0], size[0], size[1] + size[0]]
    return [size[0]] * 3


class RigMirror:
    """Collects a model's geoms once; logs their poses per frame.

    `log(data, path)` accepts anything exposing per-geom `geom_xpos`
    (ngeom, 3) and `geom_xmat` ((ngeom, 9) or (ngeom, 3, 3)) as numpy
    arrays — a raw `mujoco.MjData`, or one world sliced out of a
    batched engine. MuJoCo compiles mesh vertices into the geom frame,
    so geom_xpos/xmat is the whole transform.

    model_colors=True paints every geom with its geom_rgba (matching
    the MuJoCo render); the default keeps the yellow-rig name-prefix
    palette the standing dashboards were tuned on.
    """

    def __init__(
        self,
        model: Any,
        skip: Sequence[str] = ("floor",),
        model_colors: bool = False,
        skip_groups: Sequence[int] = (),
        skip_prefixes: Sequence[str] = (),
    ) -> None:
        import mujoco  # noqa: PLC0415 - sim extra

        self.geoms: list[int] = []  # geom ids drawn as boxes
        self.half_sizes: list[list[float]] = []
        self.colors: list[list[int]] = []  # rgb per box geom
        # (geom id, entity name, vertices, faces, rgba)
        self.meshes: list[tuple[int, str, Any, Any, list[int]]] = []
        self._mesh_logged: set[str] = set()
        for g in range(model.ngeom):
            name = model.geom(g).name
            if name in skip or int(model.geom_group[g]) in skip_groups:
                continue
            if any(name.startswith(prefix) for prefix in skip_prefixes):
                continue  # e.g. every duck but the narrated one
            size = model.geom_size[g]
            kind = int(model.geom_type[g])
            # A geom with a material takes the material's colour (the
            # geom's own rgba is the grey default then).
            mat = int(model.geom_matid[g])
            rgba_f = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
            rgba = [int(c * 255) for c in rgba_f]
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(model.geom_dataid[g])
                v0, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                f0, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                self.meshes.append(
                    (
                        g,
                        name or f"geom{g}",
                        model.mesh_vert[v0 : v0 + nv].copy(),
                        model.mesh_face[f0 : f0 + nf].copy(),
                        rgba,
                    )
                )
                continue
            half = _bounding_half(mujoco, kind, size)
            if half is None or max(half) <= 0.0:
                # Planes (infinite; mirrored as a zero box they would
                # ANCHOR the view's bounds at their own position, so a
                # robot a meter away shrinks to a corner of the frame —
                # measured on the duck, 2026-09-01) and zero-size geoms
                # draw nothing; the caller supplies its own ground.
                continue
            self.geoms.append(g)
            self.half_sizes.append(half)
            if model_colors:
                self.colors.append(rgba[:3])
            elif name.startswith("yarm"):
                self.colors.append([255, 190, 40])
            elif name.startswith("prop"):
                self.colors.append([230, 40, 40])
            else:
                self.colors.append([90, 130, 220])

    def log(self, data: Any, path: str = RIG_PATH, *, static: bool = False) -> None:
        """`static=True` for a snapshot (an artifact shown as itself): the
        poses then hold at every time, instead of landing at the clock's
        current instant and vanishing behind a paused cursor (2026-09-09)."""
        import rerun as rr  # noqa: PLC0415 - viz extra

        if self.geoms:
            rr.log(
                path,
                rr.Boxes3D(
                    centers=data.geom_xpos[self.geoms],
                    half_sizes=self.half_sizes,
                    quaternions=[
                        rr.Quaternion(xyzw=mat_to_xyzw(data.geom_xmat[g]))
                        for g in self.geoms
                    ],
                    colors=self.colors,
                    fill_mode="solid",
                ),
                static=static,
            )
        for g, name, verts, faces, rgba in self.meshes:
            entity = f"{path}/{name}"
            if entity not in self._mesh_logged:
                rr.log(
                    entity,
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=faces,
                        albedo_factor=rgba,
                    ),
                    static=True,
                )
                self._mesh_logged.add(entity)
            rr.log(
                entity,
                rr.Transform3D(
                    translation=data.geom_xpos[g],
                    mat3x3=data.geom_xmat[g].reshape(3, 3),
                ),
                static=static,
            )
