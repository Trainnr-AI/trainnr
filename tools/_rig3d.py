"""Shared Rerun 3D mirror for the yellow rig.

Every MuJoCo geom becomes an oriented solid box in Rerun (cylinders,
capsules and spheres by their bounding box), posed from geom_xpos/xmat —
so the live dashboard, the replay and the sim prediction all draw the
same shapes the physics uses, and none of them invents a skeleton.
"""

import math

import mujoco
import numpy as np
import rerun as rr


def mat_to_xyzw(flat):
    m = np.asarray(flat).reshape(3, 3)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        r = math.sqrt(1 + t)
        w = 0.5 * r
        x = (m[2, 1] - m[1, 2]) / (2 * r)
        y = (m[0, 2] - m[2, 0]) / (2 * r)
        z = (m[1, 0] - m[0, 1]) / (2 * r)
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = math.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0, 0.0, 0.0]
        q[i] = 0.5 * r
        q[j] = (m[j, i] + m[i, j]) / (2 * r)
        q[k] = (m[k, i] + m[i, k]) / (2 * r)
        w = (m[k, j] - m[j, k]) / (2 * r)
        x, y, z = q
    return [x, y, z, w]


class RigMirror:
    """Collects a model's geoms once; logs their poses per frame.

    Primitive geoms go out as one Boxes3D batch. Mesh geoms (the SO-101
    arm in the components scenes — the yellow twin has none) go out as
    real Mesh3D entities: geometry logged once, only a Transform3D per
    frame, so the Rerun rig looks like the MuJoCo window instead of a
    pile of bounding cubes. MuJoCo compiles mesh vertices into the geom
    frame, so geom_xpos/xmat is the whole transform.

    model_colors=True paints every geom with its geom_rgba (matching
    the MuJoCo render); the default keeps the yellow-rig name-prefix
    palette the standing dashboards were tuned on.
    """

    def __init__(self, model, skip=("floor",), model_colors=False, skip_groups=()):
        self.geoms, self.half_sizes, self.colors = [], [], []
        self.meshes = []  # (geom id, entity name, vertices, faces, rgba)
        self._mesh_logged: set[str] = set()
        for g in range(model.ngeom):
            name = model.geom(g).name
            if name in skip or int(model.geom_group[g]) in skip_groups:
                continue
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
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                half = size.tolist()
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                half = [size[0], size[0], size[1]]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                half = [size[0], size[0], size[1] + size[0]]
            else:
                half = [size[0]] * 3
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

    def log(self, data, path="world/rig"):
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
            )
