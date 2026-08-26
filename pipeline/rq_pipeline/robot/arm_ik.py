"""Damped-least-squares IK for one arm — the demo generator's reach.

T5's scripted demonstrations must pick parts at RANDOMIZED positions;
hand-tuned joint waypoints (the SO-101 tasks' method) cannot follow a
part that moved. This is the missing capability, kept deliberately
small: position IK to a site, with a soft "gripper points down"
orientation term, solved by damped least squares on the arm's six
joints. It edits `data.qpos` in place and reports whether it converged
— a False return is a reach the choreographer must not pretend
happened.

Demo generation is stage ④, not evaluation: the generator may read
privileged state (part poses) freely. The TRAINED policy never sees
any of this — it learns from the demonstrations' pixels and states.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def solve_arm_ik(  # noqa: PLR0913, PLR0915 - the solver: its knobs and its loop
    model: Any,
    data: Any,
    *,
    site: str,
    joints: Sequence[str],
    target_pos: Any,
    grip_geoms: Sequence[str] | None = None,
    approach_axis: Sequence[float] = (0.0, 0.0, -1.0),
    down_weight: float = 0.3,
    pos_tol: float = 0.005,
    max_iters: int = 100,
    damping: float = 1e-2,
    step_limit: float = 0.3,
) -> bool:
    """Move `joints` so the gripper reaches `target_pos`, down-ish.

    Without `grip_geoms`, the positioned point is `site`. With them
    (the two finger-pad geoms), the positioned point is THEIR MIDPOINT
    — the actual grip centre — via averaged geom Jacobians. That mode
    exists because positioning the site with a hand-measured pad
    offset failed in the worst way available: on low reaches the
    solver sacrificed the soft orientation term, the wrist pitched
    flat, the pads swung UP behind the site, and the offset chased a
    moving frame (measured: a "descend" that rose 6 cm and closed on
    air 12 cm above the part). Position what you mean to place.

    Returns True when the position error is within `pos_tol` metres.
    The orientation term aligns the site's z-axis with world -z at
    weight `down_weight` — a bias, not a constraint. `step_limit`
    (rad) clamps each iteration's joint step; joint ranges respected.
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    single_dof_joint_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0:
        raise ValueError(f"no site named {site!r}")
    geom_ids = []
    for name in grip_geoms or ():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"no geom named {name!r}")
        geom_ids.append(geom_id)
    joint_ids = []
    for name in joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"no joint named {name!r}")
        # Compared as ints: MuJoCo 3.12's pybind enum no longer equals a
        # numpy integer in a membership test (3.11 did), so a hinge stopped
        # looking like a hinge. Caught twice the same night: the WSL train
        # venv's 3.12 refused every hinge in the demo generator (2026-08-26),
        # and an accidental 3.11->3.12 lock bump on the Mac (2026-08-27).
        if int(model.jnt_type[joint_id]) not in single_dof_joint_types:
            # A free or ball joint has multi-dof addressing; the scalar
            # dq indexing below would write plausible-looking garbage.
            raise ValueError(
                f"joint {name!r} is not single-dof (hinge/slide); "
                "solve_arm_ik cannot drive it"
            )
        joint_ids.append(joint_id)
    import numpy as _np  # noqa: PLC0415

    dof_columns = [int(model.jnt_dofadr[j]) for j in joint_ids]
    qpos_rows = [int(model.jnt_qposadr[j]) for j in joint_ids]
    # An UNLIMITED joint carries range (0, 0); clamping to that froze
    # every rangeless joint at zero — caught by this module's first
    # direct test (2026-08-26). Only declared limits clamp.
    limited = model.jnt_limited[joint_ids].astype(bool)
    lower = _np.where(limited, model.jnt_range[joint_ids, 0], -_np.inf)
    upper = _np.where(limited, model.jnt_range[joint_ids, 1], _np.inf)

    target = np.asarray(target_pos, dtype=float)
    down = np.asarray(approach_axis, dtype=float)
    down = down / np.linalg.norm(down)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    geom_jac = np.zeros((3, model.nv))

    def positioned_point() -> Any:
        if geom_ids:
            return np.mean([data.geom_xpos[g] for g in geom_ids], axis=0)
        return data.site_xpos[site_id]

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        pos_err = target - positioned_point()
        site_z = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        # Rotation error that drives site_z toward `down`: the cross
        # product is the axis*sin(angle) of the needed rotation.
        rot_err = np.cross(site_z, down)
        if np.linalg.norm(pos_err) < pos_tol:
            return True
        if geom_ids:
            jacp[:] = 0.0
            for geom_id in geom_ids:
                mujoco.mj_jacGeom(model, data, geom_jac, None, geom_id)
                jacp += geom_jac / len(geom_ids)
            mujoco.mj_jacSite(model, data, geom_jac, jacr, site_id)
        else:
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, dof_columns], down_weight * jacr[:, dof_columns]])
        err = np.concatenate([pos_err, down_weight * rot_err])
        # Damped least squares: dq = J^T (J J^T + lambda I)^-1 err
        gram = jac @ jac.T + damping * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(gram, err)
        dq = np.clip(dq, -step_limit, step_limit)
        for index, row in enumerate(qpos_rows):
            data.qpos[row] = float(
                np.clip(data.qpos[row] + dq[index], lower[index], upper[index])
            )
    mujoco.mj_forward(model, data)
    return bool(np.linalg.norm(target - positioned_point()) < pos_tol)
