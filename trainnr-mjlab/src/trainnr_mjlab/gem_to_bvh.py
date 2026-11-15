"""GEM-X's world motion as the SOMA Retargeter's BVH. Run by GEM-X's own
interpreter, never imported by trainnr (it needs GEM-X's SOMA runtime):

    <GEM-X>/.venv/bin/python gem_to_bvh.py motion.pt skeleton.bvh out.bvh assets/soma

GEM-X saves SOMA parameters; the retargeter reads a BVH on the same 78
SOMA joints in which every joint carries its full local transform each
frame (centimetres, Z-Y-X Euler degrees, Y up). The joints, their order
and their rotation conventions match: SOMA's rest pose reproduces the
retargeter's own `soma_zero_frame0.bvh` (checked 2026-11-15). The
hierarchy is copied from that skeleton; Root stays at the origin.

GEM-X's world can stand tilted: on the court clip its floor leaned 12.8
degrees and the hips rose 70 cm over 22 s. `level` fits a plane to each
frame's lowest foot point and turns it level, then lifts each frame by
the lowest foot's height smoothed over a second, so slow drift goes and
a jump stays. The numbers are printed as JSON on the last line.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

FEET = ("LeftToeEnd", "RightToeEnd", "LeftFoot", "RightFoot")
FLOOR_WINDOW_S = 1.0
PARALLEL = 1e-9  # a floor normal this close to up needs no turn
CM_PER_M = 100.0


def level(
    world: np.ndarray, feet: list[int], fps: float
) -> tuple[np.ndarray, dict[str, float]]:
    """`world` (frames, joints, 4, 4) on a level floor, and what it took."""
    points = world[:, feet][:, :, :3, 3]
    low = points[np.arange(len(world)), points[:, :, 1].argmin(1)]
    plane = np.c_[low[:, 0], low[:, 2], np.ones(len(low))]
    a, b, _ = np.linalg.lstsq(plane, low[:, 1], rcond=None)[0]
    normal = np.array([-a, 1.0, -b]) / np.linalg.norm([-a, 1.0, -b])
    axis = np.cross(normal, [0.0, 1.0, 0.0])
    turn = np.eye(4)
    if np.linalg.norm(axis) > PARALLEL:
        angle = np.arcsin(min(float(np.linalg.norm(axis)), 1.0))
        turn[:3, :3] = Rotation.from_rotvec(
            axis / np.linalg.norm(axis) * angle
        ).as_matrix()
    world = turn @ world
    lowest = world[:, feet][:, :, 1, 3].min(1)
    k = max(1, round(FLOOR_WINDOW_S * fps))
    padded = np.pad(lowest, (k, k), mode="edge")
    # the median of each window's lower half: the floor under the feet
    floor = np.array(
        [np.median(np.sort(padded[i : i + 2 * k + 1])[:k]) for i in range(len(lowest))]
    )
    world[:, :, 1, 3] -= floor[:, None]
    return world, {
        "tilt_deg": round(float(np.degrees(np.arccos(normal[1]))), 2),
        "lift_m_min": round(float(floor.min()), 3),
        "lift_m_max": round(float(floor.max()), 3),
    }


def main(motion_pt: str, skeleton_bvh: str, out_bvh: str, soma_assets: str) -> None:
    import torch  # noqa: PLC0415 - GEM-X's environment
    from gem.common.soma import (  # noqa: PLC0415
        SomaFixedGs1BodyAdapter,
        SomaRuntimeConfig,
    )
    from gem.inference.timing import prediction_fps  # noqa: PLC0415

    result: dict[str, Any] = torch.load(
        motion_pt, map_location="cpu", weights_only=False
    )
    params = result["pred_soma_params_global"]
    fps = float(prediction_fps(result))
    frames = int(torch.as_tensor(result["length"]).item())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    body = SomaFixedGs1BodyAdapter(
        SomaRuntimeConfig(
            data_root=soma_assets,
            device=device,
            lod="mid",
            apply_correctives=False,
            enable_procedural_transforms=True,
            procedural_reference="runtime",
        )
    ).to(device)
    names = list(body.runtime.public_joint_names)
    parents = [int(p) for p in body.runtime.public_joint_parent_ids]
    chunks = []
    with torch.inference_mode():
        for start in range(0, frames, 64):
            stop = min(start + 64, frames)
            values = {
                k: params[k][0, start:stop].to(device, torch.float32)
                for k in ("global_orient", "body_pose", "transl")
            }
            for k in ("identity_coeffs", "scale_params"):
                x = params[k][0]
                x = x[None].expand(stop - start, -1) if x.ndim == 1 else x[start:stop]
                values[k] = x.to(device, torch.float32)
            out = body(
                **values,
                pose2rot=True,
                apply_correctives=False,
                repose_to_bind_pose=True,
            )
            chunks.append(out["transforms"].cpu().numpy())
    world, floor = level(np.concatenate(chunks), [names.index(f) for f in FEET], fps)
    world[:, :, :3, 3] *= CM_PER_M
    local = np.empty_like(world)
    local[:, 0] = np.eye(4)
    for j in range(1, len(names)):
        p = parents[j]
        local[:, j] = (
            world[:, j] if p == 0 else np.linalg.inv(world[:, p]) @ world[:, j]
        )
    euler = Rotation.from_matrix(local[:, :, :3, :3].reshape(-1, 3, 3)).as_euler(
        "ZYX", degrees=True
    )
    rows = np.concatenate(
        [local[:, :, :3, 3], euler.reshape(frames, len(names), 3)], axis=2
    ).reshape(frames, -1)
    head = open(skeleton_bvh, encoding="utf-8").read().split("MOTION")[0]
    with open(out_bvh, "w", encoding="utf-8") as f:
        f.write(f"{head}MOTION\nFrames: {frames}\nFrame Time: {1.0 / fps:.6f}\n")
        np.savetxt(f, rows, fmt="%.6f")
    print(json.dumps({"frames": frames, "fps": fps, "floor": floor}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
