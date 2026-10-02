#!/usr/bin/env python3
"""The imported 2F-85 closing on a block: the USD-born bundle
(`robots/robotiq-2f85-isaac`) on a bench with a 30 mm cube between its
pads, the drive ramped to its closed target, seen in both viewers —
MuJoCo stills through the offscreen renderer and the Rerun stream
(the Studio when it listens, a recording file always). The evidence
behind docs/e2e-research/77 §4.

    cd trainnr && uv run --extra sim --extra viz python ../tools/show-2f85-isaac.py
    # --stills docs/figures/usd-import  writes open/closed PNGs there
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
from trainnr.bundles.locate import bundle_file  # noqa: E402
from trainnr.viz import RigMirror, open_stream, viewer_file  # noqa: E402

BUNDLE = "robotiq-2f85-isaac"
APP_ID = "trainnr-show-2f85-isaac"


@dataclass(frozen=True)
class Bench:
    """The block and the bench, in metres: the gripper stands at the
    origin with its fingers up; a cube sits between the pads."""

    block_half: float = 0.015
    block_height: float = 0.135  # the pads' centre height in the open pose
    block_mass: float = 0.05
    close_over_s: float = 1.0
    hold_s: float = 1.0
    frame_hz: float = 60.0
    ramp_target: float | None = None  # None: the drive's own ctrlrange top
    # Held: the block touches the gripper on at least two geoms and has
    # not fallen to the bench (a 30 mm cube on the bench sits at -0.045).
    held_contacts: int = 2
    held_above_m: float = 0.05


def scene(bench: Bench) -> mujoco.MjSpec:
    """The bundle's own spec with a block and a floor added around it."""
    spec = mujoco.MjSpec.from_file(str(bundle_file(BUNDLE, f"{BUNDLE}.xml")))
    spec.worldbody.add_geom(
        name="bench",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.5, 0.5, 0.05],
        pos=[0, 0, -0.06],
        rgba=[0.9, 0.9, 0.9, 1],
    )
    block = spec.worldbody.add_body(name="block", pos=[0, 0, bench.block_height])
    block.add_freejoint()
    block.add_geom(
        name="block",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[bench.block_half] * 3,
        mass=bench.block_mass,
        rgba=[0.85, 0.35, 0.2, 1],
    )
    return spec


def still(renderer: mujoco.Renderer, data: mujoco.MjData, path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, 0.09)
    cam.distance = 0.42
    cam.azimuth = 150
    cam.elevation = -18
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[2] = 1  # visuals, not hulls
    renderer.update_scene(data, cam, opt)
    Image.fromarray(renderer.render()).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stills", type=Path, default=None, help="folder for open/closed PNGs"
    )
    parser.add_argument(
        "--stream-file", type=Path, default=viewer_file("runs/show-2f85-isaac", "close")
    )
    args = parser.parse_args(argv)
    bench = Bench()
    spec = scene(bench)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)  # poses before the first still
    rr = open_stream(APP_ID, file=args.stream_file)
    mirror = RigMirror(model, skip=("bench",), model_colors=True, skip_groups=(3,))
    top = (
        bench.ramp_target
        if bench.ramp_target is not None
        else float(model.actuator_ctrlrange[0][1])
    )
    renderer = mujoco.Renderer(model, 480, 640) if args.stills else None
    if renderer is not None:
        args.stills.mkdir(parents=True, exist_ok=True)
        still(renderer, data, args.stills / "bundle_open.png")
    steps = int((bench.close_over_s + bench.hold_s) / model.opt.timestep)
    every = max(1, round(1.0 / (bench.frame_hz * model.opt.timestep)))
    block = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint")
    for step in range(steps):
        data.ctrl[0] = top * min(1.0, step * model.opt.timestep / bench.close_over_s)
        mujoco.mj_step(model, data)
        if step % every == 0:
            rr.set_time("sim_time", duration=data.time)
            mirror.log(data)
            rr.log(
                "gripper/finger_rad",
                rr.Scalars([float(data.qpos[model.jnt_qposadr[finger]])]),
            )
            rr.log("gripper/target_rad", rr.Scalars([float(data.ctrl[0])]))
            rr.log("block/height_m", rr.Scalars([float(data.xpos[block][2])]))
    contacts = sum(
        1
        for c in data.contact[: data.ncon]
        if block in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
    )
    print(
        f"closed: finger {data.qpos[model.jnt_qposadr[finger]]:.4f} rad of {top}, "
        f"block at z={data.xpos[block][2]:.4f} m ({bench.block_height} at start), "
        f"{contacts} contacts on the block, stream {args.stream_file}"
    )
    if renderer is not None:
        still(renderer, data, args.stills / "bundle_closed.png")
        renderer.close()
    held = contacts >= bench.held_contacts and data.xpos[block][2] > bench.held_above_m
    print("held" if held else "NOT held")
    return 0 if held else 1


if __name__ == "__main__":
    sys.exit(main())
