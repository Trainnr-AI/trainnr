"""Many worlds: N ALOHA 2 rigs in a grid, each with its own start and its
own dynamics, stepping together in both viewers.

    cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python \
        ../tools/show-many.py --worlds 16 --dr 0.3

The Isaac-Lab picture on the MuJoCo stack: the bundle's transfer-cube
scene is attached `--worlds` times into one MjSpec on a grid, every
copy gets a different cube start and its joint damping and actuator
gains scaled by a per-world factor drawn within +-`--dr` of the
IDENTIFIED values (domain randomisation centred on the measurement,
never on a guess — docs/22 §2.1), and all copies replay the same human
demonstration. Same commands, different physics: the spread across
worlds is the dynamics uncertainty made visible.

    MuJoCo window     the grid, live
    Rerun world/rig   the mesh-true mirror of every world
          worlds/<n>  per-world cube height and damping/gain factors
          stage       notes

This is the visualisation layer. Thousands of worlds for RL training
belong to the GPU-batched simulator (MJX-Warp, ladder rung T6), which
needs the MJX-patched model Menagerie ships alongside this one.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

from _lab import (
    bootstrap,
    frame_viewer,
    hold_until_closed,
    load_demo_actions,
    rr_session,
)

bootstrap()
from rq_pipeline.tasks.aloha2 import (  # noqa: E402
    ACT_SIM_LOOK,
    CUBE_HALF,
    CUBE_HOME,
    CUBE_SPAWN_X,
    CUBE_SPAWN_Y,
    LOOKS,
    PUBLIC_TRANSFER_CUBE_DEMOS,
    SERVOS,
    build_transfer_cube,
    ctrl_from_act_sim_action,
)
from rq_pipeline.tasks.scene import (  # noqa: E402
    GeomGroup,
    NominalOptions,
    grid_of,
    pin_nominal_options,
    set_render_budget,
)
from rq_pipeline.viz import RigMirror  # noqa: E402

NEUTRAL_QPOS = [0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084] * 2
LIFTED_M = 0.05  # cube centre 3 cm above resting counts as lifted
SIM_HZ = round(1 / NominalOptions.TIMESTEP)
# The Rerun mirror costs one transform log per mesh per world (~900 for
# 16 worlds); at 50 Hz that starves the physics loop (measured: the
# MuJoCo window lagged seconds behind). 5 Hz keeps the grid live.
LOG_EVERY = 100
SYNC_EVERY = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument(
        "--dr", type=float, default=0.3, help="+- fraction on damping/gains"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default=PUBLIC_TRANSFER_CUBE_DEMOS)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--look", default=ACT_SIM_LOOK, choices=LOOKS)
    return parser.parse_args()


def grid_scene(worlds: int, look: str):
    """One MjSpec holding `worlds` copies of the transfer-cube scene."""
    scene, prefixes = grid_of(
        f"aloha2-many-{worlds}",
        (build_transfer_cube(look=look).spec for _ in range(worlds)),
        disable_floor_contacts=True,
    )
    pin_nominal_options(scene)
    set_render_budget(scene)
    scene.visual.map.znear = 0.01
    return scene, prefixes


def cube_qpos_address(model, prefix: str) -> int:
    """The cube's free joint is unnamed; reach it through its body."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "cube")
    return int(model.jnt_qposadr[model.body_jntadr[body]])


def randomise(model, prefixes, dr: float, rng):
    """Scale each world's joint damping and actuator gains by its own factor."""
    factors = []
    for prefix in prefixes:
        damping_factor = float(rng.uniform(1 - dr, 1 + dr))
        gain_factor = float(rng.uniform(1 - dr, 1 + dr))
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith(prefix) and "cube" not in name:
                dof = model.jnt_dofadr[j]
                model.dof_damping[dof] *= damping_factor
        for a in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if name and name.startswith(prefix):
                # position actuator: gainprm[0] = kp, biasprm[1] = -kp
                model.actuator_gainprm[a, 0] *= gain_factor
                model.actuator_biasprm[a, 1] *= gain_factor
        factors.append((damping_factor, gain_factor))
    return factors


def place_worlds(model, data, prefixes, rng) -> None:
    """Every world: neutral arms, its own cube start inside the spawn box."""
    for prefix in prefixes:
        for j, value in enumerate(NEUTRAL_QPOS):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + JOINT_NAMES[j]
            )
            data.qpos[model.jnt_qposadr[joint]] = value
        adr = cube_qpos_address(model, prefix)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "cube")
        # The cube body sits at CUBE_HOME inside its world; frames are
        # compiled away, so body_pos is already in the grid's frame.
        world_origin = model.body_pos[body] - np.array(CUBE_HOME)
        start = world_origin + np.array(
            [rng.uniform(*CUBE_SPAWN_X), rng.uniform(*CUBE_SPAWN_Y), CUBE_HALF]
        )
        data.qpos[adr : adr + 7] = [*start, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    scene, prefixes = grid_scene(args.worlds, args.look)
    model = scene.compile()
    factors = randomise(model, prefixes, args.dr, rng)
    data = mujoco.MjData(model)
    place_worlds(model, data, prefixes, rng)

    actions = load_demo_actions(args.dataset, args.episode)
    ctrl_per_step = [ctrl_from_act_sim_action(a) for a in actions]

    rr_session(f"robotiq-many-{args.worlds}", mode="spawn")
    mirror = RigMirror(
        model, model_colors=True, skip_groups=(GeomGroup.COLLISION, GeomGroup.HIDDEN)
    )
    for n, (damping, gain) in enumerate(factors):
        rr.log(f"worlds/{n:02d}/damping_factor", rr.Scalars(damping), static=True)
        rr.log(f"worlds/{n:02d}/gain_factor", rr.Scalars(gain), static=True)
    cube_z_adr = [cube_qpos_address(model, p) + 2 for p in prefixes]
    rr.log(
        "stage",
        rr.TextLog(
            f"{args.worlds} worlds, dr +-{args.dr:.0%}, replaying demo {args.episode}"
        ),
    )
    print(f"{args.worlds} worlds, {model.nv} dofs, dr +-{args.dr:.0%}", flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        side = int(np.ceil(np.sqrt(args.worlds)))
        frame_viewer(viewer, 2.2 * side)
        clock = 0.0
        for k in range(len(ctrl_per_step) * 10):
            if k % 10 == 0:
                ctrl = ctrl_per_step[k // 10]
                for n in range(args.worlds):
                    data.ctrl[n * SERVOS : (n + 1) * SERVOS] = ctrl
            mujoco.mj_step(model, data)
            clock += 1.0 / SIM_HZ
            if k % LOG_EVERY == 0:
                rr.set_time("sim_time", duration=clock)
                mirror.log(data)
                for n, adr in enumerate(cube_z_adr):
                    rr.log(f"worlds/{n:02d}/cube_z", rr.Scalars(float(data.qpos[adr])))
            if k % SYNC_EVERY == 0:
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(0.002)
        heights = [float(data.qpos[adr]) for adr in cube_z_adr]
        lifted = sum(h > LIFTED_M for h in heights)
        rr.log(
            "stage",
            rr.TextLog(f"demo replayed in {args.worlds} worlds: {lifted} cubes lifted"),
        )
        print(
            f"done: {lifted}/{args.worlds} cubes lifted - close the window to exit",
            flush=True,
        )
        hold_until_closed(viewer)


JOINT_NAMES = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/left_finger",
    "left/right_finger",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
    "right/left_finger",
    "right/right_finger",
]


if __name__ == "__main__":
    main()
