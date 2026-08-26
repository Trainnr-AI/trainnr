"""Two identical deck picks, one nanometre apart, in both viewers.

The deck pick-present-stow cycle is open-loop: the arm ramps through
DECK_PICK_SEQUENCE and the cube goes wherever the contact takes it.
This runs that cycle twice from starts that differ by 1 nm — a distance
no physical rig could hold and no measurement could resolve — and draws
both cube paths together. They separate by centimetres, which is why
`test_pick_present_stow_cycle` asserting a 1 cm landing tolerance passed
on the Mac and failed on the WSL box: it pinned a number the physics
does not pin. (The test now asserts the deck landing, not the pocket.)

    cd pipeline && GALLIUM_DRIVER=d3d12 WGPU_BACKEND=vulkan \
        uv run --extra sim --extra viz python ../tools/show-cargo-chaos.py

(The two variables are WSL's GPU routing for the MuJoCo and Rerun
windows respectively — see show-aloha2.py's header; harmless elsewhere.)

Both viewers open: the MuJoCo window shows the live run (the second
lap), Rerun collects both laps' cube trails on a shared timeline. Close
either window to stop.

On Linux/WSL plain `python` drives the passive viewer; macOS needs
`mjpython` (the repo's other viewer tools say so in their headers).
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

from _lab import bootstrap, rr_session

bootstrap()
from rq_pipeline.tasks.components import (  # noqa: E402
    DECK_PICK_SEQUENCE,
    TRAY_CENTRE_X,
    TRAY_CENTRE_Y,
    compose,
)

from _rig3d import RigMirror  # noqa: E402

CROUCH = [0.0, -1.9, 1.9, 1.3, 0.0, 0.3]
CUBE_QPOS = 15  # free joint: cube x, y, z
SIM_HZ = 500
# The two starts. 1 nm is ~1/50th of a hydrogen atom's width across an
# 18 mm cube: physically meaningless, numerically decisive.
PERTURBATIONS = (0.0, 1e-9)
LAP_COLORS = ([80, 200, 255], [255, 120, 60])


def build():
    scene = compose(car=True, arm=True, cargo=True)
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def run_lap(model, data, lap, mirror=None, viewer=None):
    """One full cycle; logs the cube path and returns where it landed.

    With a mirror the whole rig is drawn (the live lap); without one
    only the cube is drawn, as a ghost box — two full rigs a nanometre
    apart would z-fight into shimmer.
    """
    trail = []
    clock = 0.0

    def ramp(pose, seconds, previous, stage):
        nonlocal clock
        for step in range(int(seconds * SIM_HZ)):
            alpha = min(1.0, step / 600)
            if step % 10 == 0:
                blend = (1 - alpha) * np.array(previous) + alpha * np.array(pose)
                data.ctrl[:] = [0.0, 0.0, *blend]
            mujoco.mj_step(model, data)
            clock += 1.0 / SIM_HZ
            if step % 10 == 0:
                cube = data.qpos[CUBE_QPOS : CUBE_QPOS + 3].copy()
                trail.append(cube)
                rr.set_time("sim_time", duration=clock)
                if mirror is not None:
                    mirror.log(data, path=f"world/lap{lap}")
                else:
                    rr.log(
                        f"world/lap{lap}/cube_ghost",
                        rr.Boxes3D(
                            centers=[cube],
                            half_sizes=[[0.012, 0.012, 0.015]],
                            colors=[[*LAP_COLORS[lap], 120]],
                            fill_mode="solid",
                        ),
                    )
                rr.log(
                    f"world/lap{lap}/cube_path",
                    rr.LineStrips3D([trail], colors=[LAP_COLORS[lap]], radii=0.0008),
                )
                rr.log(f"cube/lap{lap}/x", rr.Scalars(float(cube[0])))
                rr.log(f"cube/lap{lap}/z", rr.Scalars(float(cube[2])))
                rr.log(
                    f"cube/lap{lap}/dist_from_tray",
                    rr.Scalars(float(abs(cube[0] - TRAY_CENTRE_X))),
                )
                rr.log(f"stage/lap{lap}", rr.TextLog(stage))
            if viewer is not None:
                if not viewer.is_running():
                    raise SystemExit("viewer closed")
                if step % 10 == 0:
                    viewer.sync()
                    time.sleep(0.004)
        return pose

    previous = ramp(CROUCH, 2.0, list(data.qpos[9:15]), "CROUCH")
    names = (
        "HOVER",
        "DESCEND",
        "ALIGN",
        "GRIP",
        "CARRY",
        "PRESENT",
        "CARRY back",
        "ALIGN (jaw opens)",
        "HOVER",
    )
    for (pose, seconds), name in zip(DECK_PICK_SEQUENCE, names, strict=True):
        previous = ramp(pose, seconds, previous, name)
    return data.qpos[CUBE_QPOS : CUBE_QPOS + 3].copy()


def main() -> None:
    rr_session("robotiq-cargo-chaos", mode="spawn")
    # The tray pocket the cube is supposed to come home to.
    rr.log(
        "world/tray_centre",
        rr.Points3D(
            [[TRAY_CENTRE_X, TRAY_CENTRE_Y, 0.086]],
            colors=[[60, 255, 120]],
            radii=0.0025,
        ),
        static=True,
    )

    landings = []
    for lap, perturb in enumerate(PERTURBATIONS):
        model, data = build()
        data.qpos[CUBE_QPOS] += perturb
        print(f"lap {lap}: start x offset {perturb:+.0e} m")
        # Only the last lap drives the interactive window and the full
        # Rerun rig; earlier laps draw just their cube. The runs stay
        # bit-identical apart from their start (viewers don't step).
        if lap == len(PERTURBATIONS) - 1:
            mirror = RigMirror(model, model_colors=True)
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 0.6
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                viewer.cam.lookat[:] = [0.05, 0.0, 0.05]
                landings.append(run_lap(model, data, lap, mirror, viewer))
                # Hold the window so the final poses can be inspected;
                # close it (or Ctrl-C) to get the summary table.
                print("laps done - close the MuJoCo window to finish")
                while viewer.is_running():
                    time.sleep(0.2)
        else:
            landings.append(run_lap(model, data, lap))

    print(f"\n{'lap':>4}  {'start offset':>13}  {'landed x':>9}  {'|x - tray|':>11}")
    for lap, (perturb, land) in enumerate(zip(PERTURBATIONS, landings, strict=True)):
        print(
            f"{lap:>4}  {perturb:>13.0e}  {land[0]:>9.4f}  "
            f"{abs(land[0] - TRAY_CENTRE_X):>11.4f}"
        )
    separation = float(np.linalg.norm(landings[0][:2] - landings[1][:2]))
    print(f"\nstarts differed by {PERTURBATIONS[1] * 1000:.0e} mm")
    print(f"landings differ by {separation * 1000:.1f} mm")
    print(f"the test's tolerance is {0.01 * 1000:.0f} mm")


if __name__ == "__main__":
    main()
