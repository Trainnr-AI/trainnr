"""Live viewer: the real rig's twin trying the rear pick, looping.

    cd pipeline && uv run --extra sim mjpython ../tools/show-yellow.py
"""

import time

import mujoco
import mujoco.viewer

from rq_pipeline.tasks.yellow import REAR_GRASP_POINT, compose_rig

scene = compose_rig(car=True)
cube = scene.worldbody.add_body(name="prop", pos=[*REAR_GRASP_POINT, 0.0125])
cube.add_freejoint()
cube.add_geom(
    name="prop_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.0125, 0.0125, 0.0125],
    mass=0.015,
    friction=[2.0, 0.02, 0.001],
    rgba=[0.85, 0.15, 0.15, 1.0],
)
model = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
jadr = model.body("prop").jntadr[0]
cq = model.jnt_qposadr[jadr]

from rq_pipeline.tasks.yellow import (  # noqa: E402
    REAR_GRIP,
    REAR_HOVER,
    REAR_LIFT,
    REAR_REACH,
    REAR_TUCK,
)

SEQ = [
    (REAR_TUCK, 1.5, "tuck"),
    (REAR_HOVER, 1.5, "hover"),
    (REAR_REACH, 2.0, "reach"),
    (REAR_GRIP, 1.2, "grip"),
    (REAR_LIFT, 2.0, "lift"),
    (REAR_LIFT, 1.5, "hold"),
    (REAR_REACH, 2.0, "lower"),
    ([*REAR_REACH[:4], 0.5], 1.0, "release"),
    (REAR_TUCK, 2.0, "tuck"),
]

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [-0.15, 0.0, 0.05]
    viewer.cam.distance = 0.65
    viewer.cam.elevation = -18
    viewer.cam.azimuth = 200
    while viewer.is_running():
        for ctrl, secs, label in SEQ:
            steps = int(secs * 500)
            for step in range(steps):
                if not viewer.is_running():
                    break
                start = time.time()
                if step % 10 == 0:
                    data.ctrl[:] = [0.0, 0.0, *ctrl]
                for _ in range(4):
                    mujoco.mj_step(model, data)
                viewer.sync()
                rest = 0.008 - (time.time() - start)
                if rest > 0:
                    time.sleep(rest)
            if label == "hold":
                z = float(data.qpos[cq + 2])
                print(f"cube z at hold: {z:.3f} {'LIFTED' if z > 0.05 else 'missed'}")
            if not viewer.is_running():
                break
        # reset the cube for the next lap
        data.qpos[cq : cq + 3] = [*REAR_GRASP_POINT, 0.0125]
        data.qpos[cq + 3 : cq + 7] = [1, 0, 0, 0]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
