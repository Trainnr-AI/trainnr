"""The sim half of the sim-to-real pipeline: the yellow twin performs
the ENTIRE fetch errand in MuJoCo under the firmware's own state
machine — turn-and-glance seek on a synthetic 4 fps camera, timed spin,
reverse park, the waist-less air mime, carry, DONE — and streams the
same dashboard into Rerun as the PREDICTION the real HIL run should
match.

    cd pipeline && uv run --extra sim --extra viz mjpython \
        ../tools/sim-errand.py
"""

import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rig3d import RigMirror  # noqa: E402
from rq_pipeline.tasks.yellow import (  # noqa: E402
    AIR_PICK_SEQUENCE,
    AIR_TUCK,
    compose_rig,
)

# ---- firmware constants, mirrored (main.rs fetch_forever) ----
CREEP_M_PER_S = 0.10
TURN_RAD_PER_S = 1.4
CENTRE_DEADBAND = 0.20
AREA_ARRIVED = 1800
TURN_BURST_S = 0.150
GLANCE_S = 0.700
CREEP_LEG_S = 1.500
ARRIVE_GLANCES = 3
SPIN_S = 2.250
BACK_S = 1.400
CARRY_S = 1.500

# ---- twin duty calibration (tools/sim-errand.py trial, 2026-08-24) ----
CREEP_DUTY = 0.12  # ~0.10 m/s
TURN_DUTY = 0.50  # ~1.4 rad/s

# ---- synthetic camera: 4 fps, ~35 deg half-FOV, area ~ 1/dist^2,
# calibrated so AREA_ARRIVED corresponds to arriving ~0.30 m out ----
CAM_PERIOD_S = 0.25
CAM_HALF_FOV = 0.6
CAM_AREA_K = AREA_ARRIVED * 0.30**2

PROP_XY = (0.95, 0.20)

scene = compose_rig(car=True)
cube = scene.worldbody.add_body(name="prop", pos=[*PROP_XY, 0.0125])
cube.add_freejoint()
cube.add_geom(
    name="prop_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.0125, 0.0125, 0.0125],
    mass=0.015,
    friction=[2.0, 0.02, 0.001],
    rgba=[0.9, 0.15, 0.15, 1.0],
)
model = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


def chassis_pose():
    w, qz = data.qpos[3], data.qpos[6]
    return data.qpos[0], data.qpos[1], 2 * math.atan2(qz, w)


def synthetic_blob():
    """What the front camera would report: (x_err, area) or None."""
    x, y, yaw = chassis_pose()
    dx, dy = PROP_XY[0] - x, PROP_XY[1] - y
    bearing = math.atan2(dy, dx) - yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    if abs(bearing) > CAM_HALF_FOV:
        return None
    dist = math.hypot(dx, dy)
    if dist < 0.05:
        return None
    # Pixel-x convention matches the real camera: POSITIVE = prop to the
    # RIGHT of centre = NEGATIVE bearing. (First cut had this backwards
    # and the sim car politely orbited away from the prop forever.)
    return (-bearing / CAM_HALF_FOV, CAM_AREA_K / dist**2)


# ---- Rerun mirror (same entity paths as tools/replay-errand.py) ----
JOINT_NAMES = ["base", "waist", "shoulder", "wrist", "jaw"]
rig_geoms, rig_half_sizes, rig_colors = [], [], []
for g in range(model.ngeom):
    name = model.geom(g).name
    if name == "floor":
        continue
    size = model.geom_size[g]
    kind = int(model.geom_type[g])
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = size.tolist()
    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = [size[0], size[0], size[1]]
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = [size[0], size[0], size[1] + size[0]]
    else:
        half = [size[0]] * 3
    rig_geoms.append(g)
    rig_half_sizes.append(half)
    if name.startswith("yarm"):
        rig_colors.append([255, 190, 40])
    elif name.startswith("prop"):
        rig_colors.append([230, 40, 40])
    else:
        rig_colors.append([90, 130, 220])




rr.init("yellow-rig-sim-errand", spawn=False)
try:
    rr.connect_grpc()
except Exception:
    rr.spawn()


def log_frame(t, pose, stage=None):
    rr.set_time("run", duration=t)
    x, y, yaw = chassis_pose()
    trail.append([x, y, 0.005])
    rr.log("world/car/trail", rr.LineStrips3D([trail], colors=[[80, 220, 120]]))
    mirror.log(data)
    for name, val in zip(JOINT_NAMES, pose):
        rr.log(f"arm/{name}", rr.Scalars(val))
    rr.log("car/heading", rr.Scalars(yaw))
    if stage:
        rr.log("stage", rr.TextLog(stage))


# ---- the errand, firmware-shaped ----
# (state, since) with states mirroring fetch_forever + fetch_arm
duty = (0.0, 0.0)
arm_pose = list(AIR_TUCK)
trail = []
state = "glance"
state_since = 0.0
arrive_streak = 0
search_left = True
glance_burst = None
mime_index = -1
mime_since = 0.0
mime_prev = list(AIR_TUCK)
stage_note = "fetch SEEKING (sim)"
last_cam = -1.0
blob = None
done_at = None

t = 0.0
DT = 0.002
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.6
    viewer.cam.elevation = -30
    last_log = -1.0
    while viewer.is_running():
        wall = time.time()
        if t - last_cam >= CAM_PERIOD_S:
            last_cam = t
            blob = synthetic_blob()

        if state == "glance":
            duty = (0.0, 0.0)
            if t - state_since >= GLANCE_S:
                if blob is None:
                    arrive_streak = 0
                    state, state_since = "burst", t
                    glance_burst = TURN_DUTY if search_left else -TURN_DUTY
                else:
                    x_err, area = blob
                    centred = abs(x_err) <= CENTRE_DEADBAND
                    if centred and area >= AREA_ARRIVED:
                        arrive_streak += 1
                        if arrive_streak >= ARRIVE_GLANCES:
                            state, state_since = "spin", t
                            stage_note = "fetch ARRIVED, spinning (sim)"
                        else:
                            state_since = t
                    elif not centred:
                        arrive_streak = 0
                        search_left = x_err < 0
                        state, state_since = "burst", t
                        # x_err > 0 = prop right of centre -> clockwise
                        glance_burst = -TURN_DUTY if x_err > 0 else TURN_DUTY
                    else:
                        arrive_streak = 0
                        state, state_since = "creep", t
        elif state == "burst":
            duty = (-glance_burst, glance_burst)
            if t - state_since >= TURN_BURST_S:
                state, state_since = "glance", t
        elif state == "creep":
            duty = (CREEP_DUTY, CREEP_DUTY)
            keep = blob is not None and abs(blob[0]) <= CENTRE_DEADBAND and blob[1] < AREA_ARRIVED
            if t - state_since >= CREEP_LEG_S or not keep:
                state, state_since = "glance", t
        elif state == "spin":
            duty = (-TURN_DUTY, TURN_DUTY)
            if t - state_since >= SPIN_S:
                state, state_since = "back", t
                stage_note = "fetch SPUN, backing up (sim)"
        elif state == "back":
            duty = (-CREEP_DUTY, -CREEP_DUTY)
            if t - state_since >= BACK_S:
                state, state_since = "pick", t
                stage_note = "fetch PARKED, arm's turn (sim)"
                mime_index, mime_since = 0, t
                mime_prev = list(arm_pose)
        elif state == "pick":
            duty = (0.0, 0.0)
            pose, hold = AIR_PICK_SEQUENCE[mime_index]
            blend = 1.2
            dt_m = t - mime_since
            if dt_m < blend:
                a = dt_m / blend
                arm_pose = [p + (q - p) * a for p, q in zip(mime_prev, pose)]
            elif dt_m < blend + hold:
                arm_pose = list(pose)
            else:
                mime_prev = list(pose)
                mime_index += 1
                mime_since = t
                if mime_index >= len(AIR_PICK_SEQUENCE):
                    state, state_since = "carry", t
                    stage_note = "fetch CARRYING (sim)"
        elif state == "carry":
            duty = (CREEP_DUTY, CREEP_DUTY)
            if t - state_since >= CARRY_S:
                state, state_since = "done", t
                stage_note = "fetch DONE (sim) — this is the prediction"
                done_at = t
        elif state == "done":
            duty = (0.0, 0.0)
            if done_at is not None and t - done_at > 4.0:
                break

        data.ctrl[:] = [duty[0], duty[1], *arm_pose]
        for _ in range(8):
            mujoco.mj_step(model, data)
        t += 8 * DT
        x, y, _ = chassis_pose()
        viewer.cam.lookat[:] = [x, y, 0.08]
        viewer.sync()
        if t - last_log >= 0.04:
            last_log = t
            note = stage_note
            stage_note = None
            log_frame(t, arm_pose, note)
        rest = 8 * DT - (time.time() - wall)
        if rest > 0:
            time.sleep(rest)
print("sim errand complete — this is what the real run should look like")
