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
import time

import mujoco
import mujoco.viewer
import rerun as rr

from _lab import bootstrap, rr_session

bootstrap()
from rq_pipeline.tasks.yellow import (  # noqa: E402
    AIR_PICK_SEQUENCE,
    AIR_TUCK,
    REAL_CAR,
    SLEW_RAD_PER_S,
    air_mime_pose,
    compose_rig,
    salute_pose,
)
from rq_pipeline.viz import RigMirror  # noqa: E402


def _mime_total_s() -> float:
    total, prev = 0.0, AIR_TUCK
    for pose, hold in AIR_PICK_SEQUENCE:
        total += (
            max(abs(a - b) for a, b in zip(prev, pose, strict=True)) / SLEW_RAD_PER_S
            + hold
        )
        prev = pose
    return total


MIME_TOTAL_S = _mime_total_s()

# ---- firmware constants, mirrored (main.rs fetch_forever) ----
CREEP_M_PER_S = 0.10
TURN_RAD_PER_S = 1.4
CENTRE_DEADBAND = 0.20
AREA_ARRIVED = 1800
TURN_BURST_S = 0.150
GLANCE_S = 0.700
ARRIVE_GLANCES = 3
ARC_M_PER_S = 0.08
ARC_RAD_PER_S = 0.9
ARC_X_MAX = 0.55
SPIN_S = 2.250
BACK_S = 1.400
CARRY_S = 1.500

# ---- twin duty calibration (tools/sim-errand.py trial, 2026-08-24) ----
CREEP_DUTY = 0.12  # ~0.10 m/s
TURN_DUTY = 0.50  # ~1.4 rad/s
_M_PER_S_PER_DUTY = 0.93
_HALF_TRACK = REAL_CAR.track / 2
# The errand's timing, mirrored from main.rs's fetch loop.
ARRIVED_M = 0.05
SETTLE_S = 0.3  # main.rs settles 300 ms
DONE_HOLD_S = 4.0
LOG_PERIOD_S = 0.04


def duty_pair(v, w):
    """(forward m/s, turn rad/s) -> (left, right) wheel duty."""
    return (
        (v - w * _HALF_TRACK) / _M_PER_S_PER_DUTY,
        (v + w * _HALF_TRACK) / _M_PER_S_PER_DUTY,
    )


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
    if dist < ARRIVED_M:
        return None
    # Pixel-x convention matches the real camera: POSITIVE = prop to the
    # RIGHT of centre = NEGATIVE bearing. (First cut had this backwards
    # and the sim car politely orbited away from the prop forever.)
    return (-bearing / CAM_HALF_FOV, CAM_AREA_K / dist**2)


# ---- Rerun mirror (same entity paths as tools/replay-errand.py) ----
JOINT_NAMES = ["base", "waist", "shoulder", "wrist", "jaw"]
mirror = RigMirror(model)

rr_session("yellow-rig-sim-errand")


def log_frame(t, pose, stage=None):
    rr.set_time("run", duration=t)
    x, y, yaw = chassis_pose()
    trail.append([x, y, 0.005])
    rr.log("world/car/trail", rr.LineStrips3D([trail], colors=[[80, 220, 120]]))
    mirror.log(data)
    for name, val in zip(JOINT_NAMES, pose, strict=True):
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
glance_from = 0
cam_frames = 0
arrive_streak = 0
search_left = True
glance_burst = None
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
            cam_frames += 1
            blob = synthetic_blob()

        if state == "glance":
            # Glance-and-flow (mirrors main.rs): duty HOLDS while
            # waiting for a fresh frame; a blob in view keeps the car
            # moving — straight when centred, an arc toward it when
            # off-centre. Stop-and-burst only for lost or far blobs,
            # arrival confirmed standing still.
            if cam_frames != glance_from or t - state_since >= GLANCE_S:
                state_since, glance_from = t, cam_frames
                if blob is None:
                    arrive_streak = 0
                    duty = (0.0, 0.0)
                    state, state_since = "burst", t
                    glance_burst = TURN_DUTY if search_left else -TURN_DUTY
                else:
                    x_err, area = blob
                    centred = abs(x_err) <= CENTRE_DEADBAND
                    if centred and area >= AREA_ARRIVED:
                        duty = (0.0, 0.0)
                        arrive_streak += 1
                        if arrive_streak >= ARRIVE_GLANCES:
                            state, state_since = "arrive_settle", t
                            stage_note = "fetch ARRIVED, spinning (sim)"
                    elif centred:
                        arrive_streak = 0
                        duty = (CREEP_DUTY, CREEP_DUTY)
                    elif abs(x_err) < ARC_X_MAX:
                        arrive_streak = 0
                        search_left = x_err < 0
                        w = ARC_RAD_PER_S if x_err < 0 else -ARC_RAD_PER_S
                        duty = duty_pair(ARC_M_PER_S, w)
                    else:
                        arrive_streak = 0
                        search_left = x_err < 0
                        duty = (0.0, 0.0)
                        state, state_since = "burst", t
                        glance_burst = -TURN_DUTY if x_err > 0 else TURN_DUTY
        elif state == "burst":
            duty = (-glance_burst, glance_burst)
            if t - state_since >= TURN_BURST_S:
                state, state_since = "glance", t
                glance_from = cam_frames
                duty = (0.0, 0.0)
        elif state == "arrive_settle":
            duty = (0.0, 0.0)
            if t - state_since >= SETTLE_S:
                state, state_since = "spin", t
        elif state == "spin":
            duty = (-TURN_DUTY, TURN_DUTY)
            if t - state_since >= SPIN_S:
                state, state_since = "spun_settle", t
                stage_note = "fetch SPUN, backing up (sim)"
        elif state == "spun_settle":
            duty = (0.0, 0.0)
            if t - state_since >= SETTLE_S:
                state, state_since = "back", t
        elif state == "back":
            duty = (-CREEP_DUTY, -CREEP_DUTY)
            if t - state_since >= BACK_S:
                state, state_since = "pick", t
                stage_note = "fetch PARKED, arm's turn (sim)"
        elif state == "pick":
            duty = (0.0, 0.0)
            # The one shared reconstruction, at the firmware's slew.
            arm_pose = air_mime_pose(t - state_since)
            if t - state_since >= MIME_TOTAL_S:
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
            if done_at is not None and t - done_at > DONE_HOLD_S:
                break

        if state not in ("pick", "carry", "done"):
            wave = salute_pose(t)
            arm_pose = wave if wave is not None else list(AIR_TUCK)
        data.ctrl[:] = [duty[0], duty[1], *arm_pose]
        for _ in range(8):
            mujoco.mj_step(model, data)
        t += 8 * DT
        x, y, _ = chassis_pose()
        viewer.cam.lookat[:] = [x, y, 0.08]
        viewer.sync()
        if t - last_log >= LOG_PERIOD_S:
            last_log = t
            note = stage_note
            stage_note = None
            log_frame(t, arm_pose, note)
        rest = 8 * DT - (time.time() - wall)
        if rest > 0:
            time.sleep(rest)
print("sim errand complete — this is what the real run should look like")
