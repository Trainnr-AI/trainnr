"""Replay a fetch errand whole: car from measured odometry, arm on the
recorded stage cues — MuJoCo window + Rerun stream, side by side.

    cd trainnr && uv run --extra sim --extra viz mjpython \
        ../tools/replay-errand.py ../recordings/<run>.wire
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import rerun as rr

from _lab import bootstrap, rr_session

bootstrap()
from trainnr.collect.frames import STATUS_HZ  # noqa: E402
from trainnr.collect.wire import parse_recording  # noqa: E402
from trainnr.tasks.yellow import (  # noqa: E402
    AIR_TUCK,
    air_mime_pose,
    compose_rig,
    salute_pose,
)
from trainnr.viz import RigMirror  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("wire", type=Path, help="the .wire recording")
args = parser.parse_args()
WIRE = args.wire
if not WIRE.exists():
    sys.exit(f"no such recording: {WIRE}")
TICK = 1.0 / STATUS_HZ

rec = parse_recording(WIRE)

# --- timeline: the status index at which each stage note lands.
# The wire has no timestamps; line order is the only clock, and one
# status line is one 20 ms tick.
marks = []
count = 0
for line in WIRE.read_text(errors="replace", encoding="utf-8").splitlines():
    s = line.strip()
    if s.startswith("<"):
        s = s[1:].strip()
    if s.startswith("# fetch"):
        marks.append((count, s[2:]))
    elif s.startswith("n="):
        count += 1

pick_at = next((c for c, t in marks if "arm PICKING" in t), None)
print(f"{count} status frames; PICKING@{pick_at}")

# --- twin ---
scene = compose_rig(car=True)
model = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
free_j = next(
    j
    for j in range(model.njnt)
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
)
free_q = model.jnt_qposadr[free_j]
z0 = float(data.qpos[free_q + 2])
arm_joints = [
    model.jnt_qposadr[model.joint(n).id]
    for n in ("yarm_base_yaw", "yarm_waist", "yarm_shoulder", "yarm_wrist")
]
jaw_l = model.jnt_qposadr[model.joint("yarm_jaw_l_hinge").id]
jaw_r = model.jnt_qposadr[model.joint("yarm_jaw_r_hinge").id]
JOINT_NAMES = ["base", "waist", "shoulder", "wrist", "jaw"]

mirror = RigMirror(model)


# --- arm pose as a function of status index ---
def mime_pose(i):
    # The salute fires at the STARTING GUN (SALUTE_GO is set the moment
    # the host connects — servo.rs), not at ARRIVED as an earlier cut
    # of this file guessed; the mime follows the PICKING cue at the
    # firmware's true slew rate via yellow.air_mime_pose.
    if pick_at is not None and i >= pick_at:
        return air_mime_pose((i - pick_at) * TICK)
    wave = salute_pose(i * TICK)
    return wave if wave is not None else AIR_TUCK


rr_session(f"yellow-rig-{WIRE.stem}")

trail = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.4
    viewer.cam.elevation = -25
    for i, st in enumerate(rec.statuses):
        start = time.time()
        yaw = st.heading
        data.qpos[free_q : free_q + 3] = [st.x, st.y, z0]
        data.qpos[free_q + 3 : free_q + 7] = [
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2),
        ]
        pose = mime_pose(i)
        for adr, val in zip(arm_joints, pose[:4], strict=True):
            data.qpos[adr] = val
        data.qpos[jaw_l] = pose[4]
        data.qpos[jaw_r] = pose[4]
        mujoco.mj_forward(model, data)
        viewer.cam.lookat[:] = [st.x, st.y, 0.08]
        viewer.sync()

        if i % 2 == 0:
            rr.set_time("run", duration=i * TICK)
            trail.append([st.x, st.y, 0.005])
            rr.log("world/car/trail", rr.LineStrips3D([trail], colors=[[80, 160, 255]]))
            mirror.log(data)
            for name, val in zip(JOINT_NAMES, pose, strict=True):
                rr.log(f"arm/{name}", rr.Scalars(val))
            rr.log("car/heading", rr.Scalars(yaw))
        for c, txt in marks:
            if c == i:
                rr.log("stage", rr.TextLog(txt))
        rest = TICK - (time.time() - start)
        if rest > 0:
            time.sleep(rest)
        if not viewer.is_running():
            break
print("replay complete")
