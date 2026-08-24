"""Replay a fetch errand whole: car from measured odometry, arm on the
recorded stage cues — MuJoCo window + Rerun stream, side by side.

    cd pipeline && uv run --extra sim --extra viz mjpython \
        ../tools/replay-errand.py ../recordings/<run>.wire
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
from rq_pipeline.collect.wire import parse_recording
from rq_pipeline.tasks.yellow import compose_rig, AIR_PICK_SEQUENCE, AIR_TUCK

WIRE = Path(sys.argv[1])
TICK = 0.02  # one status frame

rec = parse_recording(WIRE)

# --- timeline: the status index at which each stage note lands.
# The wire has no timestamps; line order is the only clock, and one
# status line is one 20 ms tick.
marks = []
count = 0
for line in WIRE.read_text(errors="replace").splitlines():
    s = line.strip()
    if s.startswith("<"):
        s = s[1:].strip()
    if s.startswith("# fetch"):
        marks.append((count, s[2:]))
    elif s.startswith("n="):
        count += 1

pick_at = next((c for c, t in marks if "arm PICKING" in t), None)
arrive_at = next((c for c, t in marks if "ARRIVED" in t), None)
print(f"{count} status frames; ARRIVED@{arrive_at} PICKING@{pick_at}")

# --- twin ---
scene = compose_rig(car=True)
model = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
free_j = next(j for j in range(model.njnt) if int(model.jnt_type[j]) == 0)
free_q = model.jnt_qposadr[free_j]
z0 = float(data.qpos[free_q + 2])
arm_joints = [model.jnt_qposadr[model.joint(n).id] for n in
              ("yarm_base_yaw", "yarm_waist", "yarm_shoulder", "yarm_wrist")]
jaw_l = model.jnt_qposadr[model.joint("yarm_jaw_l_hinge").id]
jaw_r = model.jnt_qposadr[model.joint("yarm_jaw_r_hinge").id]
JOINT_NAMES = ["base", "waist", "shoulder", "wrist", "jaw"]

# Mirror MuJoCo's actual geometry into Rerun: every rig geom as an
# oriented box (cylinders/spheres approximated by their bounding box),
# so both viewers show the same shapes, not a stick abstraction.
rig_geoms = []
rig_half_sizes = []
rig_colors = []
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
    rig_colors.append([255, 190, 40] if name.startswith("yarm") else [90, 130, 220])



# --- arm pose as a function of status index ---
def mime_pose(i):
    # salute: two claw waves right after ARRIVED
    if arrive_at is not None and arrive_at <= i < arrive_at + 70 and (pick_at is None or i < pick_at):
        phase = ((i - arrive_at) // 18) % 2
        jaw = 0.35 if phase == 0 else -0.5
        return [*AIR_TUCK[:4], jaw]
    if pick_at is None or i < pick_at:
        return AIR_TUCK
    t = (i - pick_at) * TICK
    prev = AIR_TUCK
    for pose, hold in AIR_PICK_SEQUENCE:
        blend = 1.2
        if t < blend:
            a = t / blend
            return [p + (q - p) * a for p, q in zip(prev, pose)]
        t -= blend
        if t < hold:
            return pose
        t -= hold
        prev = pose
    return AIR_PICK_SEQUENCE[-1][0]

rr.init(f"yellow-rig-{WIRE.stem}", spawn=False)
try:
    rr.connect_grpc()
except Exception:
    rr.spawn()

trail = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.4
    viewer.cam.elevation = -25
    for i, st in enumerate(rec.statuses):
        start = time.time()
        yaw = st.heading
        data.qpos[free_q:free_q + 3] = [st.x, st.y, z0]
        data.qpos[free_q + 3:free_q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        pose = mime_pose(i)
        for adr, val in zip(arm_joints, pose[:4]):
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
            for name, val in zip(JOINT_NAMES, pose):
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
