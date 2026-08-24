"""Live Rerun view of a rig session: tail a growing .wire, plot everything —
and draw the rig in realtime 3D, posed by odometry and the stage notes.

    cd pipeline && uv run --extra sim --extra viz python ../tools/rig-rerun.py <file.wire>

Tails the recording as `hil-host --record` writes it (works on finished
files too — it replays then keeps watching). Spawns the Rerun viewer and
logs, on the 50 Hz seq clock:

    world/rig             the twin's geoms as oriented boxes, live
    world/car/trail       odometry (x, y) trail in 3D
    drive/duty            commanded duty (%)
    drive/ticks/left      cumulative encoder ticks
    drive/ticks/right
    drive/angle/left      wheel angle (rev) via the bundle's ticks/rev
    drive/angle/right
    drive/errors/left     encoder decode errors — should stay ~flat
    drive/errors/right
    pose/path             odometry (x, y) trail (2D plot)
    stage                 fetch stage notes as text events

The wire carries no arm telemetry (the PCA commands never cross the
bus), so the arm is drawn on the firmware's own schedule, cued by the
stage notes — a reconstruction, honest to the program if not the metal.
"""

import math
import sys
import time
from pathlib import Path

import mujoco
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rig3d import RigMirror  # noqa: E402
from rq_pipeline.bundles.profile import load_profile  # noqa: E402
from rq_pipeline.collect.frames import STATUS_HZ  # noqa: E402
from rq_pipeline.collect.wire import parse_status  # noqa: E402
from rq_pipeline.tasks.yellow import (  # noqa: E402
    AIR_PICK_SEQUENCE,
    AIR_TUCK,
    compose_rig,
)

REPO = Path(__file__).resolve().parent.parent
PROFILE = load_profile(REPO / "robots" / "rig-drivetrain")
TICKS_PER_REV = PROFILE.ticks_per_revolution
TICK = 1.0 / STATUS_HZ

path = Path(sys.argv[1])
rr.init("rig-session", spawn=True)

# --- the twin, posed kinematically from the wire ---
model = compose_rig(car=True).compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
mirror = RigMirror(model)
free_j = next(j for j in range(model.njnt) if int(model.jnt_type[j]) == 0)
free_q = model.jnt_qposadr[free_j]
z0 = float(data.qpos[free_q + 2])
arm_adrs = [
    model.jnt_qposadr[model.joint(n).id]
    for n in ("yarm_base_yaw", "yarm_waist", "yarm_shoulder", "yarm_wrist")
]
jaw_adrs = [
    model.jnt_qposadr[model.joint(n).id]
    for n in ("yarm_jaw_l_hinge", "yarm_jaw_r_hinge")
]


def mime_pose(i, arrive_at, pick_at):
    """Arm pose at status index i, on the firmware's schedule."""
    if arrive_at is not None and arrive_at <= i < arrive_at + 70:
        if pick_at is None or i < pick_at:
            jaw = 0.35 if ((i - arrive_at) // 18) % 2 == 0 else -0.5
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


seen = 0
first_seq = None
trail = []
count = 0
arrive_at = None
pick_at = None
print(f"tailing {path} (ticks/rev {TICKS_PER_REV}, ctrl-c to stop)")
while True:
    if path.exists():
        lines = path.read_text(errors="replace").splitlines()
        for raw in lines[seen:]:
            line = raw.strip()
            if line.startswith("<"):
                line = line[1:].lstrip()
            if line.startswith("# fetch"):
                if "ARRIVED" in line and arrive_at is None:
                    arrive_at = count
                if "arm PICKING" in line and pick_at is None:
                    pick_at = count
                rr.log("stage", rr.TextLog(line[2:]))
                continue
            status = parse_status(line)
            if status is None:
                continue
            if first_seq is None:
                first_seq = status.seq
            t = (status.seq - first_seq) / STATUS_HZ
            rr.set_time("wire", duration=t)
            rr.log("drive/duty", rr.Scalars(status.duty_percent))
            rr.log("drive/ticks/left", rr.Scalars(status.ticks_left))
            rr.log("drive/ticks/right", rr.Scalars(status.ticks_right))
            rr.log("drive/angle/left", rr.Scalars(status.ticks_left / TICKS_PER_REV))
            rr.log("drive/angle/right", rr.Scalars(status.ticks_right / TICKS_PER_REV))
            rr.log("drive/errors/left", rr.Scalars(status.errors_left))
            rr.log("drive/errors/right", rr.Scalars(status.errors_right))
            trail.append((status.x, status.y))
            if len(trail) > 1:
                rr.log("pose/path", rr.LineStrips2D([trail]))
            if count % 2 == 0:
                data.qpos[free_q : free_q + 3] = [status.x, status.y, z0]
                half = status.heading / 2
                data.qpos[free_q + 3 : free_q + 7] = [
                    math.cos(half), 0.0, 0.0, math.sin(half),
                ]
                pose = mime_pose(count, arrive_at, pick_at)
                for adr, val in zip(arm_adrs, pose[:4]):
                    data.qpos[adr] = val
                for adr in jaw_adrs:
                    data.qpos[adr] = pose[4]
                mujoco.mj_forward(model, data)
                mirror.log(data)
                rr.log(
                    "world/car/trail",
                    rr.LineStrips3D(
                        [[[x, y, 0.005] for x, y in trail]],
                        colors=[[80, 160, 255]],
                    ),
                )
            count += 1
        seen = len(lines)
    time.sleep(0.25)
