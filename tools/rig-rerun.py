"""Live Rerun view of a rig session: tail a growing .wire, plot everything —
and draw the rig in realtime 3D, posed by odometry and the stage notes.

    cd pipeline && uv run --extra sim --extra viz python ../tools/rig-rerun.py <file.wire>

Tails the recording as `hil-host --record` writes it (works on finished
files too — it replays then keeps watching). Logs, on the 50 Hz seq clock:

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
bus), so the arm is drawn on the firmware's own schedule — the salute
at the starting gun, the mime cued by the PICKING note, both from
yellow.py's single reconstruction at the firmware's true slew. A
reconstruction, honest to the program if not the metal; putting arm
pulses on the wire is the queued fix.
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
    AIR_TUCK,
    air_mime_pose,
    compose_rig,
    salute_pose,
)

REPO = Path(__file__).resolve().parent.parent
PROFILE = load_profile(REPO / "robots" / "rig-drivetrain")
TICKS_PER_REV = PROFILE.ticks_per_revolution
TICK = 1.0 / STATUS_HZ

if len(sys.argv) < 2:
    sys.exit("usage: rig-rerun.py <file.wire>")
path = Path(sys.argv[1])
rr.init("rig-session", spawn=False)
try:
    rr.connect_grpc()
except Exception:
    rr.spawn()

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


def arm_pose_at(t: float, pick_t: float | None) -> list[float]:
    """Arm reconstruction at wire-time t (seconds since first status)."""
    if pick_t is not None and t >= pick_t:
        return air_mime_pose(t - pick_t)
    wave = salute_pose(t)
    return wave if wave is not None else list(AIR_TUCK)


first_seq = None
trail = []
frame_index = 0
pick_t = None
partial = ""
handle = None
waited = 0.0
print(f"tailing {path} (ticks/rev {TICKS_PER_REV}, ctrl-c to stop)")
while True:
    if handle is None:
        if path.exists():
            handle = path.open("r", errors="replace")
        else:
            waited += 0.25
            if waited == 5.0:
                print(f"warning: {path} has not appeared after 5 s — typo?")
            time.sleep(0.25)
            continue
    # Incremental read with a partial-line buffer: a read can land
    # mid-line, and consuming a torn stage note ("# fetch ARR") would
    # lose its cue forever. The old whole-file re-read was also O(n^2).
    chunk = handle.read()
    if not chunk:
        time.sleep(0.25)
        continue
    chunk = partial + chunk
    lines = chunk.split("\n")
    partial = lines.pop()
    for raw in lines:
        line = raw.strip()
        if line.startswith("<"):
            line = line[1:].lstrip()
        if line.startswith("# fetch"):
            if "arm PICKING" in line and pick_t is None and first_seq is not None:
                pick_t = frame_index * TICK
            if first_seq is not None:
                rr.log("stage", rr.TextLog(line[2:]))
            continue
        status = parse_status(line)
        if status is None:
            continue
        if first_seq is None:
            first_seq = status.seq
        # seq is the device's own clock and survives dropped lines;
        # the mime clock must ride it too or the arm drifts late by
        # the cumulative drop count.
        frame_index = status.seq - first_seq
        t = frame_index * TICK
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
        if frame_index % 2 == 0:
            data.qpos[free_q : free_q + 3] = [status.x, status.y, z0]
            half = status.heading / 2
            data.qpos[free_q + 3 : free_q + 7] = [
                math.cos(half), 0.0, 0.0, math.sin(half),
            ]
            pose = arm_pose_at(t, pick_t)
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
