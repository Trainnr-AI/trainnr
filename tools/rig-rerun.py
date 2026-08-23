"""Live Rerun view of a rig session: tail a growing .wire, plot everything.

    cd pipeline && uv run --extra viz python ../tools/rig-rerun.py <file.wire>

Tails the recording as `hil-host --record` writes it (works on finished
files too — it replays then keeps watching). Spawns the Rerun viewer and
logs, on the 50 Hz seq clock:

    drive/duty            commanded duty (%)
    drive/ticks/left      cumulative encoder ticks
    drive/ticks/right
    drive/angle/left      wheel angle (rev) via the bundle's ticks/rev
    drive/angle/right
    drive/errors/left     encoder decode errors — should stay ~flat
    drive/errors/right
    pose/path             odometry (x, y) trail
"""

import sys
import time
from math import tau
from pathlib import Path

import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from rq_pipeline.bundles.profile import load_profile  # noqa: E402
from rq_pipeline.collect.frames import STATUS_HZ  # noqa: E402
from rq_pipeline.collect.wire import parse_status  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROFILE = load_profile(REPO / "robots" / "rig-drivetrain")
TICKS_PER_REV = PROFILE.ticks_per_revolution

path = Path(sys.argv[1])
rr.init("rig-session", spawn=True)

seen = 0
first_seq = None
trail = []
print(f"tailing {path} (ticks/rev {TICKS_PER_REV}, ctrl-c to stop)")
while True:
    if path.exists():
        lines = path.read_text(errors="replace").splitlines()
        for raw in lines[seen:]:
            line = raw.strip()
            if line.startswith("<"):
                line = line[1:].lstrip()
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
        seen = len(lines)
    time.sleep(0.25)
