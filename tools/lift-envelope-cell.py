"""One cell of the lift's expert envelope: the scripted pick at a damping/gain
scale, N study-jitter trials, one process per cell (repeated MjSpec compiles
in one process segfault, 2026-09-04). Prints `d<damping> g<gain> k/N`.

    cd pipeline && uv run --extra sim python ../tools/lift-envelope-cell.py 1.0 0.4 [t…
"""

import sys

import numpy as np
from rq_pipeline.physics.mujoco_backend import MuJoCoBackend, keyframe_state
from rq_pipeline.physics.servo_dr import scale_servo_dynamics
from rq_pipeline.tasks.so101 import build_lift_study, scripted_pick

d, g = float(sys.argv[1]), float(sys.argv[2])
ARGV_TRIALS = 3  # position of the optional trials argument
DEFAULT_TRIALS = 4  # the coarse grid's per-cell count; 12 placed the cliff
n = int(sys.argv[ARGV_TRIALS]) if len(sys.argv) > ARGV_TRIALS else DEFAULT_TRIALS
ok = 0
for trial in range(n):
    task = build_lift_study()
    scale_servo_dynamics(task.spec, damping_scale=d, gain_scale=g)
    model = task.spec.compile()
    b = MuJoCoBackend()
    b.load_model(model)
    p = task.protocol
    home = keyframe_state(model, p.home) if p.home else b.default_initial_state()
    st = b.stepper(p.perturb(trial, home), p.steps)
    for tick in range(p.steps // p.control_interval):
        st.advance(
            np.asarray(
                scripted_pick(tick * p.control_interval, st.data.sensordata),
                dtype=float,
            ),
            p.control_interval,
        )
    ok += bool(p.success(st.states, st.sensors))
print(f"d{d} g{g} {ok}/{n}")
