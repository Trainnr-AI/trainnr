"""Sweep MuJoCo's constraint solvers on the kitting scene — measured.

    cd pipeline && uv run --extra sim python ../tools/solver-study.py

One scripted kitting episode (trial 0, the proven band) per solver
configuration, judged by the task's own referee. Columns: did the
episode still succeed, wall time, worst contact penetration, and mean
constraint-solver iterations per control tick. The point is the
docs/e2e-research/47 question: which solver x cone x integrator choices
preserve OUR physics, and what each costs — numbers, not folklore.
(Naming collision, pinned: "newton" below is MuJoCo's mjSOL_NEWTON
constraint solver, not the NVIDIA Newton engine.)
"""

import sys
import time

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: E402
from rq_pipeline.tasks.aloha2 import (  # noqa: E402
    build_kitting,
    scripted_kitting_episode,
)

SOLVERS = {
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
}
CONES = {
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
# The shipped baseline first, then one axis moved at a time, then the
# full cross of solver x cone at the baseline integrator.
CONFIGS = [
    ("newton", "elliptic", "euler"),  # what every result so far ran on
    ("newton", "elliptic", "implicitfast"),
    ("newton", "pyramidal", "euler"),
    ("cg", "elliptic", "euler"),
    ("cg", "pyramidal", "euler"),
    ("pgs", "pyramidal", "euler"),  # PGS requires pyramidal (dual solver)
]


def run_config(solver: str, cone: str, integrator: str) -> dict:
    task = build_kitting()
    model = task.spec.compile()
    model.opt.solver = SOLVERS[solver]
    model.opt.cone = CONES[cone]
    model.opt.integrator = INTEGRATORS[integrator]
    initial = keyframe_state(model, "neutral_pose")
    initial = task.protocol.perturb(0, initial)

    worst_pen = 0.0
    niters: list[int] = []

    def probe(tick: int, live: mujoco.MjData) -> None:
        nonlocal worst_pen
        if live.ncon:
            worst_pen = min(worst_pen, float(live.contact.dist.min()))
        niters.append(int(live.solver_niter[0]))

    t0 = time.perf_counter()
    try:
        states, sensors, _actions = scripted_kitting_episode(
            model, initial, on_control=probe
        )
        verdict = "PASS" if task.protocol.success(states, sensors) else "fail"
    except RuntimeError as error:
        verdict = f"IK refused ({error})"
    wall = time.perf_counter() - t0
    return {
        "verdict": verdict,
        "wall_s": wall,
        "worst_pen_mm": -1000.0 * worst_pen,
        "mean_niter": float(np.mean(niters)) if niters else float("nan"),
    }


print(
    f"{'solver':<8}{'cone':<11}{'integrator':<14}{'verdict':<10}"
    f"{'wall_s':>7}{'pen_mm':>8}{'niter':>7}"
)
for solver, cone, integrator in CONFIGS:
    r = run_config(solver, cone, integrator)
    print(
        f"{solver:<8}{cone:<11}{integrator:<14}{r['verdict']:<10}"
        f"{r['wall_s']:>7.1f}{r['worst_pen_mm']:>8.2f}{r['mean_niter']:>7.1f}",
        flush=True,
    )
print(
    "\npen_mm = worst contact interpenetration; niter = mean constraint-"
    "solver\niterations per sampled tick. Baseline row first; the referee "
    "is the judge.",
    file=sys.stderr,
)
