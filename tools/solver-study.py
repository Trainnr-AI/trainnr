"""Sweep MuJoCo's constraint solvers on the kitting scene — measured.

    cd pipeline && uv run --extra sim python ../tools/solver-study.py

One scripted kitting episode (trial 0 of the declared band) per solver
configuration, judged by the task's own referee. Columns: verdict,
wall time, worst contact penetration, mean constraint-solver
iterations — and the docs/48 §3 diagnostics: peak tangential SLIP at
the pad-part contacts (efc_vel rows, the direct measure of what
impratio buys; elliptic rows only — pyramidal facet rows are not
velocities, printed as -) and peak pad normal FORCE (mj_contactForce).
The impratio rows at the bottom sweep the anti-slip knob itself.
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
from rq_pipeline.tasks.scene import apply_options  # noqa: E402

# The shipped baseline first, then one axis moved at a time, then the
# solver x cone cross, then the impratio sweep (elliptic + newton).
CONFIGS = [
    ("newton", "elliptic", "euler", 10),  # what every result so far ran on
    ("newton", "elliptic", "implicitfast", 10),
    ("newton", "pyramidal", "euler", 10),
    ("cg", "elliptic", "euler", 10),
    ("cg", "pyramidal", "euler", 10),
    ("pgs", "pyramidal", "euler", 10),  # PGS requires pyramidal (dual solver)
    ("newton", "elliptic", "euler", 1),  # impratio off: MuJoCo's default
    ("newton", "elliptic", "euler", 3),
    ("newton", "elliptic", "euler", 30),
]

PADS = tuple(
    f"{arm}/{finger}_g1" for arm in ("left", "right") for finger in ("left", "right")
)
PARTS = ("part_left_geom", "part_right_geom")


NORMAL_FORCE = 0  # mj_contactForce's wrench: [normal, tangent, tangent, torques]
MM_PER_M = 1000.0


def run_config(solver: str, cone: str, integrator: str, impratio: float) -> dict:
    task = build_kitting()
    model = task.spec.compile()
    apply_options(
        model.opt, solver=solver, cone=cone, integrator=integrator, impratio=impratio
    )
    initial = keyframe_state(model, "neutral_pose")
    initial = task.protocol.perturb(0, initial)

    pad_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in PADS}
    part_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in PARTS}
    elliptic = model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    min_slip_rows = 3  # normal + two sliding directions

    worst_pen = 0.0
    peak_slip = 0.0
    peak_force = 0.0
    niters: list[int] = []
    wrench = np.zeros(6)

    def probe(tick: int, live: mujoco.MjData) -> None:
        nonlocal worst_pen, peak_slip, peak_force
        if live.ncon:
            worst_pen = min(worst_pen, float(live.contact.dist.min()))
        niters.append(int(live.solver_niter[0]))
        for i in range(live.ncon):
            g1, g2 = int(live.contact.geom[i][0]), int(live.contact.geom[i][1])
            pair = {g1, g2}
            if not (pair & pad_ids and pair & part_ids):
                continue
            mujoco.mj_contactForce(model, live, i, wrench)
            peak_force = max(peak_force, float(wrench[NORMAL_FORCE]))
            addr = int(live.contact.efc_address[i])
            # Elliptic rows are constraint-space VELOCITIES: rows
            # addr+1, addr+2 are the two sliding directions. Pyramidal
            # rows are facet combinations, not velocities — skipped.
            if elliptic and addr >= 0 and int(live.contact.dim[i]) >= min_slip_rows:
                slip = float(np.hypot(live.efc_vel[addr + 1], live.efc_vel[addr + 2]))
                peak_slip = max(peak_slip, slip)

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
        "worst_pen_mm": -MM_PER_M * worst_pen,
        "mean_niter": float(np.mean(niters)) if niters else float("nan"),
        "slip_mm_s": MM_PER_M * peak_slip if elliptic else float("nan"),
        "grip_n": peak_force,
    }


def main() -> None:
    print(
        f"{'solver':<8}{'cone':<11}{'integrator':<14}{'impr':>5} {'verdict':<9}"
        f"{'wall_s':>7}{'pen_mm':>8}{'niter':>7}{'slip_mm_s':>10}{'grip_N':>8}"
    )
    for solver, cone, integrator, impratio in CONFIGS:
        r = run_config(solver, cone, integrator, impratio)
        slip = (
            f"{r['slip_mm_s']:>10.0f}" if np.isfinite(r["slip_mm_s"]) else f"{'-':>10}"
        )
        print(
            f"{solver:<8}{cone:<11}{integrator:<14}{impratio:>5g}{'':<1}{r['verdict']:<9}"
            f"{r['wall_s']:>7.1f}{r['worst_pen_mm']:>8.2f}{r['mean_niter']:>7.1f}"
            f"{slip}{r['grip_n']:>8.2f}",
            flush=True,
        )
    print(
        "\npen_mm = worst interpenetration; niter = mean solver iterations per"
        "\ntick; slip = peak tangential pad-part slip (mm/s, elliptic rows only);"
        "\ngrip_N = peak pad normal force. Baseline row first; the referee judges.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
