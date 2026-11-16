"""Motion tracking training: the G1 learns to follow a reference motion
(mjlab's tracking task, BeyondMimic's shape), as a trainnr job like a walk.

    cd trainnr-mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m \\
        trainnr_mjlab.tracking_train <reference recording> --log-dir <run folder> \\
        [--envs 4096] [--iterations 6000]

The reference is a recording made from video (`trainnr_mjlab.video_reference`)
whose `raw/motion.npz` is the motion mjlab reads: joint and body poses and
velocities at a fixed rate. The run is everything a walk run is in the Studio:
a job in Running now with its iteration and reward, the watched world
streamed live into the viewer (`trainnr_mjlab.recorder`), and a run
folder with its identity, checkpoints, console and saved stream, so the
project lists it as an experiment.

mjlab's own `scripts/train.py` is not used: it trains one world unless
told otherwise, logs to Weights & Biases by default, and is invisible to
the Studio (measured 2026-11-15: two runs learned nothing on one world,
and the user could not see either).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trainnr.mcp_jobs import default_jobs_root, jobs_dir_of, track
from trainnr.project.kinds import IDENTITY_FILE
from trainnr.robots.recording import MANIFEST_FILE as RECORDING_FILE

from trainnr_mjlab.walk_train import TRAIN_STREAM, Tee, TrainingTicker
from trainnr_mjlab.walks import Identity, use_project

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"
ROBOT = "unitree_g1"
MOTION_FILE = "raw/motion.npz"
# The job table's word, the door's own (`train_tracking`).
TRAIN_KIND = "train-tracking"
ENVS = 4096  # mjlab's tracking recipe; one world learns nothing
ITERATIONS = 6000
EVERY = 100


# MuJoCo-Warp prints "linesearch iterations limit reached" (and "solver
# iterations limit reached") from the GPU each time a world's solver hits
# mjlab's limits (20 line-search steps, 10 iterations): with 4096 G1
# worlds that was 100 million lines, 9.3 GB of job log in 400 iterations
# (2026-11-16). The prints alone are turned off; the solver, its limits
# and every other warning stay as mjlab sets them.
QUIET_WARNINGS = ("LS_ITERATIONS", "ITERATIONS")


def quiet_line_search(sim_cfg: Any) -> Any:
    """`sim_cfg` as a SimulationCfg that also clears the warning's bit, in
    mjlab's own hook for MuJoCo-Warp options (`apply_wp_opt`, after the
    model is built and before its CUDA graphs are captured; the built
    model is read-only)."""
    from dataclasses import fields  # noqa: PLC0415

    import mujoco_warp as mjw  # noqa: PLC0415
    from mjlab.sim.sim import SimulationCfg  # noqa: PLC0415

    class QuietSimulationCfg(SimulationCfg):
        def apply_wp_opt(self, wp_opt: Any) -> None:
            super().apply_wp_opt(wp_opt)
            for name in QUIET_WARNINGS:
                wp_opt.warn_overflow = int(wp_opt.warn_overflow) & ~int(
                    getattr(mjw.OverflowType, name)
                )

    return QuietSimulationCfg(
        **{f.name: getattr(sim_cfg, f.name) for f in fields(sim_cfg)}
    )


def main() -> None:  # noqa: PLR0915 - one trainer, each step named
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("reference", type=Path, help="the reference recording")
    parser.add_argument("--reference-stamp", default=None)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--envs", type=int, default=ENVS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--every", type=int, default=EVERY)
    parser.add_argument("--frame-every", type=int, default=400)
    args = parser.parse_args()

    reference = args.reference.resolve()
    motion = reference / MOTION_FILE
    if not motion.is_file():
        raise SystemExit(f"{reference}: no {MOTION_FILE}")
    record: dict[str, Any] = json.loads((reference / RECORDING_FILE).read_text())

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: PLC0415
    from trainnr.viz import TRAINING_APP_PREFIX, viewer_file  # noqa: PLC0415

    from trainnr_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    use_project(args.project)
    cfg = load_env_cfg(TASK_ID)
    cfg.commands["motion"].motion_file = str(motion)
    cfg.scene.num_envs = args.envs
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.sim = quiet_line_search(cfg.sim)
    print(
        f"[tracking] MuJoCo-Warp's {', '.join(QUIET_WARNINGS)} warnings are not printed"
    )
    agent = load_rl_cfg(TASK_ID)
    agent.max_iterations = args.iterations
    agent.logger = "tensorboard"  # mjlab's default is Weights & Biases: a network call

    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.recorders = {
        "rerun": RerunRecorderCfg(
            app_id=f"{TRAINING_APP_PREFIX}tracking-{log_dir.name}",
            every=args.every,
            frame_every=args.frame_every,
            file=str(viewer_file(log_dir, TRAIN_STREAM)),
        )
    }
    identity = {
        Identity.ROBOT: record.get("census", {}).get("robot", ROBOT),
        Identity.TASK: TASK_ID,
        Identity.SEED: cfg.seed,
        "reference": args.reference_stamp or reference.name,
        "footage": record.get("census", {}).get("footage"),
        "quiet_warnings": list(QUIET_WARNINGS),
    }
    (log_dir / IDENTITY_FILE).write_text(json.dumps(identity, indent=1))
    sys.stdout = Tee(sys.stdout, log_dir / "train.log")
    sys.stderr = Tee(sys.stderr, log_dir / "train.log")
    print(f"[tracking] identity: {identity}")
    print(f"[tracking] {args.envs} envs on {device}, {args.iterations} iterations")

    with track(
        TRAIN_KIND,
        jobs_dir=jobs_dir_of(args.project or default_jobs_root()),
        name=log_dir.name,
        viewer=str(viewer_file(log_dir, TRAIN_STREAM)),
        stream=f"{TRAINING_APP_PREFIX}tracking-{log_dir.name}",
    ) as run:
        run.stage(f"building {args.envs} G1 worlds on {device}")
        env = ManagerBasedRlEnv(cfg, device=device)
        runner = MotionTrackingOnPolicyRunner(
            RslRlVecEnvWrapper(env), asdict(agent), log_dir=str(log_dir), device=device
        )
        run.progress(
            0, args.iterations, "iterations", f"iteration 0 of {args.iterations}"
        )
        sys.stdout = Tee(sys.stdout, None, on_line=TrainingTicker(run).line)
        started = time.perf_counter()
        runner.learn(num_learning_iterations=args.iterations)
        wall = time.perf_counter() - started
        env.close()
        run.stage(f"done: {args.iterations} iterations in {wall:.0f} s")
        print(f"[tracking] done in {wall:.0f} s - checkpoints in {log_dir}")


if __name__ == "__main__":
    main()
