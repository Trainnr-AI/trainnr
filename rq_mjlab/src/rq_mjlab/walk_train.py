"""Flagship walk training: microduck learns to walk — smoke or for real.

    # G3, the real run (their PPO recipe, archived log_dir, checkpoints):
    cd rq_mjlab && MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 \\
        LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m rq_mjlab.walk_train \\
        [--envs 4096] [--iterations 8000] [--log-root ../runs/microduck-walk]

    # G2's two-minute box gate (borrowed velocity-family cfg, no archive):
    cd rq_mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
        uv run python -m rq_mjlab.walk_train --agent smoke

One driver, two agents (`walk_smoke.py` was this file minus the agent
cfg — folded 2026-09-01). Both build `microduck_walk_env_cfg` (the
certified stack: stamped robot, certified xl330.m6, declared-basis DR)
and print the run identity every record must carry.

`--agent g3` is the run docs/e2e-research/63 G3 asks for: Pollen's own
PPO recipe, transcribed number-for-number from
`microduck_rl:src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
at d424a0c (`MicroduckRlCfg`): (512, 256, 128) actor/critic with obs
normalization, clip 0.2, entropy 0.01, 5 epochs x 4 minibatches,
lr 1e-3 adaptive at KL 0.01, gamma 0.99, lam 0.95, 24 steps/env,
save every 250. Their symmetry variant ships DISABLED in production
(`ENABLE_SYMMETRY = False`) and is omitted here with them. Their
max_iterations is 50k (days); the gate asks for a GAIT, which shows
far earlier — iterations are a knob, checkpoints land every
save_interval, and the run archives itself under a timestamped
log_dir with identity.json.

`--agent smoke` is G2's gate: MOTION under the real manager stack, not
a gait — a short burst on the velocity family's borrowed PPO shape,
recorder on, nothing archived.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Per-agent knob defaults; an explicit flag always wins.
from rq_mjlab.walks import DEFAULT_ROBOT, ROBOTS, use_project, walk_spec

DEFAULTS = {
    "g3": {"envs": 4096, "iterations": 8000, "every": 100},
    "smoke": {"envs": 256, "iterations": 20, "every": 10},
}


# The saved stream of a training run, inside its folder (docs/76 §10.5).
TRAIN_STREAM = "train"


def smoke_agent(iterations: int) -> Any:
    """G2's borrowed cfg: the velocity family's PPO shape, smoke-scale."""
    from mjlab.tasks.registry import load_rl_cfg  # noqa: PLC0415

    agent = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    agent.max_iterations = iterations
    agent.experiment_name = "microduck-walk-smoke"
    return agent


def g3_agent(iterations: int) -> Any:
    """Their recipe, transcribed (docs/e2e-research/63 §2.2)."""
    from mjlab.rl.config import (  # noqa: PLC0415
        RslRlModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    # One recipe for both networks; Any so it unpacks into either.
    model_cfg: dict[str, Any] = dict(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            **model_cfg,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(**model_cfg),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),  # rest matches theirs
        num_steps_per_env=24,
        save_interval=250,
        max_iterations=iterations,
        experiment_name="microduck-walk",
        logger="tensorboard",
    )


def _span(text: str) -> float | str:
    """A declared span as a number, or the word `identified` for the
    bundle's own interval (microduck_walk.IDENTIFIED)."""
    if text == "identified":
        return text
    return float(text)


class Tee:
    """A stream that writes to the console and to a file, line-buffered
    so a reader of the file sees an iteration as soon as it is printed."""

    def __init__(self, console: Any, path: Path) -> None:
        self._console = console
        self._file = path.open("a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, text: str) -> int:
        self._console.write(text)
        self._file.write(text)
        return len(text)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._console.fileno()


def main() -> None:  # noqa: PLR0915 - one CLI, each knob named
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", choices=sorted(DEFAULTS), default="g3")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="a project root: its robots are searched first (the Go2 lives there)",
    )
    parser.add_argument(
        "--robot",
        choices=ROBOTS,
        default=DEFAULT_ROBOT,
        help="which walk (rq_mjlab.walks): the microduck on its certified bundle, "
        "or mjlab's own Go1 flat task around its derived gains",
    )
    parser.add_argument(
        "--head",
        choices=("free", "pinned"),
        default="free",
        help="pinned takes the four head joints out of the action space - their "
        "servos hold neutral (docs/07 2026-09-05: free heads park at their "
        "limits and pass through the shoulders); recorded in the identity",
    )
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--log-root", type=Path, default=None, help="default ../runs/<robot>-walk"
    )
    parser.add_argument("--every", type=int, default=None)
    parser.add_argument("--frame-every", type=int, default=400)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="the actuator bundle to train on (microduck only; default the "
        "shipped xl330.m6; robots/actuator-bundles/xl330-refit.m6 is our refit "
        "with its interval)",
    )
    parser.add_argument(
        "--dr-span",
        type=_span,
        default=None,
        help="declared actuator-DR span around the identified point (the bundle's "
        "fit, or Go1's derived gains); 0 = none; default the walk's own "
        "(the walk C1 study's arms: 0 / 0.10 / 0.30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the env's seed (mjlab's default 42 when unset): a study's replicates "
        "name theirs, and the identity records it",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="archive here instead of <log-root>/<timestamp> (a study names its arms)",
    )
    parser.add_argument(
        "--task-stamp",
        default=None,
        help="the declared environment's version (a project's task): cited by the run",
    )
    parser.add_argument(
        "--no-recorder",
        action="store_true",
        help="headless run (a pod with no Studio listening on :9876)",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="a captured scene's folder (a project's scenes/<name>): the walk "
        "trains on its heightfield from the course's start, the head camera "
        "seeing its splat (docs/78 E2; the Go2 only)",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="on a scene, train without the head camera (the rate without pictures)",
    )
    args = parser.parse_args()
    knobs = DEFAULTS[args.agent]
    envs = args.envs if args.envs is not None else knobs["envs"]
    iterations = args.iterations if args.iterations is not None else knobs["iterations"]
    every = args.every if args.every is not None else knobs["every"]

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from rq_pipeline.viz import viewer_file  # noqa: PLC0415

    from rq_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    use_project(args.project)
    spec = walk_spec(args.robot)
    span = spec.default_span if args.dr_span is None else args.dr_span
    cfg, identity = spec.env_cfg(
        dr_span=span or None,
        pin_scale=None,
        bundle=args.bundle,
        head=args.head,
        scene=args.scene,
        cameras=not args.no_cameras,
    )
    print(f"[train] actuator {identity['actuator']}; dr_basis: {identity['dr_basis']}")
    if args.scene is not None and not args.no_cameras:
        from rq_mjlab.scene_stage import render_splats  # noqa: PLC0415

        # the scene's gaussians into every camera mjlab renders (docs/78 E2)
        gaussians = render_splats(args.scene)
        identity["cameras"] = f"{identity['cameras']} over {gaussians} splats"
    if args.scene is not None:
        print(f"[train] scene {identity['scene']}; cameras: {identity['cameras']}")
    log_root = args.log_root or Path(f"../runs/{args.robot}-walk")
    cfg.scene.num_envs = envs
    if args.seed is not None:
        cfg.seed = args.seed
    identity = {**identity, "seed": cfg.seed}
    if args.task_stamp:
        identity["task"] = args.task_stamp
    # The smoke gate archives nothing; g3 archives itself with identity,
    # and its saved stream lands inside that folder (docs/76 §10.5).
    log_dir = None
    if args.agent == "g3":
        log_dir = args.log_dir or log_root / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
    cfg.recorders = (
        {}
        if args.no_recorder
        else {
            "rerun": RerunRecorderCfg(
                app_id=f"rq-walk-{args.agent}",
                every=every,
                frame_every=args.frame_every,
                file=str(viewer_file(log_dir, TRAIN_STREAM)) if log_dir else None,
            )
        }
    )

    agent = smoke_agent(iterations) if args.agent == "smoke" else spec.agent(iterations)
    if log_dir is not None:
        (log_dir / "identity.json").write_text(json.dumps(identity, indent=1))
        # The console also lands in the run folder as train.log: the
        # project's live view reads it while the run trains (docs/77).
        sys.stdout = Tee(sys.stdout, log_dir / "train.log")
        sys.stderr = Tee(sys.stderr, log_dir / "train.log")
        print(f"[g3] log_dir: {log_dir}")

    tag = args.agent
    print(f"[{tag}] identity: {identity}")
    print(f"[{tag}] {envs} envs on {device}, {iterations} iterations")

    env = ManagerBasedRlEnv(cfg, device=device)
    runner = MjlabOnPolicyRunner(
        RslRlVecEnvWrapper(env),
        asdict(agent),
        log_dir=None if log_dir is None else str(log_dir),
        device=device,
    )
    started = time.perf_counter()
    runner.learn(num_learning_iterations=iterations)
    wall = time.perf_counter() - started
    env_steps = envs * agent.num_steps_per_env * iterations
    # the rate as run, in the log: what a scene's cameras cost is a
    # measured number (docs/78 E2), never a guess
    print(
        f"[{tag}] {env_steps} env-steps in {wall:.0f} s = {env_steps / wall:.0f} "
        f"steps/s ({envs} envs x {agent.num_steps_per_env} x {iterations})"
    )
    env.close()
    print(f"[{tag}] done" + (f" - checkpoints in {log_dir}" if log_dir else ""))


if __name__ == "__main__":
    main()
