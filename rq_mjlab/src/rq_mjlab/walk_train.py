"""Flagship G3: microduck learns to walk for real, on this box.

    cd rq_mjlab && MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 \\
        LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m rq_mjlab.walk_train \\
        [--envs 4096] [--iterations 8000] [--log-root ../runs/microduck-walk]

The real training run docs/e2e-research/63 G3 asks for: the certified
stack (stamped robot, certified xl330.m6, declared-basis DR) under
Pollen's own PPO recipe, transcribed number-for-number from
`microduck_rl:src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
at d424a0c (`MicroduckRlCfg`): (512, 256, 128) actor/critic with obs
normalization, clip 0.2, entropy 0.01, 5 epochs x 4 minibatches,
lr 1e-3 adaptive at KL 0.01, gamma 0.99, lam 0.95, 24 steps/env,
save every 250. Their symmetry variant ships DISABLED in production
(`ENABLE_SYMMETRY = False`) and is omitted here with them. Their
max_iterations is 50k (days); the gate asks for a GAIT, which shows
far earlier - iterations are a knob, checkpoints land every
save_interval, and the run archives itself under a timestamped
log_dir. The RerunRecorder streams the watched duck into the Studio
at gentle rates; the run identity (robot stamp, actuator stamp, DR
basis) prints at launch and lands in the log_dir as identity.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--log-root", type=Path, default=Path("../runs/microduck-walk"))
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=400)
    parser.add_argument(
        "--no-recorder",
        action="store_true",
        help="headless run (a pod with no Studio listening on :9876)",
    )
    args = parser.parse_args()

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import (  # noqa: PLC0415
        MjlabOnPolicyRunner,
        RslRlVecEnvWrapper,
    )
    from mjlab.rl.config import (  # noqa: PLC0415
        RslRlModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415
    from rq_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    cfg, identity = microduck_walk_env_cfg()
    cfg.scene.num_envs = args.envs
    cfg.recorders = (
        {}
        if args.no_recorder
        else {
            "rerun": RerunRecorderCfg(
                app_id="rq-walk-g3", every=args.every, frame_every=args.frame_every
            )
        }
    )

    model_cfg = dict(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    agent = RslRlOnPolicyRunnerCfg(
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
        max_iterations=args.iterations,
        experiment_name="microduck-walk",
        logger="tensorboard",
    )

    log_dir = args.log_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "identity.json").write_text(json.dumps(identity, indent=1))
    print(f"[g3] identity: {identity}")
    print(f"[g3] {args.envs} envs on {device}, {args.iterations} iterations")
    print(f"[g3] log_dir: {log_dir}")

    env = ManagerBasedRlEnv(cfg, device=device)
    runner = MjlabOnPolicyRunner(
        RslRlVecEnvWrapper(env), asdict(agent), log_dir=str(log_dir), device=device
    )
    runner.learn(num_learning_iterations=args.iterations)
    env.close()
    print(f"[g3] done - checkpoints in {log_dir}")


if __name__ == "__main__":
    main()
