"""The flagship's G2 gate: microduck learns to walk for two minutes,
live in the Studio.

    cd rq_mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
        uv run python -m rq_mjlab.walk_smoke [--iterations 20] [--envs 256]

Builds `microduck_walk_env_cfg` (the certified stack: stamped robot,
certified actuator, declared-basis DR), attaches the RerunRecorder so
the watched duck's joints and rewards stream to the Studio's :9876,
and runs rsl-rl for a short burst — G2 asks for MOTION under the real
manager stack, not a gait (G3, the paid run, asks for the gait). The
rl cfg borrows the velocity family's PPO shape with smoke-scale
iterations; G3 gets its own tuned cfg. Prints the run identity
(robot stamp, actuator stamp, DR basis) the way every record must
carry it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_rl_cfg  # noqa: PLC0415

    from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415
    from rq_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    cfg, identity = microduck_walk_env_cfg()
    cfg.scene.num_envs = args.envs
    cfg.recorders = {"rerun": RerunRecorderCfg(every=args.every)}

    agent = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    agent.max_iterations = args.iterations
    agent.experiment_name = "microduck-walk-smoke"

    print(f"[smoke] identity: {identity}")
    print(f"[smoke] {args.envs} envs on {device}, {args.iterations} iterations")
    env = ManagerBasedRlEnv(cfg, device=device)
    runner = MjlabOnPolicyRunner(
        RslRlVecEnvWrapper(env), asdict(agent), log_dir=None, device=device
    )
    runner.learn(num_learning_iterations=args.iterations)
    env.close()
    print("[smoke] done - the ducks moved; the story is in the viewer")


if __name__ == "__main__":
    main()
