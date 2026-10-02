"""mjlab's own cartpole, narrating itself into the Studio.

    cd trainnr-mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
        uv run python -m trainnr_mjlab.demo_recorder [--steps 2000] [--envs 64]

Loads `Mjlab-Cartpole-Balance` exactly as mjlab registers it, attaches
the `RerunRecorder` to the env cfg's own `recorders` slot, and drives
random actions — the recorder streams world 0's joints, reward and
episode boundaries to the Studio's :9876 (or any Rerun viewer listening
there). This is B4's proof and also the first time this package's
pieces run inside mjlab's REAL manager stack rather than the tests'
harness; anything the framework does differently from the harness fails
here, loudly, before a training run pays for it.
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--task", default="Mjlab-Cartpole-Balance")
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args()

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg  # noqa: PLC0415

    from trainnr_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.envs
    cfg.recorders = {"rerun": RerunRecorderCfg(every=args.every)}
    env = ManagerBasedRlEnv(cfg, device=device)
    print(f"[demo] {args.task}: {args.envs} envs on {device}, streaming to :9876")
    env.reset()
    action_dim = env.action_manager.total_action_dim
    for step in range(args.steps):
        actions = torch.rand((args.envs, action_dim), device=device) * 2 - 1
        env.step(actions)
        if step % 500 == 0:
            print(f"[demo] step {step}")
    env.close()
    print("[demo] done — the story is in the viewer")


if __name__ == "__main__":
    main()
