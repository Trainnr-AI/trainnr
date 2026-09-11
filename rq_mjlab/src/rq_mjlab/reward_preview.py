"""See the reward before a training run: a short rollout of the task
under a controller that has learned nothing, every reward term
streamed per step into the Studio beside the 3D world, and a summary
of what each term paid.

    cd rq_mjlab && ../tools/wsl-run.sh uv run python -m rq_mjlab.reward_preview \\
        --robot go2 --project <root> [--controller untrained|stand] [--seconds 5] \\
        [--out <task folder>/preview-untrained.json]

Two controllers: `untrained`, the recipe's actor at its random
initialisation (what iteration 0 of a run does); `stand`, zero actions
(the default posture held). Nothing here is a judgment of a policy -
it is what the reward function says about behaviour nobody trained,
so a term that is silent (a misnamed sensor), a term that dominates,
or a scale that surprises shows before a run is paid for. The field's
tools show terms during a run (docs/33); this shows them before one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rq_mjlab.walks import DEFAULT_ROBOT, ROBOTS, use_project, walk_spec

CONTROLLERS = ("untrained", "stand")
APP_ID = "rq-reward-preview"
FRAME_EVERY = 5  # control steps between camera frames (~30 ms each)


def summarize(steps: list[dict[str, float]], fell: int) -> dict[str, Any]:
    """Per term: mean, min and max over the rollout; the total's mean;
    how many times the world fell and reset."""
    names = sorted({k for row in steps for k in row})
    terms = {}
    for name in names:
        values = [row[name] for row in steps if name in row]
        terms[name] = {
            "mean": round(sum(values) / len(values), 5),
            "min": round(min(values), 5),
            "max": round(max(values), 5),
        }
    total = [sum(row.values()) for row in steps]
    return {
        "steps": len(steps),
        "falls": fell,
        "total_mean": round(sum(total) / len(total), 5) if total else 0.0,
        "terms": terms,
    }


def preview(  # noqa: PLR0913 - the preview's knobs, each named
    *,
    robot: str,
    controller: str,
    seconds: float,
    device: str,
    address: str | None,
    seed: int,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415

    from rq_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    if controller not in CONTROLLERS:
        raise ValueError(f"controller is one of {CONTROLLERS}, got {controller!r}")
    spec = walk_spec(robot)
    cfg, identity = spec.env_cfg(dr_span=None, pin_scale=None)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    recorder = RerunRecorderCfg(app_id=APP_ID, every=1, frame_every=FRAME_EVERY)
    if address:
        recorder.address = address
    cfg.recorders = {"rerun": recorder}
    agent = spec.agent(1)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    unwrapped = env.unwrapped
    if controller == "untrained":
        runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)
        policy = runner.get_inference_policy(device=device)
    else:

        def policy(_obs: Any) -> Any:
            return torch.zeros(
                1, unwrapped.action_manager.total_action_dim, device=device
            )

    ticks = max(1, round(seconds / unwrapped.step_dt))
    obs = env.get_observations()
    rows: list[dict[str, float]] = []
    fell = 0
    for _ in range(ticks):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        rows.append(
            {
                name: float(values[0])
                for name, values in unwrapped.reward_manager.get_active_iterable_terms(
                    0
                )
            }
        )
        if bool(dones[0]) and bool(unwrapped.termination_manager.terminated[0]):
            fell += 1
    env.close()
    return {
        "schema": "trainnr-reward-preview/1",
        "robot": robot,
        "controller": controller,
        "seconds": seconds,
        "identity": identity,
        "recorder": {"app_id": APP_ID, "address": recorder.address},
        **summarize(rows, fell),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=ROBOTS)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--controller", default="untrained", choices=CONTROLLERS)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--address", default=None, help="a Rerun gRPC endpoint")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    use_project(args.project)
    result = preview(
        robot=args.robot,
        controller=args.controller,
        seconds=args.seconds,
        device=args.device,
        address=args.address,
        seed=args.seed,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    width = max(len(n) for n in result["terms"]) if result["terms"] else 8
    print(
        f"[preview] {args.robot} {args.controller}: {result['steps']} steps, "
        f"{result['falls']} falls, total mean {result['total_mean']:+.3f}"
    )
    for name, t in sorted(result["terms"].items(), key=lambda kv: kv[1]["mean"]):
        stats = f"mean {t['mean']:+8.4f}  min {t['min']:+8.4f}  max {t['max']:+8.4f}"
        print(f"[preview] {name:{width}s} {stats}")
    if args.out is not None:
        print(f"[preview] summary -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
