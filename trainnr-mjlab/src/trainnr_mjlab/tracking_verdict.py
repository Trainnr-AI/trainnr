"""A tracking policy's evaluation: paired trials through its reference,
pushed and lagged as its tracking task declares, as a trainnr job the
Studio watches:

    uv run python -m trainnr_mjlab.tracking_verdict <run>/model_N.pt \\
        --reference <reference recording> [--trials 40] [--seed 1000] \\
        [--pushes | --no-pushes] [--lag-ms 20] [--task-stamp <task>]

One world per trial, all from the clip's first frame (mjlab's play mode:
the whole clip, no time limit), with training's conditions judged rather
than switched off: observation noise, the start pose and velocity drawn
around the reference's, and the pushes. Each trial draws its command lag
in [0, lag_ms], whole control ticks, the policy's action reaching the
robot that many ticks late. A trial succeeds when the clip plays to its
end with no termination (mjlab's terminations are the tracking bounds).

Every trial is an `EpisodeRecord` appended to `verdict/records-<suffix>.jsonl`
beside the checkpoint; the count and the exact Clopper-Pearson interval
land in `verdict/walk-verdict-<suffix>.json`, the file the project turns
into a policy and an evaluation (trainnr.project.live). World 0 streams
live into the Studio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from trainnr.evaluate.records import EpisodeRecord, append_records
from trainnr.mcp_jobs import default_jobs_root, jobs_dir_of, track
from trainnr.project.kinds import IDENTITY_FILE
from trainnr.stats.intervals import clopper_pearson
from trainnr.tasks.tracking import SUCCESS

from trainnr_mjlab.checkpoint_guard import require_tensor_only
from trainnr_mjlab.tracking_train import TASK_ID, quiet_line_search
from trainnr_mjlab.walk_verdict import instrument_for
from trainnr_mjlab.walks import use_project

MOTION_FILE = "raw/motion.npz"
CONTROL_HZ = 50  # the tracking task's control rate
KIND = "evaluate-tracking"
VERDICT_DIR = "verdict"
# The project imports `walk-verdict-*.json` from a run's verdict folder
# (trainnr.envs.rsl_rl_log.VERDICT_GLOB); a tracking verdict is named into it.
VERDICT_PREFIX = "walk-verdict-"
STREAM_FILE = "evaluation.rrd"


def lag_ticks(lag_ms: float, trials: int, seed: int) -> np.ndarray:
    """Each trial's command lag in whole control ticks, drawn uniformly in
    [0, lag_ms] from the evaluation's seed: the same trial, the same lag."""
    most = round(lag_ms * CONTROL_HZ / 1000)
    return np.random.default_rng(seed).integers(0, most + 1, size=trials)


def suffix_of(device: str, *, pushes: bool, lag_ms: float) -> str:
    """What tells this evaluation's files from another's of the checkpoint."""
    tag = "cuda" if device.startswith("cuda") else "cpu"
    return f"tracking-{'push' if pushes else 'still'}-lag{lag_ms:g}-{tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-stamp", default=None)
    parser.add_argument("--task-stamp", default=None)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--pushes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lag-ms", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.checkpoint.resolve().parent
    with track(
        KIND,
        jobs_dir=jobs_dir_of(args.project or default_jobs_root()),
        name=f"{run_dir.name} {args.checkpoint.stem}, {args.trials} trials",
        viewer=str(run_dir / VERDICT_DIR / STREAM_FILE),
        stream=f"trainnr-eval-{run_dir.name}-{args.checkpoint.stem}",
    ) as run:
        judge(args, run)


def judge(args: argparse.Namespace, run: Any) -> None:  # noqa: PLR0915 - the evaluation's procedure, in order
    import torch  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")

    from dataclasses import asdict  # noqa: PLC0415

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415
    from mjlab.tasks.tracking.mdp.commands import (  # noqa: PLC0415
        MotionCommand,
        MotionCommandCfg,
    )
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: PLC0415

    from trainnr_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    use_project(args.project)
    run_dir = args.checkpoint.resolve().parent
    out = run_dir / VERDICT_DIR
    out.mkdir(exist_ok=True)
    trained = json.loads((run_dir / IDENTITY_FILE).read_text())
    reference_stamp = args.reference_stamp or trained.get("reference")
    source = args.task_stamp or reference_stamp
    instrument = instrument_for(device)

    # play mode for the whole clip from its first frame; training's
    # conditions put back, so the claim is judged, not a sanitized demo
    cfg = load_env_cfg(TASK_ID, play=True)
    training = load_env_cfg(TASK_ID)
    motion_cfg, trained_cfg = cfg.commands["motion"], training.commands["motion"]
    assert isinstance(motion_cfg, MotionCommandCfg)
    assert isinstance(trained_cfg, MotionCommandCfg)
    motion_cfg.motion_file = str(args.reference.resolve() / MOTION_FILE)
    motion_cfg.pose_range = dict(trained_cfg.pose_range)
    motion_cfg.velocity_range = dict(trained_cfg.velocity_range)
    cfg.observations["actor"].enable_corruption = True
    if args.pushes:
        cfg.events["push_robot"] = training.events["push_robot"]
    cfg.scene.num_envs = args.trials
    cfg.seed = args.seed
    cfg.sim = quiet_line_search(cfg.sim)
    cfg.recorders = {
        "rerun": RerunRecorderCfg(
            app_id=f"trainnr-eval-{run_dir.name}-{args.checkpoint.stem}",
            every=2,
            frame_every=10**9,
            file=str(out / STREAM_FILE),
        )
    }
    reference = np.load(args.reference / MOTION_FILE)
    steps = len(reference["joint_pos"])
    ticks = lag_ticks(args.lag_ms, args.trials, args.seed)
    protocol = {
        "trials": args.trials,
        "seed": args.seed,
        "criterion": SUCCESS,
        "reference": reference_stamp,
        "pushes": args.pushes,
        "lag_ms": args.lag_ms,
        "observation_noise": True,
        "start": "the clip's first frame, pose and velocity drawn as in training",
        "instrument": instrument,
    }

    run.stage(f"building {args.trials} worlds on {device}")
    agent = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MotionTrackingOnPolicyRunner(
        wrapped, asdict(agent), log_dir=None, device=device
    )
    require_tensor_only(args.checkpoint)  # before any unpickling
    runner.load(
        str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    command = env.command_manager.get_term("motion")
    assert isinstance(command, MotionCommand)

    n = args.trials
    lag = torch.as_tensor(ticks, device=device)
    history: list[Any] = []
    alive = torch.ones(n, dtype=torch.bool, device=device)
    ended = torch.full((n,), steps, device=device)
    body_err = torch.zeros(n, device=device)
    joint_err = torch.zeros(n, device=device)
    obs = wrapped.get_observations()
    run.stage(f"{n} trials through {steps / CONTROL_HZ:.1f} s of reference")
    with torch.inference_mode():
        for i in range(steps):
            if i % 50 == 0:
                fallen = int((~alive).sum())
                run.progress(i, steps, "steps", f"step {i} of {steps}, {fallen} fallen")
            history = [policy(obs), *history][: int(ticks.max()) + 1]
            # each world's action from `lag` ticks ago (the newest when the
            # rollout is younger than its lag)
            stacked = torch.stack(history)
            index = torch.clamp(lag, max=len(history) - 1)
            action = stacked[index, torch.arange(n, device=device)]
            obs, _, dones, _ = wrapped.step(action)
            body = (
                (command.body_pos_relative_w - command.robot_body_pos_w)
                .norm(dim=-1)
                .mean(dim=-1)
            )
            joint = (command.joint_pos - command.robot_joint_pos).abs().mean(dim=-1)
            body_err += torch.where(alive, body, 0.0)
            joint_err += torch.where(alive, joint, 0.0)
            if i < steps - 1:
                fell = dones.bool() & alive
                ended = torch.where(fell, i + 1, ended)
                alive &= ~fell
    env.close()

    successes = int(alive.sum())
    low, high = clopper_pearson(successes, n)
    policy_name = args.checkpoint.stem
    records = [
        EpisodeRecord(
            source=source,
            policy=policy_name,
            trial=trial,
            success=bool(alive[trial]),
            steps=int(ended[trial]),
            instrument=instrument,
            protocol=protocol,
            seed=args.seed,
            events=({"index": 0, "name": "survived", "passed": bool(alive[trial])},),
            variations={
                "lag_ms": int(ticks[trial]) * 1000 // CONTROL_HZ,
                "mean_body_pos_error_m": round(
                    float(body_err[trial] / ended[trial]), 4
                ),
                "mean_joint_error_rad": round(
                    float(joint_err[trial] / ended[trial]), 4
                ),
            },
        )
        for trial in range(n)
    ]
    suffix = suffix_of(device, pushes=args.pushes, lag_ms=args.lag_ms)
    append_records(out / f"records-{suffix}.jsonl", records)
    verdict = {
        "source": source,
        "policy": policy_name,
        "instrument": instrument,
        "protocol": protocol,
        "identity": {**trained, "tracking_task": args.task_stamp},
        "funnel": {"survived": successes},
        "successes": successes,
        "trials": n,
        "ci95": [round(low, 4), round(high, 4)],
        "median_body_pos_error_m": float(
            np.median([r.variations["mean_body_pos_error_m"] for r in records])
        ),
    }
    verdict_path = out / f"{VERDICT_PREFIX}{suffix}.json"
    if verdict_path.exists():  # never silently replaced (walk_verdict's rule)
        previous = json.loads(verdict_path.read_text())
        old = previous.get("protocol", {})
        backup = out / (
            f"{VERDICT_PREFIX}{suffix}.{previous.get('policy')}"
            f".seed{old.get('seed')}.n{previous.get('trials')}.json"
        )
        if not backup.exists():
            verdict_path.rename(backup)
    verdict_path.write_text(json.dumps(verdict, indent=1))
    run.stage(f"done: {successes}/{n} danced to the end, CP95 [{low:.2f}, {high:.2f}]")
    summary = {k: verdict[k] for k in ("successes", "trials", "ci95")}
    print(f"[verdict] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
