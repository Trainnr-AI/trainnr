"""The locomotion certificate — G4, in C1's shape.

    cd rq_mjlab && LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python -m \\
        rq_mjlab.walk_verdict ../runs/microduck-walk/<stamp>/model_7999.pt \\
        [--trials 40] [--device cuda:0|cpu] [--seed 1000]

One seeded episode per trial through the SAME certified env the
checkpoint trained under (full DR, pushes on — the robustness claim is
judged, not a sanitized demo). Per episode, two declared milestones:

- ``survived``: the episode ended by time_out, never by fell_over or
  nan_state.
- ``tracked``: the episode-mean planar velocity error, measured in the
  base frame against the commanded twist, closes at least half the
  gap that standing still would leave — err_ratio = mean‖v - v*‖ /
  max(mean‖v*‖, ERR_FLOOR) < 0.5. Dimensionless, so a gentle command
  is not an easier exam than a brisk one.

Success = survived AND tracked. Every row is an
`rq_pipeline.evaluate.records.EpisodeRecord` — the run's identity as
the stamped source, the mjlab/mujoco/warp/device instrument string,
the drawn command and errors under `variations` — appended to
records.jsonl beside the checkpoint; the fold and the exact
Clopper-Pearson interval land in walk-verdict.json. Run once per
instrument (cuda:0 and cpu — warp's two device backends); the
plain-CPU-MuJoCo cross-check belongs to C4's deployment manifest,
where the policy leaves torch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The judgment's constants, declared where the certificate cites them.
ERR_RATIO_BOUND = 0.5  # tracked = closes at least half the standing-still gap
ERR_FLOOR = 0.1  # m/s; below this commanded speed the ratio's denominator floors


@dataclass(frozen=True)
class EpisodeOutcome:
    """One episode's measured facts, before any threshold is applied."""

    steps: int
    fell: bool
    mean_err: float  # mean |v_xy - v*_xy| over the episode, m/s
    mean_cmd: float  # mean ‖v*_xy‖ over the episode, m/s

    @property
    def err_ratio(self) -> float:
        return self.mean_err / max(self.mean_cmd, ERR_FLOOR)

    @property
    def survived(self) -> bool:
        return not self.fell

    @property
    def tracked(self) -> bool:
        return self.err_ratio < ERR_RATIO_BOUND

    @property
    def success(self) -> bool:
        return self.survived and self.tracked


@dataclass(frozen=True)
class WorldEpisode:
    """One world's first episode: the judgment plus, when captured, the
    trajectory a dataset writer needs (D2) - the actor's observation
    and the action at every control tick, the qpos to replay for
    frames, and the commanded twist the episode began under."""

    outcome: EpisodeOutcome
    command: list[float]
    observations: Any = None  # (T, obs) float32
    actions: Any = None  # (T, nu) float32
    qpos: Any = None  # (T, nq) float32


def instrument_for(device: str) -> str:
    """The stamp every row carries: the batched stack and its device."""
    import mjlab  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    tag = "cuda" if device.startswith("cuda") else "cpu"
    return (
        f"mjlab-{getattr(mjlab, '__version__', '1.6.0')}"
        f"+mujoco-{mujoco.__version__}+warp-{wp.config.version}+{tag}"
    )


def rollout_episodes(
    env, policy, trials: int, *, capture: bool = False
) -> list[WorldEpisode]:
    """One completed episode per world, judged from the live managers:
    the commanded twist from the command manager, the base-frame
    velocity from the entity, the cause of death from the termination
    manager. Worlds that finish early keep stepping (the env auto-
    resets) but only each world's FIRST episode is recorded. With
    `capture`, the per-tick observation/action/qpos of that first
    episode ride along (the rollout->dataset writer's raw material)."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from rq_mjlab.actuator import as_torch  # noqa: PLC0415

    unwrapped = env.unwrapped
    device = unwrapped.device
    err_sum = torch.zeros(trials, device=device)
    cmd_sum = torch.zeros(trials, device=device)
    steps = torch.zeros(trials, dtype=torch.long, device=device)
    open_worlds = torch.ones(trials, dtype=torch.bool, device=device)
    fell = torch.zeros(trials, dtype=torch.bool, device=device)
    recorded_steps = torch.zeros(trials, dtype=torch.long, device=device)
    device_qpos = as_torch(unwrapped.sim.data.qpos) if capture else None
    trace: list[list[tuple[Any, Any, Any]]] = [[] for _ in range(trials)]

    obs = env.get_observations()  # a TensorDict, not the (obs, extras) pair
    first_command = unwrapped.command_manager.get_command("twist").clone()
    while open_worlds.any():
        with torch.inference_mode():
            actions = policy(obs)
        if capture and device_qpos is not None:
            actor = obs["actor"].detach().cpu().numpy()
            act = actions.detach().cpu().numpy()
            pose = device_qpos.detach().cpu().numpy()
            still_open = open_worlds.cpu().numpy()
            for i in np.flatnonzero(still_open):
                trace[i].append((actor[i], act[i], pose[i]))
        obs, _, dones, _ = env.step(actions)
        command = unwrapped.command_manager.get_command("twist")
        velocity = unwrapped.scene["robot"].data.root_link_lin_vel_b
        err = torch.linalg.norm(velocity[:, :2] - command[:, :2], dim=1)
        cmd = torch.linalg.norm(command[:, :2], dim=1)
        err_sum += err * open_worlds
        cmd_sum += cmd * open_worlds
        steps += open_worlds.long()
        closing = open_worlds & dones.bool()
        if closing.any():
            fell |= closing & unwrapped.termination_manager.terminated
            recorded_steps = torch.where(closing, steps, recorded_steps)
            open_worlds &= ~closing

    episodes = []
    for i in range(trials):
        outcome = EpisodeOutcome(
            steps=int(recorded_steps[i]),
            fell=bool(fell[i]),
            mean_err=float(err_sum[i] / recorded_steps[i]),
            mean_cmd=float(cmd_sum[i] / recorded_steps[i]),
        )
        arrays: dict[str, Any] = {}
        if capture and trace[i]:
            observations, acts, poses = zip(*trace[i], strict=True)
            arrays = {
                "observations": np.stack(observations).astype(np.float32),
                "actions": np.stack(acts).astype(np.float32),
                "qpos": np.stack(poses).astype(np.float32),
            }
        episodes.append(
            WorldEpisode(outcome, [float(v) for v in first_command[i]], **arrays)
        )
    return episodes


def rollout_outcomes(env, policy, trials: int) -> list[EpisodeOutcome]:
    """The certificate's view: judgments only (no trajectories)."""
    return [episode.outcome for episode in rollout_episodes(env, policy, trials)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    import warp as wp  # noqa: PLC0415

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")

    from dataclasses import asdict  # noqa: PLC0415

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from rq_pipeline.bundles.hashing import fields_hash  # noqa: PLC0415
    from rq_pipeline.evaluate.records import (  # noqa: PLC0415
        EpisodeRecord,
        append_records,
    )
    from rq_pipeline.stats.intervals import clopper_pearson  # noqa: PLC0415

    from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415
    from rq_mjlab.walk_train import g3_agent  # noqa: PLC0415

    cfg, identity = microduck_walk_env_cfg()
    cfg.scene.num_envs = args.trials
    cfg.seed = args.seed
    trained = args.checkpoint.parent / "identity.json"
    if trained.is_file() and json.loads(trained.read_text()) != identity:
        raise SystemExit(f"identity mismatch: this env is {identity}")

    devicetag = "cuda" if device.startswith("cuda") else "cpu"
    instrument = instrument_for(device)
    source = f"microduck-walk@{fields_hash(identity)}"
    protocol = {
        "trials": args.trials,
        "seed": args.seed,
        "criterion": f"survived and err_ratio<{ERR_RATIO_BOUND}",
        "err_floor_mps": ERR_FLOOR,
        "dr_basis": identity["dr_basis"],
    }
    print(f"[verdict] {source} on {instrument}, {args.trials} trials")

    agent = g3_agent(iterations=1)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)
    runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    outcomes = rollout_outcomes(
        env, runner.get_inference_policy(device=device), args.trials
    )
    env.close()

    records = [
        EpisodeRecord(
            source=source,
            policy=args.checkpoint.stem,
            trial=trial,
            success=outcome.success,
            steps=outcome.steps,
            instrument=instrument,
            protocol=protocol,
            seed=args.seed,
            events=(
                {"index": 0, "name": "survived", "passed": outcome.survived},
                {"index": 1, "name": "tracked", "passed": outcome.tracked},
            ),
            variations={
                "mean_err_mps": round(outcome.mean_err, 4),
                "mean_cmd_mps": round(outcome.mean_cmd, 4),
                "err_ratio": round(outcome.err_ratio, 4),
            },
        )
        for trial, outcome in enumerate(outcomes)
    ]
    out_dir = args.checkpoint.parent / "verdict"
    out_dir.mkdir(exist_ok=True)
    append_records(out_dir / f"records-{devicetag}.jsonl", records)

    survived = sum(o.survived for o in outcomes)
    tracked = sum(o.tracked for o in outcomes)
    successes = sum(o.success for o in outcomes)
    low, high = clopper_pearson(successes, args.trials)
    row = {
        "source": source,
        "policy": args.checkpoint.stem,
        "instrument": instrument,
        "protocol": protocol,
        "identity": identity,
        "funnel": {"survived": survived, "tracked": tracked},
        "successes": successes,
        "trials": args.trials,
        "ci95": [round(low, 4), round(high, 4)],
        "median_err_ratio": round(
            sorted(o.err_ratio for o in outcomes)[len(outcomes) // 2], 4
        ),
    }
    verdict_path = out_dir / f"walk-verdict-{devicetag}.json"
    verdict_path.write_text(json.dumps(row, indent=1))
    print(
        f"[verdict] survived {survived}/{args.trials}, tracked "
        f"{tracked}/{args.trials} -> success {successes}/{args.trials}, "
        f"CP95 [{low:.3f}, {high:.3f}]"
    )
    print(f"[verdict] rows in {out_dir}")


if __name__ == "__main__":
    main()
