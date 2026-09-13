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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.evaluate.tracking import (
    ERR_FLOOR_MPS,
    ERR_RATIO_BOUND,
    TrackingOutcome,
    criterion_text,
)
from rq_pipeline.viz import viewer_file

from rq_mjlab.envelope import checkpoint_iteration, pin_command_envelope
from rq_mjlab.walks import DEFAULT_ROBOT, ROBOTS, use_project, walk_spec

# The judgment's rule lives in `rq_pipeline.evaluate.tracking`, shared
# with the deployment gate; these names stay for the callers here.
ERR_FLOOR = ERR_FLOOR_MPS
EpisodeOutcome = TrackingOutcome

# The observation group the actor reads and the capture records: what a
# captured (T, obs) row IS, and the key a labeler must hand back to the
# actor — rsl-rl actors index a TensorDict of groups, never a bare tensor.
ACTOR_OBS_GROUP = "actor"


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
    env, policy, trials: int, *, capture: bool = False, max_ticks: int | None = None
) -> list[WorldEpisode]:
    """One completed episode per world, judged from the live managers:
    the commanded twist from the command manager, the base-frame
    velocity from the entity, the cause of death from the termination
    manager. Worlds that finish early keep stepping (the env auto-
    resets) but only each world's FIRST episode is recorded. With
    `capture`, the per-tick observation/action/qpos of that first
    episode ride along (the rollout->dataset writer's raw material).
    With `max_ticks`, a world still open at that tick is closed as
    survived so far (the stills tool wants one frame, not a verdict)."""
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
            actor = obs[ACTOR_OBS_GROUP].detach().cpu().numpy()
            act = actions.detach().cpu().numpy()
            pose = device_qpos.detach().cpu().numpy()
            still_open = open_worlds.cpu().numpy()
            for world in np.flatnonzero(still_open):
                trace[world].append((actor[world], act[world], pose[world]))
        obs, _, dones, _ = env.step(actions)
        command = unwrapped.command_manager.get_command("twist")
        velocity = unwrapped.scene["robot"].data.root_link_lin_vel_b
        err = torch.linalg.norm(velocity[:, :2] - command[:, :2], dim=1)
        cmd = torch.linalg.norm(command[:, :2], dim=1)
        err_sum += err * open_worlds
        cmd_sum += cmd * open_worlds
        steps += open_worlds.long()
        closing = open_worlds & dones.bool()
        if max_ticks is not None:
            closing |= open_worlds & (steps >= max_ticks)
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


class VerdictFeed:
    """The certificate, watched (docs/66 §0): world 0's chase view while
    a student is judged, every trial's verdict as it lands, the fold at
    the end - into the Studio's one ingest address. Best-effort: no
    rerun-sdk means one loud line and an unwatched verdict."""

    ROOT = "verdict"

    def __init__(self, run_name: str, file: Path | None = None) -> None:
        import rerun as rr  # noqa: PLC0415 - viz extra
        from rq_pipeline.viz import STUDIO_ADDRESS, open_stream  # noqa: PLC0415

        self._rr: Any = rr
        open_stream(f"rq-verdict-{run_name}", address=STUDIO_ADDRESS, file=file)
        self._tick = 0

    @classmethod
    def connect(cls, run_name: str, file: Path | None = None) -> VerdictFeed | None:
        try:
            return cls(run_name, file)
        except ImportError:
            print(
                "no rerun-sdk in this venv - the verdict runs UNWATCHED",
                file=sys.stderr,
            )
            return None

    def frame(self, image: Any) -> None:
        self._tick += 1
        self._rr.set_time("tick", sequence=self._tick)
        self._rr.log(f"{self.ROOT}/camera", self._rr.Image(image))

    def outcomes(self, outcomes: list[EpisodeOutcome]) -> None:
        rr = self._rr
        # The trial rows live on THEIR clock only: without the reset they
        # also carried the last tick and the viewer stacked all eight on
        # one x (the operator's screenshot, 2026-09-03).
        rr.reset_time()
        for trial, outcome in enumerate(outcomes):
            rr.set_time("trial", sequence=trial)
            rr.log(f"{self.ROOT}/err_ratio", rr.Scalars(outcome.err_ratio))
            rr.log(f"{self.ROOT}/survived", rr.Scalars(float(outcome.survived)))
            verdict = (
                "PASS" if outcome.success else ("fell" if outcome.fell else "drifted")
            )
            rr.log(
                f"{self.ROOT}/log",
                rr.TextLog(
                    f"trial {trial}: {verdict} (err_ratio {outcome.err_ratio:.2f})"
                ),
            )

    def done(self, line: str) -> None:
        self._rr.log(f"{self.ROOT}/log", self._rr.TextLog(line))


def blanked(images: Any) -> Any:
    """The camera, blanked: zeros of the frames' own shape and dtype.
    The control the campaign 3 plan asks first (docs/07 2026-09-03):
    a student judged WITHOUT its image, with the full state still in
    hand — if the score holds, the image was never used, and the next
    distill trains without the privileged state."""
    import numpy as np  # noqa: PLC0415

    return np.zeros_like(images)


CAMERA_SIGHTED = "chase"
CAMERA_BLANKED = "blanked"


STATE_GIVEN = "actor"
STATE_BLANKED = "blanked"


def verdict_suffix(
    devicetag: str, *, student: bool, blank_camera: bool, blank_state: bool = False
) -> str:
    """The certificate file's suffix — a control run must never land on
    the sighted run's file (`walk-verdict-student-cuda.json` is a
    campaign's headline number): `blank-` for the camera control,
    `blank-state-` for the state control, `blank-both-` for both."""
    if not student:
        return devicetag
    control = {
        (False, False): "",
        (True, False): "blank-",
        (False, True): "blank-state-",
        (True, True): "blank-both-",
    }[(blank_camera, blank_state)]
    return f"student-{control}{devicetag}"


class StudentPolicy:
    """A vision student (a LeRobot checkpoint) as the env's per-tick
    policy: the SAME chase camera the press wrote its frames with
    (walk_press.ChaseCamera, off a one-robot CPU mirror) renders every
    world's view from the live qpos, the actor observation rides along
    as the state, and the bridge answers from the train venv."""

    def __init__(  # noqa: PLR0913 - the student's six facts, named
        self,
        env,
        checkpoint: Path,
        *,
        python: Path,
        horizon: int,
        stride: int,
        frame_size: tuple[int, int],
        device: str,
        blank_camera: bool = False,
        blank_state: bool = False,
    ) -> None:
        from rq_pipeline.envs.policy_bridge import BridgePolicy  # noqa: PLC0415

        from rq_mjlab.actuator import as_torch  # noqa: PLC0415
        from rq_mjlab.walk_press import CAMERA_KEY, ChaseCamera  # noqa: PLC0415

        self._env = env
        self._qpos = as_torch(env.unwrapped.sim.data.qpos)
        self._camera = ChaseCamera(*frame_size)
        self._bridge = BridgePolicy(
            python, checkpoint, camera=CAMERA_KEY, horizon=horizon, device=device
        )
        self._first = True
        self.feed: VerdictFeed | None = None
        self._ticks = 0
        # The student's clock is the DATASET's: frames every `stride`
        # control ticks (walk_press --frame-every), so a chunk step is
        # `stride` ticks long. Asked every tick, the first student was
        # driven 5x too fast and fell in 18 ticks (2026-09-02).
        self._stride = stride
        self._held: Any = None
        self._blank = blank_camera
        # The complementary control (2026-09-04): the camera blanked
        # collapsed the student to 0/40, which proves the image is USED,
        # not that the state is unused — a blank frame is out of
        # distribution. Zeroing the state instead says whether the
        # student is purely visual.
        self._blank_state = blank_state

    def __call__(self, obs) -> Any:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if self._held is not None and self._ticks % self._stride:
            self._ticks += 1
            return self._held
        state = obs["actor"].detach().cpu().numpy()
        if self._blank_state:
            state = blanked(state)
        qpos = self._qpos.detach().cpu().numpy()
        images = np.stack([self._camera.frame_at(row) for row in qpos])
        if self._blank:
            images = blanked(images)
        if self.feed is not None:
            self.feed.frame(images[0])
        resets = np.full(len(state), self._first, dtype=bool)
        self._first = False
        actions = self._bridge.act(state, images, resets)
        self._held = torch.as_tensor(actions, device=self._env.unwrapped.device)
        self._ticks += 1
        return self._held

    def close(self) -> None:
        self._bridge.close()
        self._camera.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
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
        help="which walk (rq_mjlab.walks) the checkpoint belongs to",
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--student",
        type=Path,
        default=None,
        help="a LeRobot checkpoint (pretrained_model dir): judge the vision "
        "student distilled from this teacher's data instead of the teacher",
    )
    parser.add_argument(
        "--horizon", type=int, default=2, help="executed chunk steps before a replan"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="control ticks per chunk step: the press's --frame-every (1 for "
        "this gait - held 5 ticks even the teacher falls, measured 2026-09-02)",
    )
    parser.add_argument(
        "--student-python",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "pipeline"
        / ".venv-train"
        / "bin"
        / "python",
        help="the interpreter with LeRobot (the bridge's server side)",
    )
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=240)
    parser.add_argument(
        "--judge-at-scale",
        type=float,
        default=None,
        help="the envelope probe: judge with every law parameter pinned at "
        "fit x SCALE (no draw); the certificate file carries the scale",
    )
    parser.add_argument(
        "--judge-span",
        type=float,
        default=None,
        help="judge under law DR drawn from a declared ±SPAN around the fit "
        "(the mismatch matrix's span columns); without it the non-at-fit "
        "certificate judges under LAW_DR_SPAN for every policy, whatever "
        "span it trained under — NOT each policy's own span (2026-09-04)",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="the actuator bundle the checkpoint trained on (microduck only); the "
        "identity check refuses a mismatch",
    )
    parser.add_argument(
        "--judge-param",
        default="all",
        help="with --judge-at-scale: pin only this mismatch axis (the walk's "
        "PIN_AXES, e.g. kt / R / friction on the microduck, kp / kd / armature "
        "on the Go1); 'all' moves every parameter together",
    )
    parser.add_argument(
        "--judge-at-fit",
        action="store_true",
        help="judge with NO law DR — the bundle's point fit exactly, the pinned "
        "truth the walk C1 study's arms share (pushes stay on)",
    )
    parser.add_argument(
        "--blank-state",
        action="store_true",
        help="judge the student with its state vector zeroed (image still "
        "given): the complement of --blank-camera",
    )
    parser.add_argument(
        "--blank-camera",
        action="store_true",
        help="judge the student with its image zeroed (state still given): "
        "the control that says whether the image is used at all",
    )
    parser.add_argument(
        "--no-studio", action="store_true", help="do not stream to the Studio"
    )
    return parser.parse_args()


def write_certificate(  # noqa: PLR0913 - every fact of one certificate, named
    out_dir: Path,
    suffix: str,
    outcomes: list[EpisodeOutcome],
    *,
    source: str,
    policy_name: str,
    instrument: str,
    protocol: dict[str, Any],
    identity: dict[str, Any],
    trials: int,
) -> None:
    """The fold and the exact interval, printed and written beside the rows."""
    from rq_pipeline.stats.intervals import clopper_pearson  # noqa: PLC0415

    survived = sum(o.survived for o in outcomes)
    tracked = sum(o.tracked for o in outcomes)
    successes = sum(o.success for o in outcomes)
    low, high = clopper_pearson(successes, trials)
    row = {
        "source": source,
        "policy": policy_name,
        "instrument": instrument,
        "protocol": protocol,
        "identity": identity,
        "funnel": {"survived": survived, "tracked": tracked},
        "successes": successes,
        "trials": trials,
        "ci95": [round(low, 4), round(high, 4)],
        "median_err_ratio": round(
            sorted(o.err_ratio for o in outcomes)[len(outcomes) // 2], 4
        ),
    }
    verdict_path = out_dir / f"walk-verdict-{suffix}.json"
    if verdict_path.exists():
        # A certificate is never silently replaced: an 8-trial smoke run
        # once overwrote the flagship's 40-trial file (found 2026-09-04
        # by an audit reading the rows). The rows are appended and stay
        # the primary artifact; the previous file is kept beside the new.
        stamp_prev = json.loads(verdict_path.read_text())
        # Named by what it judged too: two checkpoints of one run judged
        # under the same suffix would otherwise share a backup name, and
        # the second rotation would overwrite the first (2026-09-11).
        previous = out_dir / (
            f"walk-verdict-{suffix}.{stamp_prev.get('policy', 'policy')}"
            f".seed{stamp_prev['protocol'].get('seed')}"
            f".n{stamp_prev['trials']}.json"
        )
        if not previous.exists():
            verdict_path.rename(previous)
    verdict_path.write_text(json.dumps(row, indent=1))
    print(
        f"[verdict] survived {survived}/{trials}, tracked "
        f"{tracked}/{trials} -> success {successes}/{trials}, "
        f"CP95 [{low:.3f}, {high:.3f}]"
    )
    print(f"[verdict] rows in {out_dir}")


def main() -> None:  # noqa: PLR0912, PLR0915 - the certificate's whole procedure, in order
    args = parse_args()

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

    if args.judge_at_fit and args.judge_span is not None:
        raise SystemExit("--judge-at-fit and --judge-span exclude each other")
    use_project(args.project)
    spec = walk_spec(args.robot)
    default_span = spec.default_span
    judge_span = default_span if args.judge_span is None else args.judge_span
    if args.judge_param != "all" and args.judge_at_scale is None:
        raise SystemExit("--judge-param needs --judge-at-scale")
    cfg, identity = spec.env_cfg(
        dr_span=None if args.judge_at_fit else judge_span,
        pin_scale=args.judge_at_scale,
        pin_axis=args.judge_param,
        bundle=args.bundle,
    )
    cfg.scene.num_envs = args.trials
    cfg.seed = args.seed
    trained = args.checkpoint.parent / "identity.json"
    if trained.is_file():
        trained_identity = json.loads(trained.read_text())
        # Robot and actuator must match; the DR basis may differ on
        # purpose (a policy trained under one span is judged at the fit),
        # and the certificate records both.
        for key in ("robot", "actuator"):
            if trained_identity.get(key) != identity.get(key):
                raise SystemExit(f"identity mismatch on {key}: this env is {identity}")
        identity = {
            **identity,
            "task": trained_identity.get("task"),
            "trained_dr_basis": trained_identity.get("dr_basis"),
            # The training run's seed (walk_train --seed; mjlab's default
            # 42 when the run predates the knob) — a replicate's name.
            "seed": trained_identity.get("seed", 42),
        }

    devicetag = "cuda" if device.startswith("cuda") else "cpu"
    instrument = instrument_for(device)
    # The environment the certificate cites: the declared task the run was
    # trained against (its content stamp, the project's environment card)
    # when the run recorded one; else the walk spec's identity hash.
    source = identity.get("task") or f"{spec.source_prefix}@{fields_hash(identity)}"
    agent = spec.agent(1)
    # The commands the checkpoint trained under, not the curriculum's
    # first stage a fresh env would restart at (rq_mjlab.envelope).
    envelope = pin_command_envelope(
        cfg, checkpoint_iteration(args.checkpoint.stem), agent.num_steps_per_env
    )
    protocol = {
        "trials": args.trials,
        "seed": args.seed,
        "criterion": criterion_text(),
        "err_ratio_bound": ERR_RATIO_BOUND,
        "err_floor_mps": ERR_FLOOR,
        "dr_basis": identity["dr_basis"],
        "commands": envelope["commands"],
        "command_basis": envelope["basis"],
    }
    print(f"[verdict] {source} on {instrument}, {args.trials} trials")
    print(f"[verdict] commands: {envelope['commands']} ({envelope['basis']})")

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
    policy: Any = runner.get_inference_policy(device=device)
    policy_name = args.checkpoint.stem
    if args.student is not None:
        # The student's identity: its checkpoint directory's content hash
        # (the same stamp a bundle gets), named by its run step.
        from rq_pipeline.bundles.hashing import stamp as stamp_of  # noqa: PLC0415

        policy = StudentPolicy(
            env,
            args.student,
            python=args.student_python,
            horizon=args.horizon,
            stride=args.stride,
            frame_size=(args.frame_width, args.frame_height),
            device=devicetag,
            blank_camera=args.blank_camera,
            blank_state=args.blank_state,
        )
        policy_name = stamp_of(f"student-{args.student.parent.name}", args.student)
        protocol["student"] = {
            "teacher": args.checkpoint.stem,
            "horizon": args.horizon,
            "stride": args.stride,
            # Every row says what the student SAW: the camera, or nothing.
            "camera": CAMERA_BLANKED if args.blank_camera else CAMERA_SIGHTED,
            "state": STATE_BLANKED if args.blank_state else STATE_GIVEN,
        }
    feed = (
        None
        if args.no_studio
        else VerdictFeed.connect(
            policy_name.split("@")[0],
            # The saved stream lands in the run the checkpoint belongs to.
            viewer_file(args.checkpoint.parent, f"verdict-{args.checkpoint.stem}"),
        )
    )
    if isinstance(policy, StudentPolicy):
        policy.feed = feed
    outcomes = rollout_outcomes(env, policy, args.trials)
    if isinstance(policy, StudentPolicy):
        policy.close()
    env.close()
    if feed is not None:
        feed.outcomes(outcomes)

    records = [
        EpisodeRecord(
            source=source,
            policy=policy_name,
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
    suffix = verdict_suffix(
        devicetag,
        student=args.student is not None,
        blank_camera=args.blank_camera,
        blank_state=args.blank_state,
    )
    if args.judge_at_fit:
        suffix = f"at-fit-{suffix}"
        protocol["judged_at"] = "the bundle's point fit (no law DR)"
    elif args.judge_span is not None:
        suffix = f"under-pm{args.judge_span:g}-{suffix}"
        protocol["judged_at"] = f"law DR: {identity['dr_basis']}"
    else:
        protocol["judged_at"] = f"law DR: {identity['dr_basis']}"
    if args.judge_at_scale is not None:
        axis = "" if args.judge_param == "all" else f"{args.judge_param}-"
        suffix = f"at-x{args.judge_at_scale:g}-{axis}{suffix}"
        moved = "every law parameter" if args.judge_param == "all" else args.judge_param
        protocol["judged_at"] = f"{moved} at fit x {args.judge_at_scale:g}"
    append_records(out_dir / f"records-{suffix}.jsonl", records)

    write_certificate(
        out_dir,
        suffix,
        outcomes,
        source=source,
        policy_name=policy_name,
        instrument=instrument,
        protocol=protocol,
        identity=identity,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
