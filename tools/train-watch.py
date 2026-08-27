"""Watch a policy learn: train in the background, play every checkpoint.

    # WSL / Linux (the train venv, with the Rerun viewer binary on PATH):
    cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python \\
        ../tools/train-watch.py \\
        --steps 20000 --save-freq 1000 --batch-size 8 --name t2-act
    # macOS: the passive viewer needs mjpython, and none of the GL variables:
    cd pipeline && mjpython ../tools/train-watch.py --steps 20000 --name t2-act

`lerobot-train` runs as a subprocess (ACT on the public ALOHA
transfer-cube demos by default). This process tails its log and, each
time a checkpoint is written, loads it through LeRobot's own processors
(`rq_pipeline.envs.lerobot_policy`) and plays ONE episode through the
gymnasium env on the aloha2-nominal bundle — the identified dynamics, not
the trainer's simulator — in both viewers:

    MuJoCo window            the rig, live, every checkpoint's episode
    Rerun  train/*           loss, l1, kld, grad norm, lr  (train_step)
           eval/<step>/...   the top camera the policy sees, 10 Hz;
                             object height; the referee's verdict
           world/rig         the mesh-true mirror during each episode
           stage             which checkpoint is playing, and how it did

Episodes here are a WINDOW onto training, not the evaluation: one
trial per checkpoint, the same paired start every time (trial 0), so
successive checkpoints are comparable by eye. The certificate is the
harness's job, with all trials, after training ends — and
`lerobot-train --env.type=robotiq` evaluates in-loop through the same
env without this tool at all.

    # a run that is ALREADY going — this box's chain, or a rented card's
    # run mirrored here by `cloud-gpu follow` — as a dashboard, no torch needed:
    cd pipeline && uv run --extra sim --extra viz python ../tools/train-watch.py \\
        --follow runs/t5-cloud-act

`--follow` streams a run from its files: the chain's run manifest (every
parameter, the dataset's provenance) as a text panel at step 0, the
trainer's metrics lines as `train/*`, its in-loop summaries and our
per-episode records as `eval/*` (success rate and the milestone funnel
per checkpoint), the eval videos, `nvidia-smi` samples as `machine/*`,
and the trainer's resolved `train_config.json` when the first
checkpoint lands. With `--play-checkpoints` (train venv) it also plays
each checkpoint in both viewers as it appears, as the training mode
does.

Training mode requires the train venv (LeRobot + torch) and the sim
extras; `--follow` needs only the sim and viz extras.
"""

import argparse
import json
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rerun as rr

from _lab import (
    PREVIEW_EVERY_TICKS,
    bootstrap,
    hold_until_closed,
    lerobot_train_command,
    rr_session,
    verdict_word,
)

bootstrap()
from rq_pipeline.envs.contract import InfoKeys, ObservationKeys  # noqa: E402
from rq_pipeline.envs.lerobot_train_log import (  # noqa: E402
    GPU_LOG_FILE,
    TRAIN_LOG_FILE,
    FileFollower,
    RunManifest,
    parse_eval_line,
    parse_train_line,
    watch_dir_for,
)
from rq_pipeline.evaluate.records import (  # noqa: E402
    fold,
    funnel,
    passes,
    read_records,
)
from rq_pipeline.tasks.aloha2 import (  # noqa: E402
    ACT_SIM_LOOK,
    CUBE_Z_STATE_INDEX,
    LOOKS,
    PUBLIC_TRANSFER_CUBE_DEMOS,
    RIG,
    TRANSFER_CUBE,
    ActionSpace,
    act_sim_state,
    ctrl_from_act_sim_action,
)
from rq_pipeline.tasks.registry import tasks  # noqa: E402
from rq_pipeline.tasks.scene import GeomGroup  # noqa: E402

# The ALOHA 2 tasks; every builder takes `look`, and the first free body's
# z sits at CUBE_Z_STATE_INDEX in either scene (the cube, or the right
# arm's part).
ALOHA_TASKS = {
    entry.name: entry.build for entry in tasks().values() if entry.rig == RIG
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="t2-act")
    parser.add_argument("--policy", default="act")
    parser.add_argument("--dataset", default=PUBLIC_TRANSFER_CUBE_DEMOS)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--runs", default="runs")
    parser.add_argument(
        "--play",
        default=None,
        help="no training: play this checkpoint directory (pretrained_model) "
        "for --trials paired starts in both viewers, then hold",
    )
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument(
        "--follow",
        default=None,
        metavar="RUN_DIR",
        help="no training: stream a run that is going (or went) from its files — "
        "runs/<name>-act of the chain, here or mirrored by `cloud-gpu follow`",
    )
    parser.add_argument(
        "--play-checkpoints",
        action="store_true",
        help="with --follow: also play each checkpoint in both viewers as it "
        "appears (needs the weights and the train venv)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="with --follow: stream what is there and exit instead of tailing",
    )
    parser.add_argument(
        "--rrd",
        type=Path,
        default=None,
        help="with --follow: also save the stream to this .rrd file — the run "
        "as one portable record anyone with the Rerun viewer can open",
    )
    parser.add_argument(
        "--replay-demos",
        type=int,
        default=1,
        help="human demonstrations from the dataset to replay through the bundle "
        "before training's first checkpoint (0 to skip)",
    )
    parser.add_argument(
        "--look",
        default=ACT_SIM_LOOK,
        choices=LOOKS,
        help="scene appearance for the played episodes: the ACT simulator's "
        "(what the public demos look like) or the bundle's own",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for the policy and trainer; default: cuda if present,"
        " else mps, else cpu",
    )
    parser.add_argument(
        "--task",
        default=TRANSFER_CUBE,
        choices=sorted(ALOHA_TASKS),
        help="which ALOHA 2 task the checkpoint plays (kitting for T5's)",
    )
    parser.add_argument(
        "--executed-horizon",
        type=int,
        default=None,
        help="execute this many control ticks of each predicted chunk before "
        "asking the policy again (the protocol's horizon, hashed); default: "
        "the checkpoint's own n_action_steps",
    )
    parser.add_argument(
        "--openpi",
        default=None,
        metavar="HOST[:PORT]",
        help="play a policy served by openpi (pi0 / pi0.5) instead of a "
        "checkpoint; needs --executed-horizon and the 'remote' extra",
    )
    parser.add_argument(
        "--action-space",
        default=ActionSpace.ACT_SIM,
        choices=(ActionSpace.ACT_SIM, ActionSpace.BUNDLE),
        help="what the checkpoint speaks: gym-aloha's normalised grippers "
        "(public-demo checkpoints) or the bundle's own ctrl (our demos, T5)",
    )
    return parser.parse_args()


def train_command(args, output_dir, device):
    return lerobot_train_command(
        policy=args.policy,
        device=device,
        dataset=args.dataset,
        output_dir=output_dir,
        job_name=args.name,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
    )


def log_train_line(line: str) -> int | None:
    """One trainer line to Rerun: a metrics line as `train/*` at its step
    (returned), an in-loop summary as `eval/inloop/pc_success`; anything
    else is not ours to parse."""
    parsed = parse_train_line(line)
    if parsed is not None:
        rr.set_time("train_step", sequence=parsed.step)
        for name, value in parsed.metrics.items():
            rr.log(f"train/{METRIC_NAMES.get(name, name)}", rr.Scalars(value))
        rr.log("train/samples", rr.Scalars(float(parsed.samples)))
        rr.log("train/epochs", rr.Scalars(parsed.epochs))
        return parsed.step
    summary = parse_eval_line(line)
    if summary is not None:
        rr.log("eval/inloop/pc_success", rr.Scalars(summary.pc_success))
        rr.log("eval/inloop/eval_s", rr.Scalars(summary.eval_s))
        rr.log(
            "stage",
            rr.TextLog(
                f"in-loop eval: {summary.pc_success:.0f}% of "
                f"{summary.n_episodes} in {summary.eval_s:.0f} s"
            ),
        )
    return None


# LeRobot's abbreviations, spelled out for the panel.
METRIC_NAMES = {
    "grdn": "grad_norm",
    "smp/s": "samples_per_s",
    "updt_s": "update_s",
    "data_s": "dataloading_s",
}


def tail_training(process, seen_steps):
    """Stream lerobot-train's stdout: metrics to Rerun, all else through."""
    for raw in process.stdout:
        line = raw.rstrip()
        step = log_train_line(line)
        if step is not None:
            seen_steps.append(step)
        elif "Training:" not in line and line:
            print(f"[train] {line}", flush=True)


def scheduled(policy, executed_horizon: int, nu: int):
    """A chunk policy executed on OUR horizon: `(act, reset)`."""
    from rq_pipeline.evaluate.scheduler import ActionScheduler  # noqa: PLC0415

    scheduler = ActionScheduler(policy, executed_horizon=executed_horizon, nu=nu)
    return scheduler.act, scheduler.reset


def openpi_controller(address: str, env, executed_horizon: int | None):
    """A policy served by openpi, on the protocol's horizon; the first
    camera feeds their `cam_high`, the env's instruction is the prompt."""
    from rq_pipeline.envs.openpi_policy import (  # noqa: PLC0415
        DEFAULT_PORT,
        OpenpiKeys,
        OpenpiRequest,
        openpi_chunk_policy,
    )

    if executed_horizon is None:
        raise SystemExit(
            "--openpi needs --executed-horizon: the server's chunk is executed "
            "on OUR horizon"
        )
    host, _, port = address.partition(":")
    policy = openpi_chunk_policy(
        host,
        int(port) if port else DEFAULT_PORT,
        request=OpenpiRequest(
            cameras={env.cameras[0].key: OpenpiKeys.ALOHA_CAMERAS[0]},
            prompt=env.task_description,
        ),
    )
    return scheduled(policy, executed_horizon, env.action_space.shape[0])


def checkpoint_controller(  # noqa: PLR0913 - one controller, every knob named
    path: Path,
    action_space: str,
    instruction: str,
    device: str,
    *,
    executed_horizon: int | None = None,
    nu: int | None = None,
):
    """The checkpoint as `act(observation) -> ctrl` plus its reset; the
    gym-aloha convention (normalised grippers) wrapped around the model
    call when the checkpoint speaks it. With `executed_horizon`, the
    checkpoint's chunk is executed on that horizon instead of its own."""
    from rq_pipeline.envs.lerobot_policy import load_policy  # noqa: PLC0415

    loaded = load_policy(path, instruction=instruction, device=device)
    if executed_horizon is not None:
        if nu is None:
            raise ValueError("executed_horizon needs nu")
        act, reset = scheduled(loaded.as_chunk_policy(), executed_horizon, nu)
        loaded = replace(loaded, act=act, reset=reset)
    if action_space != ActionSpace.ACT_SIM:
        return loaded.act, loaded.reset

    def act(observation):
        raw = dict(observation)
        raw[ObservationKeys.AGENT_POS] = act_sim_state(
            observation[ObservationKeys.AGENT_POS]
        )
        return ctrl_from_act_sim_action(loaded.act(raw))

    return act, loaded.reset


class Watcher:
    """The env, the viewer, the mirror: plays one episode per checkpoint."""

    def __init__(
        self, look: str, task: str, device: str, executed_horizon: int | None = None
    ):
        import mujoco.viewer  # noqa: PLC0415 - the window, only when playing
        from rq_pipeline.envs.robotiq import RobotiqEnv, bundle_source  # noqa: PLC0415

        from _rig3d import RigMirror  # noqa: PLC0415

        self.executed_horizon = executed_horizon
        built = ALOHA_TASKS[task](look=look)
        self.env = RobotiqEnv(built, source=bundle_source(built.bundle_dir))
        self.device = device
        self.mirror = RigMirror(
            self.env.model, model_colors=True, skip_groups=(GeomGroup.COLLISION,)
        )
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.viewer.cam.distance = 1.6
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -35
        self.viewer.cam.lookat[:] = [0.05, 0.0, 0.1]
        self.played = set()

    def _episode(self, controller, prefix: str, note: str, trial: int = 0):
        """One episode from paired start `trial`, driven by
        `controller(tick, observation) -> ctrl` over the env's raw
        observation; logs frames, mirror, object height; returns the
        referee's verdict."""
        env = self.env
        dt = 1.0 / env.metadata["render_fps"]
        observation, info = env.reset(seed=trial)
        rr.log("stage", rr.TextLog(note))
        top = env.cameras[0].key
        for tick in range(env.max_episode_steps):
            rr.set_time("sim_time", duration=tick * dt)
            if tick % PREVIEW_EVERY_TICKS == 0:
                rr.log(
                    f"{prefix}/top", rr.Image(observation[ObservationKeys.PIXELS][top])
                )
            observation, _reward, _terminated, _truncated, info = env.step(
                np.asarray(controller(tick, observation), dtype=float)
            )
            self.mirror.log(env.data, path="world/rig")
            rr.log(
                f"{prefix}/object_z",
                rr.Scalars(float(env.states[env.physics_step - 1, CUBE_Z_STATE_INDEX])),
            )
            if not self.viewer.is_running():
                raise SystemExit("viewer closed")
            self.viewer.sync()
        return bool(info[InfoKeys.IS_SUCCESS])

    def replay_demo(self, dataset_id: str, episode: int):
        """Play a human demonstration's actions open-loop through the bundle.

        The target behaviour, in our simulator, before any checkpoint
        exists — and a live check of the frame and gripper mapping: the
        demo's arm motion should look like the dataset's video. The
        cube sits at our paired start, not wherever theirs was, so the
        grasp itself need not land; the choreography is the point.
        """
        from _lab import load_demo_actions  # noqa: PLC0415 - LeRobot's dataset

        actions = load_demo_actions(dataset_id, episode)
        rr.set_time("train_step", sequence=0)

        def controller(tick, observation):
            del observation
            return ctrl_from_act_sim_action(actions[min(tick, len(actions) - 1)])

        success = self._episode(
            controller,
            f"demo/{episode:03d}",
            f"human demo {episode} from {dataset_id}: replaying {len(actions)} actions",
        )
        verdict = verdict_word(success)
        rr.log(
            "stage",
            rr.TextLog(
                f"human demo {episode} replayed: {verdict} "
                "(open-loop, cube at our paired start)"
            ),
        )
        print(f"[watch] demo {episode}: {verdict}", flush=True)

    def play(self, checkpoint_dir: Path, step: int, action_space: str):
        rr.set_time("train_step", sequence=step)
        rr.log("stage", rr.TextLog(f"checkpoint {step}: loading"))
        act, reset = checkpoint_controller(
            checkpoint_dir,
            action_space,
            self.env.task_description,
            self.device,
            executed_horizon=self.executed_horizon,
            nu=self.env.action_space.shape[0],
        )
        reset()
        success = self._episode(
            lambda _tick, observation: act(observation),
            f"eval/{step:06d}",
            f"checkpoint {step}: playing trial 0",
        )
        rr.set_time("train_step", sequence=step)
        rr.log("eval/success", rr.Scalars(1.0 if success else 0.0))
        verdict = verdict_word(success)
        rr.log("stage", rr.TextLog(f"checkpoint {step}: {verdict}"))
        print(f"[watch] checkpoint {step}: {verdict}", flush=True)
        self.played.add(step)

    def poll(self, checkpoints_dir: Path, action_space: str):
        if not checkpoints_dir.exists():
            return
        for entry in sorted(checkpoints_dir.iterdir()):
            if not entry.name.isdigit():
                continue
            step = int(entry.name)
            pretrained = entry / "pretrained_model"
            if step in self.played or not (pretrained / "model.safetensors").exists():
                continue
            self.play(pretrained, step, action_space)


@dataclass
class Follower:
    """A run's files → Rerun: the manifest as a panel, the log's metrics,
    the records' funnel per checkpoint, the eval videos, the GPU samples,
    the trainer's resolved config. Idempotent per file: each call to
    `poll()` logs only what is new."""

    run_dir: Path
    play: bool = False
    watcher: "Watcher | None" = None
    action_space: str = ActionSpace.BUNDLE
    manifest_shown: bool = False
    configs_shown: set = None
    videos_shown: set = None
    passes_shown: int = 0
    final_shown: bool = False
    last_step: int = 0

    def __post_init__(self) -> None:
        self.watch_dir = watch_dir_for(self.run_dir)
        self.log = FileFollower(self.watch_dir / TRAIN_LOG_FILE)
        self.gpu = FileFollower(self.watch_dir / GPU_LOG_FILE)
        self.configs_shown = set()
        self.videos_shown = set()

    # -- pieces ---------------------------------------------------------

    def manifest(self) -> None:
        if self.manifest_shown or not (self.watch_dir / "run.json").exists():
            return
        manifest = RunManifest.read(self.watch_dir)
        rr.set_time("train_step", sequence=0)
        rr.log(
            "config/run",
            rr.TextDocument(manifest.as_markdown(), media_type=rr.MediaType.MARKDOWN),
            static=True,
        )
        rr.log(
            "stage",
            rr.TextLog(
                f"{manifest.name}: {manifest.policy}, {manifest.steps} steps, "
                f"batch {manifest.batch_size}, started {manifest.started}"
            ),
        )
        self.manifest_shown = True

    def trainer_lines(self) -> None:
        for line in self.log.new_lines():
            step = log_train_line(line)
            if step is not None:
                self.last_step = step

    def gpu_samples(self) -> None:
        for line in self.gpu.new_lines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < GPU_SAMPLE_FIELDS:
                continue
            try:
                stamp = parts[0].replace("/", "-")
                util = float(parts[1].split()[0])
                mem = float(parts[2].split()[0])
            except (ValueError, IndexError):
                continue
            rr.set_time("wall", timestamp=np.datetime64(stamp.replace(" ", "T")[:26]))
            rr.log("machine/gpu_util_pct", rr.Scalars(util))
            rr.log("machine/gpu_mem_mib", rr.Scalars(mem))

    def checkpoints(self) -> list[int]:
        checkpoints = self.run_dir / "checkpoints"
        if not checkpoints.exists():
            return []
        steps = sorted(int(e.name) for e in checkpoints.iterdir() if e.name.isdigit())
        for step in steps:
            config = (
                checkpoints / f"{step:06d}" / "pretrained_model" / "train_config.json"
            )
            if step in self.configs_shown or not config.exists():
                continue
            rr.set_time("train_step", sequence=step)
            rr.log(
                "config/trainer",
                rr.TextDocument(
                    json.dumps(
                        json.loads(config.read_text(encoding="utf-8")), indent=1
                    ),
                    media_type=rr.MediaType.TEXT,
                ),
            )
            rr.log("stage", rr.TextLog(f"checkpoint {step}: written"))
            self.configs_shown.add(step)
        return steps

    def inloop_records(self, steps: list[int]) -> None:
        path = self.run_dir / "inloop-episodes.jsonl"
        if not path.exists():
            return
        chunks = passes(read_records(path))
        for index, chunk in enumerate(
            chunks[self.passes_shown :], start=self.passes_shown
        ):
            if index >= len(steps):
                return  # the checkpoint this pass belongs to is not here yet
            step = steps[index]
            rr.set_time("train_step", sequence=step)
            for score in fold(chunk):
                rr.log(
                    "eval/inloop/success_rate",
                    rr.Scalars(score.successes / score.trials),
                )
                rr.log(
                    "stage",
                    rr.TextLog(
                        f"checkpoint {step}: in-loop {score.successes}/{score.trials}"
                    ),
                )
            for counts in funnel(chunk).values():
                for milestone, count in zip(MILESTONES, counts, strict=False):
                    rr.log(
                        f"eval/inloop/funnel/{milestone}",
                        rr.Scalars(count / len(chunk)),
                    )
            self.passes_shown = index + 1

    def videos(self, steps: list[int]) -> None:
        for step in steps:
            # LeRobot nests the suite (`robotiq_0/`) under the step directory.
            for video in sorted(
                (self.run_dir / "eval" / f"videos_step_{step:06d}").rglob("*.mp4")
            ):
                if video in self.videos_shown:
                    continue
                rr.set_time("train_step", sequence=step)
                rr.log(f"eval/video/{video.stem}", rr.AssetVideo(path=video))
                self.videos_shown.add(video)

    def final_eval(self) -> None:
        """The paired evaluation beside the run (`<name>-eval`), once it is there."""
        eval_dir = self.run_dir.parent / self.run_dir.name.replace("-act", "-eval")
        path = eval_dir / "episodes.jsonl"
        if self.final_shown or not path.exists():
            return
        records = read_records(path)
        chunks = passes(records)
        rr.set_time(
            "train_step",
            sequence=max(self.last_step, max(self.configs_shown, default=0)),
        )
        for score in fold(chunks[0]):
            rr.log(
                "eval/final/success_rate", rr.Scalars(score.successes / score.trials)
            )
            rr.log(
                "stage",
                rr.TextLog(
                    f"{score.name}: {score.successes}/{score.trials} distinct starts"
                    + (
                        f" (+{sum(len(c) for c in chunks[1:])} padding episodes)"
                        if len(chunks) > 1
                        else ""
                    )
                ),
            )
        for counts in funnel(chunks[0]).values():
            for milestone, count in zip(MILESTONES, counts, strict=False):
                rr.log(
                    f"eval/final/funnel/{milestone}", rr.Scalars(count / len(chunks[0]))
                )
        for video in sorted(eval_dir.glob("videos/**/*.mp4")):
            rr.log(f"eval/final/video/{video.stem}", rr.AssetVideo(path=video))
        self.final_shown = True

    # -- the loop ----------------------------------------------------------

    def poll(self) -> None:
        self.manifest()
        self.trainer_lines()
        self.gpu_samples()
        steps = self.checkpoints()
        self.inloop_records(steps)
        self.videos(steps)
        self.final_eval()
        if self.play and self.watcher is not None:
            self.watcher.poll(self.run_dir / "checkpoints", self.action_space)


MILESTONES = ("part_moved", "part_lifted", "one_in_slot", "both_in_slot")
GPU_SAMPLE_FIELDS = 3  # nvidia-smi: timestamp, utilization.gpu, memory.used


def follow(args) -> None:
    run_dir = Path(args.follow)
    rr_session(
        f"robotiq-follow-{run_dir.name}", mode="spawn", world_up=args.play_checkpoints
    )
    if args.rrd is not None:
        args.rrd.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(args.rrd))
    watcher = None
    if args.play_checkpoints:
        from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

        watcher = Watcher(
            args.look, args.task, best_device(args.device), args.executed_horizon
        )
    follower = Follower(
        run_dir,
        play=args.play_checkpoints,
        watcher=watcher,
        action_space=args.action_space,
    )
    print(f"[follow] {run_dir}: streaming to Rerun; Ctrl-C to stop", flush=True)
    try:
        while True:
            follower.poll()
            if args.once:
                break
            if watcher is not None and not watcher.viewer.is_running():
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    print(f"[follow] last step seen: {follower.last_step}", flush=True)


def _exit_on_sigterm(signum, frame):
    # A plain SIGTERM would skip `finally` and orphan the trainer.
    raise SystemExit(128 + signum)


def play_only(args) -> None:
    """Watch one checkpoint: every paired start, both viewers, no training."""
    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

    device = best_device(args.device)
    if args.openpi:
        name = f"openpi-{args.openpi.replace(':', '-')}"
        rr_session(f"robotiq-play-{name}", mode="spawn")
        watcher = Watcher(args.look, args.task, device, args.executed_horizon)
        act, reset = openpi_controller(args.openpi, watcher.env, args.executed_horizon)
    else:
        checkpoint = Path(args.play)
        name = checkpoint.parent.name
        rr_session(f"robotiq-play-{name}", mode="spawn")
        watcher = Watcher(args.look, args.task, device, args.executed_horizon)
        act, reset = checkpoint_controller(
            checkpoint,
            args.action_space,
            watcher.env.task_description,
            device,
            executed_horizon=args.executed_horizon,
            nu=watcher.env.action_space.shape[0],
        )
    successes = 0
    for trial in range(args.trials):
        reset()
        rr.set_time("train_step", sequence=trial)
        success = watcher._episode(
            lambda _tick, observation: act(observation),
            f"play/trial{trial}",
            f"{name}: trial {trial}",
            trial=trial,
        )
        successes += int(success)
        verdict = verdict_word(success)
        rr.log("stage", rr.TextLog(f"{name}: trial {trial} {verdict}"))
        print(f"[play] trial {trial}: {verdict}", flush=True)
    rr.log("stage", rr.TextLog(f"{name}: {successes}/{args.trials}"))
    print(
        f"[play] {successes}/{args.trials} - close the MuJoCo window to exit",
        flush=True,
    )
    hold_until_closed(watcher.viewer)


def main() -> None:
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    args = parse_args()
    if args.follow:
        follow(args)
        return
    if args.play or args.openpi:
        play_only(args)
        return
    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

    output_dir = Path(args.runs) / args.name
    device = best_device(args.device)
    rr_session(f"robotiq-train-watch-{args.name}", mode="spawn")

    process = subprocess.Popen(
        train_command(args, output_dir, device),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    seen_steps: list[int] = []
    threading.Thread(
        target=tail_training, args=(process, seen_steps), daemon=True
    ).start()

    watcher = Watcher(args.look, args.task, device, args.executed_horizon)
    checkpoints = output_dir / "checkpoints"
    try:
        for episode in range(args.replay_demos):
            watcher.replay_demo(args.dataset, episode)
        while process.poll() is None:
            watcher.poll(checkpoints, args.action_space)
            if not watcher.viewer.is_running():
                process.terminate()
                break
            time.sleep(2.0)
        watcher.poll(checkpoints, args.action_space)  # the final checkpoint
        rr.log("stage", rr.TextLog(f"training exited with {process.returncode}"))
        print("[watch] training finished - close the MuJoCo window to exit", flush=True)
        hold_until_closed(watcher.viewer)
    finally:
        if process.poll() is None:
            process.terminate()


if __name__ == "__main__":
    main()
