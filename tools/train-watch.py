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
           play/<step>/...   the top camera the policy sees, 10 Hz
                             (play/trial<k>/... under --play);
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
parameter, the dataset's provenance) as a text panel from step 0, every
stage's line as `stage`, the trainer's metrics as `train/*`, our
per-episode records as `eval/*` (success rate and the milestone funnel
per checkpoint pass), the eval videos, `nvidia-smi` samples as
`machine/*`, and the trainer's resolved `train_config.json` at every
checkpoint. With `--play-checkpoints` (train venv) it also plays each
checkpoint in both viewers as it appears, as the training mode does.

Training mode requires the train venv (LeRobot + torch) and the sim
extras; `--follow` needs only the sim and viz extras.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from _lab import (
    PREVIEW_EVERY_TICKS,
    PREVIEW_JPEG_QUALITY,
    bootstrap,
    frame_viewer,
    hold_until_closed,
    lerobot_train_command,
    rr_session,
    verdict_word,
)

bootstrap()
from rq_pipeline.envs.contract import InfoKeys, ObservationKeys  # noqa: E402
from rq_pipeline.envs.lerobot_train_log import (  # noqa: E402
    CHAIN_LOG_FILE,
    GPU_LOG_FILE,
    METRIC_NAMES,
    RUN_MANIFEST_FILE,
    TRAINER_CONFIG,
    FileFollower,
    RunLayout,
    RunManifest,
    StepClock,
    TrainLine,
    is_stage_line,
    parse_eval_line,
    parse_gpu_line,
    parse_train_line,
)
from rq_pipeline.evaluate.records import (  # noqa: E402
    EpisodeRecord,
    fold,
    funnel,
    milestones,
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
from rq_pipeline.viz import sinks  # noqa: E402

# The ALOHA 2 tasks; every builder takes `look`, and the first free body's
# z sits at CUBE_Z_STATE_INDEX in either scene (the cube, or the right
# arm's part).
ALOHA_TASKS = {
    entry.name: entry.build for entry in tasks().values() if entry.rig == RIG
}
Controller = Callable[[int, dict[str, Any]], Any]  # (tick, observation) -> ctrl
Act = Callable[[dict[str, Any]], Any]
Reset = Callable[[], None]
POLL_PERIOD_S = 2.0
LEGACY_TRAIN_LOG_FILE = "train.log"  # the sidecar's trainer-only log before 2026-08-28


class Timelines:
    """The viewer picks one timeline, and a series that exists only on
    another is invisible (the 2026-08-28 screenshot): every panel is
    logged on STEP and WALL both; a played episode adds SIM."""

    STEP = "train_step"
    WALL = "wall"
    SIM = "sim_time"
    VIDEO = "video_time"


def parse_args() -> argparse.Namespace:
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
        help="any mode: also save the stream to this .rrd file — the run "
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


# ---- logging to Rerun -------------------------------------------------------


def at_step(step: int, when: datetime | None = None) -> None:
    """Both timelines: the step, and the wall clock — the line's own stamp
    when it carries one, else now."""
    rr.set_time(Timelines.STEP, sequence=step)
    moment = when or datetime.now(UTC).replace(tzinfo=None)
    rr.set_time(Timelines.WALL, timestamp=np.datetime64(moment))


def note(text: str, echo: str | None = None) -> None:
    """A line in the `stage` panel, and on the terminal when asked."""
    rr.log("stage", rr.TextLog(text))
    if echo is not None:
        print(echo, flush=True)


def log_train_line(line: str) -> TrainLine | None:
    """One trainer line to Rerun: a metrics line as `train/*` at its step,
    an in-loop summary as a `stage` note; anything else is not ours."""
    parsed = parse_train_line(line)
    if parsed is not None:
        at_step(parsed.step, parsed.stamp)
        for name, value in parsed.metrics.items():
            rr.log(f"train/{METRIC_NAMES.get(name, name)}", rr.Scalars(value))
        rr.log("train/samples", rr.Scalars(float(parsed.samples)))
        rr.log("train/epochs", rr.Scalars(parsed.epochs))
        return parsed
    summary = parse_eval_line(line)
    if summary is not None:
        note(
            f"in-loop eval: {summary.pc_success:.0f}% of {summary.n_episodes} "
            f"in {summary.eval_s:.0f} s"
        )
    return None


def log_video(entity: str, path: Path, step: int) -> None:
    """A video the viewer can play: the asset, and one frame reference per
    frame on its own timeline (an asset alone shows nothing)."""
    asset = rr.AssetVideo(path=path)
    rr.log(entity, asset)
    stamps = np.asarray(asset.read_frame_timestamps_nanos())
    if not len(stamps):
        return
    rr.send_columns(
        entity,
        indexes=[
            rr.TimeColumn(Timelines.VIDEO, duration=stamps * 1e-9),
            rr.TimeColumn(Timelines.STEP, sequence=np.full(len(stamps), step)),
        ],
        columns=rr.VideoFrameReference.columns_nanos(stamps),
    )


def tail_training(process: subprocess.Popen) -> None:
    """Stream lerobot-train's stdout: metrics to Rerun, all else through."""
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip()
        if log_train_line(line) is None and "Training:" not in line and line:
            print(f"[train] {line}", flush=True)


# ---- controllers ------------------------------------------------------------


def scheduled(policy: Any, executed_horizon: int, nu: int) -> tuple[Act, Reset]:
    """A chunk policy executed on OUR horizon: `(act, reset)`."""
    from rq_pipeline.evaluate.scheduler import ActionScheduler  # noqa: PLC0415

    scheduler = ActionScheduler(policy, executed_horizon=executed_horizon, nu=nu)
    return scheduler.act, scheduler.reset


def openpi_controller(
    address: str, env: Any, executed_horizon: int | None
) -> tuple[Act, Reset]:
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
) -> tuple[Act, Reset]:
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

    def act(observation: dict[str, Any]) -> Any:
        raw = dict(observation)
        raw[ObservationKeys.AGENT_POS] = act_sim_state(
            observation[ObservationKeys.AGENT_POS]
        )
        return ctrl_from_act_sim_action(loaded.act(raw))

    return act, loaded.reset


# ---- the viewers -------------------------------------------------------------


class Watcher:
    """The env, the viewer, the mirror: plays one episode per checkpoint."""

    def __init__(
        self, look: str, task: str, device: str, executed_horizon: int | None = None
    ) -> None:
        import mujoco.viewer  # noqa: PLC0415 - the window, only when playing
        from rq_pipeline.envs.robotiq import RobotiqEnv, bundle_source  # noqa: PLC0415
        from rq_pipeline.viz import RigMirror  # noqa: PLC0415, sinks

        self.executed_horizon = executed_horizon
        built = ALOHA_TASKS[task](look=look)
        self.env = RobotiqEnv(built, source=bundle_source(built.bundle_dir))
        self.device = device
        self.mirror = RigMirror(
            self.env.model, model_colors=True, skip_groups=(GeomGroup.COLLISION,)
        )
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        frame_viewer(self.viewer, 1.6, elevation=-35, lookat=(0.05, 0.0, 0.1))
        self.played: set[int] = set()

    def episode(
        self, controller: Controller, prefix: str, text: str, trial: int = 0
    ) -> bool:
        """One episode from paired start `trial`, driven by
        `controller(tick, observation) -> ctrl` over the env's raw
        observation; logs frames, mirror, object height; returns the
        referee's verdict."""
        env = self.env
        dt = 1.0 / env.metadata["render_fps"]
        observation, info = env.reset(seed=trial)
        note(text)
        top = env.cameras[0].key
        for tick in range(env.max_episode_steps):
            rr.set_time(Timelines.SIM, duration=tick * dt)
            if tick % PREVIEW_EVERY_TICKS == 0:
                rr.log(
                    f"{prefix}/top",
                    rr.Image(observation[ObservationKeys.PIXELS][top]).compress(
                        jpeg_quality=PREVIEW_JPEG_QUALITY
                    ),
                )
            observation, _reward, _terminated, _truncated, info = env.step(
                np.asarray(controller(tick, observation), dtype=float)
            )
            self.mirror.log(env.data)
            rr.log(
                f"{prefix}/object_z",
                rr.Scalars(float(env.states[env.physics_step - 1, CUBE_Z_STATE_INDEX])),
            )
            if not self.viewer.is_running():
                raise SystemExit("viewer closed")
            self.viewer.sync()
        return bool(info[InfoKeys.IS_SUCCESS])

    def replay_demo(self, dataset_id: str, episode: int) -> None:
        """Play a human demonstration's actions open-loop through the bundle.

        The target behaviour, in our simulator, before any checkpoint
        exists — and a live check of the frame and gripper mapping: the
        demo's arm motion should look like the dataset's video. The
        cube sits at our paired start, not wherever theirs was, so the
        grasp itself need not land; the choreography is the point.
        """
        from _lab import load_demo_actions  # noqa: PLC0415 - LeRobot's dataset

        actions = load_demo_actions(dataset_id, episode)
        at_step(0)
        success = self.episode(
            lambda tick, _obs: ctrl_from_act_sim_action(
                actions[min(tick, len(actions) - 1)]
            ),
            f"demo/{episode:03d}",
            f"human demo {episode} from {dataset_id}: replaying {len(actions)} actions",
        )
        verdict = verdict_word(success)
        note(
            f"human demo {episode} replayed: {verdict} "
            "(open-loop, cube at our paired start)",
            f"[watch] demo {episode}: {verdict}",
        )

    def play(self, checkpoint_dir: Path, step: int, action_space: str) -> None:
        at_step(step)
        note(f"checkpoint {step}: loading")
        act, reset = checkpoint_controller(
            checkpoint_dir,
            action_space,
            self.env.task_description,
            self.device,
            executed_horizon=self.executed_horizon,
            nu=self.env.action_space.shape[0],
        )
        reset()
        success = self.episode(
            lambda _tick, observation: act(observation),
            f"play/{step:06d}",
            f"checkpoint {step}: playing trial 0",
        )
        at_step(step)
        rr.log("eval/success", rr.Scalars(1.0 if success else 0.0))
        verdict = verdict_word(success)
        note(f"checkpoint {step}: {verdict}", f"[watch] checkpoint {step}: {verdict}")
        self.played.add(step)

    def poll(self, layout: RunLayout, action_space: str) -> None:
        for step in layout.checkpoint_steps(with_weights=True):
            if step not in self.played:
                self.play(layout.checkpoint(step), step, action_space)


@dataclass
class Follower:
    """A run's files → Rerun: the manifest as a panel, the log's metrics
    and stage lines, the records' funnel per checkpoint pass, the eval
    videos, the GPU samples, the trainer's resolved config. Each call to
    `poll()` logs only what is new."""

    layout: RunLayout
    watcher: Watcher | None = None
    action_space: str = ActionSpace.BUNDLE
    configs_shown: set[int] = field(default_factory=set)
    videos_shown: set[Path] = field(default_factory=set)
    marked: set[str] = field(default_factory=set)
    clock: StepClock = field(default_factory=StepClock)
    manifest_mtime: float = 0.0
    passes_shown: int = 0
    final_shown: bool = False
    last_step: int = 0

    def __post_init__(self) -> None:
        # The chain log, or the trainer-only log a chain before 2026-08-28
        # wrote (a pod's mirror may still carry one).
        self.log = FileFollower(self.layout.watch / CHAIN_LOG_FILE)
        self.legacy_log = FileFollower(self.layout.watch / LEGACY_TRAIN_LOG_FILE)
        self.gpu = FileFollower(self.layout.watch / GPU_LOG_FILE)

    def mark(self, path: str) -> None:
        """Point markers on a series that may hold one sample per
        checkpoint — a line needs two points to be seen at all."""
        if path not in self.marked:
            rr.log(path, rr.SeriesPoints(markers="circle"), static=True)
            self.marked.add(path)

    def scalar(self, path: str, value: float) -> None:
        self.mark(path)
        rr.log(path, rr.Scalars(value))

    def manifest(self) -> None:
        """The chain writes it at its start and again at the train stage
        (with the command and the dataset's provenance): shown whenever
        it changes."""
        path = self.layout.watch / RUN_MANIFEST_FILE
        if not path.exists() or path.stat().st_mtime == self.manifest_mtime:
            return
        self.manifest_mtime = path.stat().st_mtime
        manifest = RunManifest.read_from(self.layout.watch)
        at_step(0)
        rr.log(
            "config/run",
            rr.TextDocument(manifest.as_markdown(), media_type=rr.MediaType.MARKDOWN),
            static=True,
        )
        note(
            f"{manifest.name}: {manifest.policy}, {manifest.steps} steps, "
            f"batch {manifest.batch_size}, started {manifest.started}"
        )

    def trainer_lines(self) -> None:
        for line in self.log.new_lines() + self.legacy_log.new_lines():
            parsed = log_train_line(line)
            if parsed is not None:
                self.last_step = parsed.step
                if parsed.stamp:
                    self.clock.add(parsed.stamp, parsed.step)
            elif is_stage_line(line):
                at_step(self.last_step)
                note(line.strip())

    def gpu_samples(self) -> None:
        for line in self.gpu.new_lines():
            sample = parse_gpu_line(line)
            if sample is None:
                continue
            at_step(self.clock.at(sample.taken), sample.taken)
            self.scalar("machine/gpu_util_pct", sample.util_pct)
            self.scalar("machine/gpu_mem_mib", sample.mem_mib)

    def checkpoints(self) -> list[int]:
        steps = self.layout.checkpoint_steps()
        for step in steps:
            config = self.layout.checkpoint(step) / TRAINER_CONFIG
            if step in self.configs_shown or not config.exists():
                continue
            at_step(step)
            rr.log(
                "config/trainer",
                rr.TextDocument(
                    json.dumps(
                        json.loads(config.read_text(encoding="utf-8")), indent=1
                    ),
                    media_type=rr.MediaType.TEXT,
                ),
            )
            note(f"checkpoint {step}: written")
            self.configs_shown.add(step)
        return steps

    def log_pass(
        self, prefix: str, chunk: Sequence[EpisodeRecord], step: int, padding: int = 0
    ) -> None:
        """One evaluation's records: the rate and the milestone funnel at
        `step`, under `eval/<prefix>/`."""
        at_step(step)
        for score in fold(chunk):
            self.scalar(f"eval/{prefix}/success_rate", score.successes / score.trials)
            note(
                f"{score.name}: {score.successes}/{score.trials} at step {step}"
                + (f" (+{padding} padding episodes)" if padding else "")
            )
        for counts in funnel(chunk).values():
            for milestone, count in zip(milestones(chunk), counts, strict=True):
                self.scalar(f"eval/{prefix}/funnel/{milestone}", count / len(chunk))

    def inloop_records(self, steps: list[int]) -> None:
        path = self.layout.inloop_records
        if not path.exists():
            return
        chunks = passes(read_records(path))
        for index in range(self.passes_shown, min(len(chunks), len(steps))):
            self.log_pass("inloop", chunks[index], steps[index])
            self.passes_shown = index + 1

    def videos(self, steps: list[int]) -> None:
        for step in steps:
            # LeRobot nests the suite (`robotiq_0/`) under the step directory.
            for video in sorted(
                (self.layout.training / "eval" / f"videos_step_{step:06d}").rglob(
                    "*.mp4"
                )
            ):
                if video not in self.videos_shown:
                    at_step(step)
                    log_video(f"eval/video/{video.stem}", video, step)
                    self.videos_shown.add(video)

    def final_eval(self) -> None:
        """The paired evaluation beside the run, once its records are there."""
        path = self.layout.eval_records
        if self.final_shown or not path.exists():
            return
        chunks = passes(read_records(path))
        if not chunks:
            return  # the file exists before its first row
        step = max(self.last_step, *self.configs_shown, 0)
        self.log_pass("final", chunks[0], step, sum(len(c) for c in chunks[1:]))
        for video in sorted(self.layout.evaluation.glob("videos/**/*.mp4")):
            log_video(f"eval/final/video/{video.stem}", video, step)
        self.final_shown = True

    def poll(self) -> None:
        self.manifest()
        self.trainer_lines()
        self.gpu_samples()
        steps = self.checkpoints()
        self.inloop_records(steps)
        self.videos(steps)
        self.final_eval()
        if self.watcher is not None:
            self.watcher.poll(self.layout, self.action_space)


def dashboard_blueprint(*, training: bool, play: bool) -> rrb.Blueprint:
    """The panels in the order a reader wants them: what the run IS
    (manifest, trainer config, the stage log); how it is going (loss,
    learning rate, gradient norm, throughput; the evaluations; the
    machine; the eval videos); and, when checkpoints are played, what
    the policy sees, the rig and the object height."""
    about = (
        [
            rrb.TextDocumentView(origin="config/run", name="run"),
            rrb.TextDocumentView(origin="config/trainer", name="trainer config"),
        ]
        if training
        else []
    )
    rows = [rrb.Horizontal(*about, rrb.TextLogView(origin="stage", name="stage"))]
    curves = rrb.Horizontal(
        rrb.TimeSeriesView(
            origin="train",
            contents=["train/loss", "train/l1_loss", "train/kld_loss"],
            name="loss",
        ),
        rrb.TimeSeriesView(origin="train", contents="train/lr", name="learning rate"),
        rrb.TimeSeriesView(
            origin="train", contents="train/grad_norm", name="gradient norm"
        ),
        rrb.TimeSeriesView(
            origin="train",
            contents=["train/samples_per_s", "train/update_s", "train/dataloading_s"],
            name="throughput",
        ),
    )
    # Rerun's filters take `path/**` at the END of a path and `- path` to
    # exclude; `eval/**/top` matches nothing (measured 2026-08-28).
    rates = ["+ $origin/success_rate", "+ $origin/funnel/**"]
    evaluation = rrb.Horizontal(
        rrb.TimeSeriesView(
            origin="eval/inloop", contents=rates, name="in-loop eval (rate, funnel)"
        ),
        rrb.TimeSeriesView(
            origin="eval/final", contents=rates, name="paired eval (rate, funnel)"
        ),
        rrb.TimeSeriesView(origin="machine", name="machine (nvidia-smi)"),
    )
    videos = rrb.Horizontal(
        rrb.Spatial2DView(origin="eval/video", name="in-loop eval videos"),
        rrb.Spatial2DView(origin="eval/final/video", name="paired eval videos"),
    )
    if training:
        rows += [curves, evaluation, videos]
    if play:
        # Played episodes live under their own root, so the views need no
        # filter (a played checkpoint's `eval/<step>` under the records'
        # `eval/` took a five-line exclusion list — and `--play` logged
        # elsewhere, so its camera and height panels stayed empty).
        rows.append(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="play", name="the policy's camera"),
                rrb.Spatial3DView(origin="world", name="the rig"),
                rrb.TimeSeriesView(origin="play", name="object height"),
            )
        )
    # The training dimension is the default timeline; every series also
    # carries the wall clock, so the other choice reads too. A replay has
    # no training: its ticks all sit at one step, and only the simulated
    # clock spreads them into a curve.
    return rrb.Blueprint(
        rrb.Vertical(*rows),
        rrb.TimePanel(timeline=Timelines.STEP if training else Timelines.SIM),
        collapse_panels=False,
    )


def open_sinks(rrd: Path | None) -> None:
    """Keep the viewer AND write `rrd` when asked: `rr.save()` alone
    REPLACES the viewer connection, and the window stays empty while
    the file fills (measured 2026-08-28). Before the blueprint, so the
    file carries the layout too."""
    if rrd is not None:
        rr.set_sinks(*sinks(rr, file=rrd, wanted=True))


# ---- the modes ---------------------------------------------------------------


def follow(args: argparse.Namespace) -> None:
    layout = RunLayout.of(Path(args.follow))
    rr_session(
        f"robotiq-follow-{layout.name}", mode="spawn", world_up=args.play_checkpoints
    )
    open_sinks(args.rrd)
    rr.send_blueprint(dashboard_blueprint(training=True, play=args.play_checkpoints))
    watcher = None
    if args.play_checkpoints:
        from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

        watcher = Watcher(
            args.look, args.task, best_device(args.device), args.executed_horizon
        )
    follower = Follower(layout, watcher=watcher, action_space=args.action_space)
    print(f"[follow] {layout.training}: streaming to Rerun; Ctrl-C to stop", flush=True)
    try:
        while True:
            follower.poll()
            if args.once or (watcher is not None and not watcher.viewer.is_running()):
                break
            time.sleep(POLL_PERIOD_S)
    except KeyboardInterrupt:
        pass
    print(f"[follow] last step seen: {follower.last_step}", flush=True)


def play_only(args: argparse.Namespace) -> None:
    """Watch one checkpoint: every paired start, both viewers, no training."""
    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

    device = best_device(args.device)
    name = (
        f"openpi-{args.openpi.replace(':', '-')}"
        if args.openpi
        else Path(args.play).parent.name
    )
    rr_session(f"robotiq-play-{name}", mode="spawn")
    open_sinks(args.rrd)
    rr.send_blueprint(dashboard_blueprint(training=False, play=True))
    watcher = Watcher(args.look, args.task, device, args.executed_horizon)
    act, reset = (
        openpi_controller(args.openpi, watcher.env, args.executed_horizon)
        if args.openpi
        else checkpoint_controller(
            Path(args.play),
            args.action_space,
            watcher.env.task_description,
            device,
            executed_horizon=args.executed_horizon,
            nu=watcher.env.action_space.shape[0],
        )
    )
    successes = 0
    for trial in range(args.trials):
        reset()
        at_step(trial)
        success = watcher.episode(
            lambda _tick, observation: act(observation),
            f"play/trial{trial}",
            f"{name}: trial {trial}",
            trial=trial,
        )
        successes += int(success)
        verdict = verdict_word(success)
        note(f"{name}: trial {trial} {verdict}", f"[play] trial {trial}: {verdict}")
    note(
        f"{name}: {successes}/{args.trials}",
        f"[play] {successes}/{args.trials} - close the MuJoCo window to exit",
    )
    hold_until_closed(watcher.viewer)


def train_and_watch(args: argparse.Namespace) -> None:
    from rq_pipeline.envs.lerobot_policy import best_device  # noqa: PLC0415

    # `runs/<name>` names the run the way the chain does: `--name t2-act`
    # trains into runs/t2-act, and every reader of the layout agrees.
    layout = RunLayout.of(Path(args.runs) / args.name)
    device = best_device(args.device)
    rr_session(f"robotiq-train-watch-{layout.name}", mode="spawn")
    open_sinks(args.rrd)
    rr.send_blueprint(dashboard_blueprint(training=True, play=True))
    process = subprocess.Popen(
        lerobot_train_command(
            policy=args.policy,
            device=device,
            dataset=args.dataset,
            output_dir=layout.training,
            job_name=layout.training.name,
            steps=args.steps,
            batch_size=args.batch_size,
            save_freq=args.save_freq,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=tail_training, args=(process,), daemon=True).start()
    watcher = Watcher(args.look, args.task, device, args.executed_horizon)
    try:
        for episode in range(args.replay_demos):
            watcher.replay_demo(args.dataset, episode)
        while process.poll() is None:
            watcher.poll(layout, args.action_space)
            if not watcher.viewer.is_running():
                process.terminate()
                break
            time.sleep(POLL_PERIOD_S)
        watcher.poll(layout, args.action_space)  # the final checkpoint
        note(
            f"training exited with {process.returncode}",
            "[watch] training finished - close the MuJoCo window to exit",
        )
        hold_until_closed(watcher.viewer)
    finally:
        if process.poll() is None:
            process.terminate()


def _exit_on_sigterm(signum: int, frame: object) -> None:
    # A plain SIGTERM would skip `finally` and orphan the trainer.
    raise SystemExit(128 + signum)


def main() -> None:
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    args = parse_args()
    if args.follow:
        follow(args)
    elif args.play or args.openpi:
        play_only(args)
    else:
        train_and_watch(args)


if __name__ == "__main__":
    main()
