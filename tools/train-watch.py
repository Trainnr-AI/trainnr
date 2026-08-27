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

Requires the train venv (LeRobot + torch) and the sim extras.
"""

import argparse
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import rerun as rr

from _lab import (
    PREVIEW_EVERY_TICKS,
    bootstrap,
    hold_until_closed,
    lerobot_train_command,
    load_demo_actions,
    rr_session,
    verdict_word,
)

bootstrap()
from rq_pipeline.envs.contract import InfoKeys, ObservationKeys  # noqa: E402
from rq_pipeline.envs.lerobot_policy import best_device, load_policy  # noqa: E402
from rq_pipeline.envs.robotiq import RobotiqEnv, bundle_source  # noqa: E402
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

from _rig3d import RigMirror  # noqa: E402

# The ALOHA 2 tasks; every builder takes `look`, and the first free body's
# z sits at CUBE_Z_STATE_INDEX in either scene (the cube, or the right
# arm's part).
ALOHA_TASKS = {
    entry.name: entry.build for entry in tasks().values() if entry.rig == RIG
}

LOG_LINE = re.compile(r"step:(\d+).*?loss:([\d.]+).*?grdn:([\d.]+).*?lr:([\d.e+-]+)")
EXTRA = re.compile(r"(l1_loss|kld_loss):([\d.]+)")


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


def tail_training(process, seen_steps):
    """Stream lerobot-train's stdout: loss lines to Rerun, all else through."""
    for raw in process.stdout:
        line = raw.rstrip()
        match = LOG_LINE.search(line)
        if not match:
            if "Training:" not in line and line:
                print(f"[train] {line}", flush=True)
            continue
        step = int(match.group(1))
        rr.set_time("train_step", sequence=step)
        rr.log("train/loss", rr.Scalars(float(match.group(2))))
        rr.log("train/grad_norm", rr.Scalars(float(match.group(3))))
        rr.log("train/lr", rr.Scalars(float(match.group(4))))
        for key, value in EXTRA.findall(line):
            rr.log(f"train/{key}", rr.Scalars(float(value)))
        seen_steps.append(step)


def checkpoint_controller(path: Path, action_space: str, instruction: str, device: str):
    """The checkpoint as `act(observation) -> ctrl` plus its reset; the
    gym-aloha convention (normalised grippers) wrapped around the model
    call when the checkpoint speaks it."""
    loaded = load_policy(path, instruction=instruction, device=device)
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

    def __init__(self, look: str, task: str, device: str):
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
            checkpoint_dir, action_space, self.env.task_description, self.device
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


def _exit_on_sigterm(signum, frame):
    # A plain SIGTERM would skip `finally` and orphan the trainer.
    raise SystemExit(128 + signum)


def play_only(args) -> None:
    """Watch one checkpoint: every paired start, both viewers, no training."""
    checkpoint = Path(args.play)
    rr_session(f"robotiq-play-{checkpoint.parent.name}", mode="spawn")
    device = best_device(args.device)
    watcher = Watcher(args.look, args.task, device)
    act, reset = checkpoint_controller(
        checkpoint, args.action_space, watcher.env.task_description, device
    )
    name = checkpoint.parent.name
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
    if args.play:
        play_only(args)
        return
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

    watcher = Watcher(args.look, args.task, device)
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
