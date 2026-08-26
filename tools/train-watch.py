"""Watch a policy learn: train in the background, play every checkpoint.

    cd pipeline && GALLIUM_DRIVER=d3d12 WGPU_BACKEND=vulkan MUJOCO_GL=egl \\
        .venv-train/bin/python ../tools/train-watch.py \\
        --steps 20000 --save-freq 1000 --batch-size 8 --name t2-act

`lerobot-train` runs as a subprocess on the GPU (ACT on the public ALOHA
transfer-cube demos by default). This process tails its log and, each
time a checkpoint is written, loads it through LeRobot's own processors
and plays ONE transfer-cube episode through the gymnasium env
(rq_pipeline.envs) on the aloha2-nominal bundle — the identified
dynamics, not the trainer's simulator — in both viewers:

    MuJoCo window            the rig, live, every checkpoint's episode
    Rerun  train/*           loss, l1, kld, grad norm, lr  (train_step)
           eval/<step>/...   the top camera the policy sees, 10 Hz;
                             cube height; the referee's verdict
           world/rig         the mesh-true mirror during each episode
           stage             which checkpoint is playing, and how it did

Episodes here are a WINDOW onto training, not the evaluation: one
trial per checkpoint, the same paired start every time (trial 0), so
successive checkpoints are comparable by eye. The certificate is the
harness's job, with all trials, after training ends — and
`lerobot-train --env.type=robotiq` evaluates in-loop through the same
env without this tool at all.

Requires the train venv (LeRobot + CUDA torch) and the sim extras.
MUJOCO_GL=egl keeps the offscreen renders off the window's GL context;
the viewer window still uses GLFW.
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

from _lab import bootstrap, rr_session

HERE = Path(__file__).resolve().parent
bootstrap()
from rq_pipeline.envs.robotiq import RobotiqEnv, bundle_source  # noqa: E402
from rq_pipeline.tasks.aloha2 import (  # noqa: E402
    CUBE_Z_STATE_INDEX,
    act_sim_state,
    build_kitting,
    build_transfer_cube,
    ctrl_from_act_sim_action,
)

from _rig3d import RigMirror  # noqa: E402

# Both take `look`; the first free body's z sits at CUBE_Z_STATE_INDEX in
# either scene (the cube, or the right arm's part).
ALOHA_TASKS = {"transfer_cube": build_transfer_cube, "kitting": build_kitting}

LOG_LINE = re.compile(r"step:(\d+).*?loss:([\d.]+).*?grdn:([\d.]+).*?lr:([\d.e+-]+)")
EXTRA = re.compile(r"(l1_loss|kld_loss):([\d.]+)")
COLLISION_GROUP = 3
CAMERA_EVERY_TICKS = 5  # 50 Hz control -> 10 Hz frames in Rerun
LIVE_SHADOWSIZE = 2048


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="t2-act")
    parser.add_argument("--policy", default="act")
    parser.add_argument("--dataset", default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=250)
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
        default="act_sim",
        choices=("act_sim", "aloha2"),
        help="scene appearance for the played episodes: the ACT simulator's "
        "(what the public demos look like) or the bundle's own",
    )
    parser.add_argument(
        "--task",
        default="transfer_cube",
        choices=sorted(ALOHA_TASKS),
        help="which ALOHA 2 task the checkpoint plays (kitting for T5's)",
    )
    parser.add_argument(
        "--action-space",
        default="act_sim",
        choices=("act_sim", "bundle"),
        help="what the checkpoint speaks: gym-aloha's normalised grippers "
        "(public-demo checkpoints) or the bundle's own ctrl (our demos, T5)",
    )
    return parser.parse_args()


def train_command(args, output_dir):
    return [
        str(HERE.parent / "pipeline" / ".venv-train" / "bin" / "lerobot-train"),
        f"--policy.type={args.policy}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={args.dataset}",
        f"--output_dir={output_dir}",
        f"--job_name={args.name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        "--log_freq=50",
        f"--save_freq={args.save_freq}",
        "--wandb.enable=false",
    ]


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


def load_checkpoint(
    path: Path, action_space: str, instruction: str, device: str = "cuda"
):
    """A checkpoint as `act(observation) -> ctrl` plus its reset, through
    LeRobot's own pre/post-processors and `preprocess_observation` — the
    same path `lerobot-eval` takes, so nothing is hand-converted here."""
    import torch  # noqa: PLC0415
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    from lerobot.envs.utils import preprocess_observation  # noqa: PLC0415
    from lerobot.policies.factory import (  # noqa: PLC0415
        get_policy_class,
        make_pre_post_processors,
    )

    config = PreTrainedConfig.from_pretrained(str(path))
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(str(path), config=config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    translate = action_space == "act_sim"

    def act(observation):
        raw = dict(observation)
        if translate:
            raw["agent_pos"] = act_sim_state(observation["agent_pos"])
        batch = preprocess_observation(raw)
        batch["task"] = [instruction]
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(batch)))
        action = action.squeeze(0).cpu().numpy()
        return ctrl_from_act_sim_action(action) if translate else action

    return act, policy.reset


class Watcher:
    """The env, the viewer, the mirror: plays one episode per checkpoint."""

    def __init__(self, look: str, task: str = "transfer_cube"):
        self.env = RobotiqEnv(ALOHA_TASKS[task](look=look), source=bundle_source())
        self.env.model.vis.quality.shadowsize = LIVE_SHADOWSIZE
        self.mirror = RigMirror(
            self.env.model, model_colors=True, skip_groups=(COLLISION_GROUP,)
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
        observation; logs frames, mirror, cube height; returns the
        referee's verdict."""
        env = self.env
        dt = env.model.opt.timestep * env.protocol.control_interval
        observation, info = env.reset(seed=trial)
        rr.log("stage", rr.TextLog(note))
        for tick in range(env._max_episode_steps):
            rr.set_time("sim_time", duration=tick * dt)
            if tick % CAMERA_EVERY_TICKS == 0:
                rr.log(f"{prefix}/top", rr.Image(observation["pixels"]["top"]))
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
        return bool(info["is_success"])

    def replay_demo(self, dataset_id: str, episode: int):
        """Play a human demonstration's actions open-loop through the bundle.

        The target behaviour, in our simulator, before any checkpoint
        exists — and a live check of the frame and gripper mapping: the
        demo's arm motion should look like the dataset's video. The
        cube sits at our paired start, not wherever theirs was, so the
        grasp itself need not land; the choreography is the point.
        """
        import torch  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        dataset = LeRobotDataset(dataset_id)
        table = dataset.hf_dataset
        episodes = np.asarray([int(e) for e in table["episode_index"]])
        rows = np.flatnonzero(episodes == episode)
        actions = np.stack(
            [np.asarray(torch.as_tensor(table[int(i)]["action"])) for i in rows]
        )
        rr.set_time("train_step", sequence=0)

        def controller(tick, observation):
            del observation
            return ctrl_from_act_sim_action(actions[min(tick, len(actions) - 1)])

        success = self._episode(
            controller,
            f"demo/{episode:03d}",
            f"human demo {episode} from {dataset_id}: replaying {len(actions)} actions",
        )
        verdict = "SUCCESS" if success else "no transfer"
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
        act, reset = load_checkpoint(
            checkpoint_dir, action_space, self.env.task_description
        )
        reset()
        success = self._episode(
            lambda _tick, observation: act(observation),
            f"eval/{step:06d}",
            f"checkpoint {step}: playing trial 0",
        )
        rr.set_time("train_step", sequence=step)
        rr.log("eval/success", rr.Scalars(1.0 if success else 0.0))
        verdict = "SUCCESS" if success else "fail"
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
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    watcher = Watcher(args.look, args.task)
    act, reset = load_checkpoint(
        checkpoint, args.action_space, watcher.env.task_description
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
        verdict = "SUCCESS" if success else "fail"
        rr.log("stage", rr.TextLog(f"{name}: trial {trial} {verdict}"))
        print(f"[play] trial {trial}: {verdict}", flush=True)
    rr.log("stage", rr.TextLog(f"{name}: {successes}/{args.trials}"))
    print(
        f"[play] {successes}/{args.trials} - close the MuJoCo window to exit",
        flush=True,
    )
    while watcher.viewer.is_running():
        time.sleep(0.2)


def main() -> None:
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    args = parse_args()
    if args.play:
        play_only(args)
        return
    output_dir = Path(args.runs) / args.name
    rr_session(f"robotiq-train-watch-{args.name}", mode="spawn")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    process = subprocess.Popen(
        train_command(args, output_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    seen_steps: list[int] = []
    threading.Thread(
        target=tail_training, args=(process, seen_steps), daemon=True
    ).start()

    watcher = Watcher(args.look, args.task)
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
        while watcher.viewer.is_running():
            time.sleep(0.2)
    finally:
        if process.poll() is None:
            process.terminate()


if __name__ == "__main__":
    main()
