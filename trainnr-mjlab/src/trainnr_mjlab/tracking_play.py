"""A trained tracking policy dancing its reference once, in physics, as a
trainnr job the Studio watches:

    uv run python -m trainnr_mjlab.tracking_play <run>/model_N.pt \\
        --reference <reference recording> --footage <footage folder> --out <folder>

mjlab's play mode (the motion from its first frame, no pushes, no
observation noise, no time limit) on one world, the checkpoint's actor
driving it. The rollout streams live into the Studio (the recorder,
every step), and lands in `--out`: the stream, `play.mp4` (the policy in
physics beside the original video), and `play.json` with what happened:
steps alive, falls (episodes that ended before the clip did), and the
mean body position and joint errors against the reference.

A look, not an evaluation: one seed, no interval. The tracking task's
evaluation (paired trials, pushes and delays) is the next step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from trainnr.mcp_jobs import default_jobs_root, jobs_dir_of, track
from trainnr.video.footage import FOOTAGE_FILE, Footage

from trainnr_mjlab.checkpoint_guard import require_tensor_only
from trainnr_mjlab.tracking_train import TASK_ID, quiet_line_search
from trainnr_mjlab.walks import use_project

MOTION_FILE = "raw/motion.npz"
PLAY_FILE = "play.json"
VIDEO_FILE = "play.mp4"
STREAM_FILE = "stream.rrd"
KIND = "play-tracking"
FPS = 50  # the tracking task's control rate


def main() -> None:  # noqa: PLR0915 - one rollout, each step named
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--footage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: PLC0415
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from trainnr_mjlab.recorder import RerunRecorderCfg  # noqa: PLC0415

    use_project(args.project)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_name = args.checkpoint.resolve().parent.name
    stream = f"trainnr-play-{run_name}-{args.checkpoint.stem}"
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.commands["motion"].motion_file = str(args.reference.resolve() / MOTION_FILE)
    cfg.scene.num_envs = 1
    cfg.sim = quiet_line_search(cfg.sim)
    cfg.recorders = {
        "rerun": RerunRecorderCfg(
            app_id=stream, every=1, frame_every=10**9, file=str(out / STREAM_FILE)
        )
    }
    agent = load_rl_cfg(TASK_ID)
    footage = Footage.read(args.footage / FOOTAGE_FILE)
    reference = np.load(args.reference / MOTION_FILE)
    steps = len(reference["joint_pos"])

    with (
        tempfile.TemporaryDirectory(prefix="trainnr-play-") as tmp,
        track(
            KIND,
            jobs_dir=jobs_dir_of(args.project or default_jobs_root()),
            name=f"{run_name} {args.checkpoint.stem}",
            viewer=str(out / STREAM_FILE),
            stream=stream,
        ) as run,
    ):
        run.stage(f"loading {args.checkpoint.name}")
        env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MotionTrackingOnPolicyRunner(
            wrapped, asdict(agent), log_dir=None, device=device
        )
        require_tensor_only(args.checkpoint)  # before any unpickling
        runner.load(
            str(args.checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
        command = env.command_manager.get_term("motion")
        obs = wrapped.get_observations()
        frames = Path(tmp)
        falls, alive, body_err, joint_err = 0, 0, [], []
        with torch.inference_mode():
            for i in range(steps):
                if i % 50 == 0:
                    run.progress(
                        i, steps, "steps", f"step {i} of {steps}, {falls} falls"
                    )
                obs, _, dones, _ = wrapped.step(policy(obs))
                body_err.append(
                    float(
                        (command.body_pos_relative_w - command.robot_body_pos_w)
                        .norm(dim=-1)
                        .mean()
                    )
                )
                joint_err.append(
                    float((command.joint_pos - command.robot_joint_pos).abs().mean())
                )
                if bool(dones[0]) and i < steps - 1:
                    falls += 1
                else:
                    alive += 1
                Image.fromarray(env.render()).save(frames / f"{i:05d}.png")
        env.close()

        run.stage("the policy beside the video")
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
                "-i", str(frames / "%05d.png"), "-i", str(args.footage / footage.video),
                "-filter_complex",
                "[0:v]scale=-2:480[p];[1:v]fps=50,scale=-2:480[v];[v][p]hstack=inputs=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest",
                str(out / VIDEO_FILE),
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
        result: dict[str, Any] = {
            "checkpoint": args.checkpoint.name,
            "run": run_name,
            "reference": args.reference.name,
            "steps": steps,
            "seconds": steps / FPS,
            "falls": falls,
            "steps_alive": alive,
            "mean_body_pos_error_m": round(float(np.mean(body_err)), 4),
            "mean_joint_error_rad": round(float(np.mean(joint_err)), 4),
            "protocol": "mjlab play mode, one world, one seed: a look, no interval",
        }
        (out / PLAY_FILE).write_text(json.dumps(result, indent=1))
        run.stage(f"done: {falls} falls in {steps / FPS:.1f} s of dance")
        print(f"[play] {json.dumps(result)}")


if __name__ == "__main__":
    main()
