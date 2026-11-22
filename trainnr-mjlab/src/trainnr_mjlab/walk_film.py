"""A walk's checkpoints on film: what the policy did at each saved
iteration, rendered offscreen, as a trainnr job the Studio lists.

    cd trainnr-mjlab && uv run python -m trainnr_mjlab.walk_film \\
        <run>/model_250.pt <run>/model_500.pt ... --robot legged \\
        --project <project> [--seconds 10] [--command 0.3 0 0] [--seed 1000]

Each checkpoint walks one world at the nominal point (no actuator
randomization) from the same seed and the same reset, under one
commanded twist held for the whole film (forward at the top of the
task's declared range unless `--command` says otherwise), for
`--seconds` or until it falls. Every control tick is drawn from the
walk's chase camera, so the film runs at the control rate in real time
whatever the machine's speed. Each checkpoint lands beside the run in
`play-model_N/`: `play.mp4` and `play.json` (the command, the seconds
walked, whether and when it fell, the distance covered). With several
checkpoints, `film.mp4` beside them plays them in iteration order, the
iteration written on the picture, and `film.json` lists them; the run's
page in the Studio plays it on the iteration timeline.

A look, not an evaluation: one world, one seed, no interval.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from trainnr_mjlab.walk_stills import (
    CHECKPOINT,
    _policy_loader,
    _scene_file,
    scene_renderer,
)
from trainnr_mjlab.walks import DEFAULT_ROBOT, ROBOTS, use_project, walk_spec

PLAY_DIR = "play-{stem}"
PLAY_FILE = "play.json"
VIDEO_FILE = "play.mp4"
FILM_FILE = "film.mp4"
FILM_RECORD = "film.json"
KIND = "film-walk"
SIZE = (640, 480)
# The camera follows the trunk through a low-pass, so a gait's bounce
# does not shake the picture (the share of the gap closed each tick).
FOLLOW = 0.1
LABEL = (236, 238, 242)
FELL_LABEL = (240, 96, 96)
PROTOCOL = (
    "one world, nominal actuators, one seed, one held command: a look, no interval"
)


def film_command(cfg: Any, command: list[float] | None) -> list[float]:
    """Hold one twist for the whole film: `command` (forward, sideways,
    turn), else forward at the top of the task's declared range. The
    curriculum, standing, heading and world-frame draws are turned off so
    every checkpoint walks under the same one."""
    from trainnr_mjlab.envelope import COMMAND_TERM, CURRICULUM_TERM  # noqa: PLC0415

    if cfg.curriculum:
        cfg.curriculum.pop(CURRICULUM_TERM, None)
    term = cfg.commands[COMMAND_TERM]
    ranges = term.ranges
    vx, vy, wz = command if command is not None else (max(ranges.lin_vel_x), 0.0, 0.0)
    ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = (vx, vx), (vy, vy), (wz, wz)
    term.heading_command = False
    term.rel_standing_envs = term.rel_world_envs = term.rel_forward_envs = 0.0
    term.init_velocity_prob = 0.0
    return [float(vx), float(vy), float(wz)]


def iteration_of(checkpoint: Path) -> int | None:
    match = CHECKPOINT.search(checkpoint.name)
    return int(match.group(1)) if match else None


def _encoder(out: Path, fps: float) -> subprocess.Popen[bytes]:
    w, h = SIZE
    return subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip


def write_film(  # noqa: PLR0913 - one rollout drawn, each piece named
    scene: tuple[Any, Any, Any, Any],
    qpos: Any,
    out: Path,
    *,
    fps: float,
    title: str,
    fell: bool,
) -> None:
    """Every captured pose drawn from the chase camera into `out`, the
    title and the clock on the picture, `fell` on the last frame."""
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    model, data, renderer, camera = scene
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)
    encoder = _encoder(out, fps)
    assert encoder.stdin is not None
    lookat = None
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        trunk = np.array(data.xpos[1])  # the first body after the world
        lookat = trunk if lookat is None else lookat + FOLLOW * (trunk - lookat)
        camera.lookat[:] = lookat
        renderer.update_scene(data, camera=camera)
        picture = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(picture)
        draw.text((14, 10), title, fill=LABEL, font=font)
        draw.text((14, 40), f"{i / fps:4.1f} s", fill=LABEL, font=small)
        if fell and i == len(qpos) - 1:
            draw.text((14, 62), "fell", fill=FELL_LABEL, font=font)
        encoder.stdin.write(picture.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg could not write {out}")


def join_films(films: list[Path], out: Path) -> None:
    """The films one after another, as they are (same size, rate, codec)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        listing.write("".join(f"file '{f.resolve()}'\n" for f in films))
    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", listing.name, "-c", "copy", str(out),
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
    finally:
        Path(listing.name).unlink()


def _already(record: Path, wanted: dict[str, Any]) -> dict[str, Any] | None:
    """The play record on disk when it was filmed the same way."""
    if not (record.is_file() and (record.parent / VIDEO_FILE).is_file()):
        return None
    row: dict[str, Any] = json.loads(record.read_text())
    return row if all(row.get(k) == v for k, v in wanted.items()) else None


def film_checkpoints(  # noqa: PLR0913 - the film's knobs, each named
    checkpoints: list[Path],
    *,
    robot: str,
    seconds: float,
    command: list[float] | None,
    seed: int,
    device: str,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Film each checkpoint into `play-model_N/` beside it; with several,
    `film.mp4` and `film.json` beside them. A checkpoint already filmed
    the same way is kept. The rows, in iteration order."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from trainnr_mjlab.walk_verdict import rollout_episodes  # noqa: PLC0415
    from trainnr_mjlab.walks import adopt_legged_of  # noqa: PLC0415

    checkpoints = sorted(checkpoints, key=lambda c: (iteration_of(c) or 0, c.name))
    run_dir = checkpoints[0].parent
    adopt_legged_of(robot, checkpoints[0])  # a legged run's own robot (#100)
    spec = walk_spec(robot)
    held: list[list[float]] = []
    env, load = _policy_loader(
        spec, device, lambda cfg: held.append(film_command(cfg, command))
    )
    fps = 1.0 / float(env.unwrapped.step_dt)
    ticks = max(1, round(seconds * fps))
    wanted = {"command": held[0], "seconds_asked": seconds, "seed": seed}
    scene = scene_renderer(_scene_file(robot), spec.chase, SIZE)
    rows = []
    try:
        for k, checkpoint in enumerate(checkpoints):
            folder = run_dir / PLAY_DIR.format(stem=checkpoint.stem)
            row = _already(folder / PLAY_FILE, wanted)
            if progress is not None:
                progress(k, len(checkpoints), checkpoint.name)
            if row is None:
                torch.manual_seed(seed)
                env.unwrapped.seed(seed)
                env.reset()
                walk = rollout_episodes(
                    env, load(checkpoint), 1, capture=True, max_ticks=ticks
                )[0]
                o = walk.outcome
                iteration = iteration_of(checkpoint)
                title = (
                    f"iteration {iteration}"
                    if iteration is not None
                    else checkpoint.stem
                )
                folder.mkdir(exist_ok=True)
                write_film(
                    scene,
                    walk.qpos,
                    folder / VIDEO_FILE,
                    fps=fps,
                    title=title,
                    fell=o.fell,
                )
                start, end = walk.qpos[0][:2], walk.qpos[-1][:2]
                row = {
                    "checkpoint": checkpoint.name,
                    "iteration": iteration,
                    **wanted,
                    "seconds": round(len(walk.qpos) / fps, 2),
                    "fps": fps,
                    "fell": o.fell,
                    "fell_at_s": round(o.steps / fps, 2) if o.fell else None,
                    "distance_m": round(float(np.linalg.norm(end - start)), 3),
                    "tracked": o.success,
                    "err_ratio": round(o.err_ratio, 4),
                    "protocol": PROTOCOL,
                }
                (folder / PLAY_FILE).write_text(json.dumps(row, indent=1) + "\n")
                print(f"[film] {json.dumps(row)}", flush=True)
            rows.append(row)
    finally:
        scene[2].close()
    if len(rows) > 1:
        films = [
            run_dir / PLAY_DIR.format(stem=Path(r["checkpoint"]).stem) / VIDEO_FILE
            for r in rows
        ]
        join_films(films, run_dir / FILM_FILE)
        record = {
            "film": FILM_FILE,
            **wanted,
            "protocol": PROTOCOL,
            "checkpoints": [
                {
                    "checkpoint": r["checkpoint"],
                    "iteration": r["iteration"],
                    "play": PLAY_DIR.format(stem=Path(r["checkpoint"]).stem),
                    "seconds": r["seconds"],
                    "fell": r["fell"],
                }
                for r in rows
            ],
        }
        (run_dir / FILM_RECORD).write_text(json.dumps(record, indent=1) + "\n")
    return rows


def main() -> int:
    from trainnr.mcp_jobs import default_jobs_root, jobs_dir_of, track  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=ROBOTS)
    parser.add_argument("--project", default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--command", type=float, nargs=3, default=None,
        metavar=("FORWARD", "SIDEWAYS", "TURN"),
        help="m/s, m/s, rad/s; forward at the top of the task's range when unset",
    )  # fmt: skip
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--device", default=None, help="cuda:0 or cpu; found when unset"
    )
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen: no window needed
    use_project(args.project)
    import warp as wp  # noqa: PLC0415

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    checkpoints = [c.resolve() for c in args.checkpoints]
    with track(
        KIND,
        jobs_dir=jobs_dir_of(args.project or default_jobs_root()),
        name=f"{checkpoints[0].parent.name} film",
    ) as run:

        def progress(done: int, total: int, name: str) -> None:
            run.progress(done, total, "checkpoints", f"filming {name}")

        run.stage(f"filming {len(checkpoints)} checkpoints")
        rows = film_checkpoints(
            checkpoints,
            robot=args.robot,
            seconds=args.seconds,
            command=args.command,
            seed=args.seed,
            device=device,
            progress=progress,
        )
        fell = sum(1 for r in rows if r["fell"])
        where = (
            FILM_FILE
            if len(rows) > 1
            else f"{PLAY_DIR.format(stem=checkpoints[0].stem)}/{VIDEO_FILE}"
        )
        run.stage(f"done: {len(rows)} filmed, {fell} fell; {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
