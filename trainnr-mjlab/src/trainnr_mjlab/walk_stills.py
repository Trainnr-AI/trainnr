"""One picture per checkpoint of a walk run: what the policy looked
like every N iterations, from an actual rollout of each checkpoint.

    cd trainnr-mjlab && ../tools/wsl-run.sh uv run \\
        python -m trainnr_mjlab.walk_stills \\
        <run dir> --project <project> --robot go2 [--every 100] [--tick 100]

One world at the nominal point (no actuator randomization), the env
built once and every checkpoint's actor loaded into it in turn; each
rollout is capped at `--tick` control ticks (2 s at 50 Hz by default)
and the pose at the last tick is rendered through the project's own
scene file (the Menagerie scene with its floor) from the chase camera
the press uses. Beside the frames: `stills.json` (iteration, what the
world did in those ticks) and `sheet.png`, every still on one page in
iteration order - survived and tracking in green, fallen in red,
survived but off-command in blue. The run's presenter puts the stills
on the iteration timeline, so the Studio scrubs through them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trainnr_mjlab.checkpoint_guard import require_tensor_only
from trainnr_mjlab.walks import (
    DEFAULT_ROBOT,
    ROBOTS,
    CameraFraming,
    use_project,
    walk_spec,
)

STILLS_DIR = "stills"
STILLS_FILE = "stills.json"
SHEET_FILE = "sheet.png"
CHECKPOINT = re.compile(r"model_(\d+)\.pt$")
WIDTH, HEIGHT = 480, 360
SHEET_COLUMNS = 8
SHEET_THUMB = (240, 180)
SHEET_LABEL_H = 22
SHEET_BORDER = 3
SURVIVED_TRACKING = (72, 190, 110)
FELL = (214, 72, 72)
SURVIVED_OFF_COMMAND = (88, 144, 230)
SHEET_GROUND = (24, 26, 31)
SHEET_TEXT = (220, 224, 230)


def checkpoints_every(run_dir: Path, every: int) -> list[tuple[int, Path]]:
    """`(iteration, file)` for every `model_<n>.pt` whose n is a multiple
    of `every`, plus the last checkpoint whatever its number, ascending."""
    found: dict[int, Path] = {}
    for file in run_dir.glob("model_*.pt"):
        match = CHECKPOINT.search(file.name)
        if match:
            found[int(match.group(1))] = file
    if not found:
        return []
    last = max(found)
    keep = sorted(n for n in found if n % every == 0 or n == last)
    return [(n, found[n]) for n in keep]


def outcome_colour(row: dict[str, Any]) -> tuple[int, int, int]:
    if row["fell"]:
        return FELL
    return SURVIVED_TRACKING if row["tracked"] else SURVIVED_OFF_COMMAND


def write_sheet(rows: list[dict[str, Any]], stills_dir: Path) -> Path:
    """Every still on one page, `SHEET_COLUMNS` wide, in iteration
    order, each framed in its outcome's colour under its iteration."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    columns = min(SHEET_COLUMNS, max(1, len(rows)))
    lines = (len(rows) + columns - 1) // columns
    w, h = SHEET_THUMB
    cell_h = h + SHEET_LABEL_H
    page = Image.new("RGB", (columns * w, lines * cell_h), SHEET_GROUND)
    draw = ImageDraw.Draw(page)
    for i, row in enumerate(rows):
        x, y = (i % columns) * w, (i // columns) * cell_h
        colour = outcome_colour(row)
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=colour, width=SHEET_BORDER)
        thumb = Image.open(stills_dir / row["file"]).convert("RGB")
        thumb.thumbnail((w - 2 * SHEET_BORDER, h - 2 * SHEET_BORDER))
        page.paste(thumb, (x + SHEET_BORDER, y + SHEET_BORDER))
        draw.text((x + 6, y + h + 4), f"iteration {row['iteration']}", fill=SHEET_TEXT)
    out = stills_dir / SHEET_FILE
    page.save(out)
    return out


def scene_renderer(scene_xml: Path, chase: CameraFraming) -> tuple[Any, Any, Any, Any]:
    """The robot's scene and a renderer framing it the way the walk's own
    chase camera does (`walks.CameraFraming`, declared by the walk)."""
    import mujoco  # noqa: PLC0415

    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.visual.global_.offwidth = WIDTH
    spec.visual.global_.offheight = HEIGHT
    model = spec.compile()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = chase.distance
    camera.elevation = chase.elevation
    camera.azimuth = chase.azimuth
    return model, data, mujoco.Renderer(model, height=HEIGHT, width=WIDTH), camera


def render_pose(scene: tuple[Any, Any, Any, Any], qpos: Any, out: Path) -> None:
    import imageio.v3 as iio  # noqa: PLC0415
    import mujoco  # noqa: PLC0415

    model, data, renderer, camera = scene
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    camera.lookat[:] = data.xpos[1]  # the base, the first body after the world
    renderer.update_scene(data, camera=camera)
    iio.imwrite(out, renderer.render())


def _scene_file(robot: str) -> Path:
    from trainnr.bundles.locate import find_bundle  # noqa: PLC0415

    folder = find_bundle(robot)
    if folder is None:
        raise FileNotFoundError(f"no robot bundle named {robot!r}")
    for name in (f"scene_{robot}.xml", f"{robot}.xml"):
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(f"{folder}: no scene file for {robot}")


def _policy_loader(spec: Any, device: str) -> tuple[Any, Any]:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415

    cfg, _ = spec.env_cfg(dr_span=None, pin_scale=None)
    cfg.scene.num_envs = 1
    agent = spec.agent(1)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=agent.clip_actions
    )
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device=device)

    def load(checkpoint: Path) -> Any:
        require_tensor_only(checkpoint)  # before any unpickling
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
        )
        return runner.get_inference_policy(device=device)

    return env, load


def write_stills(  # noqa: PLR0913 - the tool's knobs, each named
    run_dir: Path,
    *,
    robot: str,
    every: int,
    tick: int,
    seed: int,
    device: str,
    err_ratio_bound: float,
) -> list[dict[str, Any]]:
    """Render one still per selected checkpoint; the rows written to
    `stills.json`. A still already on disk for an iteration is kept."""
    import torch  # noqa: PLC0415

    from trainnr_mjlab.walk_verdict import rollout_episodes  # noqa: PLC0415

    selected = checkpoints_every(run_dir, every)
    if not selected:
        raise FileNotFoundError(f"{run_dir}: no model_*.pt checkpoint")
    stills_dir = run_dir / STILLS_DIR
    stills_dir.mkdir(exist_ok=True)
    record = stills_dir / STILLS_FILE
    rows: list[dict[str, Any]] = (
        json.loads(record.read_text()) if record.is_file() else []
    )
    done = {row["iteration"] for row in rows if (stills_dir / row["file"]).is_file()}
    rows = [row for row in rows if row["iteration"] in done]
    todo = [(n, f) for n, f in selected if n not in done]
    if todo:
        env, load = _policy_loader(walk_spec(robot), device)
        scene = scene_renderer(_scene_file(robot), walk_spec(robot).chase)
        for n, checkpoint in todo:
            torch.manual_seed(seed)
            env.unwrapped.seed(seed)
            walk = rollout_episodes(
                env, load(checkpoint), 1, capture=True, max_ticks=tick
            )[0]
            o = walk.outcome
            file = f"model_{n}.png"
            render_pose(
                scene, walk.qpos[min(tick, len(walk.qpos)) - 1], stills_dir / file
            )
            err_ratio = o.err_ratio  # the certificate's own rule (evaluate.tracking)
            rows.append(
                {
                    "iteration": n,
                    "checkpoint": checkpoint.name,
                    "file": file,
                    "tick": min(tick, o.steps),
                    "fell": o.fell,
                    "tracked": o.success,
                    "err_ratio": round(err_ratio, 4),
                    "seed": seed,
                }
            )
            state = "fell" if o.fell else "up"
            print(f"model_{n}: {state} at tick {o.steps}, err ratio {err_ratio:.2f}")
        scene[2].close()
    rows.sort(key=lambda r: r["iteration"])
    record.write_text(json.dumps(rows, indent=1) + "\n")
    write_sheet(rows, stills_dir)
    return rows


def main() -> int:
    from trainnr_mjlab.walk_verdict import ERR_RATIO_BOUND  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=ROBOTS)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument(
        "--tick", type=int, default=100, help="control ticks rolled out (50 Hz)"
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--device", default=None, help="cuda:0 or cpu; found when unset"
    )
    args = parser.parse_args()
    use_project(args.project)
    import warp as wp  # noqa: PLC0415

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    rows = write_stills(
        args.run_dir.resolve(),
        robot=args.robot,
        every=args.every,
        tick=args.tick,
        seed=args.seed,
        device=device,
        err_ratio_bound=ERR_RATIO_BOUND,
    )
    up = sum(1 for r in rows if not r["fell"])
    where = args.run_dir / STILLS_DIR
    print(f"{len(rows)} stills -> {where} ({up} up at tick {args.tick}); {SHEET_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
