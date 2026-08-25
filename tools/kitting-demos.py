"""Generate scripted kitting demonstrations — T5's data source.

    cd pipeline && uv run --extra sim python ../tools/kitting-demos.py \
        [episodes] [out] [--max-attempts N]

Each episode: spawn both parts uniformly in the PROVEN band (the front
half of the spawn box — the far band's closing-plane edge is a known
open item, see test_aloha2_kitting), randomize joint damping and
actuator gains within ±30% of the bundle's values (domain
randomization centred on the identified parameters — the show-many
pattern), run the scripted choreography, and keep the episode ONLY if
the task's own referee scores it a success — the same judge policies
will face. Saves per episode: states/actions/top-camera JPEG frames +
a manifest with the draw, the DR scales, and the verdict. The WSL
box's train venv converts the batch to a LeRobot dataset for T5
training; this side stays torch-free.
"""

import argparse
import json
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: E402
from rq_pipeline.tasks.aloha2 import (  # noqa: E402
    PART_SPAWN,
    PART_STATE_SLICE,
    KittingStats,
    build_kitting,
    scripted_kitting_episode,
)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("episodes", nargs="?", type=int, default=10)
parser.add_argument("out", nargs="?", type=Path, default=Path("runs/kitting-demos"))
parser.add_argument(
    "--max-attempts",
    type=int,
    default=None,
    help="give up after this many draws (default: 20 * episodes) — the DR"
    " sampler can in principle draw only unreachable dynamics",
)
args = parser.parse_args()
EPISODES, OUT = args.episodes, args.out
MAX_ATTEMPTS = args.max_attempts if args.max_attempts is not None else 20 * EPISODES
DR_SPAN = 0.30  # +-30% around the bundle's identified/nominal values
FRAME_EVERY = 5  # control ticks between saved frames (10 Hz at 50 Hz control)
SEED = 20260826

task = build_kitting()
rng = np.random.default_rng(SEED)
OUT.mkdir(parents=True, exist_ok=True)

kept = 0
attempt = 0
while kept < EPISODES:
    if attempt >= MAX_ATTEMPTS:
        print(
            f"gave up after {attempt} attempts: kept {kept}/{EPISODES} -> {OUT}",
            file=sys.stderr,
        )
        sys.exit(1)
    attempt += 1
    # Domain randomization: recompile the scene with scaled dynamics.
    spec = build_kitting().spec
    damping_scale = float(1.0 + rng.uniform(-DR_SPAN, DR_SPAN))
    gain_scale = float(1.0 + rng.uniform(-DR_SPAN, DR_SPAN))
    for joint in spec.joints:
        if joint.name.startswith(("left/", "right/")):
            joint.damping[0] = joint.damping[0] * damping_scale
    for actuator in spec.actuators:
        actuator.gainprm[0] = actuator.gainprm[0] * gain_scale
    model = spec.compile()

    initial = keyframe_state(model, "neutral_pose")
    draws = {}
    for arm in ("right", "left"):
        (x_low, x_high), (y_low, y_high) = PART_SPAWN[arm]
        # The proven band: the front HALF of the spawn box in y.
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_low + 0.5 * (y_high - y_low)))
        part = PART_STATE_SLICE[arm]
        initial[part.start] = x
        initial[part.start + 1] = y
        draws[arm] = (x, y)

    renderer = mujoco.Renderer(model, height=480, width=640)
    episode_dir = OUT / f"episode_{kept:04d}"
    frames_dir = episode_dir / "frames"
    frames: list[tuple[int, Path]] = []

    # renderer/frames bound as defaults — a late-binding closure would see
    # only the LAST attempt's objects (ruff B023).
    def snap(
        tick: int, live: mujoco.MjData, renderer=renderer, frames=frames
    ) -> None:
        if tick % (FRAME_EVERY * 10) != 0:  # tick is a physics step here
            return
        renderer.update_scene(live, camera="top")
        frames.append((tick, renderer.render().copy()))

    stats = KittingStats()
    try:
        states, sensors, actions = scripted_kitting_episode(
            model, initial, on_control=snap, stats=stats
        )
        succeeded = task.protocol.success(states, sensors)
    except RuntimeError as error:
        # An unreachable draw under this DR sample is a discard, not a
        # crash — the library keeps its honesty (it raises), the
        # generator keeps its throughput (it filters).
        print(f"attempt {attempt}: discard (IK: {error})", file=sys.stderr)
        succeeded = False
    finally:
        renderer.close()
    print(
        f"attempt {attempt}: {'KEEP' if succeeded else 'discard'} "
        f"(damping x{damping_scale:.2f}, gain x{gain_scale:.2f}, "
        f"retries {len(stats.retries)})",
        file=sys.stderr,
    )
    if not succeeded:
        continue

    episode_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        episode_dir / "trajectory.npz",
        states=states.astype(np.float32),
        sensors=sensors.astype(np.float32),
        actions=actions.astype(np.float32),
    )
    for tick, frame in frames:
        Image.fromarray(frame).save(
            frames_dir / f"{tick:06d}.jpg", quality=85
        )
    (episode_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "attempt": attempt,
                "draws": draws,
                "damping_scale": damping_scale,
                "gain_scale": gain_scale,
                "retries": stats.retries,
                "control_hz": 50,
                "frame_every_control_ticks": FRAME_EVERY,
                "action_semantics": "14 commanded joint positions (ctrl order)",
                "verdict": "success (task referee)",
            },
            indent=1,
        )
    )
    kept += 1

print(f"kept {kept}/{attempt} episodes -> {OUT}", file=sys.stderr)
