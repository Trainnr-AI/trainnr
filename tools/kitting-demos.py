"""Generate scripted kitting demonstrations — T5's data source.

    cd pipeline && uv run --env-file wsl.env --extra sim \
        python ../tools/kitting-demos.py \
        [episodes] [out] [--frame-every 1] [--dr-span 0.10] [--seed S]

(On WSL the offscreen renderer reaches the GPU only through the variables
in pipeline/wsl.env — Mesa's EGL default is llvmpipe at ~300 ms per frame,
which turned a 1 s scripted episode into a 460 s one on 2026-08-26.)

Each episode: spawn both parts uniformly over the task's DECLARED band
(`KittingSpec.part_spawn`, the same band the harness evaluates on; until
2026-08-27 this drew the front half only, a workaround for the near-base
corners the expert could not grasp — fixed, docs/07), randomize joint damping and
actuator gains within ±30% of the bundle's values (domain
randomization centred on the identified parameters — the show-many
pattern), run the scripted choreography, and keep the episode ONLY if
the task's own referee scores it a success — the same judge policies
will face. Saves per episode: states/actions/top-camera JPEG frames +
a manifest with the draw, the DR scales, the verdict and the expert's
own stamp (`expert_stamp`, the choreography named by content). The WSL
box's train venv converts the batch to a LeRobot dataset for T5
training; this side stays torch-free.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from rq_pipeline.collect.kitting_export import Manifest, write_episode  # noqa: E402
from rq_pipeline.physics.mujoco_backend import keyframe_state  # noqa: E402
from rq_pipeline.tasks.aloha2 import (  # noqa: E402
    PART_STATE_SLICE,
    KittingStats,
    build_kitting,
    expert_stamp,
    scale_dynamics,
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
parser.add_argument(
    "--frame-every",
    type=int,
    default=5,
    help="control ticks between saved frames: 5 = 10 Hz previews, 1 = 50 Hz,"
    " the rate the harness's vision rollout observes at (T5 training data)",
)
parser.add_argument("--seed", type=int, default=20260826)
parser.add_argument(
    "--first-episode",
    type=int,
    default=0,
    help="number kept episodes from here: N generators with disjoint ranges"
    " and seeds fill ONE batch directory in parallel (the render is the"
    " cost — 95 s/episode measured 2026-08-27 — and the box has cores)",
)
parser.add_argument(
    "--dr-span",
    type=float,
    default=0.30,
    help="domain-randomisation half-width around the bundle's values (0.30 ="
    " +-30%%). Measured 2026-08-26 with the gain scaled correctly (both kp"
    " terms): the scripted expert keeps 8/10 at 0.10 and 10/10 at 0.30, no"
    " retries. (An earlier one-sided gain scale moved the setpoints and"
    " read as 'nominal only'; T5's first batch ran at 0 because of it.)",
)
args = parser.parse_args()
EPISODES, OUT = args.episodes, args.out
FIRST_EPISODE = args.first_episode
MAX_ATTEMPTS = args.max_attempts if args.max_attempts is not None else 20 * EPISODES
DR_SPAN = args.dr_span  # +-span around the bundle's identified/nominal values
FRAME_EVERY = args.frame_every
SEED = args.seed

task = build_kitting()
EXPERT = expert_stamp()
rng = np.random.default_rng(SEED)
print(f"expert {EXPERT}, task {task.stamp}", file=sys.stderr)
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
    # (`scale_dynamics` scales BOTH kp terms of a position servo; this
    # tool once scaled gainprm[0] alone, which moves the setpoint, not
    # the stiffness — the "expert only works at nominal" artifact.)
    spec = build_kitting().spec
    damping_scale = float(1.0 + rng.uniform(-DR_SPAN, DR_SPAN))
    gain_scale = float(1.0 + rng.uniform(-DR_SPAN, DR_SPAN))
    scale_dynamics(spec, damping_scale=damping_scale, gain_scale=gain_scale)
    model = spec.compile()

    initial = keyframe_state(model, task.protocol.home)
    draws = {}
    for arm in ("right", "left"):
        (x_low, x_high), (y_low, y_high) = task.task_spec.part_spawn[arm]
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))
        part = PART_STATE_SLICE[arm]
        initial[part.start] = x
        initial[part.start + 1] = y
        draws[arm] = (x, y)

    camera = task.cameras[0]
    renderer = mujoco.Renderer(model, height=camera.height, width=camera.width)
    frames: list[tuple[int, Any]] = []

    # renderer/frames bound as defaults — a late-binding closure would see
    # only the LAST attempt's objects (ruff B023).
    def snap(
        tick: int,
        live: mujoco.MjData,
        renderer=renderer,
        frames=frames,
        camera=camera,
    ) -> None:
        if tick % (FRAME_EVERY * task.protocol.control_interval) != 0:
            return  # tick is a physics step here
        renderer.update_scene(live, camera=camera.camera_name)
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

    write_episode(
        OUT,
        FIRST_EPISODE + kept,
        states=states,
        sensors=sensors,
        actions=actions,
        frames=frames,
        manifest=Manifest(
            seed=SEED,
            attempt=attempt,
            draws=draws,
            dr_span=DR_SPAN,
            damping_scale=damping_scale,
            gain_scale=gain_scale,
            retries=stats.retries,
            control_hz=task.control_hz,
            frame_every_control_ticks=FRAME_EVERY,
            expert=EXPERT,
        ),
    )
    kept += 1

print(
    f"kept {kept}/{attempt} episodes -> {OUT} "
    f"(episodes {FIRST_EPISODE}..{FIRST_EPISODE + kept - 1})",
    file=sys.stderr,
)
