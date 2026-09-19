"""Press planner demonstrations on an SO-101 task (docs/66 D3).

    cd pipeline && uv run --env-file wsl.env --extra sim --extra viz \\
        python ../tools/planner-demos.py TASK [out] [--episodes 8] [--seed S] \\
        [--frame-every 5] [--dr-span 0.0] [--first-episode K] [--no-studio] \\
        [--shards N] [--parallel M]

The planner (`rq_pipeline.collect.choreography.PickPlacePlanner`) reads
the object and the goal off each seated scene, writes the beats, and
executes them by chained IK; the task's own referee keeps or discards.
Dynamics are drawn per episode from ±`--dr-span` around nominal (0 =
the nominal condition). Streams to the Studio by default (docs/66 §0).
"""

import argparse
import sys
from pathlib import Path

from _lab import PREVIEW_EVERY_TICKS, bootstrap

bootstrap()

from rq_pipeline.collect.planner_demos import generate_planned_demos  # noqa: E402
from rq_pipeline.collect.press_feed import PRESS_STREAM, StudioPressFeed  # noqa: E402
from rq_pipeline.collect.shards import (  # noqa: E402
    ShardSpec,
    plan_shards,
    record_shard,
    run_sharded_tool,
)
from rq_pipeline.tasks.registry import resolve  # noqa: E402
from rq_pipeline.tasks.so101 import planner_rig  # noqa: E402
from rq_pipeline.viz import viewer_file  # noqa: E402

DEFAULT_OUT = Path("runs/planner-demos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "task", help="a registered SO-101 task: lift, block_stack, tool_insert"
    )
    parser.add_argument("out", nargs="?", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--frame-every",
        type=int,
        default=PREVIEW_EVERY_TICKS,
        help="control ticks between saved frames (5 = 10 Hz previews, 1 = 50 Hz)",
    )
    parser.add_argument(
        "--dr-span",
        type=float,
        default=0.0,
        help="servo damping/gain half-width around nominal (0 = nominal)",
    )
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="press in N runs with disjoint ranges and seeds (docs/66 D4), merged",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="shards at a time (3+ EGL contexts livelock on WSL, measured 2026-08-28)",
    )
    parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-studio",
        action="store_true",
        help="do not stream this run to the Studio (docs/66 §0 streams by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or DEFAULT_OUT / args.task
    span = args.dr_span
    basis = (
        "nominal servo dynamics"
        if span == 0
        else f"guessed span ±{span:g} around nominal (folklore DR)"
    )

    def say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    if args.shards > 1:
        plan = plan_shards(args.episodes, args.shards, args.seed, args.first_episode)
        sys.exit(run_sharded_tool(sys.argv, out, plan, parallel=args.parallel, say=say))
    run_name = (
        out.name if args.shard_index is None else f"{out.name}-s{args.shard_index}"
    )
    feed = (
        None
        if args.no_studio
        else StudioPressFeed.connect(
            run_name, say=say, file=viewer_file(args.out, PRESS_STREAM)
        )
    )
    batch = generate_planned_demos(
        out,
        task_factory=resolve(args.task).build,
        rig=planner_rig(args.task),
        dr={"damping": (1 - span, 1 + span), "gain": (1 - span, 1 + span)},
        basis=basis,
        episodes=args.episodes,
        seed=args.seed,
        frame_every=args.frame_every,
        first_episode=args.first_episode,
        feed=feed,
        say=say,
    )
    if args.shard_index is not None:
        spec = ShardSpec(args.shard_index, args.first_episode, args.episodes, args.seed)
        record_shard(out, spec, batch)
    if not batch.complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
