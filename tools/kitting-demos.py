"""Generate scripted kitting demonstrations — T5's data source.

    cd pipeline && uv run --env-file wsl.env --extra sim \\
        python ../tools/kitting-demos.py \\
        [episodes] [out] [--frame-every 1] [--dr-span 0.30] [--seed S] \\
        [--first-episode K] [--shards N] [--parallel M]

(On WSL the offscreen renderer reaches the GPU only through the variables
in pipeline/wsl.env — Mesa's EGL default is llvmpipe at ~300 ms per frame,
which turned a 1 s scripted episode into a 460 s one on 2026-08-26.)

The generator itself is `rq_pipeline.collect.kitting_demos.generate_demos`
(what it draws, keeps and writes is documented and tested there); this
is its command line. The WSL box's train venv converts the batch to a
LeRobot dataset for T5 training; this side stays torch-free.
"""

import argparse
import sys
from pathlib import Path

from _lab import PREVIEW_EVERY_TICKS, bootstrap

bootstrap()

from rq_pipeline.collect.kitting_demos import DR_SPAN, generate_demos  # noqa: E402
from rq_pipeline.collect.press_feed import PRESS_STREAM, StudioPressFeed  # noqa: E402
from rq_pipeline.collect.shards import (  # noqa: E402
    ShardSpec,
    plan_shards,
    record_shard,
    run_sharded_tool,
)
from rq_pipeline.viz import viewer_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("episodes", nargs="?", type=int, default=10)
    parser.add_argument("out", nargs="?", type=Path, default=Path("runs/kitting-demos"))
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="give up after this many draws (default: 20 per wanted episode)",
    )
    parser.add_argument(
        "--frame-every",
        type=int,
        default=PREVIEW_EVERY_TICKS,
        help="control ticks between saved frames: 5 = 10 Hz previews, 1 = 50 Hz,"
        " the rate the harness's vision rollout observes at (T5 training data)",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--first-episode",
        type=int,
        default=0,
        help="number kept episodes from here: N generators with disjoint ranges"
        " and seeds fill ONE batch directory in parallel",
    )
    parser.add_argument(
        "--no-studio",
        action="store_true",
        help="do not stream this run to the Studio (docs/66 §0 streams by default)",
    )
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
        "--dr-span",
        type=float,
        default=DR_SPAN,
        help="domain-randomisation half-width around the bundle's values "
        "(0.30 = +-30%%)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    # docs/66 §0: everything that happens streams to the Studio -- the
    # press run is watchable by default, opted out per run.
    if args.shards > 1:
        plan = plan_shards(args.episodes, args.shards, args.seed, args.first_episode)
        code = run_sharded_tool(
            sys.argv, args.out, plan, parallel=args.parallel, say=say
        )
        sys.exit(code)
    name = (
        args.out.name
        if args.shard_index is None
        else f"{args.out.name}-s{args.shard_index}"
    )
    feed = (
        None
        if args.no_studio
        else StudioPressFeed.connect(
            name, say=say, file=viewer_file(args.out, PRESS_STREAM)
        )
    )
    batch = generate_demos(
        args.out,
        episodes=args.episodes,
        seed=args.seed,
        dr_span=args.dr_span,
        frame_every=args.frame_every,
        first_episode=args.first_episode,
        max_attempts=args.max_attempts,
        feed=feed,
        say=say,
    )
    if args.shard_index is not None:
        spec = ShardSpec(args.shard_index, args.first_episode, args.episodes, args.seed)
        record_shard(args.out, spec, batch)
    if not batch.complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
