"""Accept or reject a task: the scripted expert must pass its referee on
every paired trial, and the do-nothing floor must pass none.

    cd pipeline && uv run --extra sim python ../tools/accept-task.py kitting
    cd pipeline && uv run --extra sim python ../tools/accept-task.py kitting \\
        --tray-y 0.9            # a variant: the tray out of reach -> REJECTED

Prints the verdict, the counts, the expert's funnel and every reason;
exits 1 on a rejection so a generator loop can read it. Records go to
`--record` as the same JSONL rows every evaluation writes.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.envs.robotiq import bundle_source  # noqa: E402
from rq_pipeline.tasks.acceptance import accept  # noqa: E402
from rq_pipeline.tasks.aloha2 import (  # noqa: E402
    KITTING,
    KITTING_SPEC,
    build_kitting,
    scripted_kitting_episode,
)

# The tasks that have an expert to review them, by name. A task without
# one cannot be accepted here: acceptance IS the expert's verdict.
EXPERTS = {KITTING: (build_kitting, scripted_kitting_episode)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("task", choices=sorted(EXPERTS))
    parser.add_argument(
        "--record", type=Path, default=None, help="append the rows here"
    )
    parser.add_argument(
        "--tray-y", type=float, default=None, help="kitting variant: move the tray"
    )
    parser.add_argument(
        "--in-slot-xy",
        type=float,
        default=None,
        help="kitting variant: the referee's radius",
    )
    args = parser.parse_args()
    build, expert = EXPERTS[args.task]
    spec = KITTING_SPEC
    if args.tray_y is not None:
        spec = replace(spec, tray_center=(spec.tray_center[0], args.tray_y))
    if args.in_slot_xy is not None:
        spec = replace(spec, in_slot_xy_m=args.in_slot_xy)
    task = build(spec=spec)

    def run_expert(model, initial):
        return expert(model, initial, spec=spec)

    verdict = accept(
        task, run_expert, source=bundle_source(task.bundle_dir), record_to=args.record
    )
    print(verdict, flush=True)
    sys.exit(0 if verdict.accepted else 1)


if __name__ == "__main__":
    main()
