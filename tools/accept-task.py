"""Accept or reject a task: the scripted expert must pass its referee on
every paired trial, and the do-nothing floor must pass none.

    cd pipeline && uv run --extra sim python ../tools/accept-task.py kitting
    cd pipeline && uv run --extra sim python ../tools/accept-task.py kitting \\
        --tray-y 0.9            # a variant: the tray out of reach -> REJECTED
    cd pipeline && uv run --extra sim python ../tools/accept-task.py \\
        --project ../projects/aloha-kitting --name tray-far   # a declared task

Prints the verdict, the counts, the expert's funnel and every reason;
exits 1 on a rejection so a generator loop can read it. Records go to
`--record` as the same JSONL rows every evaluation writes. With
`--project` and `--name`, the task is the project's declared variant
(its `task.json`, overlay included), the records land beside it and the
verdict is written as `acceptance.json` for the index and the Studio.
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
from rq_pipeline.envs.robotiq import bundle_source  # noqa: E402
from rq_pipeline.physics.backend import instrument_stamp  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.kinds import (  # noqa: E402
    ACCEPTANCE_FILE,
    ACCEPTANCE_SCHEMA,
    TASK_FILE,
)
from rq_pipeline.project.locate import Project  # noqa: E402
from rq_pipeline.tasks.acceptance import accept  # noqa: E402
from rq_pipeline.tasks.aloha2 import KITTING_SPEC  # noqa: E402
from rq_pipeline.tasks.experts import EXPERTS, expert_for  # noqa: E402
from rq_pipeline.tasks.overlay import build_from_reference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("task", nargs="?", choices=sorted(EXPERTS))
    parser.add_argument("--project", type=Path, default=None, help="a project root")
    parser.add_argument("--name", default=None, help="a declared task in the project")
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
    if args.project is not None or args.name is not None:
        if args.project is None or args.name is None:
            parser.error("--project and --name go together")
        sys.exit(0 if review_declared(args.project, args.name) else 1)
    if args.task is None:
        parser.error("name a task, or a --project and --name")
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


def review_declared(project_root: Path, name: str) -> bool:
    """Review the project's declared task `name`; write its verdict."""
    project = Project(project_root.resolve())
    folder = project.folder("tasks") / name
    ref_path = folder / TASK_FILE
    if not ref_path.is_file():
        raise SystemExit(f"no task {name!r} in {project.root} (no {ref_path})")
    ref = json.loads(ref_path.read_text())
    task = build_from_reference(ref)
    expert = expert_for(ref["task_id"])
    spec = task.task_spec

    def run_expert(model, initial):
        return expert(model, initial, spec=spec)

    verdict = accept(
        task,
        run_expert,
        source=bundle_source(task.bundle_dir),
        record_to=folder / "acceptance-records.jsonl",
    )
    print(verdict, flush=True)
    record = {
        "schema": ACCEPTANCE_SCHEMA,
        "task": verdict.task,
        "accepted": verdict.accepted,
        "reasons": list(verdict.reasons),
        "refusals": list(verdict.refusals),
        "expert_successes": verdict.expert_successes,
        "floor_successes": verdict.floor_successes,
        "trials": verdict.trials,
        "funnel": {k: list(v) for k, v in verdict.funnel.items()},
        "milestones": [name for name, _ in task.protocol.milestones],
        "instrument": instrument_stamp("mujoco", mujoco.__version__),
        "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "records": "acceptance-records.jsonl",
    }
    # Atomic, like the index: a reader never sees half a verdict.
    staging = folder / (ACCEPTANCE_FILE + ".tmp")
    staging.write_text(json.dumps(record, indent=1) + "\n")
    staging.replace(folder / ACCEPTANCE_FILE)
    # The verdict is part of the task's record: the index and the Studio
    # see it the moment the job ends.
    write_index(project, index_project(project))
    return verdict.accepted


if __name__ == "__main__":
    main()
