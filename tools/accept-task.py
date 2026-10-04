"""Accept or reject a task: the scripted expert must pass its referee on
every paired trial, and the do-nothing floor must pass none.

    cd trainnr && uv run --extra sim python ../tools/accept-task.py kitting
    cd trainnr && uv run --extra sim python ../tools/accept-task.py kitting \\
        --overlay '{"tray_center": [0.0, 0.9]}'   # the tray out of reach -> REJECTED
    cd trainnr && uv run --extra sim python ../tools/accept-task.py \\
        --project ../projects/aloha-kitting --name tray-far   # a declared task

Prints the verdict, the counts, the expert's funnel and every reason;
exits 1 on a rejection so a generator loop can read it. Records go to
`--record` as the same JSONL rows every evaluation writes. With
`--project` and `--name`, the task is the project's declared variant
(its `task.json`, overlay included), the records land beside it and the
verdict is written as `acceptance.json` for the index and the Studio.
"""

import argparse
import ast
import contextlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _lab import bootstrap

bootstrap()

import mujoco  # noqa: E402
from trainnr.envs.gymnasium_env import bundle_source  # noqa: E402
from trainnr.mcp_actions import TRAINNR_MJLAB_DIR, walk_train_argv  # noqa: E402
from trainnr.mcp_jobs import prepare_uv  # noqa: E402
from trainnr.physics.backend import instrument_stamp  # noqa: E402
from trainnr.project import index_project, write_index  # noqa: E402
from trainnr.project.kinds import (  # noqa: E402
    ACCEPTANCE_FILE,
    ACCEPTANCE_SCHEMA,
)
from trainnr.project.locate import Project  # noqa: E402
from trainnr.project.task_ref import (  # noqa: E402
    TaskReference,
    read_task_reference,
)
from trainnr.tasks.acceptance import accept  # noqa: E402
from trainnr.tasks.experts import expert_for, experts  # noqa: E402
from trainnr.tasks.overlay import build_from_reference, build_variant  # noqa: E402
from trainnr.tasks.walks import (  # noqa: E402
    bundle_walk_shape_missing,
    walk_robot,
    walk_shape_sentence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    reviewable = sorted(entry.name for entry in experts().values())
    parser.add_argument("task", nargs="?", choices=reviewable)
    parser.add_argument("--project", type=Path, default=None, help="a project root")
    parser.add_argument("--name", default=None, help="a declared task in the project")
    parser.add_argument(
        "--record", type=Path, default=None, help="append the rows here"
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="a variant: JSON with the spec fields you change "
        "(list_task_families lists them)",
    )
    args = parser.parse_args()
    if args.project is not None or args.name is not None:
        if args.project is None or args.name is None:
            parser.error("--project and --name go together")
        sys.exit(0 if review_declared(args.project, args.name) else 1)
    if args.task is None:
        parser.error("name a task, or a --project and --name")
    overlay = json.loads(args.overlay) if args.overlay else None
    task, spec = build_variant(args.task, overlay)
    expert = expert_for(args.task)

    def run_expert(model: Any, initial: Any) -> Any:
        return expert(model, initial, spec=spec)

    verdict = accept(
        task, run_expert, source=bundle_source(task.bundle_dir), record_to=args.record
    )
    print(verdict, flush=True)
    sys.exit(0 if verdict.accepted else 1)


SMOKE_ENVS = 2
SMOKE_ITERATIONS = 2
SMOKE_AGENT = "smoke"
SMOKE_DONE_MARK = "[smoke] done"  # the trainer's last line (trainnr_mjlab.walk_train)
ACCEPTANCE_LOG = "acceptance.log"
Runner = Callable[..., subprocess.CompletedProcess[Any]]


def review_walk(  # noqa: PLR0913 - the review's inputs and its two seams
    project: Project,
    folder: Path,
    ref: TaskReference,
    robot: str,
    *,
    run: Runner = subprocess.run,
    prepare: Callable[[Sequence[str], Path], None] = prepare_uv,
    shape_missing: Callable[[str], list[str]] = bundle_walk_shape_missing,
) -> bool:
    """A walk's acceptance is learnability: the environment builds from
    the project's robot and a few PPO iterations run — trainnr_mjlab's smoke,
    in its own venv, with the declared span, through the SAME command
    line the train door spawns (launch environment included). The
    identity it prints is the record. First, the robot's shape: a bundle
    without the parts the trainer reads by name (`tasks.walks.WALK_SHAPES`)
    is refused in one sentence, not after a traceback in the log
    (stranger test 2026-10-03)."""
    missing = shape_missing(robot)
    if missing:
        reason = walk_shape_sentence(robot, missing)
        _write_walk_verdict(
            project,
            folder,
            ref,
            accepted=False,
            gate="the robot's shape, before the learnability smoke",
            reasons=[reason],
            identity={},
            log=None,
        )
        print(f"REJECTED: {ref.stamp} — {reason}", flush=True)
        return False
    argv = walk_train_argv(
        agent=SMOKE_AGENT,
        robot=robot,
        project=str(project.root),
        envs=SMOKE_ENVS,
        iterations=SMOKE_ITERATIONS,
        dr_span=ref.dr_span if ref.dr_span is not None else 0.0,
        recorder=False,
    )
    log_path = folder / ACCEPTANCE_LOG
    # the door's line is `uv run --no-sync`: its environment is made ready
    # here, as the job runner does for a door (a failure is refused by name)
    prepare(argv, TRAINNR_MJLAB_DIR)
    with log_path.open("w") as log:
        result = run(
            argv,
            cwd=TRAINNR_MJLAB_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    lines = log_path.read_text(errors="replace").splitlines()
    identity: dict = {}
    for line in lines:
        if "identity:" in line:
            with contextlib.suppress(ValueError, SyntaxError):
                identity = ast.literal_eval(line.split("identity:", 1)[1].strip())
    accepted = result.returncode == 0 and any(SMOKE_DONE_MARK in line for line in lines)
    reasons = (
        []
        if accepted
        else [f"the learnability smoke exited {result.returncode}; see acceptance.log"]
    )
    _write_walk_verdict(
        project,
        folder,
        ref,
        accepted=accepted,
        gate=(
            f"learnability smoke: {SMOKE_ENVS} environments, "
            f"{SMOKE_ITERATIONS} PPO iterations"
        ),
        reasons=reasons,
        identity=identity,
        log=log_path.name,
    )
    return accepted


def _write_walk_verdict(  # noqa: PLR0913 - the verdict's fields, each named
    project: Project,
    folder: Path,
    ref: TaskReference,
    *,
    accepted: bool,
    gate: str,
    reasons: list[str],
    identity: dict,
    log: str | None,
) -> None:
    """The walk verdict on disk — atomic, like the index, and the index
    rewritten so the Studio sees it the moment the job ends."""
    record = {
        "schema": ACCEPTANCE_SCHEMA,
        "task": ref.stamp,
        "accepted": accepted,
        "gate": gate,
        "reasons": reasons,
        "refusals": [],
        "identity": identity,
        "instrument": identity.get("actuator", "unrecorded"),
        "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "log": log,
    }
    if log is not None:
        word = "ACCEPTED" if accepted else "REJECTED"
        print(f"{word}: {ref.stamp} — {gate}", flush=True)
    staging = folder / (ACCEPTANCE_FILE + ".tmp")
    staging.write_text(json.dumps(record, indent=1) + "\n")
    staging.replace(folder / ACCEPTANCE_FILE)
    write_index(project, index_project(project))


def review_declared(project_root: Path, name: str) -> bool:
    """Review the project's declared task `name`; write its verdict."""
    project = Project(project_root.resolve()).use()
    try:
        ref = read_task_reference(project, name)
    except (FileNotFoundError, ValueError) as why:
        raise SystemExit(str(why)) from why
    robot = walk_robot(ref.task_id)
    if robot is not None:
        return review_walk(project, ref.folder, ref, robot)
    task = build_from_reference({"task_id": ref.task_id, "spec": ref.spec})
    expert = expert_for(ref.task_id)
    spec = task.task_spec
    folder = ref.folder

    def run_expert(model: Any, initial: Any) -> Any:
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
