"""A project's record that it uses a task — by id AND content stamp.

Seven tasks live in Python (`tasks/registry.py`), and a project that
trains against one has nothing on disk saying so; its loop map would
show "task declared" dark under a trained policy. This file is the
proof: `tasks/<task_id>/task.json` naming the registered task and the
content stamp it had when used. If the task's constants change tomorrow,
the stamp changes, and every project that used the old one still says
exactly which version — the same lineage the dataset's provenance
sidecar carries for its bundle.

A task authored as a spec overlay (docs/76 §7) writes the same file
with `kind: "declared"` and the overlay fields; a registered task
writes `kind: "registered"`. The indexer reads either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rq_pipeline.project.kinds import TASK_FILE
from rq_pipeline.project.locate import Project, plain_name

REGISTERED = "registered"
DECLARED = "declared"


def write_task_reference(  # noqa: PLR0913 - one record, each field named
    project: Project,
    task_id: str,
    stamp: str | None,
    *,
    kind: str = REGISTERED,
    spec: dict[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    """Record a task in the project. `stamp` is the task's content stamp
    (`Task.stamp`), or None when the task carries no spec — recorded as
    `unstamped` rather than invented. A declared variant carries its
    `name` (the folder) and its full `spec`. Idempotent: the same task
    and stamp write the same bytes."""
    if "/" not in task_id:
        raise ValueError(
            f"task ids are namespaced, like robotiq/kitting; got {task_id!r}"
        )
    folder = project.folder("tasks") / (name or task_id.replace("/", "--"))
    folder.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "task_id": task_id,
        "stamp": stamp if stamp is not None else "unstamped",
        "kind": kind,
    }
    if name is not None:
        record["name"] = name
    if spec is not None:
        record["spec"] = spec
    out = folder / TASK_FILE
    out.write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def declare_task(
    project: Project, task_id: str, name: str, overlay: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Declare a variant into the project under `name`: built for real
    (the scene compiles or the declaration fails), stamped by content,
    written as a task reference of kind `declared` with its full spec."""
    from dataclasses import asdict  # noqa: PLC0415

    from rq_pipeline.tasks.overlay import build_variant, jsonable  # noqa: PLC0415
    from rq_pipeline.tasks.registry import resolve  # noqa: PLC0415

    plain_name(name, "task name")
    folder = project.folder("tasks") / name
    if folder.exists():
        raise FileExistsError(f"{name!r} is already a task in this project")
    task, spec = build_variant(task_id, overlay)
    full_id = resolve(task_id).task_id
    path = write_task_reference(
        project,
        full_id,
        task.stamp,
        kind=DECLARED,
        spec=jsonable(asdict(spec)),
        name=name,
    )
    return {
        "name": name,
        "task_id": full_id,
        "stamp": task.stamp,
        "path": str(path.parent.relative_to(project.root)),
        "overlay": jsonable(dict(overlay or {})),
    }
