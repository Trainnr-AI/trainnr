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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trainnr import safe_write
from trainnr.project.kinds import TASK_FILE
from trainnr.project.locate import Project, plain_name

REGISTERED = "registered"
DECLARED = "declared"
UNSTAMPED = "unstamped"
TASKS_FOLDER = "tasks"


@dataclass(frozen=True)
class TaskReference:
    """A project's `task.json`, read: the family id, the content stamp
    (or `unstamped`), the kind, the declared spec when there is one."""

    name: str  # the folder under tasks/
    task_id: str
    stamp: str
    kind: str
    folder: Path
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def dr_span(self) -> float | None:
        """A declared walk's randomization span, when the spec has one."""
        value = self.spec.get("dr_span")
        return float(value) if value is not None else None

    @property
    def fit(self) -> str | None:
        """A declared walk's joint fit (its stamp, or its recording's),
        when the spec names one (2026-09-25; trainnr_mjlab.fit_walk)."""
        value = self.spec.get(FIT_FIELD)
        return str(value) if value else None


FIT_FIELD = "fit"  # task.json's spec field naming the fit a walk trains under


def read_task_reference(project: Project, name: str) -> TaskReference:
    """The project's task `name`, or a refusal by name (FileNotFoundError)."""
    plain_name(name, "task name")
    folder = project.folder(TASKS_FOLDER) / name
    path = folder / TASK_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no task {name!r} in {project.root} (no {path})")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "task_id" not in raw:
        raise ValueError(f"{path}: no task_id recorded")
    return TaskReference(
        name=name,
        task_id=str(raw["task_id"]),
        stamp=str(raw.get("stamp", UNSTAMPED)),
        kind=str(raw.get("kind", REGISTERED)),
        folder=folder,
        spec=dict(raw.get("spec") or {}),
    )


def task_references(project: Project) -> list[TaskReference]:
    """Every task the project records, by folder name."""
    root = project.folder(TASKS_FOLDER)
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(p for p in root.iterdir() if (p / TASK_FILE).is_file()):
        found.append(read_task_reference(project, folder.name))
    return found


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
            f"task ids are namespaced, like trainnr/kitting; got {task_id!r}"
        )
    folder = project.folder(TASKS_FOLDER) / (name or task_id.replace("/", "--"))
    folder.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "task_id": task_id,
        "stamp": stamp if stamp is not None else UNSTAMPED,
        "kind": kind,
    }
    if name is not None:
        record["name"] = name
    if spec is not None:
        record["spec"] = spec
    out = folder / TASK_FILE
    safe_write.write_text(out, json.dumps(record, indent=1, sort_keys=True) + "\n")
    return out


def declare_task(
    project: Project, task_id: str, name: str, overlay: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Declare a variant into the project under `name`: built for real
    (the scene compiles or the declaration fails), stamped by content,
    written as a task reference of kind `declared` with its full spec."""
    from dataclasses import asdict  # noqa: PLC0415

    from trainnr.tasks.overlay import build_variant, jsonable  # noqa: PLC0415
    from trainnr.tasks.registry import resolve  # noqa: PLC0415

    plain_name(name, "task name")
    folder = project.folder(TASKS_FOLDER) / name
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
