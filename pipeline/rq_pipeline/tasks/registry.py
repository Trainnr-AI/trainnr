"""The task registry: tasks declare themselves; the env reads the list.

Until 2026-08-26 the generic env module imported every task builder by
name into a table — the generic layer depended on every concrete task,
and a third party could not add one without editing us. Now a builder
registers itself:

    @register("pour", rig="acme-arm")
    def build_pour(*, look: str = "acme") -> Task: ...

and an installed package advertises its module through the
`rq_pipeline.tasks` entry-point group, which `tasks()` imports on
demand. Task ids are `namespace/name` — gymnasium's own id grammar
minus the version — so the built-ins are `robotiq/kitting`,
`robotiq/reach`, …, and a stranger's `acme/pour` never collides.
Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rq_pipeline.plugins import load_group

BUILTIN_NAMESPACE = "robotiq"
ENTRY_POINT_GROUP = "rq_pipeline.tasks"
# The modules that register the built-in tasks; imported by `tasks()` so
# a checkout works before its entry points are installed.
BUILTIN_MODULES = ("rq_pipeline.tasks.aloha2", "rq_pipeline.tasks.so101")


@dataclass(frozen=True)
class TaskEntry:
    """A registered builder and the rig it composes."""

    task_id: str  # namespace/name
    rig: str  # the bundle family, e.g. "aloha2", "so101"
    build: Callable[..., Any]

    @property
    def name(self) -> str:
        return self.task_id.rpartition("/")[2]


_REGISTRY: dict[str, TaskEntry] = {}


def register(name: str, *, rig: str, namespace: str = BUILTIN_NAMESPACE):
    """Decorate a builder `build(**kwargs) -> Task`; refuses a duplicate id."""
    if "/" in name or "/" in namespace or not name or not namespace:
        raise ValueError(f"task ids are namespace/name, got {namespace!r}/{name!r}")
    task_id = f"{namespace}/{name}"

    def decorate(build: Callable[..., Any]) -> Callable[..., Any]:
        existing = _REGISTRY.get(task_id)
        if existing is not None and existing.build is not build:
            raise ValueError(f"task {task_id!r} registered twice")
        _REGISTRY[task_id] = TaskEntry(task_id=task_id, rig=rig, build=build)
        return build

    return decorate


def tasks() -> Mapping[str, TaskEntry]:
    """Every registered task: the built-ins plus installed plugins."""
    load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_REGISTRY)


def resolve(task: str) -> TaskEntry:
    """`robotiq/kitting` or, for a built-in, bare `kitting`."""
    known = tasks()
    if task in known:
        return known[task]
    builtin = f"{BUILTIN_NAMESPACE}/{task}"
    if builtin in known:
        return known[builtin]
    raise KeyError(f"no task {task!r}; the registry knows {sorted(known)}")
