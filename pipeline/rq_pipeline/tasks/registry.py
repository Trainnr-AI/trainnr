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
from dataclasses import dataclass, replace
from typing import Any

from rq_pipeline.plugins import load_group

BUILTIN_NAMESPACE = "robotiq"
ENTRY_POINT_GROUP = "rq_pipeline.tasks"
# The modules that register the built-in tasks; imported by `tasks()` so
# a checkout works before its entry points are installed.
BUILTIN_MODULES = (
    "rq_pipeline.tasks.aloha2",
    "rq_pipeline.tasks.so101",
    "rq_pipeline.tasks.walks",
)


Expert = Callable[..., Any]
"""A scripted policy that solves a task's family: `expert(model,
initial_state, *, spec=...) -> (episode, stats, ...)`, the callable the
acceptance critic runs on every paired trial (`tasks/acceptance.py`)."""


@dataclass(frozen=True)
class TaskEntry:
    """A registered builder and the rig it composes. `walk` marks a
    locomotion family (rq_mjlab builds it; the rig IS the walk robot);
    `expert` is the scripted policy that reviews the family, when one
    is registered (`register_expert`)."""

    task_id: str  # namespace/name
    rig: str  # the bundle family, e.g. "aloha2", "so101"
    build: Callable[..., Any]
    walk: bool = False
    expert: Expert | None = None

    @property
    def name(self) -> str:
        return self.task_id.rpartition("/")[2]


_REGISTRY: dict[str, TaskEntry] = {}


def _task_id(name: str, namespace: str) -> str:
    if "/" in name or "/" in namespace or not name or not namespace:
        raise ValueError(f"task ids are namespace/name, got {namespace!r}/{name!r}")
    return f"{namespace}/{name}"


def register(
    name: str, *, rig: str, namespace: str = BUILTIN_NAMESPACE, walk: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a builder `build(**kwargs) -> Task`; refuses a duplicate id.
    `walk=True` marks a locomotion family whose rig is the walk robot."""
    task_id = _task_id(name, namespace)

    def decorate(build: Callable[..., Any]) -> Callable[..., Any]:
        existing = _REGISTRY.get(task_id)
        if existing is not None and existing.build is not build:
            raise ValueError(f"task {task_id!r} registered twice")
        expert = existing.expert if existing is not None else None
        _REGISTRY[task_id] = TaskEntry(
            task_id=task_id, rig=rig, build=build, walk=walk, expert=expert
        )
        return build

    return decorate


def register_expert(
    name: str, *, namespace: str = BUILTIN_NAMESPACE
) -> Callable[[Expert], Expert]:
    """Decorate the scripted policy that reviews the family `name`: the
    acceptance critic runs it on every paired trial. The family must be
    registered first (same module order as its builder), and a family
    reviews with one expert."""
    task_id = _task_id(name, namespace)

    def decorate(expert: Expert) -> Expert:
        entry = _REGISTRY.get(task_id)
        if entry is None:
            raise KeyError(f"no task {task_id!r} to review: register its builder first")
        if entry.expert is not None and entry.expert is not expert:
            raise ValueError(f"task {task_id!r} already has an expert")
        _REGISTRY[task_id] = replace(entry, expert=expert)
        return expert

    return decorate


def tasks() -> Mapping[str, TaskEntry]:
    """Every registered task: the built-ins plus installed plugins."""
    load_group(ENTRY_POINT_GROUP, BUILTIN_MODULES)
    return dict(_REGISTRY)


def walk_entries() -> dict[str, TaskEntry]:
    """The locomotion families, by id."""
    return {task_id: e for task_id, e in tasks().items() if e.walk}


def expert_entries() -> dict[str, TaskEntry]:
    """The families a scripted expert reviews, by id."""
    return {task_id: e for task_id, e in tasks().items() if e.expert is not None}


def resolve(task: str) -> TaskEntry:
    """`robotiq/kitting` or, for a built-in, bare `kitting`."""
    known = tasks()
    if task in known:
        return known[task]
    builtin = f"{BUILTIN_NAMESPACE}/{task}"
    if builtin in known:
        return known[builtin]
    raise KeyError(f"no task {task!r}; the registry knows {sorted(known)}")
