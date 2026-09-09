"""A task declared by conversation: a spec overlay over a registered
family, built for real and named by its content. Writing it into a
project is the project layer's job (`project/task_ref.declare_task`).

A family is a registered builder that takes a `spec` (kitting's
`KittingSpec`, the lift study's `LiftStudySpec`); every other builder
composes a fixed scene and is refused here by name — a variant of it
would be authored work, not an overlay. The overlay is the dataclass's
own fields, so an agent reads them (`spec_fields`) and writes only what
it changes; the rest stays the family's default. The stamp is
`Task.stamp`, a content hash over the whole spec, so two agents writing
the same numbers get the same name.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from rq_pipeline.tasks.registry import TaskEntry, resolve, tasks

SPEC_PARAMETER = "spec"


def families() -> dict[str, TaskEntry]:
    """The task ids whose builder takes a `spec`, so a variant can be
    declared over them."""
    return {
        task_id: entry for task_id, entry in tasks().items() if _default_spec(entry)
    }


def _default_spec(entry: TaskEntry) -> Any | None:
    try:
        parameter = inspect.signature(entry.build).parameters.get(SPEC_PARAMETER)
    except (TypeError, ValueError):
        return None
    default = parameter.default if parameter is not None else None
    return default if is_dataclass(default) and not isinstance(default, type) else None


def spec_fields(task_id: str) -> dict[str, dict[str, Any]]:
    """Every field of the family's spec: its type and its default, the
    way the agent should read them before writing an overlay."""
    entry = resolve(task_id)
    default = _default_spec(entry)
    if default is None:
        raise ValueError(_not_a_family(entry.task_id))
    return {
        f.name: {
            "type": _type_name(f.type),
            "default": jsonable(getattr(default, f.name)),
        }
        for f in fields(default)
    }


def build_variant(task_id: str, overlay: dict[str, Any] | None) -> tuple[Any, Any]:
    """The family's builder run with its spec replaced by the overlay.
    Returns (task, spec). Refuses an unknown field by name, listing the
    real ones."""
    entry = resolve(task_id)
    default = _default_spec(entry)
    if default is None:
        raise ValueError(_not_a_family(entry.task_id))
    overlay = dict(overlay or {})
    known = {f.name: getattr(default, f.name) for f in fields(default)}
    unknown = sorted(set(overlay) - set(known))
    if unknown:
        raise ValueError(
            f"{entry.task_id} has no field {', '.join(unknown)}; "
            f"its fields are {', '.join(sorted(known))}"
        )
    coerced = {k: _coerce(v, known[k]) for k, v in overlay.items()}
    spec = replace(default, **coerced)
    return entry.build(**{SPEC_PARAMETER: spec}), spec


def build_from_reference(ref: dict[str, Any]) -> Any:
    """The task a project's `task.json` names, rebuilt exactly: with its
    overlay when it carries one, the family default otherwise."""
    task_id = ref.get("task_id", "")
    if ref.get("spec"):
        task, _ = build_variant(task_id, ref["spec"])
        return task
    return resolve(task_id).build()


def acceptance_path(task_folder: Path) -> Path:
    return task_folder / ACCEPTANCE_FILE


ACCEPTANCE_FILE = "acceptance.json"
ACCEPTANCE_SCHEMA = "trainnr-acceptance/1"


# -- helpers ---------------------------------------------------------------------


def _not_a_family(task_id: str) -> str:
    names = ", ".join(sorted(families())) or "none"
    return (
        f"{task_id} composes a fixed scene; a variant needs a family with a spec: "
        f"{names}"
    )


def _type_name(t: Any) -> str:
    return t if isinstance(t, str) else getattr(t, "__name__", str(t))


def _coerce(value: Any, default: Any) -> Any:
    """JSON gives lists and plain dicts; the spec wants what its default
    is shaped like (tuples of numbers, mappings of tuples)."""
    if isinstance(default, tuple) and isinstance(value, (list, tuple)):
        inner = default[0] if default else None
        return tuple(_coerce(v, inner) for v in value)
    if isinstance(default, dict) and isinstance(value, dict):
        sample = next(iter(default.values()), None)
        return {k: _coerce(v, sample) for k, v in value.items()}
    return value


def jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return [jsonable(x) for x in v]
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return v
