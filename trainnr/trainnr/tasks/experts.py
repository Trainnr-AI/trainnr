"""The scripted experts that review tasks: a family's expert is registered
on its task entry (`registry.register_expert`), so a third party's family
brings its own reviewer without editing us. Acceptance IS the expert's
verdict, so a family without one cannot be accepted — it is named as such.
"""

from __future__ import annotations

from trainnr.tasks.registry import Expert, TaskEntry, expert_entries, resolve


def experts() -> dict[str, TaskEntry]:
    """The families a scripted expert reviews, by id."""
    return expert_entries()


def expert_for(task_id: str) -> Expert:
    """The expert that solves `task_id`'s family, or a refusal by name."""
    entry = resolve(task_id)
    if entry.expert is not None:
        return entry.expert
    known = ", ".join(sorted(experts()))
    raise ValueError(
        f"no scripted expert reviews {entry.task_id}; acceptance exists for {known}"
    )
