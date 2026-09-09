"""The scripted experts that review tasks: by family, the builder and
the expert that solves it. Acceptance IS the expert's verdict, so a
family without one cannot be accepted — it is named as such."""

from __future__ import annotations

from typing import Any

from rq_pipeline.tasks.aloha2 import KITTING, build_kitting, scripted_kitting_episode
from rq_pipeline.tasks.registry import resolve

EXPERTS: dict[str, tuple[Any, Any]] = {
    KITTING: (build_kitting, scripted_kitting_episode)
}


def expert_for(task_id: str) -> Any:
    """The expert that solves `task_id`'s family, or a refusal by name."""
    entry = resolve(task_id)
    for name, (_build, expert) in EXPERTS.items():
        if resolve(name).task_id == entry.task_id:
            return expert
    known = ", ".join(sorted(resolve(n).task_id for n in EXPERTS))
    raise ValueError(
        f"no scripted expert reviews {entry.task_id}; acceptance exists for {known}"
    )
