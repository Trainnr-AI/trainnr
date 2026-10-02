"""The command envelope a checkpoint trained under: the velocity ranges
its curriculum had reached when it was saved.

mjlab's velocity task widens the commanded twist in stages keyed on the
environment step counter (`command_vel`, stages at 0, 5000 and 10000
iterations of 24 steps for the Go2). A fresh environment starts the
curriculum over, so a certificate or an export built from the config
alone judges and describes the FIRST stage whatever the checkpoint saw
- both certificates of the Go2 run and its manifest did (2026-09-11).
`pin_command_envelope` sets the config's ranges to the stage the
checkpoint's iteration had reached, removes the curriculum so nothing
moves them back, and says which stage that was.
"""

from __future__ import annotations

import re
from typing import Any

COMMAND_TERM = "twist"
CURRICULUM_TERM = "command_vel"
STAGES_KEY = "velocity_stages"
RANGE_KEYS = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")
CHECKPOINT = re.compile(r"model_(\d+)")


def checkpoint_iteration(name: str) -> int | None:
    """`model_7999` -> 7999; None when the name carries no iteration."""
    match = CHECKPOINT.search(name)
    return int(match.group(1)) if match else None


def stage_reached(stages: list[dict[str, Any]], steps: int) -> int:
    """The index of the last stage whose step the counter has passed."""
    reached = -1
    for i, stage in enumerate(stages):
        if steps >= int(stage.get("step", 0)):
            reached = i
    return reached


def command_ranges(cfg: Any) -> dict[str, list[float]]:
    ranges = cfg.commands[COMMAND_TERM].ranges
    return {
        k: list(getattr(ranges, k))
        for k in RANGE_KEYS
        if getattr(ranges, k, None) is not None
    }


def pin_command_envelope(
    cfg: Any, iteration: int | None, steps_per_iteration: int
) -> dict[str, Any]:
    """Set the config's command ranges to what the checkpoint trained
    under and drop the curriculum. Returns `{"commands": ranges,
    "basis": ...}` for a certificate's protocol or a manifest. A
    checkpoint whose name carries no iteration keeps the config's own
    ranges, said so in the basis."""
    term = cfg.curriculum.pop(CURRICULUM_TERM, None) if cfg.curriculum else None
    stages = list((term.params or {}).get(STAGES_KEY, [])) if term else []
    if iteration is None or not stages:
        why = "no curriculum" if not stages else "iteration unknown"
        return {
            "commands": command_ranges(cfg),
            "basis": f"the task's declared ranges ({why})",
        }
    # The checkpoint model_<n> is saved after iteration n has run.
    steps = (iteration + 1) * steps_per_iteration
    reached = stage_reached(stages, steps)
    ranges = cfg.commands[COMMAND_TERM].ranges
    for stage in stages[: reached + 1]:
        for key in RANGE_KEYS:
            if key in stage:
                setattr(ranges, key, tuple(stage[key]))
    return {
        "commands": command_ranges(cfg),
        "basis": (
            f"command curriculum stage {reached + 1} of {len(stages)}, "
            f"reached by iteration {iteration}"
        ),
    }
