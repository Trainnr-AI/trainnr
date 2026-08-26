"""What the SO-101 scripted experts score, per physics build.

An expert's rate is a fact about the code AND the instrument. The two
venvs run different MuJoCo versions, and on 2026-08-26 the same
scripted stacker and inserter, the same task, the same four paired
starts, scored 4/4 on the locked sim venv's MuJoCo 3.11.0 and 3/4 on
the train venv's 3.12.0 (both fail trial 3 under 3.12; lift clears
both). That is why every record carries `instrument`. The pins below
assert the rate MEASURED on the instrument they run on and skip on an
instrument nobody has measured, instead of one number pretending to
hold across builds.
"""

from __future__ import annotations

import unittest

# Keys: the registry names (tasks.so101.LIFT / BLOCK_STACK / TOOL_INSERT)
# and MuJoCoBackend.instrument ("mujoco-<version>").
SO101_EXPERT_RATE: dict[str, dict[str, float]] = {
    "lift": {"mujoco-3.11.0": 1.0, "mujoco-3.12.0": 1.0},
    "block_stack": {"mujoco-3.11.0": 1.0, "mujoco-3.12.0": 0.75},
    "tool_insert": {"mujoco-3.11.0": 1.0, "mujoco-3.12.0": 0.75},
}


def expected_expert_rate(task: str, instrument: str, case: unittest.TestCase) -> float:
    """The measured rate of `task`'s expert on `instrument`, or skip the pin."""
    by_instrument = SO101_EXPERT_RATE[task]
    if instrument not in by_instrument:
        case.skipTest(
            f"{task} expert rate not measured on {instrument}; "
            f"measured on {sorted(by_instrument)}"
        )
    return by_instrument[instrument]
