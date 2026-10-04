"""What the SO-101 scripted experts score, per physics build.

An expert's rate is a fact about the code AND the instrument. The two
venvs run different MuJoCo versions, and on 2026-08-26 the same
scripted stacker and inserter, the same task, the same four paired
starts, scored 4/4 on the locked sim venv's MuJoCo 3.11.0 and 3/4 on
the train venv's 3.12.0 (both fail trial 3 under 3.12; lift clears
both). And the same 3.11.0 on the Mac's arm64 and this box's x86_64
disagree on the kitting solver sweep (docs/e2e-research/47 §7.1), so
the stamp names the architecture too. That is why every record
carries `instrument`. The pins below
assert the rate MEASURED on the instrument they run on and skip on an
instrument nobody has measured, instead of one number pretending to
hold across builds.

Measured the same night, so nobody re-chases it: the compiled models
are identical (24 arrays: friction, solref, solimp, masses, gains,
ranges) and so are the options; the expert's 360 control vectors are
identical; the state trajectories agree to the last bit for 84 steps
and part at step 85 by 1.6e-15, the contact count first differs at
tick 17 (10 vs 9 contacts) and the difference grows through contact
to centimetres by the drop — cube A ends on cube B under 3.11.0 and
on the table under 3.12.0 for trial 3, the (-2 mm, +2 mm) pick
corner. Engine numerics, amplified by contact, on an expert whose
margin at that corner is thin. Not a code path; not a bug.
"""

from __future__ import annotations

import unittest

from trainnr.tasks.so101 import BLOCK_STACK, LIFT, TOOL_INSERT

# Keys: the registry names and MuJoCoBackend.instrument
# ("mujoco-<version>+<arch>"). The arm64
# rows are the Mac's: its suite asserted 1.0 on 3.11.0 and passed
# (physics-newton branch, 2026-08-27).
SO101_EXPERT_RATE: dict[str, dict[str, float]] = {
    LIFT: {
        "mujoco-3.11.0+x86_64": 1.0,
        "mujoco-3.11.0+arm64": 1.0,
        "mujoco-3.12.0+x86_64": 1.0,
    },
    BLOCK_STACK: {
        "mujoco-3.11.0+x86_64": 1.0,
        "mujoco-3.11.0+arm64": 1.0,
        "mujoco-3.12.0+x86_64": 0.75,
    },
    TOOL_INSERT: {
        "mujoco-3.11.0+x86_64": 1.0,
        "mujoco-3.11.0+arm64": 1.0,
        "mujoco-3.12.0+x86_64": 0.75,
    },
}


# The planner expert (a note D3) on the same tasks, measured 2026-09-02
# with the default PlannerKnobs: the scripted experts' ceiling, matched.
SO101_PLANNER_RATE: dict[str, dict[str, float]] = {
    LIFT: {"mujoco-3.11.0+x86_64": 1.0},
    BLOCK_STACK: {"mujoco-3.11.0+x86_64": 1.0},
    TOOL_INSERT: {"mujoco-3.11.0+x86_64": 1.0},
}


def expected_expert_rate(task: str, instrument: str, case: unittest.TestCase) -> float:
    """The measured rate of `task`'s expert on `instrument`, or skip the pin."""
    return _measured(SO101_EXPERT_RATE, "expert", task, instrument, case)


def expected_planner_rate(task: str, instrument: str, case: unittest.TestCase) -> float:
    """The measured rate of the planner on `task` and `instrument`, or skip."""
    return _measured(SO101_PLANNER_RATE, "planner", task, instrument, case)


def _measured(
    table: dict[str, dict[str, float]],
    who: str,
    task: str,
    instrument: str,
    case: unittest.TestCase,
) -> float:
    by_instrument = table[task]
    if instrument not in by_instrument:
        case.skipTest(
            f"{task} {who} rate not measured on {instrument}; "
            f"measured on {sorted(by_instrument)}"
        )
    return by_instrument[instrument]
