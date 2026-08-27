"""Kitting — T5's industrial task (docs/31): parts into a tray's slots,
one part per arm so the task is bimanual by reach. `KittingSpec` is the
task as DATA (named by `Task.stamp`, admitted by `tasks/acceptance.py`);
`build_kitting` composes the scene and the referee around it. The
scripted expert that demonstrates it is `expert.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rq_pipeline.protocol import EpisodeProtocol, Placement
from rq_pipeline.tasks.aloha2.rig import (
    _HOLD_STEPS,
    ACT_SIM_Y_SHIFT,
    ALOHA2_LOOK,
    ALOHA_TOP_CAMERAS,
    BUNDLE_XML,
    HOME_KEYFRAME,
    KITTING,
    MOVED_M,
    PART_ORDER,
    RIG,
    SERVOS,
    _add_free_box,
    _add_top_camera_and_referees,
    _corner_fraction,
    _task_scene,
)
from rq_pipeline.tasks.registry import register
from rq_pipeline.tasks.scene import (
    TABLE_GEOM,
    add_slot_walls,
)
from rq_pipeline.tasks.task import CONTROL_INTERVAL, PAIRED_TRIALS, Task

# ------------------------------------------------------------- kitting --
# T5's industrial task (docs/31): parts into a tray's slots, one part
# per arm so the task is bimanual by reach. Everything below serves two
# consumers: `build_kitting` gives the harness the scene and the judge;
# `kitting_waypoints` gives the DEMO GENERATOR (tools/kitting-demos.py)
# the Cartesian choreography it converts to joint-space commands via
# IK — chained solves, each warm-starting the next, because a cold IK
# jump to a low target can stall in a local minimum (measured: a lone
# 6 cm-height target failed at 69 mm where the same target approached
# from 10 cm above converges).


@dataclass(frozen=True)
class KittingSpec:
    """The kitting task as DATA: the tray, the parts, where they start,
    what counts as placed, how long an episode is, and the sentence a
    policy is judged under. Today's numbers are the defaults; a variant
    is `replace(KITTING_SPEC, ...)`, named by `Task.stamp`, and admitted
    only by `tasks/acceptance.py` — the scripted expert must pass its
    referee and the do-nothing floor must not. The scene's structure
    (arms, table, cameras, the parts' state slices) stays code."""

    tray_center: tuple[float, float] = (0.0, -0.02)
    slot_offset_x: float = 0.09  # one slot per arm, mirrored about the tray centre
    slot_half: float = 0.045
    slot_wall: float = 0.008
    slot_wall_height: float = 0.015
    part_half: float = 0.02
    # Spawn bands, one per arm side, inside the proven reach envelope. A
    # plain dict: `asdict` deep-copies for the stamp, and a MappingProxyType
    # cannot be copied (measured 2026-08-28).
    # (transfer_cube's spawn box, mirrored for the left arm).
    part_spawn: Mapping[str, tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=lambda: {
            "right": (
                (0.14, 0.24),
                (0.28 + ACT_SIM_Y_SHIFT, 0.44 + ACT_SIM_Y_SHIFT),
            ),
            "left": (
                (-0.24, -0.14),
                (0.28 + ACT_SIM_Y_SHIFT, 0.44 + ACT_SIM_Y_SHIFT),
            ),
        }
    )
    # 28 s at 500 Hz: two sequential pick-places PLUS the closed-loop
    # corrections and one grasp retry per arm. The budget was 18 s and
    # every robustness fix shifted which trial's endgame got truncated —
    # the whack-a-mole was the clock, not the choreography.
    steps: int = 14000
    trials: int = PAIRED_TRIALS
    # The paired starts: the spawn box's corners pulled in by this fraction
    # of each side (`_corner_fraction`).
    spawn_inset: float = 0.2
    # The referee: a part is in its slot within this radius of the slot
    # centre and below this height (resting on the tray floor).
    in_slot_xy_m: float = 0.035
    in_slot_z_m: float = 0.045
    # A lift that left the part below this never lifted it.
    lift_check_z_m: float = 0.05
    # The sentence the dataset was exported with (collect/kitting_export.py)
    # and the policy is judged under — one string, both places read it.
    instruction: str = "kit both parts into their slots"

    @property
    def slot_centers(self) -> dict[str, tuple[float, float]]:
        """The part that starts on the RIGHT goes into the right slot.
        Derived, not restated — the tray centre once sat beside
        hand-copied slot y values."""
        cx, cy = self.tray_center
        return {
            "right": (cx + self.slot_offset_x, cy),
            "left": (cx - self.slot_offset_x, cy),
        }

    @property
    def part_home(self) -> dict[str, tuple[float, float, float]]:
        """Each part's authored resting pose: the middle of its band."""
        return {
            arm: (
                (x_low + x_high) / 2,
                (y_low + y_high) / 2,
                self.part_half,
            )
            for arm, ((x_low, x_high), (y_low, y_high)) in self.part_spawn.items()
        }


KITTING_SPEC = KittingSpec()
PART_HOME = KITTING_SPEC.part_home  # each part's authored resting pose


def part_body(arm: str) -> str:
    """The kitting part each arm places, by body name."""
    return f"part_{arm}"


# FULLPHYSICS: qpos = 16 arm + 7 right part + 7 left part; positions at
# 17..19 and 24..26. Pinned by test.
PART_STATE_SLICE = {"right": slice(17, 20), "left": slice(24, 27)}


def part_qpos_slice(arm: str) -> slice:
    """The part's position in `data.qpos`: the FULLPHYSICS state carries
    time first, so its slice sits one to the right of qpos's."""
    state = PART_STATE_SLICE[arm]
    return slice(state.start - 1, state.stop - 1)


KITTING_INSTRUCTION = KITTING_SPEC.instruction


TRAY_RGBA = (0.35, 0.25, 0.15, 1.0)
PART_RGBA = {"right": (1, 0, 0, 1), "left": (0, 0.55, 1, 1)}


@register(KITTING, rig=RIG)
def build_kitting(
    bundle_xml: Path = BUNDLE_XML,
    look: str = ALOHA2_LOOK,
    *,
    spec: KittingSpec = KITTING_SPEC,
) -> Task:
    """Kitting: each arm places its side's part into its slot. `spec`
    is the task's data (the shipped one by default); a variant composes
    the same scene structure around different numbers."""
    import numpy as np  # noqa: PLC0415

    scene = _task_scene("kitting", bundle_xml, look)

    # The tray: two shallow square wells built from wall boxes, static.
    slot_centers = spec.slot_centers
    for arm, centre in slot_centers.items():
        add_slot_walls(
            scene,
            f"slot_{arm}",
            centre,
            offset=(spec.slot_half, spec.slot_half),
            long_half=(spec.slot_half + spec.slot_wall,) * 2,
            wall=spec.slot_wall,
            height=spec.slot_wall_height,
            rgba=TRAY_RGBA,
        )

    for arm in PART_ORDER:
        _add_free_box(
            scene, part_body(arm), spec.part_home[arm], spec.part_half, PART_RGBA[arm]
        )
    _add_top_camera_and_referees(scene)

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        fx, fy = _corner_fraction(trial, inset=spec.spawn_inset)
        for arm in PART_ORDER:
            (x_low, x_high), (y_low, y_high) = spec.part_spawn[arm]
            part = PART_STATE_SLICE[arm]
            initial[part.start] = x_low + fx * (x_high - x_low)
            initial[part.start + 1] = y_low + fy * (y_high - y_low)
        return initial

    def in_slot(part: Any, arm: str) -> bool:
        cx, cy = slot_centers[arm]
        return bool(
            np.hypot(part[0] - cx, part[1] - cy) < spec.in_slot_xy_m
            and part[2] < spec.in_slot_z_m
        )

    def success(states: Any, sensors: Any) -> bool:
        del sensors
        tail = states[-_HOLD_STEPS:]
        return all(
            in_slot(row[PART_STATE_SLICE[arm]], arm)
            for arm in slot_centers
            for row in tail
        )

    # Order-free by construction (either arm may go first): any part
    # touched, any part lifted, one in its slot, both in their slots.
    def parts(states: Any, step: int) -> dict[str, Any]:
        return {arm: states[step, PART_STATE_SLICE[arm]] for arm in slot_centers}

    def part_moved(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return any(
            np.linalg.norm(part[:2] - states[0, PART_STATE_SLICE[arm]][:2]) > MOVED_M
            for arm, part in parts(states, step).items()
        )

    def part_lifted(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return any(
            part[2] > spec.lift_check_z_m for part in parts(states, step).values()
        )

    def one_in_slot(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return any(in_slot(part, arm) for arm, part in parts(states, step).items())

    def both_in_slot(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return all(in_slot(part, arm) for arm, part in parts(states, step).items())

    return Task(
        name=KITTING,
        spec=scene,
        cameras=ALOHA_TOP_CAMERAS,
        state_width=SERVOS,
        instruction=spec.instruction,
        bundle_dir=Path(bundle_xml).parent,
        task_spec=spec,
        protocol=EpisodeProtocol(
            trials=spec.trials,
            steps=spec.steps,
            control_interval=CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
            placements=tuple(
                Placement(
                    part_body(arm),
                    TABLE_GEOM,
                    x=spec.part_spawn[arm][0],
                    y=spec.part_spawn[arm][1],
                )
                for arm in PART_ORDER
            ),
            home=HOME_KEYFRAME,
            milestones=(
                ("part_moved", part_moved),
                ("part_lifted", part_lifted),
                ("one_in_slot", one_in_slot),
                ("both_in_slot", both_in_slot),
            ),
        ),
    )


# Per-segment (dz above part/slot, gripper normalized, seconds).
