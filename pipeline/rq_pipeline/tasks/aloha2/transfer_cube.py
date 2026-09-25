"""Transfer cube — gym-aloha's protocol on the identified rig: the cube
spawns on the right arm's side, the right arm picks it up, hands it to
the left, and success (their reward 4) is the LEFT gripper holding the
cube clear of the table at the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rq_pipeline.protocol import EpisodeProtocol, Placement
from rq_pipeline.tasks.aloha2.rig import (
    _HELD_RADIUS_M,
    _HOLD_STEPS,
    _STEPS,
    _TRANSFERRED_HEIGHT_M,
    ALOHA2_LOOK,
    ALOHA_TOP_CAMERAS,
    BUNDLE_XML,
    CUBE_BODY,
    CUBE_HALF,
    CUBE_HOME,
    CUBE_SPAWN_INSET,
    CUBE_SPAWN_X,
    CUBE_SPAWN_Y,
    CUBE_STATE_SLICE,
    HOME_KEYFRAME,
    LEFT_GRIPPER_POS_SLICE,
    MOVED_M,
    RIG,
    SERVOS,
    TRANSFER_CUBE,
    _add_free_box,
    _add_top_camera_and_referees,
    _task_scene,
)
from rq_pipeline.tasks.registry import register
from rq_pipeline.tasks.scene import (
    TABLE_GEOM,
    corner_fraction,
)
from rq_pipeline.tasks.task import CONTROL_INTERVAL, PAIRED_TRIALS, Task

TRANSFER_CUBE_INSTRUCTION = "transfer the cube to the left gripper"


@register(TRANSFER_CUBE, rig=RIG)
def build_transfer_cube(bundle_xml: Path = BUNDLE_XML, look: str = ALOHA2_LOOK) -> Task:
    """Transfer cube, gym-aloha's protocol on the identified rig.

    `look` selects the appearance: the bundle's own (`aloha2`) or the
    ACT simulator's (`act_sim`) for checkpoints trained on its renders.
    """
    import numpy as np  # noqa: PLC0415

    scene = _task_scene("transfer-cube", bundle_xml, look)
    # gym-aloha's red_box, verbatim: 2 cm half-size, red.
    _add_free_box(scene, CUBE_BODY, CUBE_HOME, CUBE_HALF, (1, 0, 0, 1))
    _add_top_camera_and_referees(scene)

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        fx, fy = corner_fraction(trial, inset=CUBE_SPAWN_INSET)
        initial[CUBE_STATE_SLICE.start] = CUBE_SPAWN_X[0] + fx * (
            CUBE_SPAWN_X[1] - CUBE_SPAWN_X[0]
        )
        initial[CUBE_STATE_SLICE.start + 1] = CUBE_SPAWN_Y[0] + fy * (
            CUBE_SPAWN_Y[1] - CUBE_SPAWN_Y[0]
        )
        return initial

    def success(states: Any, sensors: Any) -> bool:
        tail_cube = states[-_HOLD_STEPS:, CUBE_STATE_SLICE]
        tail_left = sensors[-_HOLD_STEPS:, LEFT_GRIPPER_POS_SLICE]
        lifted = bool(np.min(tail_cube[:, 2]) > _TRANSFERRED_HEIGHT_M)
        held = bool(
            np.max(np.linalg.norm(tail_cube - tail_left, axis=1)) < _HELD_RADIUS_M
        )
        return lifted and held

    # The chain the referee's own ingredients imply: touched, lifted,
    # brought to the left gripper. The verdict stays the hold window.
    def cube_moved(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        start = states[0, CUBE_STATE_SLICE][:2]
        now = states[step, CUBE_STATE_SLICE][:2]
        return bool(np.linalg.norm(now - start) > MOVED_M)

    def cube_lifted(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return bool(states[step, CUBE_STATE_SLICE][2] > _TRANSFERRED_HEIGHT_M)

    def cube_at_left(states: Any, sensors: Any, step: int) -> bool:
        cube = states[step, CUBE_STATE_SLICE]
        left = sensors[step, LEFT_GRIPPER_POS_SLICE]
        return bool(np.linalg.norm(cube - left) < _HELD_RADIUS_M)

    return Task(
        name=TRANSFER_CUBE,
        spec=scene,
        cameras=ALOHA_TOP_CAMERAS,
        state_width=SERVOS,
        instruction=TRANSFER_CUBE_INSTRUCTION,
        bundle_dir=Path(bundle_xml).parent,
        protocol=EpisodeProtocol(
            trials=PAIRED_TRIALS,
            steps=_STEPS,
            control_interval=CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
            placements=(
                Placement(CUBE_BODY, TABLE_GEOM, x=CUBE_SPAWN_X, y=CUBE_SPAWN_Y),
            ),
            home=HOME_KEYFRAME,
            milestones=(
                ("cube_moved", cube_moved),
                ("cube_lifted", cube_lifted),
                ("cube_at_left", cube_at_left),
            ),
        ),
    )
