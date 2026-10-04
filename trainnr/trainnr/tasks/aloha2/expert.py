"""The scripted kitting expert — the demo generator's choreography.

Cartesian beats per arm (`kitting_waypoints`), each turned into a joint
target by chained IK (a cold jump to a low target stalls in a local
minimum; each solve warm-starts the next), closed-loop corrections on
the low beats, one grasp retry re-planned from where the part is, and a
park between arms. `KittingChoreography` holds every measured knob and
`expert_stamp` names the whole by content, so a demo batch's provenance
says which expert produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trainnr.bundles.hashing import content_stamp
from trainnr.collect.choreography import Episode, Waypoint
from trainnr.tasks.aloha2.kitting import (
    KITTING,
    KITTING_SPEC,
    KittingSpec,
    part_qpos_slice,
)
from trainnr.tasks.aloha2.rig import (
    ARM_CTRL_SLICES,
    ARM_IK_JOINTS,
    FINGERS,
    NEUTRAL_CTRL,
    PART_ORDER,
    SERVOS_PER_ARM,
    gripper_ctrl_from_normalized,
)
from trainnr.tasks.registry import register_expert

KITTING_EXPERT = "kitting-expert"  # the stamp's name half


class KittingChoreography:
    """The choreographer's measured knobs, in one place; the stories that
    set them stay beside the calls. GRASP_TILT: near-horizontal, 45
    degrees down, pointing inward (x flips with the arm)."""

    GRASP_TILT = (0.7, 0.0, -0.71)
    # The fingers close along world y: the parts are axis-aligned boxes,
    # so the pads meet their y faces squarely. Without it the closing
    # plane pitched 19 degrees at the near-base corners (the log 2026-08-27).
    GRASP_CLOSING_AXIS = (0.0, 1.0, 0.0)
    # Where the closing plane is pinned: the GRASP beats, where the pads
    # must meet the part's faces squarely. Not the hovers (pinned there,
    # the far corner's 12 cm hover is out of reach: trial 0 refused) and
    # not the place over the slot (the part is already in the gripper,
    # and the left slot beside its own base is unreachable with the
    # finger axis pinned) — both measured 2026-08-27.
    CLOSING_PLANE_SEGMENTS = ("descend", "close", "lift")
    # After its place, an arm PARKS: it folds back to the neutral pose,
    # out of the shared workspace, before the other arm works. Without
    # it the right arm's retreat pose left its forearm mid-table across
    # the left arm's path, and the left gripper shoved it — measured
    # 2026-08-27: -55 N.m of constraint torque on the left shoulder, the
    # pads stalled 6 cm above the part, and not one finger contact.
    PARK_SECONDS = 1.5
    IK_DOWN_WEIGHT = 0.5
    IK_CLOSING_WEIGHT = 0.5
    IK_POS_TOL_M = 0.008
    IK_MAX_ITERS = 250
    IK_DAMPING = 5e-3
    CORRECTION_ROUNDS = 3
    CORRECTION_BITE_M = 0.05  # linear correction only holds locally: 5 cm bites
    CORRECTION_DONE_M = 0.008  # closed-loop grip corrections stop inside this
    PAD_FLOOR_Z_M = 0.02  # the pads never need to go under 2 cm
    CORRECTION_SECONDS = 0.5
    # Which beats track the SLOT rather than the part, which get the
    # closed-loop correction rounds, and after which the lift is checked
    # and the grasp retried once — choreography facts, so the stamp
    # covers them (they lived outside this struct until 2026-08-28).
    SLOT_ANCHORED_SEGMENTS = ("above_slot", "lower", "open", "retreat")
    CORRECTED_SEGMENTS = ("descend", "lower")
    RETRY_AFTER_SEGMENT = "lift"

    @classmethod
    def fields(cls) -> dict[str, Any]:
        """The choreography as data — every constant above plus the
        segment table — so `expert_stamp` can name it by content."""
        constants = {
            name: value
            for name, value in vars(cls).items()
            if name.isupper() and not name.startswith("_")
        }
        constants["SEGMENTS"] = _PICK_PLACE_SEGMENTS
        constants["PART_ORDER"] = PART_ORDER
        return constants


# The grasp approach: near-horizontal, tilted 45 degrees down and
# pointing inward (away from the arm's own base). A straight-down
# approach is IMPOSSIBLE on this gripper: the base housing's collision
# meshes reach the table before the pads reach a 4 cm part (measured
# via contacts: 1-8 cm of penetration in every vertical solve; tilted
# solves are clean). An adaptive base-toward-target axis was tried and
# REVERTED — its y-tilt degraded the wrist pose and every trial failed.


def grasp_axis(arm: str, target_xy: Any) -> tuple[float, float, float]:
    """Fixed inward tilt — the adaptive base-to-target version made
    every trial fail (the y-tilt degrades the wrist pose); the fixed
    axis carried 2/4. target_xy stays in the signature for the day a
    smarter axis earns its way back with evidence."""
    del target_xy
    x, y, z = KittingChoreography.GRASP_TILT
    return (-x if arm == "right" else x, y, z)


_PICK_PLACE_SEGMENTS = (
    ("above_part", 0.10, 1.0, 1.2),
    ("descend", 0.005, 1.0, 1.0),
    ("close", 0.005, 0.0, 0.6),
    ("lift", 0.12, 0.0, 1.0),
    ("above_slot", 0.12, 0.0, 1.4),
    ("lower", 0.045, 0.0, 1.0),
    ("open", 0.045, 1.0, 0.6),
    ("retreat", 0.14, 1.0, 0.8),
)


@dataclass
class KittingStats:
    """What the choreographer observed about its own run — typed so a
    consumer misspelling a key gets an AttributeError, not a silent
    empty default. Each retry records (arm, physics_step, part_z)."""

    retries: list[tuple[str, int, float]] = field(default_factory=list)
    steps_used_before_hold: int = 0
    truncated: bool = False


def expert_stamp() -> str:
    """`kitting-expert@<hash>`: the scripted expert named by its
    choreography's content, the way a task is named by its spec and a
    bundle by its files. A demo batch's manifests carry it, so a
    choreography change (2026-08-27's closing plane and park beat)
    shows in the provenance instead of hiding behind an unchanged task
    stamp."""
    return content_stamp(KITTING_EXPERT, KittingChoreography.fields())


def kitting_waypoints(
    arm: str, part_xy: Any, *, spec: KittingSpec = KITTING_SPEC
) -> list[Waypoint]:
    """The Cartesian choreography for one arm.

    Targets before `above_slot` track the PART's spawn position;
    from `above_slot` on they track the slot. The demo generator turns
    each into a joint waypoint via chained IK.
    """
    slot = spec.slot_centers[arm]
    slot_anchored = KittingChoreography.SLOT_ANCHORED_SEGMENTS
    return [
        Waypoint(
            name,
            (*(slot if name in slot_anchored else part_xy)[:2], spec.part_half + dz),
            grip,
            secs,
        )
        for name, dz, grip, secs in _PICK_PLACE_SEGMENTS
    ]


@dataclass(frozen=True)
class ArmSession:
    """One arm's names and ids for the choreography, resolved once."""

    model: Any
    arm: str
    ctrl_slice: slice
    gripper_index: int
    pad_names: tuple[str, ...]
    pad_ids: tuple[int, ...]
    joint_qpos: tuple[int, ...]  # qpos address of each IK joint, in order

    @classmethod
    def of(cls, model: Any, arm: str) -> ArmSession:
        import mujoco  # noqa: PLC0415 - sim extra

        ctrl_slice = ARM_CTRL_SLICES[arm]
        pad_names = tuple(f"{arm}/{finger}_g1" for finger in FINGERS)
        return cls(
            model,
            arm,
            ctrl_slice,
            ctrl_slice.start + SERVOS_PER_ARM - 1,
            pad_names,
            tuple(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in pad_names
            ),
            tuple(
                int(
                    model.jnt_qposadr[
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    ]
                )
                for n in ARM_IK_JOINTS[arm]
            ),
        )

    def part_qpos(self, data: Any) -> Any:
        return data.qpos[part_qpos_slice(self.arm)]

    def grip_error(self, data: Any, target_xyz: Any) -> Any:
        import numpy as np  # noqa: PLC0415

        grip_center = 0.5 * (
            data.geom_xpos[self.pad_ids[0]] + data.geom_xpos[self.pad_ids[1]]
        )
        return grip_center - np.asarray(target_xyz)

    def reach(
        self,
        episode: Episode,
        target_xyz: Any,
        grip: float,
        secs: float,
        *,
        closing: bool,
    ) -> None:
        """Solve the reach on the scratch copy, then command it over `secs`.
        A reach the IK cannot make is refused: the choreography must not
        pretend it happened."""
        from trainnr.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

        knobs = KittingChoreography
        scratch = episode.scratch
        scratch.qpos[:] = episode.data.qpos
        reached = solve_arm_ik(
            self.model,
            scratch,
            site=f"{self.arm}/gripper",
            joints=ARM_IK_JOINTS[self.arm],
            grip_geoms=self.pad_names,
            target_pos=target_xyz,
            approach_axis=grasp_axis(self.arm, target_xyz[:2]),
            down_weight=knobs.IK_DOWN_WEIGHT,
            closing_axis=knobs.GRASP_CLOSING_AXIS if closing else None,
            closing_weight=knobs.IK_CLOSING_WEIGHT,
            pos_tol=knobs.IK_POS_TOL_M,
            max_iters=knobs.IK_MAX_ITERS,
            damping=knobs.IK_DAMPING,
        )
        if not reached:
            raise RuntimeError(
                f"IK failed: {self.arm} arm to {target_xyz} — the choreography "
                "must not pretend a reach happened"
            )
        target_ctrl = episode.ctrl.copy()
        for joint, address in enumerate(self.joint_qpos):
            target_ctrl[self.ctrl_slice.start + joint] = scratch.qpos[address]
        target_ctrl[self.gripper_index] = gripper_ctrl_from_normalized(grip)
        episode.advance(secs, target_ctrl)

    def park(self, episode: Episode) -> None:
        """Fold back to the neutral pose, out of the shared workspace."""
        import numpy as np  # noqa: PLC0415

        parked = episode.ctrl.copy()
        parked[self.ctrl_slice] = np.asarray(NEUTRAL_CTRL, dtype=float)[self.ctrl_slice]
        parked[self.gripper_index] = gripper_ctrl_from_normalized(1.0)
        episode.advance(KittingChoreography.PARK_SECONDS, parked)


def _run_arm(
    episode: Episode, session: ArmSession, spec: KittingSpec, stats: KittingStats | None
) -> None:
    """One arm's pick-and-place over its part's ACTUAL position: the
    beats in order, closed-loop corrections on the low ones, one grasp
    retry re-planned from where the part is, then the park."""
    import numpy as np  # noqa: PLC0415

    knobs = KittingChoreography
    plan = kitting_waypoints(
        session.arm, session.part_qpos(episode.data)[:2].copy(), spec=spec
    )
    index = 0
    grasp_retried = False
    while index < len(plan):
        if episode.stepper.done:
            if stats is not None:
                stats.truncated = True
            return
        name, target_xyz, grip, secs = plan[index]
        closing = name in knobs.CLOSING_PLANE_SEGMENTS
        index += 1
        session.reach(episode, target_xyz, grip, secs, closing=closing)
        part_z = float(session.part_qpos(episode.data)[2])
        if (
            name == knobs.RETRY_AFTER_SEGMENT
            and part_z < spec.lift_check_z_m
            and not grasp_retried
        ):
            if stats is not None:
                stats.retries.append(
                    (session.arm, episode.stepper.step, round(part_z, 3))
                )
            grasp_retried = True
            plan = kitting_waypoints(
                session.arm, session.part_qpos(episode.data)[:2].copy(), spec=spec
            )
            index = 0
            continue
        if name in knobs.CORRECTED_SEGMENTS:
            for _ in range(knobs.CORRECTION_ROUNDS):
                error = session.grip_error(episode.data, target_xyz)
                if float(np.linalg.norm(error)) < knobs.CORRECTION_DONE_M:
                    break
                bite = np.clip(error, -knobs.CORRECTION_BITE_M, knobs.CORRECTION_BITE_M)
                corrected = (
                    target_xyz[0] - bite[0],
                    target_xyz[1] - bite[1],
                    max(knobs.PAD_FLOOR_Z_M, target_xyz[2] - bite[2]),
                )
                session.reach(
                    episode, corrected, grip, knobs.CORRECTION_SECONDS, closing=closing
                )
    session.park(episode)


@register_expert(KITTING)
def scripted_kitting_episode(
    model: Any,
    initial_state: Any,
    *,
    on_control: Any = None,
    stats: KittingStats | None = None,
    spec: KittingSpec = KITTING_SPEC,
) -> tuple[Any, Any, Any]:
    """One scripted kitting demonstration, privileged, R7-correct.

    Arms act sequentially (right, then left), each running the
    choreography over ITS part's actual position read from state —
    which is why this is the demo GENERATOR's path, not a harness
    policy. Returns (states, sensors, actions): per-physics-step
    FULLPHYSICS states and sensors shaped exactly like the harness
    rollout (so the task's `success` judges them unchanged), and the
    50 Hz commanded-position actions a dataset records.

    `on_control(step, data)` is the generator's hook (render frames).
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    from trainnr.physics.mujoco_backend import Stepper  # noqa: PLC0415

    episode = Episode(
        Stepper(model, initial_state, spec.steps),
        mujoco.MjData(model),
        on_control,
        np.array(NEUTRAL_CTRL, dtype=float),
    )
    for arm in PART_ORDER:
        _run_arm(episode, ArmSession.of(model, arm), spec, stats)
    if stats is not None:
        stats.steps_used_before_hold = episode.stepper.step
    while not episode.stepper.done:
        episode.advance(1.0, episode.ctrl)
    return episode.stepper.states, episode.stepper.sensors, np.asarray(episode.actions)
