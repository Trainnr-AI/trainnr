"""The choreography's one home: beats over privileged poses, executed
by chained IK - and the planner that writes the beats itself.

`Waypoint` and `Episode` are the kitting expert's machinery
(tasks/aloha2/expert.py), moved here so a second rig can use them.
`Gripper` is what a planner needs to know about one arm, resolved once
on a compiled model. `PickPlacePlanner` is the second expert source
(after hand-written beats): given an object's pose and size and a goal,
it writes the beats (hover, descend, close, lift, carry, lower, open,
retract) that the kitting choreography had by hand - so a new rigid
pick/place task is "scene + referee", no joint vectors, no per-task
authoring. Demo generation reads privileged state freely; the trained
policy never sees any of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from trainnr.tasks.task import CONTROL_INTERVAL


class Waypoint(NamedTuple):
    """One choreography beat — typed because the tuple's arity already
    lied once (an annotation described a removed 3-tuple shape while
    the code appended and unpacked four)."""

    name: str
    target: tuple[float, float, float]
    grip: float
    seconds: float


@dataclass
class Episode:
    """The rollout the choreographer drives: the stepper, its live data,
    a scratch copy for IK, the control vector as it is blended, and the
    50 Hz actions a dataset records."""

    stepper: Any
    scratch: Any
    on_control: Any
    ctrl: Any
    actions: list[Any] = field(default_factory=list)

    @property
    def data(self) -> Any:
        return self.stepper.data

    def advance(self, seconds: float, target_ctrl: Any) -> None:
        """Blend `ctrl` toward `target_ctrl` linearly over `seconds`,
        one control tick at a time, recording each commanded row."""
        import numpy as np  # noqa: PLC0415

        tick_seconds = self.stepper.model.opt.timestep * CONTROL_INTERVAL
        controls = max(1, round(seconds / tick_seconds))
        start = self.ctrl.copy()
        target = np.asarray(target_ctrl, dtype=float)
        for tick in range(controls):
            if self.stepper.done:
                return
            self.ctrl = start + (tick + 1) / controls * (target - start)
            self.actions.append(self.ctrl.copy())
            if self.on_control is not None:
                self.on_control(self.stepper.step, self.data)
            self.stepper.advance(self.ctrl, CONTROL_INTERVAL)


_ABOVE_THE_BASE_M = 1e-6  # no azimuth this close to the base axis
CORRECTED_AXES = {"xyz": (1.0, 1.0, 1.0), "xy": (1.0, 1.0, 0.0)}


@dataclass(frozen=True)
class Gripper:
    """One arm's grasping facts on a compiled model. `pocket` is the
    grasp centre in the SITE's frame - measured once per rig (the
    SO-101's from the pose whose closed jaws centre on a known point),
    so a planner asks for the object where the JAWS meet, not where
    some site happens to sit."""

    site: str
    site_id: int
    ik_joints: tuple[str, ...]
    joint_ids: tuple[int, ...]
    joint_qpos: tuple[int, ...]
    joint_ctrl: tuple[int, ...]
    jaw_ctrl: int
    jaw_open: float
    jaw_closed: float
    pocket: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grip_geoms: tuple[str, ...] | None = None
    # The world line the jaws close along: fixed, or None for tangential
    # around the arm's first joint (the fingers straddle from the sides
    # whatever the azimuth).
    closing: tuple[float, float, float] | None = None

    @classmethod
    def resolve(  # noqa: PLR0913 - a rig's facts, each named
        cls,
        model: Any,
        *,
        site: str,
        ik_joints: tuple[str, ...],
        jaw_ctrl: int,
        jaw_open: float,
        jaw_closed: float,
        pocket: tuple[float, float, float] = (0.0, 0.0, 0.0),
        grip_geoms: tuple[str, ...] | None = None,
        closing: tuple[float, float, float] | None = None,
    ) -> Gripper:
        import mujoco  # noqa: PLC0415 - sim extra

        def joint_id(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        ctrl_of_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        joint_ids = tuple(joint_id(n) for n in ik_joints)
        return cls(
            site=site,
            site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site),
            ik_joints=tuple(ik_joints),
            joint_ids=joint_ids,
            joint_qpos=tuple(int(model.jnt_qposadr[j]) for j in joint_ids),
            joint_ctrl=tuple(ctrl_of_joint[j] for j in joint_ids),
            jaw_ctrl=jaw_ctrl,
            jaw_open=jaw_open,
            jaw_closed=jaw_closed,
            pocket=pocket,
            grip_geoms=grip_geoms,
            closing=closing,
        )

    def closing_axis(self, data: Any, target: Any) -> tuple[float, float, float]:
        """The world line to close along at `target`."""
        import numpy as np  # noqa: PLC0415

        if self.closing is not None:
            return self.closing
        radial = (
            np.asarray(target, dtype=float)[:2]
            - np.asarray(data.xanchor[self.joint_ids[0]])[:2]
        )
        reach = float(np.linalg.norm(radial))
        if reach < _ABOVE_THE_BASE_M:
            return (1.0, 0.0, 0.0)
        return (float(-radial[1] / reach), float(radial[0] / reach), 0.0)

    def jaw(self, fraction: float) -> float:
        """0 = closed, 1 = open, as this rig's ctrl value."""
        clipped = min(1.0, max(0.0, float(fraction)))
        return self.jaw_closed + clipped * (self.jaw_open - self.jaw_closed)

    def grasp_point(self, data: Any) -> Any:
        """Where the jaws meet right now: the site plus the pocket."""
        import numpy as np  # noqa: PLC0415

        rotation = np.asarray(data.site_xmat[self.site_id]).reshape(3, 3)
        return np.asarray(data.site_xpos[self.site_id]) + rotation @ np.asarray(
            self.pocket
        )


class Beat(NamedTuple):
    """One planned move: the grasp point's target, the jaw fraction,
    the seconds to blend there, the world line the jaws close along
    (None leaves the roll free) and which axes the reach is corrected
    on closed-loop: "xyz" for the free-space beats the descents start
    from (droop), "xy" for the descents themselves, which END in
    contact - pushing their height error into a table, a cube or a
    pocket wall jams the jaw (measured 2026-09-02 on the insert)."""

    name: str
    target: tuple[float, float, float]
    jaw: float
    seconds: float
    closing: tuple[float, float, float] | None = None
    corrected: str | None = None


@dataclass(frozen=True)
class PlannerKnobs:
    """The planner's geometry and timing - a content-stamped record, so
    a planned batch says which planner made it."""

    hover_clearance_m: float = 0.06  # above the object's top
    grasp_depth_m: float = 0.0  # grasp centre relative to the object's centre
    carry_height_m: float = 0.10  # above the table, object in hand
    place_clearance_m: float = 0.01  # above the goal's resting height
    retract_height_m: float = 0.10
    # Timing: the worst case (every correction round taken) must fit
    # the SO-101 tasks' 7.2 s horizons with the referee's hold tail to
    # spare - 6.9 s here; the typical plan takes well under 5 s.
    hover_seconds: float = 0.6
    descend_seconds: float = 0.6
    close_seconds: float = 0.4
    lift_seconds: float = 0.6
    carry_seconds: float = 0.8
    lower_seconds: float = 0.6
    open_seconds: float = 0.3
    correction_rounds: int = 3
    correction_done_m: float = 0.003
    correction_seconds: float = 0.2
    ik_pos_tol_m: float = 0.005
    ik_down_weight: float = 0.5
    ik_max_iters: int = 250
    ik_damping: float = 1e-2

    @property
    def stamp(self) -> str:
        """`planner@<hash of every knob>` - the expert stamp a planned
        batch carries, so two batches from different knobs never share."""
        from dataclasses import asdict  # noqa: PLC0415

        from trainnr.bundles.hashing import content_stamp  # noqa: PLC0415

        return content_stamp("planner", asdict(self))


@dataclass(frozen=True)
class Place:
    """Put the object down with its centre at `xyz`, released from
    `clearance_m` above it (None = the knobs' default; a walled pocket
    the fingers cannot enter wants its wall height plus a margin)."""

    xyz: tuple[float, float, float]
    clearance_m: float | None = None


@dataclass(frozen=True)
class Lift:
    """Hold the object clear at `height_m` above the table."""

    height_m: float


class PickPlacePlanner:
    """Beats from poses: pick an object by a top-down pinch, then lift
    it or place it. The beats are Cartesian targets for the grasp point;
    `run` turns each into joints by IK on a scratch copy and commands
    it over its seconds - the kitting choreography's loop, generic."""

    def __init__(self, knobs: PlannerKnobs | None = None) -> None:
        self.knobs = knobs or PlannerKnobs()

    def plan(
        self,
        object_xyz: Any,
        object_half: Any,
        goal: Lift | Place,
        closing: tuple[float, float, float],
    ) -> list[Beat]:
        """The beats for picking the object and doing `goal` with it.
        `closing` is the world line the jaws close along, held through
        the whole plan so the object keeps the yaw it was picked with."""
        k = self.knobs
        x, y, z = (float(v) for v in object_xyz)
        top = z + float(object_half[2])
        grasp_z = z + k.grasp_depth_m
        hover = (x, y, top + k.hover_clearance_m)
        beats = [
            Beat("hover", hover, 1.0, k.hover_seconds, closing, "xyz"),
            Beat("descend", (x, y, grasp_z), 1.0, k.descend_seconds, closing, "xy"),
            Beat("close", (x, y, grasp_z), 0.0, k.close_seconds, closing),
            Beat("lift", (x, y, k.carry_height_m), 0.0, k.lift_seconds, closing),
        ]
        if isinstance(goal, Lift):
            hold = (x, y, goal.height_m)
            beats.append(Beat("hold", hold, 0.0, k.lift_seconds, closing))
            return beats
        gx, gy, gz = (float(v) for v in goal.xyz)
        clearance = (
            k.place_clearance_m if goal.clearance_m is None else goal.clearance_m
        )
        release = (gx, gy, gz + clearance)
        carry = (gx, gy, k.carry_height_m)
        retract = (gx, gy, k.retract_height_m)
        beats += [
            Beat("carry", carry, 0.0, k.carry_seconds, closing, "xyz"),
            Beat("lower", release, 0.0, k.lower_seconds, closing, "xy"),
            Beat("open", release, 1.0, k.open_seconds, closing),
            Beat("retract", retract, 1.0, k.lift_seconds, closing),
        ]
        return beats

    def reach(
        self, episode: Episode, gripper: Gripper, beat: Beat, target: Any
    ) -> None:
        """Solve the beat's target for the grasp point on the scratch
        copy (two passes: the pocket's world offset depends on the
        solved orientation), then command it. A reach the IK cannot
        make is refused - the choreography must not pretend."""
        import numpy as np  # noqa: PLC0415

        from trainnr.robot.arm_ik import solve_arm_ik  # noqa: PLC0415

        k = self.knobs
        scratch = episode.scratch
        scratch.qpos[:] = episode.data.qpos
        site_target = np.asarray(target, dtype=float)
        for _ in range(2):
            reached = solve_arm_ik(
                episode.stepper.model,
                scratch,
                site=gripper.site,
                joints=gripper.ik_joints,
                grip_geoms=gripper.grip_geoms,
                target_pos=site_target,
                down_weight=k.ik_down_weight,
                closing_axis=beat.closing,
                pos_tol=k.ik_pos_tol_m,
                max_iters=k.ik_max_iters,
                damping=k.ik_damping,
            )
            rotation = np.asarray(scratch.site_xmat[gripper.site_id]).reshape(3, 3)
            site_target = np.asarray(target, dtype=float) - rotation @ np.asarray(
                gripper.pocket
            )
        if not reached:
            raise RuntimeError(
                f"IK failed: {beat.name} to {tuple(round(float(v), 3) for v in target)}"
            )
        target_ctrl = episode.ctrl.copy()
        for qpos_address, ctrl_index in zip(
            gripper.joint_qpos, gripper.joint_ctrl, strict=True
        ):
            target_ctrl[ctrl_index] = scratch.qpos[qpos_address]
        target_ctrl[gripper.jaw_ctrl] = gripper.jaw(beat.jaw)
        episode.advance(beat.seconds, target_ctrl)

    def run(self, episode: Episode, gripper: Gripper, beats: list[Beat]) -> None:
        """Every beat in order; the corrected ones re-reach from the
        measured grasp-point error until it is within tolerance."""
        import numpy as np  # noqa: PLC0415

        k = self.knobs
        for beat in beats:
            if episode.stepper.done:
                return
            self.reach(episode, gripper, beat, beat.target)
            if not beat.corrected:
                continue
            mask = CORRECTED_AXES[beat.corrected]
            for _ in range(k.correction_rounds):
                error = mask * (
                    gripper.grasp_point(episode.data) - np.asarray(beat.target)
                )
                if float(np.linalg.norm(error)) < k.correction_done_m:
                    break
                corrected = tuple(float(v) for v in np.asarray(beat.target) - error)
                self.reach(
                    episode,
                    gripper,
                    beat._replace(seconds=k.correction_seconds),
                    corrected,
                )
