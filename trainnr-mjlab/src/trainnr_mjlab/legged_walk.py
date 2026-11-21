"""A walk for any onboarded legged robot (#100): microduck's recipe (the
transcription `microduck_walk` cites line by line) over a `LeggedRobot`
that names everything the recipe reads off a robot: its bundle and model
file, the actuator (a certified or identified BAM bundle, or the model's
own actuators, declared and so recorded), the trunk, the feet, the joints
the policy drives and the ones it holds, the stand pose, the reset
height, the command ranges and the pose reward's widths.

`microduck_walk.microduck_walk_env_cfg` is this recipe with microduck's
values; a declared legged walk (`trainnr.tasks.walks`, family
`legged-walk`) is this recipe with the values its spec names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from trainnr.bundles.hashing import STAMP_SEPARATOR

from trainnr_mjlab.actuator import BamActuatorCfg
from trainnr_mjlab.dr import bam_param_dr_event
from trainnr_mjlab.entity import entity_from_bundle
from trainnr_mjlab.events import bam_expansion_event
from trainnr_mjlab.linter import lint
from trainnr_mjlab.microduck_walk import (
    AIR_TIME_WINDOW,
    ANG_VEL_Z,
    ARMATURE,
    COMMAND_THRESHOLD,
    FOOT_FRICTION,
    FOOT_TARGET_HEIGHT,
    HEAD_FREE,
    HEAD_PINNED,
    IDENTIFIED,
    LAW_DR_SPAN,
    LIN_VEL_X,
    LIN_VEL_Y,
    LS_ITERATIONS,
    MASS_INERTIA,
    NCONMAX,
    PHYSICS_DT,
    PLAY_PUSH_INTERVAL_S,
    POSE_STD_STANDING,
    POSE_STD_WALKING,
    PUSH_INTERVAL_S,
    PUSH_VELOCITY,
    SOLVER_ITERATIONS,
    TRACK_ANG_STD,
    TRACK_LIN_STD,
    TRUNK_COM_OFFSET,
    UPRIGHT_STD,
    WALKING_THRESHOLD,
)

# The actuator stamp a walk on the model's own actuators records: the
# robot's own content (its MJCF carries the gains), named as declared.
MODEL_ACTUATORS = "{}-model-actuators@{}"
MODEL_DR_BASIS = "none: the model's own actuators, declared, not identified"


@dataclass(frozen=True)
class LeggedRobot:
    """Everything the walk recipe reads off a robot. `actuator` None is
    the model's own actuators (declared); `pinned` are joint patterns held
    at the default pose and left out of the action; `init_state` the stand
    pose (None: mjlab's default, every joint at zero)."""

    name: str
    robot_dir: Path
    model_file: str
    trunk: str
    foot_sites: tuple[str, ...]
    foot_geoms: tuple[str, ...]
    servo_joints: tuple[str, ...]
    pose_joints: tuple[str, ...]
    reset_z: tuple[float, float]
    actuator: Path | None = None
    pinned: tuple[str, ...] = ()
    init_state: EntityCfg.InitialStateCfg | None = None
    without_floor: bool = False  # an onboarded scene's floor plane dropped
    # the robot's limits over several joints (#121) and what breaking one
    # does in training: LIMITS_TERMINATE ends the episode, LIMITS_COST pays
    limits_file: Path | None = None
    limits_mode: str = "terminate"
    lin_vel_x: tuple[float, float] = LIN_VEL_X
    lin_vel_y: tuple[float, float] = LIN_VEL_Y
    ang_vel_z: tuple[float, float] = ANG_VEL_Z
    pose_std_standing: Mapping[str, float] = field(
        default_factory=lambda: dict(POSE_STD_STANDING)
    )
    pose_std_walking: Mapping[str, float] = field(
        default_factory=lambda: dict(POSE_STD_WALKING)
    )


def legged_walk_env_cfg(  # noqa: PLR0915 - the recipe's knobs; one linear transcription
    robot: LeggedRobot,
    *,
    play: bool = False,
    law_dr_span: float | str | None = LAW_DR_SPAN,
    law_pin_scale: float | None = None,
    law_pin_only: tuple[str, ...] | None = None,
) -> tuple[ManagerBasedRlEnvCfg, dict[str, str]]:
    """The walk cfg and its identity: robot stamp, actuator stamp, the
    law-DR basis and whether joints are held. `law_dr_span` is the
    declared span around a BAM bundle's point fit (None or 0: none);
    a walk on the model's own actuators has no law to randomise."""
    cfg = make_velocity_env_cfg()
    if cfg.sim.mujoco.timestep != PHYSICS_DT:
        raise AssertionError(
            f"mjlab's base timestep moved to {cfg.sim.mujoco.timestep}; this "
            f"transcription is pinned against {PHYSICS_DT} (50 Hz control) — "
            "re-derive the delay steps and rates before trusting it"
        )

    # Their solver raise for the duck.
    cfg.sim.nconmax = NCONMAX
    cfg.sim.mujoco.iterations = SOLVER_ITERATIONS
    cfg.sim.mujoco.ls_iterations = LS_ITERATIONS

    # The robot: its stamped bundle; the actuator: a certified (or
    # identified) BAM bundle, or the model's own actuators, declared.
    bam = robot.actuator is not None
    actuator: Any = (
        BamActuatorCfg.from_bundle(
            robot.actuator, target_names_expr=robot.servo_joints, physics_dt=PHYSICS_DT
        )
        if robot.actuator is not None
        # the walk sends position targets: a model whose actuators are not
        # position actuators is refused by mjlab, not driven limp
        else XmlActuatorCfg(
            target_names_expr=robot.servo_joints, command_field="position"
        )
    )
    entity, robot_stamp = entity_from_bundle(
        robot.robot_dir,
        robot.model_file,
        (actuator,),
        init_state=robot.init_state,
        without_floor=robot.without_floor,
    )
    actuator_stamp = (
        actuator.stamp
        if bam
        else MODEL_ACTUATORS.format(*robot_stamp.split(STAMP_SEPARATOR, 1))
    )
    cfg.scene.entities = {"robot": entity}
    cfg.viewer.body_name = robot.trunk

    # Flat terrain (their walk; height scan leaves both obs groups).
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Sensors: their three, on our model's names. Their self-collision
    # sensor targets their `self_collision_only`-classed geoms; the
    # subtree-vs-subtree form covers the same contacts on 1.6.
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=robot.foot_geoms, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=robot.trunk, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=robot.trunk, entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=s, entity="robot") for s in robot.foot_sites
        ),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )
    cfg.scene.sensors = (feet_ground, self_collision, foot_height_scan)

    # Action: scale 1.0, EXPLICIT — the value their deploy flag
    # `--action-scale 0.8` contradicts (57 §5, confirmed at the source);
    # the C4 manifest exists so that contradiction cannot ship.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    if robot.pinned:
        # pinned joints hold their default pose, out of the action
        joint_pos_action.actuator_names = (f"^(?!{'|'.join(robot.pinned)}).*",)

    # Commands: their ranges; heading control off (their duck steers by
    # yaw-rate command, not heading pursuit).
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=robot.lin_vel_x,
        lin_vel_y=robot.lin_vel_y,
        ang_vel_z=robot.ang_vel_z,
        # No heading range: 1.6's UniformVelocityCommand refuses a
        # heading range while heading_command is off (caught by the
        # box's first real env build, 2026-09-01).
        heading=None,
    )
    twist.heading_command = False

    # Observations: actor loses base_lin_vel (critic keeps it —
    # privileged); flat terrain removes height_scan from both.
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Rewards: their numbers onto 1.6's terms.
    rewards = cfg.rewards
    rewards["track_linear_velocity"].weight = 2.0
    rewards["track_linear_velocity"].params["std"] = TRACK_LIN_STD
    rewards["track_angular_velocity"].weight = 2.0
    rewards["track_angular_velocity"].params["std"] = TRACK_ANG_STD
    rewards["upright"].weight = 2.0
    rewards["upright"].params["std"] = UPRIGHT_STD
    rewards["upright"].params["asset_cfg"].body_names = (robot.trunk,)
    rewards["upright"].params.pop("terrain_sensor_names", None)
    rewards["pose"].params["std_standing"] = dict(robot.pose_std_standing)
    rewards["pose"].params["std_walking"] = dict(robot.pose_std_walking)
    rewards["pose"].params["std_running"] = dict(robot.pose_std_walking)
    rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=robot.pose_joints
    )
    rewards["pose"].params["walking_threshold"] = WALKING_THRESHOLD
    rewards["air_time"].weight = 3.0
    rewards["air_time"].params["threshold_min"] = AIR_TIME_WINDOW[0]
    rewards["air_time"].params["threshold_max"] = AIR_TIME_WINDOW[1]
    rewards["air_time"].params["command_threshold"] = COMMAND_THRESHOLD
    for name in ("foot_clearance", "foot_slip"):
        rewards[name].params["asset_cfg"].site_names = robot.foot_sites
    rewards["foot_clearance"].params["target_height"] = FOOT_TARGET_HEIGHT
    rewards["foot_clearance"].params["command_threshold"] = COMMAND_THRESHOLD
    rewards["foot_swing_height"].params["target_height"] = FOOT_TARGET_HEIGHT
    rewards["foot_swing_height"].params["command_threshold"] = COMMAND_THRESHOLD
    rewards["foot_slip"].weight = -0.1  # their "deliberately weak"
    rewards["foot_slip"].params["command_threshold"] = COMMAND_THRESHOLD
    rewards["action_rate_l2"].weight = -0.1  # stage-0; their ramp is a curriculum
    rewards["body_ang_vel"].weight = -0.05
    rewards["body_ang_vel"].params["asset_cfg"].body_names = (robot.trunk,)
    rewards["angular_momentum"].weight = -0.02
    rewards.pop("soft_landing", None)  # theirs: "velocity removes it"
    rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision.name},
    )

    # Terminations: base's, plus the NaN guard — theirs in spirit,
    # 1.6's native `nan_detection` in mechanism.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=envs_mdp.nan_detection, time_out=False
    )
    limits = _limits_terms(cfg, robot)

    # Events: their numbers through 1.6's native functions, plus ours.
    events = cfg.events
    events["foot_friction"].params["asset_cfg"].geom_names = robot.foot_geoms
    events["foot_friction"].params["ranges"] = FOOT_FRICTION
    events["reset_base"].params["pose_range"]["z"] = robot.reset_z
    if play:
        # `play` is for WATCHING a policy, so the disturbances it must
        # survive stay on and come more often (theirs: a viewer sees
        # recovery), while observation corruption goes off — a viewer
        # wants the true state, not the actor's noisy view. Stated
        # because the first cut changed only the push interval, which
        # reads as an inversion rather than a mode (review 2026-09-01).
        events["push_robot"].interval_range_s = PLAY_PUSH_INTERVAL_S
        cfg.observations["actor"].enable_corruption = False
    else:
        events["push_robot"].interval_range_s = PUSH_INTERVAL_S
    events["push_robot"].params["velocity_range"] = {
        "x": PUSH_VELOCITY,
        "y": PUSH_VELOCITY,
    }
    events["base_com"].params["asset_cfg"].body_names = (robot.trunk,)
    # 1.6's Ranges dict keys are ENTITY-NAME patterns, not axes (the
    # box's env build refused 'x' as a body name, 2026-09-01); a plain
    # tuple applies over body_com_offset's default axes [0, 1, 2] —
    # exactly their all-axes +-offset.
    events["base_com"].params["ranges"] = (-TRUNK_COM_OFFSET, TRUNK_COM_OFFSET)
    events["mass_inertia"] = EventTermCfg(
        func=envs_mdp.dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(robot.trunk,)),
            # their derivation: e^(2·alpha) spans the ±5 % mass range
            "alpha_range": (
                math.log(MASS_INERTIA[0]) / 2.0,
                math.log(MASS_INERTIA[1]) / 2.0,
            ),
        },
    )
    events["armature"] = EventTermCfg(
        func=envs_mdp.dr.joint_armature,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=robot.servo_joints),
            "operation": "scale",
            "ranges": ARMATURE,
        },
    )
    # Ours: the expansion event (their decorator-carrier no-op, owned),
    # and law DR from the certified bundle with a declared basis.
    if bam:
        events["bam_expansion"] = bam_expansion_event()
    if not bam:
        dr_basis = MODEL_DR_BASIS
    elif law_pin_scale is not None:
        # The envelope probe: the law at a fixed offset from the fit.
        law_dr, dr_basis = bam_param_dr_event(
            actuator, pin_scale=law_pin_scale, pin_only=law_pin_only
        )
        events["bam_param_dr"] = law_dr
    elif law_dr_span == IDENTIFIED:
        # No fallback: the bundle's interval or a refusal.
        law_dr, dr_basis = bam_param_dr_event(actuator, fallback_span=None)
        events["bam_param_dr"] = law_dr
    elif law_dr_span:
        law_dr, dr_basis = bam_param_dr_event(
            actuator, fallback_span=float(law_dr_span)
        )
        events["bam_param_dr"] = law_dr
    else:
        dr_basis = "none: the bundle's point fit exactly (no law DR)"

    # Curricula: none — see the module docstring's OMITTED list.
    cfg.curriculum = {}

    # Zero silent no-ops, by construction — the linter is the gate, not
    # a review habit.
    lint(cfg.events, (actuator,))

    identity = {
        "robot": robot_stamp,
        "actuator": actuator_stamp,
        "dr_basis": dr_basis,
        "head": HEAD_PINNED if robot.pinned else HEAD_FREE,
    }
    if limits:
        identity["limits"] = limits
    return cfg, identity


LIMITS_TERMINATE, LIMITS_COST = "terminate", "cost"
LIMITS_MODES = (LIMITS_TERMINATE, LIMITS_COST)


def _limits_terms(cfg: ManagerBasedRlEnvCfg, robot: LeggedRobot) -> str:
    """The robot's declared limits over several joints in the walk: a
    termination (a pose the robot refuses ends the episode) or a cost per
    limit broken. Returns the identity's line, or "" with no limits."""
    if robot.limits_file is None:
        return ""
    from trainnr.bundles.hashing import stamp  # noqa: PLC0415

    from trainnr_mjlab.limits_term import limits_broken, limits_count  # noqa: PLC0415

    if robot.limits_mode not in LIMITS_MODES:
        raise ValueError(
            f"limits is one of {', '.join(LIMITS_MODES)}, got {robot.limits_mode!r}"
        )
    params = {"limits_file": str(robot.limits_file)}
    if robot.limits_mode == LIMITS_TERMINATE:
        cfg.terminations["joint_limits"] = TerminationTermCfg(
            func=limits_broken, time_out=False, params=params
        )
    else:
        cfg.rewards["joint_limits"] = RewardTermCfg(
            func=limits_count, weight=-1.0, params=params
        )
    return f"{robot.limits_mode}: {stamp('limits', robot.limits_file)}"


# A declared legged walk's spec, as the trainnr side resolves it (paths
# absolute, the actuator a bundle file or MODEL): the keys it reads.
MODEL = "model"
LIBRARY = "library:"
SPEC_KEYS = frozenset(
    {
        "name",
        "robot",
        "model_file",
        "actuator",
        "stand_keyframe",
        "trunk",
        "foot_sites",
        "foot_geoms",
        "pinned_joints",
        "reset_z",
        "command",
        "dr_span",
        "limits",
    }
)
# Above the stand keyframe's height, the reset's spread (m).
RESET_ABOVE_STAND = (0.0, 0.01)
# A pose joint no width pattern names: microduck's widths for a leg's
# middle joints.
POSE_STD_FALLBACK = (0.15, 0.4)


def legged_robot(spec: Mapping[str, Any]) -> LeggedRobot:
    """A `LeggedRobot` from a declared legged walk's spec: every hinge the
    model has is driven unless pinned, the stand keyframe is the default
    pose and the reset, the pose reward's widths are microduck's by joint
    name (a fallback for names it does not know). Refuses by name a spec
    key it does not read, a model without a floating base, and a keyframe
    the model does not have."""
    import re  # noqa: PLC0415

    import mujoco  # noqa: PLC0415
    from trainnr.bundles.locate import find_bundle  # noqa: PLC0415
    from trainnr.robot.actuator_bundle import BUNDLE_STORE  # noqa: PLC0415
    from trainnr.robot.joint_limits import LIMITS_FILE  # noqa: PLC0415

    unknown = sorted(set(spec) - SPEC_KEYS)
    if unknown:
        raise ValueError(f"a legged walk's spec has no {', '.join(unknown)}")
    name = str(spec["robot"]).split("@", 1)[0]
    found = find_bundle(name)  # the project's robots first (use_project)
    if found is None:
        raise FileNotFoundError(f"no robot {name!r} in the project or the library")
    bundle_dir = Path(found)
    model = mujoco.MjModel.from_xml_path(str(bundle_dir / spec["model_file"]))
    obj = mujoco.mjtObj
    names = [mujoco.mj_id2name(model, obj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    free = [
        n
        for j, n in enumerate(names)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free) != 1:
        raise ValueError(
            f"{spec['robot']} has no floating base (one free joint) to walk on"
        )
    hinges = [
        n
        for j, n in enumerate(names)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    pinned = [
        n
        for n in hinges
        if any(re.fullmatch(p, n) for p in spec.get("pinned_joints") or [])
    ]
    exact = lambda joint: f"^{re.escape(joint)}$"  # noqa: E731
    driven = [n for n in hinges if n not in pinned]

    def widths(table: Mapping[str, float], fallback: float) -> dict[str, float]:
        return {
            exact(n): next(
                (v for p, v in table.items() if re.fullmatch(p, n)), fallback
            )
            for n in driven
        }

    init_state = None
    reset_z = tuple(spec.get("reset_z") or RESET_ABOVE_STAND)
    key = str(spec.get("stand_keyframe") or "")
    if key:
        # by name, or `#<index>` for an unnamed keyframe
        k = (
            int(key[1:])
            if key.startswith("#")
            else mujoco.mj_name2id(model, obj.mjOBJ_KEY, key)
        )
        if not 0 <= k < model.nkey:
            raise ValueError(f"{spec['robot']} has no keyframe {key!r}")
        qpos = model.key_qpos[k]
        root = int(
            model.jnt_qposadr[names.index(free[0])]
        )  # the free joint's own place
        init_state = EntityCfg.InitialStateCfg(
            pos=(float(qpos[root]), float(qpos[root + 1]), float(qpos[root + 2])),
            rot=tuple(float(v) for v in qpos[root + 3 : root + 7]),  # type: ignore[arg-type]
            joint_pos={
                exact(n): float(qpos[model.jnt_qposadr[names.index(n)]]) for n in hinges
            },
        )
    command = spec.get("command") or {}
    actuator = str(spec.get("actuator") or MODEL)
    limits = spec.get("limits") or {}
    if actuator.startswith(LIBRARY):  # a certified bundle in the library
        bundle_file: Path | None = (
            BUNDLE_STORE / f"{actuator.removeprefix(LIBRARY)}.bundle.json"
        )
    elif actuator != MODEL:  # a bundle file in the robot's project
        bundle_file = bundle_dir.resolve().parent.parent / actuator
    else:
        bundle_file = None
    return LeggedRobot(
        name=str(spec["robot"]),
        robot_dir=bundle_dir,
        model_file=str(spec["model_file"]),
        trunk=str(spec["trunk"]),
        foot_sites=tuple(spec["foot_sites"]),
        foot_geoms=tuple(spec["foot_geoms"]),
        servo_joints=tuple(exact(n) for n in hinges),
        pose_joints=tuple(exact(n) for n in driven),
        reset_z=(float(reset_z[0]), float(reset_z[1])),
        actuator=bundle_file,
        pinned=tuple(exact(n) for n in pinned),
        init_state=init_state,
        without_floor=True,
        limits_file=bundle_dir / LIMITS_FILE if limits else None,
        limits_mode=str(limits.get("mode", LIMITS_TERMINATE)),
        lin_vel_x=tuple(command.get("lin_vel_x", LIN_VEL_X)),
        lin_vel_y=tuple(command.get("lin_vel_y", LIN_VEL_Y)),
        ang_vel_z=tuple(command.get("ang_vel_z", ANG_VEL_Z)),
        pose_std_standing=widths(POSE_STD_STANDING, POSE_STD_FALLBACK[0]),
        pose_std_walking=widths(POSE_STD_WALKING, POSE_STD_FALLBACK[1]),
    )
