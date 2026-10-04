"""microduck's walk on mjlab 1.6, through the certified stack — flagship G2.

The three-source transcription docs/e2e-research/63 §2.2 scoped: mjlab
1.3's velocity base (their baseline), Pollen's 949-line delta file
(`microduck_rl:src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
at d424a0c — every number below cites it), and mjlab 1.6's velocity
base (this build's substrate). Same recipe, different guarantees:

- the robot comes from OUR stamped bundle and the actuator from the
  CERTIFIED xl330.m6 (their laptop-path params, 57 §5, cannot recur);
- law-parameter DR goes through `bam_param_dr_event` with a DECLARED
  basis (their hand-typed ±10 % scalar, folklore beside a fit treated
  as exact, becomes a recorded declaration);
- their TWO silent no-ops (`randomize_joint_friction`/`_damping` on
  fields the BAM actuator overwrites every step) are not ported — the
  linter REFUSES them by construction, and the test suite shows it;
- their decorator-carrier no-op event every env had to remember is our
  `bam_expansion_event`, registered here, once.

Adopted from them with credit (63 §1 "respect what they got right"):
the NaN termination (1.6 now ships `mdp.nan_detection` natively — their
1.3-era custom is retired upstream), the foot-sensor layout, and the
audited trunk-CoM range.

Deliberately OMITTED in G2, each with its reason:
- their mutation-based curricula (`standing_envs`, CoM-cap ramps,
  `action_rate` -0.1→-1.0, command widening): mjlab curricula rewrite
  the live cfg, and ours must stay hashable (63 §2.2's named tension).
  Stage-0 values are used throughout; the G3 re-expression decides how
  ramps return without mutating identity.
- their 61-D deployment obs contract (head/body zero-padded command
  slots, `head_pose_tracking`): deployment parity, not the walk — the
  C4 manifest work owns that boundary.
- their kp/kd DR (a custom hook into their delayed actuator): kp is a
  LAW parameter here; law DR covers the electrical side, and a gains
  span they never justified is not re-declared by us.
- their head-CoM DR: its body list names their model revision's
  assembly; ours differs, and a guessed body set is exactly what this
  stack refuses.
"""

from __future__ import annotations

import math
from pathlib import Path

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
from trainnr.paths import checkout

from trainnr_mjlab.actuator import BamActuatorCfg
from trainnr_mjlab.dr import bam_param_dr_event
from trainnr_mjlab.entity import entity_from_bundle
from trainnr_mjlab.events import bam_expansion_event
from trainnr_mjlab.linter import lint

# The checkout: `$TRAINNR_REPO`, else the ancestor that carries trainnr.
REPO = checkout()
ROBOT_DIR = REPO / "robots" / "microduck"
MODEL_FILE = "robot_walk.xml"
XL330_BUNDLE = REPO / "robots" / "actuator-bundles" / "xl330.m6.bundle.json"
# Our refit of the same model on Rhoban's public logs, with its
# bootstrap interval (docs/e2e-research/72): the bundle the
# identified-interval arm and its own point arm train on.
XL330_REFIT_BUNDLE = REPO / "robots" / "actuator-bundles" / "xl330-refit.m6.bundle.json"

# The bundle's actuated joints — everything but the freejoint; all
# fourteen are XL330s (57 §2, confirmed against the compiled model).
SERVO_JOINTS = (r"^(?!trunk_base_freejoint).*",)
TRUNK = "trunk_base"
# The head (the log 2026-09-05): four actuated joints - neck_pitch,
# head_pitch, head_yaw, head_roll - that the walk rewards nowhere
# (upstream's head is command-driven at deployment, and that command is
# not part of the walk), so the policy parks them at their limits and
# waves; the model has no collision geometry above the feet, so the
# head passes through the shoulders - on hardware a self-collision.
# `pinned` takes them out of the action space: their position servos
# hold zero (the neutral pose) and the policy commands the legs only.
HEAD_FREE, HEAD_PINNED = "free", "pinned"
HEAD_ACTUATORS = (r".*neck.*", r".*head.*")
LEG_ACTUATORS = (r"^(?!.*neck.*|.*head.*).*",)
FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")

# Their solver raise for the duck (delta file: nconmax 200 "was 35",
# iterations 30, ls_iterations 50).
NCONMAX = 200
SOLVER_ITERATIONS = 30
LS_ITERATIONS = 50

# Their command ranges (delta file; 63 §2.1).
LIN_VEL_X = (-0.4, 0.4)
LIN_VEL_Y = (-0.3, 0.3)
ANG_VEL_Z = (-1.0, 1.0)

# Their reward numbers, verbatim (delta file; comments theirs in spirit):
TRACK_LIN_STD = math.sqrt(0.1)
TRACK_ANG_STD = math.sqrt(0.5)
UPRIGHT_STD = math.sqrt(0.05)  # "2.0 / std²=0.05 ... hold the trunk level"
AIR_TIME_WINDOW = (0.125, 0.300)
COMMAND_THRESHOLD = 0.01
FOOT_TARGET_HEIGHT = 0.02  # "increased from 0.01 to penalize dragging"
POSE_STD_STANDING = {
    r".*hip_yaw.*": 0.1,
    r".*hip_roll.*": 0.05,  # their "hold the 5°-inward stance"
    r".*hip_pitch.*": 0.15,
    r".*knee.*": 0.15,
    r".*ankle.*": 0.1,
}
POSE_STD_WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,  # their "was 0.15"
}
# Pose shapes LEG joints only — their comment: head/neck are
# command-driven and a home-pull here fights the head objective.
POSE_JOINTS = (r"^(?!trunk_base_freejoint|.*neck.*|.*head.*).*",)
WALKING_THRESHOLD = 0.01

# Their DR numbers, verbatim (delta file constants block):
FOOT_FRICTION = (0.7, 1.3)  # "grippier footpad — narrowed from (0.3, 1.2)"
RESET_Z = (0.12, 0.13)
PUSH_VELOCITY = (-0.2, 0.2)
PUSH_INTERVAL_S = (3.0, 6.0)
PLAY_PUSH_INTERVAL_S = (0.5, 1.0)  # theirs: more often, so a viewer sees recovery
TRUNK_COM_OFFSET = 0.003  # their audited stage-0 value (the ramp is a curriculum)
MASS_INERTIA = (0.95, 1.05)
ARMATURE = (0.9, 1.1)

# The law-DR declaration: their ±10 % friction scalar
# (JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)) re-expressed as a
# declared span over the certified bundle's law parameters — recorded
# as the basis on the event and on every episode that samples it.
LAW_DR_SPAN = 0.10
# `law_dr_span=IDENTIFIED` draws from the bundle's own `uncertainty`
# section — the identified-interval arm the thesis names — and refuses
# a bundle that carries none (2026-09-05, docs/e2e-research/72).
IDENTIFIED = "identified"
# The mismatch axes a certificate can pin one at a time (walk_verdict
# --judge-param): the motor's strength (kt), its winding (R), and the
# whole friction family as one axis. "all" is the diagonal.
PIN_AXES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "kt": ("kt",),
    "R": ("R",),
    "friction": (
        "friction_base",
        "friction_stribeck",
        "load_friction_motor",
        "load_friction_external",
        "load_friction_motor_stribeck",
        "load_friction_external_stribeck",
        "load_friction_motor_quad",
        "load_friction_external_quad",
        "dtheta_stribeck",
    ),
}

# The physics rate this cfg is transcribed against: mjlab 1.6's base
# timestep x decimation 4 = 50 Hz control, the rate their deploy runs.
# Asserted at build so a silent upstream change fails loudly.
PHYSICS_DT = 0.005


def microduck_walk_env_cfg(  # noqa: PLR0913, PLR0915 - the recipe's knobs; one linear transcription, each statement a cited number
    *,
    play: bool = False,
    law_dr_span: float | str | None = LAW_DR_SPAN,
    law_pin_scale: float | None = None,
    law_pin_only: tuple[str, ...] | None = None,
    bundle: Path | None = None,
    head: str = HEAD_FREE,
) -> tuple[ManagerBasedRlEnvCfg, dict[str, str]]:
    """The walk cfg and its identity: robot stamp, actuator stamp, and
    the law-DR basis — the three strings a run's record must carry.

    `law_dr_span` is the DECLARED span around the bundle's point fit
    that the law-parameter DR draws from (the bundle carries no
    intervals — every bundle in the store is a point estimate,
    2026-09-04). `None` or 0 means NO law DR: the actuator runs at the
    fit exactly — the point arm of the walk C1 study, and the pinned
    truth every arm is judged at. The basis string says which."""
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

    # The robot: our stamped bundle, the certified actuator.
    actuator = BamActuatorCfg.from_bundle(
        bundle or XL330_BUNDLE, target_names_expr=SERVO_JOINTS, physics_dt=PHYSICS_DT
    )
    entity, robot_stamp = entity_from_bundle(ROBOT_DIR, MODEL_FILE, (actuator,))
    cfg.scene.entities = {"robot": entity}
    cfg.viewer.body_name = TRUNK

    # Flat terrain (their walk; height scan leaves both obs groups).
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Sensors: their three, on our model's names. Their self-collision
    # sensor targets their `self_collision_only`-classed geoms; the
    # subtree-vs-subtree form covers the same contacts on 1.6.
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=TRUNK, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=TRUNK, entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
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
    if head not in (HEAD_FREE, HEAD_PINNED):
        raise ValueError(f"head must be {HEAD_FREE!r} or {HEAD_PINNED!r}, got {head!r}")
    if head == HEAD_PINNED:
        joint_pos_action.actuator_names = LEG_ACTUATORS

    # Commands: their ranges; heading control off (their duck steers by
    # yaw-rate command, not heading pursuit).
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=LIN_VEL_X,
        lin_vel_y=LIN_VEL_Y,
        ang_vel_z=ANG_VEL_Z,
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
    rewards["upright"].params["asset_cfg"].body_names = (TRUNK,)
    rewards["upright"].params.pop("terrain_sensor_names", None)
    rewards["pose"].params["std_standing"] = POSE_STD_STANDING
    rewards["pose"].params["std_walking"] = POSE_STD_WALKING
    rewards["pose"].params["std_running"] = POSE_STD_WALKING
    rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=POSE_JOINTS
    )
    rewards["pose"].params["walking_threshold"] = WALKING_THRESHOLD
    rewards["air_time"].weight = 3.0
    rewards["air_time"].params["threshold_min"] = AIR_TIME_WINDOW[0]
    rewards["air_time"].params["threshold_max"] = AIR_TIME_WINDOW[1]
    rewards["air_time"].params["command_threshold"] = COMMAND_THRESHOLD
    for name in ("foot_clearance", "foot_slip"):
        rewards[name].params["asset_cfg"].site_names = FOOT_SITES
    rewards["foot_clearance"].params["target_height"] = FOOT_TARGET_HEIGHT
    rewards["foot_clearance"].params["command_threshold"] = COMMAND_THRESHOLD
    rewards["foot_swing_height"].params["target_height"] = FOOT_TARGET_HEIGHT
    rewards["foot_swing_height"].params["command_threshold"] = COMMAND_THRESHOLD
    rewards["foot_slip"].weight = -0.1  # their "deliberately weak"
    rewards["foot_slip"].params["command_threshold"] = COMMAND_THRESHOLD
    rewards["action_rate_l2"].weight = -0.1  # stage-0; their ramp is a curriculum
    rewards["body_ang_vel"].weight = -0.05
    rewards["body_ang_vel"].params["asset_cfg"].body_names = (TRUNK,)
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

    # Events: their numbers through 1.6's native functions, plus ours.
    events = cfg.events
    events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    events["foot_friction"].params["ranges"] = FOOT_FRICTION
    events["reset_base"].params["pose_range"]["z"] = RESET_Z
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
    events["base_com"].params["asset_cfg"].body_names = (TRUNK,)
    # 1.6's Ranges dict keys are ENTITY-NAME patterns, not axes (the
    # box's env build refused 'x' as a body name, 2026-09-01); a plain
    # tuple applies over body_com_offset's default axes [0, 1, 2] —
    # exactly their all-axes +-offset.
    events["base_com"].params["ranges"] = (-TRUNK_COM_OFFSET, TRUNK_COM_OFFSET)
    events["mass_inertia"] = EventTermCfg(
        func=envs_mdp.dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(TRUNK,)),
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=SERVO_JOINTS),
            "operation": "scale",
            "ranges": ARMATURE,
        },
    )
    # Ours: the expansion event (their decorator-carrier no-op, owned),
    # and law DR from the certified bundle with a declared basis.
    events["bam_expansion"] = bam_expansion_event()
    if law_pin_scale is not None:
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

    return cfg, {
        "robot": robot_stamp,
        "actuator": actuator.stamp,
        "dr_basis": dr_basis,
        "head": head,
    }
