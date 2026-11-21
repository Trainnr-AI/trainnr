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
from trainnr.paths import checkout

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
# The head (measured 2026-09-05): four actuated joints - neck_pitch,
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


def microduck_walk_env_cfg(  # noqa: PLR0913 - the recipe's knobs; one linear transcription, each statement a cited number
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
    from trainnr_mjlab.legged_walk import (  # noqa: PLC0415 - it imports this module
        LeggedRobot,
        legged_walk_env_cfg,
    )

    if head not in (HEAD_FREE, HEAD_PINNED):
        raise ValueError(f"head must be {HEAD_FREE!r} or {HEAD_PINNED!r}, got {head!r}")
    robot = LeggedRobot(
        name="microduck",
        robot_dir=ROBOT_DIR,
        model_file=MODEL_FILE,
        trunk=TRUNK,
        foot_sites=FOOT_SITES,
        foot_geoms=FOOT_GEOMS,
        servo_joints=SERVO_JOINTS,
        pose_joints=POSE_JOINTS,
        reset_z=RESET_Z,
        actuator=bundle or XL330_BUNDLE,
        pinned=HEAD_ACTUATORS if head == HEAD_PINNED else (),
    )
    return legged_walk_env_cfg(
        robot,
        play=play,
        law_dr_span=law_dr_span,
        law_pin_scale=law_pin_scale,
        law_pin_only=law_pin_only,
    )
