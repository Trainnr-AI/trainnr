"""SO-101 task builders — the sim side of Paper 2, one task at a time.

Each builder composes the `so101-nominal` bundle into a scene via
`MjSpec` attach and returns the spec together with the episode protocol
that judges it. Two conventions, both learned the hard way:

- **The child's contact options do not survive attach** (measured:
  MuJoCo keeps the parent's defaults and drops the arm's
  `cone=elliptic, impratio=10` with a warning) — so every builder sets
  them on the scene spec explicitly. Losing them silently would change
  contact physics between "arm alone" and "arm in scene".
- **The referee's sensors ride in the scene, after the robot's.** The
  arm contributes sensordata[0:12] (six jointpos, six jointvel); task
  builders append what their predicates need (e.g. a fingertip
  framepos). Policies legitimately see these too — a real arm computes
  its own forward kinematics — but a task must never leak its TARGET
  through a sensor.

The first task is deliberately humble: `reach` — drive the fingertip to
a fixed target and hold. It exists to prove the composition, the
prefixing, and the harness plumbing end-to-end. `lift` is the first
manipulation task: a cube in the gripper's measured "pocket", a scripted
pick as the reference policy, success = cube held clear of the table.

A third lesson from building `lift`, worth its comment: **commanded is
not achieved under gravity at kp=50** — the nominal gains sag the
shoulder ~0.1 rad, which is ~2 cm at the fingertip, so every waypoint
below was tuned against the ACHIEVED pose (measured by probe), not the
commanded one. That droop is itself nominal-model behaviour Paper 1's
identified gains will change — another reason the nominal condition is
worth measuring rather than assuming.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from trainnr.bundles.locate import bundle_file, require_bundle_file
from trainnr.evaluate.vision import ARMNETBENCH_CAMERAS
from trainnr.physics.mujoco_backend import seat_at_keyframe
from trainnr.protocol import EpisodeProtocol, Placement
from trainnr.tasks.registry import register
from trainnr.tasks.scene import (
    TABLE_GEOM,
    add_free_box,
    add_slot_walls,
    pin_nominal_options,
    set_render_budget,
)
from trainnr.tasks.task import CONTROL_INTERVAL, PAIRED_TRIALS, Task

ARM_PREFIX = "arm_"
DEFAULT_ARM_XML = bundle_file("so101-nominal", "so101.xml")

# The SO-101 arm's sensor block: six jointpos then six jointvel.
# (aloha2.py declares its own for the 14-servo rig; each rig owns its
# census constant.)
ARM_SENSOR_WIDTH = 12
# The jointpos half is what a policy sees as `agent_pos`.
STATE_WIDTH = ARM_SENSOR_WIDTH // 2


FINGERTIP_SLICE = slice(ARM_SENSOR_WIDTH, ARM_SENSOR_WIDTH + 3)

# Episode design: 500 Hz physics (scene.NominalOptions.TIMESTEP), policies at 50 Hz —
# the same order of control rate the real bus sustains.
_STEPS = 600
_HOLD_STEPS = 50
# Task names: the registry key, `Task.name`, and the gym id's last part.
REACH = "reach"
LIFT = "lift"
LIFT_STUDY = "lift-study"
BLOCK_STACK = "block_stack"
TOOL_INSERT = "tool_insert"
RIG = "so101"
_REACH_TOLERANCE_M = 0.03
# FULLPHYSICS layout: [time, qpos(6), qvel(6)]; joint angles start at 1.
_QPOS_OFFSET = 1
# The gripper: the last servo; its open and closed setpoints, and the
# wrist-flat angle every scripted pose holds (measured poses, one name).
JAW_INDEX = STATE_WIDTH - 1
JAW_OPEN = 1.3
JAW_CLOSED = -0.15
WRIST_FLAT = -1.571
# The task objects, by body name: the lift's cube, and the stack/insert pair.
CUBE_BODY = "cube"
CUBE_A_BODY = "cube_a"
CUBE_B_BODY = "cube_b"
HOME_KEYFRAME = f"{ARM_PREFIX}home"  # the bundle's `home`, prefixed by attach
JAW_BODY = f"{ARM_PREFIX}Fixed_Jaw"
FINGERTIP_SITE = "fingertip"  # added post-attach: no prefix
GRASP_SITE = "grasp"  # the planner's point: where the closed jaws meet
# Where a HELD cube sits in the fixed jaw's frame, oriented so the
# site's z is the approach (the finger line) and its y the closing line
# - measured 2026-09-02 from the scripted lift's hold, 4 trials, spread
# under 1 mm (test-pinned). Not the commanded grip pose: that one is a
# droop-compensated overshoot whose kinematic pocket is 3.5 cm off.
_GRASP_POS = (-0.0041, -0.0915, -0.0065)
_GRASP_QUAT = (0.5034, 0.4958, 0.4966, -0.5041)


def _so101_task(
    name: str,
    scene: Any,
    protocol: EpisodeProtocol,
    instruction: str,
    arm_xml: Path,
    **extra: Any,
) -> Task:
    """Every scene here carries the ArmnetBench rig (the front/top/wrist
    cameras `_scene_with_arm` places) and the six-joint state block."""
    return Task(
        name=name,
        spec=scene,
        protocol=protocol,
        cameras=ARMNETBENCH_CAMERAS,
        state_width=STATE_WIDTH,
        instruction=instruction,
        bundle_dir=Path(arm_xml).parent,
        **extra,
    )


def _scene_with_arm(name: str, arm_xml: Path) -> Any:
    import mujoco  # noqa: PLC0415 - sim extra

    arm = mujoco.MjSpec.from_file(str(require_bundle_file(arm_xml)))
    scene = mujoco.MjSpec()
    scene.modelname = name
    # Restore what attach drops — see module docstring.
    pin_nominal_options(scene)
    scene.worldbody.add_geom(
        name=TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.4, 0.4, 0.02],
        pos=[0.0, -0.2, -0.02],
    )
    frame = scene.worldbody.add_frame(pos=[0, 0, 0])
    frame.attach_body(arm.worldbody.first_body(), ARM_PREFIX, "")
    _add_armnetbench_cameras(scene)
    # The fingertip: reach's referee reads it, and the planner expert
    # positions it (collect/choreography.Gripper) - one site on every
    # scene, so a planned pick is the same reach on every so101 task.
    jaw = scene.body(JAW_BODY)
    jaw.add_site(name=FINGERTIP_SITE, pos=[0.012, -0.08, 0.0], size=[0.005] * 3)
    jaw.add_site(name=GRASP_SITE, pos=_GRASP_POS, quat=_GRASP_QUAT, size=[0.004] * 3)
    scene.add_sensor(
        name="fingertip_pos",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=FINGERTIP_SITE,
    )
    return scene


def _add_armnetbench_cameras(scene: Any) -> None:
    """The three-camera rig every ArmnetBench policy was trained on.

    front/top/wrist, matching the dataset's observation keys. Placement
    is V1 GEOMETRY — real-frame matching (sim render beside dataset
    video, tools/camera-match.py) is the calibration step, deliberately
    done by eye against the 2,499 released episodes before any
    correlation is trusted; a vision policy can fail on cosmetics, and
    that failure must not be read as a dynamics gap.
    """
    # The offscreen framebuffer defaults to 640x480; the wrist camera
    # renders 1280x720, so the scene must say so or Renderer refuses.
    set_render_budget(scene)
    scene.worldbody.add_camera(
        name="front",
        pos=[0.0, -0.85, 0.25],
        # x flipped +: with x=[-1,0,0] the -z view axis pointed AWAY
        # from the arm and rendered pure black - caught by the
        # black-frame guard on its first outing.
        xyaxes=[1, 0, 0, 0, 0.35, 0.94],
        fovy=52,
    )
    scene.worldbody.add_camera(
        name="top",
        pos=[0.0, -0.25, 0.85],
        xyaxes=[-1, 0, 0, 0, -1, 0],
        fovy=58,
    )
    # The wrist camera rides the jaw; the body exists only post-attach.
    wrist_mount = scene.body(JAW_BODY)
    wrist_mount.add_camera(
        name="wrist",
        pos=[0.0, -0.06, 0.04],
        xyaxes=[0, -1, 0, 1, 0, 1],
        fovy=70,
    )


@register(REACH, rig=RIG)
def build_reach(arm_xml: Path = DEFAULT_ARM_XML) -> Task:
    """Reach: fingertip to the home keyframe's fingertip position, held.

    The target is COMPUTED from the bundle's own home keyframe rather
    than hardcoded, so a bundle whose geometry changes moves the target
    with it instead of silently invalidating the task.
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-reach", arm_xml)

    probe_model = scene.compile()
    probe_data = mujoco.MjData(probe_model)
    seat_at_keyframe(probe_model, probe_data, HOME_KEYFRAME)
    target = tuple(float(v) for v in probe_data.sensordata[FINGERTIP_SLICE])

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        # Deterministic paired starts: base and elbow offset per trial.
        initial[_QPOS_OFFSET + 0] += -0.3 + 0.2 * trial
        initial[_QPOS_OFFSET + 2] += 0.15 * trial - 0.2
        return initial

    def success(states: Any, sensors: Any) -> bool:
        tail = sensors[-_HOLD_STEPS:, FINGERTIP_SLICE]
        distances = np.linalg.norm(tail - np.array(target), axis=1)
        return bool(distances.max() < _REACH_TOLERANCE_M)

    return _so101_task(
        REACH,
        scene,
        EpisodeProtocol(
            trials=PAIRED_TRIALS,
            steps=_STEPS,
            control_interval=CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
        ),
        "reach the target and hold",
        arm_xml,
        target=target,
    )


# ---------------------------------------------------------------- lift --

# The gripper's grasp pocket, measured by probe on 2026-08-24: with the
# droop-compensated DESCEND pose held, the closed pads centre on this
# point, and the scripted pick lifts from anywhere within +-4 mm of it
# (15/15 in the jitter sweep).
_CUBE_HOME = (-0.006, -0.255, 0.015)
_CUBE_HALF = (0.012, 0.012, 0.015)
_LIFT_STEPS = 2500
_LIFTED_HEIGHT_M = 0.06
# FULLPHYSICS layout follows declaration order, and the arm is attached
# BEFORE the cube is added: [time(1), arm qpos(6), cube qpos(7), ...] —
# the cube's height is therefore state index 1+6+2 = 9. Pinned by test,
# because this index moved once already during development when the
# declaration order did.
CUBE_Z_STATE_INDEX = 9

# Phase boundaries in physics steps (2 ms each): 1.0 s hover to settle,
# 1.2 s descend, 0.6 s squeeze, lift for the remainder.
_DESCEND_AT_STEP = 500
_GRIP_AT_STEP = 1100
_LIFT_AT_STEP = 1400

_HOVER = [0.0, -1.2, 2.0, 0.9, WRIST_FLAT, JAW_OPEN]
_DESCEND = [0.0, -1.411, 2.225, 0.682, WRIST_FLAT, JAW_OPEN]
_GRIP = [0.0, -1.411, 2.225, 0.682, WRIST_FLAT, JAW_CLOSED]
_LIFT = [0.0, -1.57, 1.57, 1.57, WRIST_FLAT, JAW_CLOSED]


def scripted_pick(step: int, sensordata: Any) -> Any:
    """The reference policy: hover, descend, squeeze, lift.

    Open-loop by design — it is the task's competence ceiling for
    scripted control, and the harness's graded ladder measures neural
    policies against exactly this kind of ceiling later.
    """
    if step < _DESCEND_AT_STEP:
        return _HOVER
    if step < _GRIP_AT_STEP:
        return _DESCEND
    if step < _LIFT_AT_STEP:
        return _GRIP
    return _LIFT


def scripted_no_close(step: int, sensordata: Any) -> Any:
    """Approach without ever closing the jaw — the graded failure."""
    control = list(scripted_pick(step, sensordata))
    control[JAW_INDEX] = JAW_OPEN
    return control


def _add_free_cube(  # noqa: PLR0913 - a geom's facts, not knobs
    scene: Any, name: str, pos: Any, half: Any, *, mass: float, friction: Any, rgba: Any
) -> None:
    """A free-floating box on the table; every task object here is one."""
    add_free_box(scene, name, pos, half, rgba=rgba, mass=mass, friction=list(friction))


def _add_cube_a(scene: Any) -> None:
    """Cube A in the measured pocket — stack and insert pick the SAME
    cube with the same high-friction skin; it exists once so the two
    tasks cannot drift apart."""
    _add_free_cube(
        scene,
        CUBE_A_BODY,
        _CUBE_HOME,
        _CUBE_HALF,
        mass=0.02,
        friction=(2.0, 0.02, 0.001),
        rgba=(0.85, 0.15, 0.15, 1.0),
    )


def _jitter_cube_a(trial: int, home: Any) -> Any:
    """The +-2 mm paired pick jitter stack and insert share."""
    initial = home.copy()
    initial[CUBE_A_STATE_SLICE.start] += -0.002 + 0.002 * (trial % 3)
    initial[CUBE_A_STATE_SLICE.start + 1] += -0.002 + 0.002 * (trial % 2) * 2
    return initial


@register(LIFT, rig=RIG)
def build_lift(arm_xml: Path = DEFAULT_ARM_XML) -> Task:
    """Lift: squeeze the cube out of the pocket and hold it clear.

    Success reads the cube's height from privileged STATE (the referee
    sees everything); no sensor carries the cube's pose, so policies
    cannot read the object they are supposed to perceive — vision comes
    later, and pretending proprioception is perception would flatter
    every policy tested here.
    """
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-lift", arm_xml)
    _add_free_cube(
        scene,
        CUBE_BODY,
        _CUBE_HOME,
        _CUBE_HALF,
        mass=0.02,
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.8, 0.1, 0.1, 1.0),
    )

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        initial[_QPOS_OFFSET + 0] += -0.004 + 0.002 * trial
        initial[_QPOS_OFFSET + 1] += 0.004 - 0.002 * trial
        return initial

    def success(states: Any, sensors: Any) -> bool:
        tail = states[-_HOLD_STEPS:, CUBE_Z_STATE_INDEX]
        return bool(np.min(tail) > _LIFTED_HEIGHT_M)

    return _so101_task(
        LIFT,
        scene,
        EpisodeProtocol(
            trials=PAIRED_TRIALS,
            steps=_LIFT_STEPS,
            control_interval=CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
            placements=(Placement(CUBE_BODY, TABLE_GEOM),),
        ),
        "lift the cube out of the pocket and hold it clear",
        arm_xml,
    )


# The paired study's lift (docs/e2e-research/62): LIFT's own perturb is
# a 4-trial ladder whose offsets grow LINEARLY with the trial index —
# past trial 3 the start walks out of the intended band, so a sized
# evaluation (62 §3: 23-39 paired trials per side) cannot use it. The
# study variant keeps the scene, the referee and the expert, and draws
# every trial's start from a BOUNDED deterministic band: hash the trial
# index (the `evaluate/variations._unit` idea, spelled locally so the
# task owns its own starts), so any two policies' trial k begin
# identically and no trial count leaves the band.
_STUDY_TRIALS = 40
_STUDY_JITTER_RAD = 0.004


def _study_jitter(trial: int, component: int) -> float:
    digest = hashlib.sha256(f"{LIFT_STUDY}|{trial}|{component}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return (2.0 * unit - 1.0) * _STUDY_JITTER_RAD


@dataclass(frozen=True)
class LiftStudySpec:
    """The study task's one datum: how many paired starts it declares.
    A study that judges on 80 matched trials rebuilds the task with
    `trials=80` and its stamp says so (`envs.gymnasium_env.make_env` sizes
    the protocol through this spec; the jitter is deterministic per
    trial index, so any count is valid)."""

    trials: int = _STUDY_TRIALS


LIFT_STUDY_SPEC = LiftStudySpec()


@register(LIFT_STUDY, rig=RIG)
def build_lift_study(
    arm_xml: Path = DEFAULT_ARM_XML, spec: LiftStudySpec = LIFT_STUDY_SPEC
) -> Task:
    """Lift with study-grade starts: bounded, deterministic, any number
    of trials. Everything else is `build_lift`'s, by construction."""
    task = build_lift(arm_xml)

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        initial[_QPOS_OFFSET + 0] += _study_jitter(trial, 0)
        initial[_QPOS_OFFSET + 1] += _study_jitter(trial, 1)
        return initial

    return replace(
        task,
        name=LIFT_STUDY,
        protocol=replace(task.protocol, trials=spec.trials, perturb=perturb),
        task_spec=spec,
    )


# --------------------------------------------------------------- stack --

# block_stack: pick cube A from the measured pocket, swing the base by
# _STACK_PHI, and DROP it onto cube B. Three measured facts shape the
# script (2026-08-24):
#
# - Near the table the arm has NO steady state above the surface — every
#   hover/descend-family command sags to table contact within ~2 s at
#   the nominal kp=50, so a "lower gently and release" place cannot
#   exist; the working release is a controlled ~4 cm drop from the
#   carry pose.
# - B's position is the MEASURED landing point of that drop
#   ((0.082, -0.203)), the same probe-first discipline as every pocket.
# - The retract after release must pull UP along the carry arc: a
#   hover-family retract sweeps low and demolishes the fresh stack
#   (measured: 0/9 with a bigger B, debris at 50-70 mm — every "drop
#   miss" in the first grid was actually a demolition). With the high
#   retract the stack survives 9/9 across +-2 mm pick jitter.
_CUBE_B_HOME = (0.082, -0.203, 0.015)
_CUBE_B_HALF = 0.014
_STACK_PHI = 0.5
_STACK_STEPS = 3600
_STACKED_HEIGHT_M = 0.038
_STACK_HORIZ_TOLERANCE_M = _CUBE_B_HALF + 0.002
_B_SETTLE_TOLERANCE_M = 0.006
# State indices (time + arm 6 + A free 7 + B free 7):
CUBE_A_STATE_SLICE = slice(7, 10)
CUBE_B_STATE_SLICE = slice(14, 17)

# Stack phase boundaries (physics steps): pick through 1400, then carry
# up, two staged base sub-swings (a single 0.5 rad jump whips the cube
# out of the pinch — measured), release, retract high.
_STACK_CARRY_AT = 1400
_STACK_SWING_HALF_AT = 1900
_STACK_SWING_FULL_AT = 2200
_STACK_RELEASE_AT = 2600
_STACK_RETURN_AT = 3200

_STACK_CARRY = [0.0, -1.57, 1.57, 1.57, WRIST_FLAT, JAW_CLOSED]
_STACK_RELEASE = [_STACK_PHI, -1.57, 1.57, 1.57, WRIST_FLAT, JAW_OPEN]
_STACK_RETURN = [0.0, -1.57, 1.57, 1.57, WRIST_FLAT, JAW_OPEN]


def scripted_stack(step: int, sensordata: Any) -> Any:
    """Pick A, staged base swing over B, open, retract high."""
    if step < _STACK_CARRY_AT:
        return scripted_pick(step, sensordata)
    if step < _STACK_SWING_HALF_AT:
        return _STACK_CARRY
    if step < _STACK_SWING_FULL_AT:
        return [_STACK_PHI / 2, *_STACK_CARRY[1:]]
    if step < _STACK_RELEASE_AT:
        return [_STACK_PHI, *_STACK_CARRY[1:]]
    if step < _STACK_RETURN_AT:
        return _STACK_RELEASE
    return _STACK_RETURN


def scripted_stack_no_release(step: int, sensordata: Any) -> Any:
    """Carries A over B but never opens — the graded near-miss."""
    control = list(scripted_stack(step, sensordata))
    control[JAW_INDEX] = JAW_CLOSED
    return control


@register(BLOCK_STACK, rig=RIG)
def build_stack(arm_xml: Path = DEFAULT_ARM_XML) -> Task:
    """block_stack: cube A ends resting ON cube B, B undisturbed."""
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-stack", arm_xml)
    _add_cube_a(scene)
    _add_free_cube(
        scene,
        CUBE_B_BODY,
        _CUBE_B_HOME,
        (_CUBE_B_HALF, _CUBE_B_HALF, 0.015),
        mass=0.06,
        friction=(2.0, 0.02, 0.001),
        rgba=(0.15, 0.35, 0.85, 1.0),
    )

    def success(states: Any, sensors: Any) -> bool:
        tail = states[-_HOLD_STEPS:]
        a = tail[:, CUBE_A_STATE_SLICE]
        b = tail[:, CUBE_B_STATE_SLICE]
        horizontal = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
        return bool(
            a[:, 2].min() > _STACKED_HEIGHT_M
            and horizontal.max() < _STACK_HORIZ_TOLERANCE_M
            and abs(b[:, 2] - _CUBE_B_HOME[2]).max() < _B_SETTLE_TOLERANCE_M
        )

    return _so101_task(
        BLOCK_STACK,
        scene,
        EpisodeProtocol(
            trials=PAIRED_TRIALS,
            steps=_STACK_STEPS,
            control_interval=CONTROL_INTERVAL,
            perturb=_jitter_cube_a,
            success=success,
            placements=(
                Placement(CUBE_A_BODY, TABLE_GEOM),
                Placement(CUBE_B_BODY, TABLE_GEOM),
            ),
        ),
        "stack cube A on cube B",
        arm_xml,
    )


# -------------------------------------------------------------- insert --

# tool_insert analogue: the same pick-swing-drop delivers cube A into a
# walled pocket instead of onto a block. Pocket centred on the measured
# free-fall landing point (scatter across the pick-jitter grid is only
# 5 x 4 mm); walls are LOW on purpose — 24 mm walls let the cube cock
# against their tops and perch (3/9), 12 mm walls seat it 9/9. The
# pocket's inner clearance is the task's precision: +-6 mm.
_SLOT_CENTRE = (0.085, -0.209)
_SLOT_INNER = (0.018, 0.017)  # half-extents of the pocket cavity
_SLOT_WALL_THICKNESS = 0.004
_SLOT_WALL_HALF_HEIGHT = 0.006
_INSERT_SEATED_Z_M = 0.019
_INSERT_MARGIN_M = 0.010
# The planner releases the cube above the walls (its pads cannot enter
# the pocket) and lets it drop, as the script does: wall height + margin.
_PLANNER_DROP_CLEARANCE_M = 2 * _SLOT_WALL_HALF_HEIGHT + 0.006

# The reference insert policy IS the stack script: same pick, same
# swing, same drop — the scene decides whether that lands on a block or
# into a pocket. One behaviour, two tasks, exactly how a real policy
# gets evaluated across a suite.
scripted_insert = scripted_stack
scripted_insert_no_release = scripted_stack_no_release


@register(TOOL_INSERT, rig=RIG)
def build_insert(arm_xml: Path = DEFAULT_ARM_XML) -> Task:
    """tool_insert: cube A seated inside the pocket, flat on the table."""
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-insert", arm_xml)
    _add_cube_a(scene)
    centre_x, centre_y = _SLOT_CENTRE
    inner_x, inner_y = _SLOT_INNER
    thickness = _SLOT_WALL_THICKNESS
    add_slot_walls(
        scene,
        "slot",
        _SLOT_CENTRE,
        offset=(inner_x + thickness, inner_y + thickness),
        long_half=(inner_x + 2 * thickness, inner_y),
        wall=thickness,
        height=_SLOT_WALL_HALF_HEIGHT,
        rgba=(0.3, 0.3, 0.35, 1.0),
    )

    def success(states: Any, sensors: Any) -> bool:
        tail = states[-_HOLD_STEPS:, CUBE_A_STATE_SLICE]
        return bool(
            np.abs(tail[:, 0] - centre_x).max()
            < inner_x - _CUBE_HALF[0] + _INSERT_MARGIN_M
            and np.abs(tail[:, 1] - centre_y).max()
            < inner_y - _CUBE_HALF[1] + _INSERT_MARGIN_M
            and tail[:, 2].max() < _INSERT_SEATED_Z_M
        )

    return _so101_task(
        TOOL_INSERT,
        scene,
        EpisodeProtocol(
            trials=PAIRED_TRIALS,
            steps=_STACK_STEPS,
            control_interval=CONTROL_INTERVAL,
            perturb=_jitter_cube_a,
            success=success,
            placements=(Placement(CUBE_A_BODY, TABLE_GEOM),),
        ),
        "seat cube A inside the pocket",
        arm_xml,
    )


# ------------------------------------------------------- the planner --
# docs/66 §3 source 2: the pick/place beats written from poses, not by
# hand. The scripted experts above stay as the measured ceiling the
# planner is compared against.
ARM_IK_JOINTS = tuple(
    f"{ARM_PREFIX}{name}"
    for name in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
)
PLANNER_LIFT_HEIGHT_M = 0.10  # comfortably above the referee's _LIFTED_HEIGHT_M


def so101_gripper(model: Any) -> Any:
    """The SO-101's grasping facts on this compiled model: the grasp
    site IS the pocket (measured, see `_GRASP_POS`), the closing line
    runs tangentially around the base (the `_GRIP` pose's, generalised
    to any azimuth)."""
    from trainnr.collect.choreography import Gripper  # noqa: PLC0415

    return Gripper.resolve(
        model,
        site=GRASP_SITE,
        ik_joints=ARM_IK_JOINTS,
        jaw_ctrl=JAW_INDEX,
        jaw_open=JAW_OPEN,
        jaw_closed=JAW_CLOSED,
    )


def planner_goal(task_name: str, model: Any, data: Any) -> Any:
    """What the planner should do with the picked object on this task,
    read off the LIVE scene (privileged, as demo generation is)."""
    import mujoco  # noqa: PLC0415 - sim extra

    from trainnr.collect.choreography import Lift, Place  # noqa: PLC0415

    if task_name in (LIFT, LIFT_STUDY):
        return Lift(PLANNER_LIFT_HEIGHT_M)
    if task_name == BLOCK_STACK:
        b = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_B_BODY)]
        return Place((float(b[0]), float(b[1]), float(b[2]) + 0.015 + _CUBE_HALF[2]))
    if task_name == TOOL_INSERT:
        return Place((*_SLOT_CENTRE, _CUBE_HALF[2]), _PLANNER_DROP_CLEARANCE_M)
    raise ValueError(f"no planner goal for task {task_name!r}")


def planner_object(task_name: str) -> str:
    """The body the planner picks on this task."""
    if task_name in (LIFT, LIFT_STUDY):
        return CUBE_BODY
    if task_name in (BLOCK_STACK, TOOL_INSERT):
        return CUBE_A_BODY
    raise ValueError(f"no planner object for task {task_name!r}")


def planner_rig(task_name: str) -> Any:
    """What the planner needs to press this task on the SO-101."""
    from functools import partial  # noqa: PLC0415

    from trainnr.collect.planner_demos import PlannerRig  # noqa: PLC0415

    return PlannerRig(
        gripper=so101_gripper,
        goal=partial(planner_goal, task_name),
        object_body=planner_object(task_name),
    )
