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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.evaluate.harness import EpisodeProtocol

ARM_PREFIX = "arm_"
DEFAULT_ARM_XML = (
    Path(__file__).resolve().parents[3] / "robots" / "so101-nominal" / "so101.xml"
)

# One definition repo-wide; components.py imports it from here (it
# already imports from this module, so this direction has no cycle).
ARM_SENSOR_WIDTH = 12  # qpos+qvel slice width of the arm's sensordata

# The arm's own sensor block: six jointpos then six jointvel.

FINGERTIP_SLICE = slice(ARM_SENSOR_WIDTH, ARM_SENSOR_WIDTH + 3)

# Episode design: 500 Hz physics (the model default), policies at 50 Hz —
# the same order of control rate the real bus sustains.
_STEPS = 600
_CONTROL_INTERVAL = 10
_TRIALS = 4
_HOLD_STEPS = 50
_REACH_TOLERANCE_M = 0.03
# FULLPHYSICS layout: [time, qpos(6), qvel(6)]; joint angles start at 1.
_QPOS_OFFSET = 1


@dataclass(frozen=True)
class SO101Task:
    """A composed scene and the protocol that scores episodes in it."""

    name: str
    spec: Any
    protocol: EpisodeProtocol
    target: tuple[float, float, float] | None = None


def _scene_with_arm(name: str, arm_xml: Path) -> Any:
    import mujoco  # noqa: PLC0415 - sim extra

    arm = mujoco.MjSpec.from_file(str(arm_xml))
    scene = mujoco.MjSpec()
    scene.modelname = name
    # Restore what attach drops — see module docstring.
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = 10
    scene.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.4, 0.4, 0.02],
        pos=[0.0, -0.2, -0.02],
    )
    frame = scene.worldbody.add_frame(pos=[0, 0, 0])
    frame.attach_body(arm.worldbody.first_body(), ARM_PREFIX, "")
    _add_armnetbench_cameras(scene)
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
    scene.visual.global_.offwidth = 1280
    scene.visual.global_.offheight = 720
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
    wrist_mount = scene.body(f"{ARM_PREFIX}Fixed_Jaw")
    wrist_mount.add_camera(
        name="wrist",
        pos=[0.0, -0.06, 0.04],
        xyaxes=[0, -1, 0, 1, 0, 1],
        fovy=70,
    )


def build_reach(arm_xml: Path = DEFAULT_ARM_XML) -> SO101Task:
    """Reach: fingertip to the home keyframe's fingertip position, held.

    The target is COMPUTED from the bundle's own home keyframe rather
    than hardcoded, so a bundle whose geometry changes moves the target
    with it instead of silently invalidating the task.
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-reach", arm_xml)
    jaw = scene.body(f"{ARM_PREFIX}Fixed_Jaw")
    jaw.add_site(name="fingertip", pos=[0.012, -0.08, 0.0], size=[0.005] * 3)
    scene.add_sensor(
        name="fingertip_pos",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="fingertip",
    )

    probe_model = scene.compile()
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_resetDataKeyframe(probe_model, probe_data, 0)  # arm_home
    mujoco.mj_forward(probe_model, probe_data)
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

    return SO101Task(
        name="reach",
        spec=scene,
        protocol=EpisodeProtocol(
            trials=_TRIALS,
            steps=_STEPS,
            control_interval=_CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
        ),
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

_HOVER = [0.0, -1.2, 2.0, 0.9, -1.571, 1.3]
_DESCEND = [0.0, -1.411, 2.225, 0.682, -1.571, 1.3]
_GRIP = [0.0, -1.411, 2.225, 0.682, -1.571, -0.15]
_LIFT = [0.0, -1.57, 1.57, 1.57, -1.571, -0.15]


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
    control[5] = 1.3
    return control


def build_lift(arm_xml: Path = DEFAULT_ARM_XML) -> SO101Task:
    """Lift: squeeze the cube out of the pocket and hold it clear.

    Success reads the cube's height from privileged STATE (the referee
    sees everything); no sensor carries the cube's pose, so policies
    cannot read the object they are supposed to perceive — vision comes
    later, and pretending proprioception is perception would flatter
    every policy tested here.
    """
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-lift", arm_xml)
    cube = scene.worldbody.add_body(name="cube", pos=list(_CUBE_HOME))
    cube.add_freejoint()
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(_CUBE_HALF),
        mass=0.02,
        rgba=[0.8, 0.1, 0.1, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        initial[1] += -0.004 + 0.002 * trial
        initial[2] += 0.004 - 0.002 * trial
        return initial

    def success(states: Any, sensors: Any) -> bool:
        tail = states[-_HOLD_STEPS:, CUBE_Z_STATE_INDEX]
        return bool(np.min(tail) > _LIFTED_HEIGHT_M)

    return SO101Task(
        name="lift",
        spec=scene,
        protocol=EpisodeProtocol(
            trials=_TRIALS,
            steps=_LIFT_STEPS,
            control_interval=_CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
        ),
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

_STACK_CARRY = [0.0, -1.57, 1.57, 1.57, -1.571, -0.15]
_STACK_RELEASE = [_STACK_PHI, -1.57, 1.57, 1.57, -1.571, 1.3]
_STACK_RETURN = [0.0, -1.57, 1.57, 1.57, -1.571, 1.3]


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
    control[5] = -0.15
    return control


def build_stack(arm_xml: Path = DEFAULT_ARM_XML) -> SO101Task:
    """block_stack: cube A ends resting ON cube B, B undisturbed."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-stack", arm_xml)
    cube_a = scene.worldbody.add_body(name="cube_a", pos=list(_CUBE_HOME))
    cube_a.add_freejoint()
    cube_a.add_geom(
        name="cube_a_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(_CUBE_HALF),
        mass=0.02,
        friction=[2.0, 0.02, 0.001],
        rgba=[0.85, 0.15, 0.15, 1.0],
    )
    cube_b = scene.worldbody.add_body(name="cube_b", pos=list(_CUBE_B_HOME))
    cube_b.add_freejoint()
    cube_b.add_geom(
        name="cube_b_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[_CUBE_B_HALF, _CUBE_B_HALF, 0.015],
        mass=0.06,
        friction=[2.0, 0.02, 0.001],
        rgba=[0.15, 0.35, 0.85, 1.0],
    )

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        initial[CUBE_A_STATE_SLICE.start] += -0.002 + 0.002 * (trial % 3)
        initial[CUBE_A_STATE_SLICE.start + 1] += -0.002 + 0.002 * (trial % 2) * 2
        return initial

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

    return SO101Task(
        name="block_stack",
        spec=scene,
        protocol=EpisodeProtocol(
            trials=_TRIALS,
            steps=_STACK_STEPS,
            control_interval=_CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
        ),
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

# The reference insert policy IS the stack script: same pick, same
# swing, same drop — the scene decides whether that lands on a block or
# into a pocket. One behaviour, two tasks, exactly how a real policy
# gets evaluated across a suite.
scripted_insert = scripted_stack
scripted_insert_no_release = scripted_stack_no_release


def build_insert(arm_xml: Path = DEFAULT_ARM_XML) -> SO101Task:
    """tool_insert: cube A seated inside the pocket, flat on the table."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    scene = _scene_with_arm("so101-insert", arm_xml)
    cube_a = scene.worldbody.add_body(name="cube_a", pos=list(_CUBE_HOME))
    cube_a.add_freejoint()
    cube_a.add_geom(
        name="cube_a_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(_CUBE_HALF),
        mass=0.02,
        friction=[2.0, 0.02, 0.001],
        rgba=[0.85, 0.15, 0.15, 1.0],
    )
    centre_x, centre_y = _SLOT_CENTRE
    inner_x, inner_y = _SLOT_INNER
    thickness = _SLOT_WALL_THICKNESS
    for label, dx, dy, sx, sy in (
        ("north", 0.0, inner_y + thickness, inner_x + 2 * thickness, thickness),
        ("south", 0.0, -(inner_y + thickness), inner_x + 2 * thickness, thickness),
        ("east", inner_x + thickness, 0.0, thickness, inner_y),
        ("west", -(inner_x + thickness), 0.0, thickness, inner_y),
    ):
        scene.worldbody.add_geom(
            name=f"slot_{label}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[sx, sy, _SLOT_WALL_HALF_HEIGHT],
            pos=[centre_x + dx, centre_y + dy, _SLOT_WALL_HALF_HEIGHT],
            rgba=[0.3, 0.3, 0.35, 1.0],
        )

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        initial[CUBE_A_STATE_SLICE.start] += -0.002 + 0.002 * (trial % 3)
        initial[CUBE_A_STATE_SLICE.start + 1] += -0.002 + 0.002 * (trial % 2) * 2
        return initial

    def success(states: Any, sensors: Any) -> bool:
        tail = states[-_HOLD_STEPS:, CUBE_A_STATE_SLICE]
        return bool(
            np.abs(tail[:, 0] - centre_x).max()
            < inner_x - _CUBE_HALF[0] + _INSERT_MARGIN_M
            and np.abs(tail[:, 1] - centre_y).max()
            < inner_y - _CUBE_HALF[1] + _INSERT_MARGIN_M
            and tail[:, 2].max() < _INSERT_SEATED_Z_M
        )

    return SO101Task(
        name="tool_insert",
        spec=scene,
        protocol=EpisodeProtocol(
            trials=_TRIALS,
            steps=_STACK_STEPS,
            control_interval=_CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
        ),
    )
