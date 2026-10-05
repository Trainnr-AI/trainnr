"""The yellow arm's MuJoCo twin, on the car — the REAL rig's double.

Link lengths measured by ruler on the assembled arm (2026-08-24, ±2 mm):
base platform top 40 mm, waist→shoulder 15 mm, shoulder→elbow 59 mm,
elbow→wrist 55 mm, wrist→gripper tip 80 mm; whole arm ~50-60 g. Every
dynamics number beyond geometry is NOMINAL hobby-servo guesswork and
labelled so — this arm has no feedback to identify against, which is
exactly why its twin matters: the twin is where the fetch behaviour
gets tuned before the metal runs it.

Joint map mirrors the PCA channels (canonical since assembly):

    ch0 base yaw · ch1 waist pitch (EMPTY since 2026-08-24 — servo
    removed after the toppling; the joint stays in the twin for the
    REAR floor-pick capability) · ch2 shoulder pitch · ch3 wrist pitch ·
    ch4 gripper (two gear-meshed jaws, one servo)

Servo pulse ↔ joint angle: 1500 µs is centre = 0 rad; SG90-class travel
is ~180° over ~1900 µs, so RAD_PER_US ≈ 0.00165. The claw's second jaw
is a mirrored joint coupled by an equality constraint — one actuator,
two jaws, like the gears on the desk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainnr.tasks.components import CAR_PREFIX, base_scene

YELLOW_RGBA = [0.95, 0.8, 0.1, 1.0]  # the rig's colour, once

# Real car geometry, measured by ruler 2026-08-24. Self-consistent:
# axle 75 mm from the rear of a 250 mm chassis = 50 mm behind centre;
# caster 50 mm from the front = 75 mm ahead of centre; axle→caster
# 125 mm ≈ the measured 130 mm wheelbase. REAR-wheel drive, single
# front caster, arm base over the rear axle (weight on the driven
# wheels). NOTE the reach fact this geometry proves: arm full reach
# 194 mm < 175 mm-to-front-edge + floor drop — the arm can only reach
# the FLOOR over the REAR edge (75 mm behind its base).


@dataclass(frozen=True)
class RealCar:
    wheel_radius: float = 0.0215
    track: float = 0.115
    chassis_half_length: float = 0.125
    chassis_half_width: float = 0.075
    chassis_half_height: float = 0.015
    axle_x: float = -0.050  # rear axle, chassis frame
    caster_x: float = 0.075  # single front caster
    mass: float = 0.9  # chassis + battery + boards, unweighed estimate


REAL_CAR = RealCar()


def add_real_car(scene: Any) -> Any:
    """The REAL rig's base: rear-drive, front caster, measured numbers."""
    import mujoco  # noqa: PLC0415 - sim extra

    g = REAL_CAR
    chassis = scene.worldbody.add_body(
        name=f"{CAR_PREFIX}chassis", pos=[0, 0, g.wheel_radius]
    )
    chassis.add_freejoint()
    chassis.add_geom(
        name=f"{CAR_PREFIX}body",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[g.chassis_half_length, g.chassis_half_width, g.chassis_half_height],
        mass=g.mass,
        rgba=[0.15, 0.25, 0.5, 1.0],
    )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        wheel = chassis.add_body(
            name=f"{CAR_PREFIX}{side}_wheel",
            pos=[g.axle_x, sign * g.track / 2.0, 0.0],
        )
        wheel.add_joint(name=f"{CAR_PREFIX}{side}", axis=[0, 1, 0], damping=0.001)
        wheel.add_geom(
            name=f"{CAR_PREFIX}{side}_tyre",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[g.wheel_radius, 0.006, 0],
            quat=[0.7071068, 0.7071068, 0, 0],
            mass=0.03,
            friction=[1.2, 0.005, 0.0001],
            rgba=[0.1, 0.1, 0.1, 1.0],
        )
    # Single front caster, per the metal. priority=1: same parking-brake
    # lesson as the demo car (friction combines by max).
    caster_r = 0.010
    chassis.add_geom(
        name=f"{CAR_PREFIX}caster",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[caster_r, 0, 0],
        pos=[g.caster_x, 0.0, -(g.wheel_radius - caster_r)],
        mass=0.01,
        priority=1,
        friction=[0.03, 0.001, 0.0001],
        rgba=[0.6, 0.6, 0.6, 1.0],
    )
    for side in ("left", "right"):
        scene.add_actuator(
            name=f"{CAR_PREFIX}{side}_motor",
            target=f"{CAR_PREFIX}{side}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gainprm=[0.05] + [0.0] * 9,
            ctrlrange=[-1.0, 1.0],
        )
        scene.add_sensor(
            name=f"{CAR_PREFIX}{side}_encoder",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=f"{CAR_PREFIX}{side}",
        )
    return chassis


# Measured geometry (m).
BASE_TOP = 0.040
WAIST_TO_SHOULDER = 0.015
SHOULDER_TO_ELBOW = 0.059
ELBOW_TO_WRIST = 0.055
WRIST_TO_TIP = 0.080

# Nominal hobby-servo actuation: SG90 stall ~0.18 N·m; a position servo
# that reaches ~stall a few degrees off centre is kp ≈ 2. MG90S is
# stronger; one number serves the twin until anyone measures.
SERVO_KP = 2.0
SERVO_FORCE = 0.22
# Reflected gearbox inertia — the physical fact that also makes the twin
# integrable: kp=2 on bare milligram links at 2 ms steps is numerically
# marginal (measured: joints settled at ~55% of command with no load);
# the servo's own geartrain inertia is real and stabilising.
SERVO_ARMATURE = 0.0005
SERVO_DAMPING = 0.03
RAD_PER_US = 0.00165
# The firmware's FETCH_MIN/MAX_US, mirrored (servo.rs) — both ends
# clamp identically, pinned by test_firmware_mirror.
PULSE_FLOOR_US = 600
PULSE_CEILING_US = 2400
YELLOW_SENSOR_WIDTH = 6  # five joints + the driven jaw

ARM_MASS = 0.055  # ruler-and-guesswork estimate, whole arm


def add_yellow_arm(scene: Any, mount: Any) -> None:
    """Attach the yellow arm's twin to a mount body (chassis or world)."""
    import mujoco  # noqa: PLC0415 - sim extra

    on_chassis = bool(mount.name)
    # On the real rig the arm base sits OVER THE REAR AXLE, facing the
    # rear (the only direction its 194 mm reach can touch the floor).
    base = mount.add_body(
        name="yarm_base",
        pos=[REAL_CAR.axle_x, 0, REAL_CAR.chassis_half_height]
        if on_chassis
        else [0, 0, 0],
        quat=[0, 0, 0, 1] if on_chassis else [1, 0, 0, 0],  # Rz(180): face rear
    )
    base.add_geom(
        name="yarm_base_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.035, BASE_TOP / 2, 0],
        pos=[0, 0, BASE_TOP / 2],
        mass=0.015,
        rgba=YELLOW_RGBA,
    )
    platform = base.add_body(name="yarm_platform", pos=[0, 0, BASE_TOP])
    platform.add_joint(
        name="yarm_base_yaw",
        axis=[0, 0, 1],
        range=[-1.57, 1.57],
        damping=SERVO_DAMPING,
        armature=SERVO_ARMATURE,
    )
    platform.add_geom(
        name="yarm_platform_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.03, 0.004, 0],
        pos=[0, 0, 0.004],
        mass=0.008,
        rgba=YELLOW_RGBA,
    )
    upper = platform.add_body(name="yarm_upper", pos=[0, 0, WAIST_TO_SHOULDER])
    upper.add_joint(
        name="yarm_waist",
        axis=[0, 1, 0],
        range=[-1.57, 1.57],
        damping=SERVO_DAMPING,
        armature=SERVO_ARMATURE,
    )
    upper.add_geom(
        name="yarm_upper_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.008, 0.012, SHOULDER_TO_ELBOW / 2],
        pos=[0, 0, SHOULDER_TO_ELBOW / 2],
        mass=0.012,
        rgba=YELLOW_RGBA,
    )
    fore = upper.add_body(name="yarm_fore", pos=[0, 0, SHOULDER_TO_ELBOW])
    fore.add_joint(
        name="yarm_shoulder",
        axis=[0, 1, 0],
        range=[-1.6, 1.6],
        damping=SERVO_DAMPING,
        armature=SERVO_ARMATURE,
    )
    fore.add_geom(
        name="yarm_fore_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.008, 0.010, ELBOW_TO_WRIST / 2],
        pos=[0, 0, ELBOW_TO_WRIST / 2],
        mass=0.010,
        rgba=YELLOW_RGBA,
    )
    hand = fore.add_body(name="yarm_hand", pos=[0, 0, ELBOW_TO_WRIST])
    hand.add_joint(
        name="yarm_wrist",
        axis=[0, 1, 0],
        range=[-1.6, 1.6],
        damping=SERVO_DAMPING,
        armature=SERVO_ARMATURE,
    )
    hand.add_geom(
        name="yarm_hand_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.010, 0.012, 0.020],
        pos=[0, 0, 0.02],
        mass=0.010,
        rgba=YELLOW_RGBA,
    )
    # The claw: two jaws on mirrored hinges, gear-meshed in metal =
    # equality-coupled here. Jaw length reaches the measured tip.
    # LATERAL pinch, matching the metal (observed 2026-08-24: the real
    # claw closes left-right across the car's width, 90° from the first
    # model). Jaws offset along hand-Y, hinging about hand-X.
    jaw_len = WRIST_TO_TIP - 0.04
    for side, sign in (("l", 1.0), ("r", -1.0)):
        jaw = hand.add_body(name=f"yarm_jaw_{side}", pos=[0, sign * 0.012, 0.04])
        jaw.add_joint(
            name=f"yarm_jaw_{side}_hinge",
            axis=[sign, 0, 0],
            range=[-0.6, 0.6],
            damping=SERVO_DAMPING,
            armature=SERVO_ARMATURE,
        )
        jaw.add_geom(
            name=f"yarm_jaw_{side}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.008, 0.004, jaw_len / 2],
            pos=[0, sign * 0.004, jaw_len / 2],
            mass=0.004,
            friction=[1.5, 0.02, 0.001],
            rgba=YELLOW_RGBA,
        )
    equality = scene.add_equality(
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        type=mujoco.mjtEq.mjEQ_JOINT,
        name1="yarm_jaw_l_hinge",
        name2="yarm_jaw_r_hinge",
    )
    equality.data[0] = 0.0
    equality.data[1] = 1.0  # jaw_l = jaw_r (axes already mirrored by sign)

    for joint in (
        "yarm_base_yaw",
        "yarm_waist",
        "yarm_shoulder",
        "yarm_wrist",
        "yarm_jaw_l_hinge",
    ):
        scene.add_actuator(
            name=f"{joint}_servo",
            target=joint,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gainprm=[SERVO_KP] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0.0, -SERVO_KP, -0.05] + [0.0] * 7,
            forcerange=[-SERVO_FORCE, SERVO_FORCE],
            ctrlrange=[-1.6, 1.6],
        )
        scene.add_sensor(
            name=f"{joint}_pos",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=joint,
        )
    scene.add_sensor(
        name="yarm_jaw_r_pos",
        type=mujoco.mjtSensor.mjSENS_JOINTPOS,
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        objname="yarm_jaw_r_hinge",
    )


def compose_rig(*, car: bool = True) -> Any:
    """The REAL rig's twin: measured car + yellow arm, as on the desk."""
    scene = base_scene("yellow-rig" if car else "yellow-arm")
    if car:
        chassis = add_real_car(scene)
        add_yellow_arm(scene, chassis)
    else:
        add_yellow_arm(scene, scene.worldbody)
    return scene


# ------------------------------------------------------------ rear pick --

# The rear floor pick, tuned in the twin 2026-08-24. Grasp point in the
# CHASSIS frame; the basin is a measured 15/15 across +-8 mm in both
# axes — the parking spec for the fetch.
# The tuning found two things a spec sheet never would: the claw's V
# tilts with the hand (no wrist roll exists to level it), so dead-centre
# approaches punched the cube's far corner until shoulder depth put both
# tips below cube-top before closing; and the working (shoulder, wrist)
# region is a RIDGE, not a point — (0.55, 0.7..0.85) all lift.
REAR_GRASP_POINT = (-0.187, 0.0)
REAR_PICK_BASIN_M = 0.008

# ctrl vectors: [base_yaw, waist, shoulder, wrist, jaw]
# Jaw sense with the LATERAL pinch (matches the metal): NEGATIVE opens,
# POSITIVE closes — inverted from the first radial-pinch model.
REAR_TUCK = [0.0, 0.2, 0.3, 0.2, -0.5]
REAR_HOVER = [0.0, 1.1, 0.55, 0.8, -0.5]
REAR_REACH = [0.0, 1.50, 0.55, 0.8, -0.5]
REAR_GRIP = [0.0, 1.50, 0.55, 0.8, 0.35]
REAR_LIFT = [0.0, 0.9, 0.55, 0.8, 0.35]

# The real arm's acrylic base toppled at the floor pick's full stretch
# (2026-08-24). The REAR_* floor pick stays as the TWIN's capability —
# the metal performs the AIR mime instead. The WAIST SERVO IS REMOVED
# from the real arm (2026-08-24, after it kept toppling): ch1 is an
# empty channel. The twin's waist joint still EXISTS (kp=2 servo, full
# range — needed for the REAR capability); the AIR poses merely hold it
# at 0, which is not the same as the metal's rigid link — under load
# the twin's waist can sag where the metal cannot.
# Yaw stays hand-aligned; shoulder and wrist carry the
# mime at moderate angles (max 0.9 rad), claw acts.
AIR_TUCK = [0.0, 0.0, 0.3, 0.2, -0.5]
AIR_REACH = [0.0, 0.0, 0.7, 0.9, -0.5]
AIR_GRAB = [0.0, 0.0, 0.7, 0.9, 0.35]
AIR_CARRY = [0.0, 0.0, 0.4, 0.4, 0.35]

# Holds mirror the firmware's FETCH_PICK table row for row (including
# the final hold) — pinned by test_firmware_mirror, so a retune here
# fails the build until servo.rs is regenerated to match.
AIR_PICK_SEQUENCE = [
    (AIR_TUCK, 1.5),
    (AIR_REACH, 2.0),
    (AIR_GRAB, 1.2),
    (AIR_CARRY, 2.0),
    (AIR_CARRY, 0.8),
]

# The firmware slews FETCH_STEP_US = 3 µs per 40 ms tick (servo.rs) —
# 75 µs/s, which through RAD_PER_US is the arm's true joint speed. The
# viewers reconstruct the mime with THIS rate; a fixed guess of 1.2 s
# per transition had the replayed arm finishing ~14 s before the metal.
FIRMWARE_SLEW_US_PER_S = 75
SLEW_RAD_PER_S = FIRMWARE_SLEW_US_PER_S * RAD_PER_US

# The salute (servo.rs fetch_arm): at the starting gun the claw runs
# 2 waves x [close 350 ms, open 350 ms] = 1.4 s, jaw only.
SALUTE_HALF_S = 0.35
SALUTE_TOTAL_S = 4 * SALUTE_HALF_S


def salute_pose(t: float) -> list[float] | None:
    """Arm pose t seconds after the starting gun, or None once done."""
    if t < 0 or t >= SALUTE_TOTAL_S:
        return None
    jaw = 0.35 if int(t / SALUTE_HALF_S) % 2 == 0 else -0.5
    return [*AIR_TUCK[:4], jaw]


def air_mime_pose(t: float) -> list[float]:
    """Arm pose t seconds after PICKING, at the firmware's true slew.

    The ONE reconstruction of the metal's mime — every viewer imports
    this instead of guessing its own blend times.
    """
    prev = AIR_TUCK
    for pose, hold in AIR_PICK_SEQUENCE:
        blend = (
            max(abs(a - b) for a, b in zip(prev, pose, strict=True)) / SLEW_RAD_PER_S
        )
        if t < blend:
            frac = t / blend if blend > 0 else 1.0
            return [a + (b - a) * frac for a, b in zip(prev, pose, strict=True)]
        t -= blend
        if t < hold:
            return list(pose)
        t -= hold
        prev = pose
    return list(AIR_PICK_SEQUENCE[-1][0])


REAR_PICK_SEQUENCE = (
    (REAR_TUCK, 1.2),
    (REAR_HOVER, 1.2),
    (REAR_REACH, 1.8),
    (REAR_GRIP, 1.0),
    (REAR_LIFT, 1.8),
)


def pose_to_pulses_us(pose: list[float]) -> list[int]:
    """Joint radians → PCA pulse widths for the metal arm.

    1500 µs is centre by construction (the horns went on against a held
    centre). Clamped to the firmware's FETCH_MIN/MAX_US band exactly as
    the metal clamps — the REAR poses' waist (1.50 rad → 2409 µs) sits
    past the SG90's physical stop, and an unclamped translation would
    command a buzz-at-stall. ⚠️ Per-joint SIGNS and spline trims are
    properties of the assembled metal, not of this model — they get
    measured against the real arm, the same promotion every sign
    constant in the firmware went through.
    """
    return [
        min(max(round(1500 + angle / RAD_PER_US), PULSE_FLOOR_US), PULSE_CEILING_US)
        for angle in pose
    ]
