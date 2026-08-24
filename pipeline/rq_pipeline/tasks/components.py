"""Composable scene components: the car, the arm, and their assemblies.

The product thesis in miniature: **components attach, in any
combination** — the car alone (the rig's differential-drive base), the
arm alone (so101-nominal), or the arm mounted on the car (the mobile
manipulator the venture actually describes). Scenes are programs, so
"optional component" is an argument, not a fork of an XML file.

Geometry comes from the Rust `RobotSpec` (crates/sim-core): 30 mm wheel
radius, 150 mm track width — flagged placeholder there and therefore
placeholder here, one source either way. Wheel encoders ride along as
jointpos sensors, mirroring the real car's only proprioception. Every
dynamics number is nominal; the drivetrain bundle's fitted parameters
slot in when Paper 0's rig session produces them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rq_pipeline.tasks.so101 import ARM_PREFIX, DEFAULT_ARM_XML

# RobotSpec::REAL_BOT geometry (placeholders there, single-sourced here).
WHEEL_RADIUS = 0.03
TRACK_WIDTH = 0.15
CHASSIS_SIZE = (0.11, 0.06, 0.02)  # half-sizes: 22 x 12 x 4 cm box
CHASSIS_CLEARANCE = 0.032
CAR_PREFIX = "car_"

# Sensor widths, for slicing sensordata downstream.
CAR_SENSOR_WIDTH = 2  # one wheel encoder per side


def _base_scene(name: str) -> Any:
    import mujoco  # noqa: PLC0415 - sim extra

    scene = mujoco.MjSpec()
    scene.modelname = name
    # Radians, explicitly: MjSpec's programmatic default is DEGREES,
    # which turned the yellow arm's range=[-1.4, 1.4] into a ±1.4°
    # straitjacket — every joint sat pinned on its own limit and the
    # servos looked 85% too weak (measured; the XML models never hit
    # this because they declare angle="radian").
    scene.compiler.degree = False
    # The arm's contact options, restored as always (attach drops them);
    # harmless for the car-only scene.
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = 10
    scene.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[4.0, 4.0, 0.1],
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.35, 0.35, 0.38, 1.0],
    )
    return scene


def add_car(scene: Any, pos: tuple[float, float] = (0.0, 0.0)) -> Any:
    """Add the differential-drive base; returns the chassis body so a
    caller can mount things on it."""
    import mujoco  # noqa: PLC0415 - sim extra

    chassis = scene.worldbody.add_body(
        name=f"{CAR_PREFIX}chassis",
        pos=[pos[0], pos[1], CHASSIS_CLEARANCE + CHASSIS_SIZE[2]],
    )
    chassis.add_freejoint()
    # Heavy and low: the battery lives in the chassis, and the first
    # drive test without it TIPPED OVER the moment the arm reached out —
    # the assembly's stability margin is a real design constraint the
    # composition surfaced immediately, which is rather the point.
    chassis.add_geom(
        name=f"{CAR_PREFIX}body",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(CHASSIS_SIZE),
        mass=1.8,
        rgba=[0.15, 0.25, 0.5, 1.0],
    )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        wheel = chassis.add_body(
            name=f"{CAR_PREFIX}{side}_wheel",
            pos=[0.03, sign * TRACK_WIDTH / 2.0, -CHASSIS_SIZE[2] - 0.002],
        )
        wheel.add_joint(name=f"{CAR_PREFIX}{side}", axis=[0, 1, 0], damping=0.001)
        wheel.add_geom(
            name=f"{CAR_PREFIX}{side}_tyre",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[WHEEL_RADIUS, 0.006, 0],
            quat=[0.7071068, 0.7071068, 0, 0],  # cylinder axis -> y (axle)
            mass=0.03,
            friction=[1.2, 0.005, 0.0001],
            rgba=[0.1, 0.1, 0.1, 1.0],
        )
    # FOUR corner casters. The design walked here one measured failure
    # at a time: a single rear caster let the arm's reach pull the CoM
    # out of the support triangle (pitched onto its face); fore-and-aft
    # casters fixed pitch but left roll free, and an arm swing beached
    # the chassis on its own edge with a tyre in the air. Corners give
    # the support polygon the whole footprint.
    for label, x, y in (
        ("caster_rl", -0.10, 0.045),
        ("caster_rr", -0.10, -0.045),
        ("caster_fl", 0.10, 0.045),
        ("caster_fr", 0.10, -0.045),
    ):
        # priority=1 is load-bearing: MuJoCo combines contact friction
        # by ELEMENT-WISE MAXIMUM, so without it these low-friction
        # casters inherit the floor's mu=1.0 and become parking brakes
        # carrying most of the robot's weight (measured: the drive
        # stalled at zero motion with 2.7 N of propulsion available).
        chassis.add_geom(
            name=f"{CAR_PREFIX}{label}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=[x, y, -CHASSIS_SIZE[2] - 0.02],
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
            gainprm=[0.05] + [0.0] * 9,  # torque per unit ctrl — nominal
            ctrlrange=[-1.0, 1.0],
        )
        scene.add_sensor(
            name=f"{CAR_PREFIX}{side}_encoder",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=f"{CAR_PREFIX}{side}",
        )
    return chassis


def attach_arm(scene: Any, mount: Any, arm_xml: Path = DEFAULT_ARM_XML) -> None:
    """Attach the so101 bundle to a mount body (the world or a chassis)."""
    import mujoco  # noqa: PLC0415 - sim extra

    arm = mujoco.MjSpec.from_file(str(arm_xml))
    on_chassis = bool(mount.name)
    # The arm's workspace lies along its OWN -y; rotate the mount so it
    # reaches along the car's +x (forward) — unrotated, it reaches
    # sideways, exits the support polygon, and the car rolls over.
    # quat, not euler: add_frame accepted euler=... and silently applied
    # no rotation at all (measured — the arm reached sideways while every
    # probe assumed forward). Rz(+90 deg) as an explicit quaternion.
    frame = mount.add_frame(
        pos=[-0.05, 0, CHASSIS_SIZE[2]] if on_chassis else [0, 0, 0],
        quat=[0.7071068, 0, 0, 0.7071068] if on_chassis else [1, 0, 0, 0],
    )
    frame.attach_body(arm.worldbody.first_body(), ARM_PREFIX, "")


# The cargo tray: measured grasp pocket of the mounted arm, mapped by
# probe (closed pads 2-3 centre on chassis-frame (0.090, 0.000) at cube
# height). Low 6 mm walls confine the cube during driving; the jaw
# grips the cube's upper half well above them.
TRAY_CENTRE_X = 0.09
# 4 mm to -y: the OPEN fixed-jaw pad descends at y +0.009..0.013, which
# grazes a centred cube's +y edge (half-width 0.012) and nudges it out
# of the pocket before the grip closes (measured — the first cargo probe
# lost 7 mm of cube position during descent). Offset, the pad clears.
TRAY_CENTRE_Y = -0.004
TRAY_INNER_HALF = 0.0145
TRAY_WALL_HALF = 0.003
CUBE_HALF = 0.012
# Deck-grasp waypoints, droop-compensated against the ACHIEVED pose
# (same discipline as the table pick in so101.py).
# The pick approaches with the base swung +0.06 rad so the fixed jaw
# descends clear of the cube, then swings back at depth to straddle it
# before the squeeze — a straight vertical descent cannot work here:
# the open fixed pad and the cube's +y face are 1 mm apart at best
# (measured across three probe rounds; -4 mm tray offset grazes,
# -7 mm cannot pinch).
DECK_HOVER = [0.06, -2.45, 2.6, 1.4, -1.571, 1.0]
DECK_DESCEND = [0.06, -2.613, 3.14, 1.166, -1.571, 1.0]
DECK_ALIGN = [0.0, -2.613, 3.14, 1.166, -1.571, 1.0]
DECK_GRIP = [0.0, -2.613, 3.14, 1.166, -1.571, -0.15]
# Carry lifts back ALONG the approach arc (the closed hover pose), then
# presents modestly — the original wide swing to a far pose sheared the
# cube out of the pinch.
DECK_CARRY = [0.06, -2.45, 2.6, 1.4, -1.571, -0.15]
DECK_PRESENT = [0.06, -2.1, 2.2, 1.35, -1.571, -0.15]

# The probed pick-present-stow cycle: ramp between waypoints in order.
# Measured end-to-end 2026-08-24: cube lifts to z 0.135, returns to
# within 3 mm of tray centre. The jaw opens DURING the return ramp
# (DECK_ALIGN carries jaw=1.0), which lowers the cube guided rather
# than dropping it.
DECK_PICK_SEQUENCE = (
    (DECK_HOVER, 1.5),
    (DECK_DESCEND, 1.5),
    (DECK_ALIGN, 0.8),
    (DECK_GRIP, 1.5),
    (DECK_CARRY, 2.0),
    (DECK_PRESENT, 2.0),
    (DECK_CARRY, 1.5),
    (DECK_ALIGN, 2.0),
    (DECK_HOVER, 1.2),
)


def add_cargo(scene: Any) -> None:
    """A walled tray on the front deck, with a cube resting in it."""
    import mujoco  # noqa: PLC0415 - sim extra

    chassis = scene.body(f"{CAR_PREFIX}chassis")
    wall = TRAY_INNER_HALF + TRAY_WALL_HALF
    # The -y ("right") wall is HALF height: the moving jaw's closing
    # sweep passes through that side, and a full wall blocks it — the
    # grip then squeezes the cube against the wall instead of pinching
    # it (measured: 5 N into the wall, zero moving-pad contact).
    # Back wall is half-height too: the jaw's descent arc passes over
    # its position and a full wall interrupts the descent 15 mm short of
    # the cube (measured: 28-42 N of pad force pressing DOWN on the wall
    # top, pads parked at x 0.074 against a cube at 0.089).
    for label, dx, dy, sx, sy, sz in (
        ("front", wall, 0.0, TRAY_WALL_HALF, wall, 0.003),
        ("back", -wall, 0.0, TRAY_WALL_HALF, wall, 0.0015),
        ("left", 0.0, wall, wall, TRAY_WALL_HALF, 0.003),
        ("right", 0.0, -wall, wall, TRAY_WALL_HALF, 0.0015),
    ):
        chassis.add_geom(
            name=f"{CAR_PREFIX}tray_{label}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[sx, sy, sz],
            pos=[TRAY_CENTRE_X + dx, TRAY_CENTRE_Y + dy, CHASSIS_SIZE[2] + sz],
            mass=0.005,
            rgba=[0.25, 0.4, 0.7, 1.0],
        )
    cube = scene.worldbody.add_body(
        name="cargo_cube",
        pos=[
            TRAY_CENTRE_X,
            TRAY_CENTRE_Y,
            CHASSIS_CLEARANCE + 2 * CHASSIS_SIZE[2] + CUBE_HALF,
        ],
    )
    cube.add_freejoint()
    # Grippy on purpose (foam-wrapped in spirit): contact friction
    # combines by max, so the cube's own coefficients govern both the
    # pad pinch and the tray floor. The torsional term matters — with
    # the default 0.005 the pinched cube pivoted out of the narrow pad
    # contact during the carry swing.
    cube.add_geom(
        name="cargo_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, 0.015],
        mass=0.02,
        friction=[2.0, 0.02, 0.001],
        rgba=[0.85, 0.15, 0.15, 1.0],
    )


# Ground pick: the cube sits on the FLOOR ahead of the car, and the car
# must park so the cube lands at GROUND_GRASP_POINT in the chassis frame
# (measured closed-pad pocket; the pick tolerates +-4 mm, 9/9 jitter
# sweep). Longer reach than the deck pick buys 6 mm of descent
# clearance; the same base-swing approach aligns only at depth.
GROUND_GRASP_POINT = (0.203, 0.005)
GROUND_HOVER = [0.06, -1.5, 2.45, 0.55, -1.571, 1.0]
GROUND_DESCEND = [0.06, -1.16, 2.247, 0.2, -1.571, 1.0]
GROUND_ALIGN = [0.0, -1.16, 2.247, 0.2, -1.571, 1.0]
GROUND_GRIP = [0.0, -1.16, 2.247, 0.2, -1.571, -0.15]
GROUND_CARRY = [0.0, -1.6, 2.45, 0.55, -1.571, -0.15]
GROUND_HOLDUP = [0.0, -1.9, 2.2, 0.9, -1.571, -0.15]

GROUND_PICK_SEQUENCE = (
    (GROUND_HOVER, 1.5),
    (GROUND_DESCEND, 1.5),
    (GROUND_ALIGN, 0.8),
    (GROUND_GRIP, 1.5),
    (GROUND_CARRY, 2.0),
    (GROUND_HOLDUP, 2.0),
)
GROUND_PLACE_SEQUENCE = (
    (GROUND_CARRY, 1.5),
    (GROUND_ALIGN, 2.0),
    (GROUND_HOVER, 1.2),
)


def add_floor_cube(scene: Any, pos: tuple[float, float]) -> None:
    """A graspable cube on the floor — the ground-pick target."""
    import mujoco  # noqa: PLC0415 - sim extra

    cube = scene.worldbody.add_body(name="cargo_cube", pos=[pos[0], pos[1], 0.015])
    cube.add_freejoint()
    cube.add_geom(
        name="cargo_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, 0.015],
        mass=0.02,
        friction=[2.0, 0.02, 0.001],
        rgba=[0.85, 0.15, 0.15, 1.0],
    )


def compose(
    *,
    car: bool,
    arm: bool,
    cargo: bool = False,
    arm_xml: Path = DEFAULT_ARM_XML,
) -> Any:
    """The assembly menu: car, arm, or the mobile manipulator.

    Actuator/sensor layout follows attach order and is stable: car first
    when present (2 motors, 2 encoders), then the arm (6 actuators, 12
    sensors). Pinned by tests per mode.
    """
    if not car and not arm:
        raise ValueError("an empty scene is not an assembly — pick a component")
    name = {
        (True, True): "rig-mobile-manipulator",
        (True, False): "rig-car",
        (False, True): "so101-standalone",
    }[(car, arm)]
    if cargo and not (car and arm):
        raise ValueError("cargo needs the mobile manipulator: car AND arm")
    scene = _base_scene(name)
    if car:
        chassis = add_car(scene)
        if arm:
            attach_arm(scene, chassis, arm_xml)
    else:
        attach_arm(scene, scene.worldbody, arm_xml)
    if cargo:
        add_cargo(scene)
    return scene
