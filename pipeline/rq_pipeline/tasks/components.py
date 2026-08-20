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
ARM_SENSOR_WIDTH = 12


def _base_scene(name: str) -> Any:
    import mujoco  # noqa: PLC0415 - sim extra

    scene = mujoco.MjSpec()
    scene.modelname = name
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
    frame = mount.add_frame(
        pos=[-0.05, 0, CHASSIS_SIZE[2]] if on_chassis else [0, 0, 0],
        euler=[0, 0, 1.5708] if on_chassis else [0, 0, 0],
    )
    frame.attach_body(arm.worldbody.first_body(), ARM_PREFIX, "")


def compose(
    *,
    car: bool,
    arm: bool,
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
    scene = _base_scene(name)
    if car:
        chassis = add_car(scene)
        if arm:
            attach_arm(scene, chassis, arm_xml)
    else:
        attach_arm(scene, scene.worldbody, arm_xml)
    return scene
