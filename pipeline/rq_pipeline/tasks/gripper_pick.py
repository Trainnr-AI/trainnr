"""gripper-pick — the first task that USES a robot imported from USD.

The `robotiq-2f85-isaac` bundle (docs/e2e-research/77 §4.1: the Isaac
Robotiq 2F-85, imported through Newton) is a gripper with a fixed root
and one drive. This family hangs it, fingers down, from a Cartesian
CARRIAGE — three slide joints with position actuators, task scaffolding
declared here and never written into the bundle — over a table with a
cube inside the stroke. The referee: the cube lifted by a stated height
above where it spawned, held there for a stated time. The scripted
expert (`scripted_gripper_pick`) approaches over the cube it reads from
privileged state, descends, closes, lifts and holds; the ladder's
failures are the same choreography never closing (`no-close`) and with
every drive cut (`limp`).

Composition facts, each measured on this bundle (2026-09-25):

- **Fingers down is a half-turn about x.** The bundle stands fingers-up
  (+z); the carriage mounts it through a frame rotated 180 degrees about
  x, so the pads close along world y and hang below the carriage.
- **The tips reach past the pads' centre.** Closed, the fingertip
  hulls span 0.122-0.162 m from the base along the finger axis; a grasp
  centred on a 40 mm cube would drive the tips into the table, so the
  expert sets the base by the TIPS (`PickChoreography.PAD_REACH_M` plus
  a clearance) and the pads close over the cube's full height.
- **A pinch needs torsional friction.** At condim 3 the cube pivots
  about the line through its two pad contacts as it rises; the cube
  carries condim 4 with the pads' own contact material (`CubeContact`).
- **Carriage state is the policy's state.** The carriage's three
  jointpos sensors are added before the gripper is attached, so
  `sensordata[0:4]` is (x, y, z, finger_joint) — the four commanded
  quantities, in ctrl order — and `agent_pos` is that block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.locate import bundle_file, require_bundle_file
from rq_pipeline.protocol import CameraSpec, EpisodeProtocol, Placement
from rq_pipeline.tasks.registry import register, register_expert
from rq_pipeline.tasks.scene import (
    TABLE_GEOM,
    add_free_box,
    add_sky,
    corner_fraction,
    pin_nominal_options,
    set_render_budget,
)
from rq_pipeline.tasks.task import CONTROL_INTERVAL, PAIRED_TRIALS, Task

GRIPPER_PICK = "gripper-pick"
BUNDLE = "robotiq-2f85-isaac"
RIG = BUNDLE  # the bundle family this task composes
BUNDLE_XML = bundle_file(BUNDLE, f"{BUNDLE}.xml")
GRIPPER_PREFIX = "gripper_"
FINGER_JOINT = f"{GRIPPER_PREFIX}finger_joint"
CUBE_BODY = "cube"


class Carriage:
    """The task's scaffolding: a Cartesian carriage the gripper hangs
    from. Three slide joints (x, y, z) with position actuators, zero at
    `HOME` — the hover over the table's centre. Not part of the bundle:
    a real 2F-85 is carried by an arm, and this is the smallest arm that
    lets the gripper's own drive be the thing under test.

    Gains: kp against the ~1.1 kg carried (the gripper's 0.99 kg plus the
    carriage) sags the z axis m*g/kp = 2.7 mm at rest; the damping is
    joint damping (integrated implicitly under the nominal Euler
    integrator, unlike an actuator's kv) near critical, 2*sqrt(kp*m)."""

    BODY = "carriage"
    AXES = (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0)))
    HOME = (0.0, 0.0, 0.30)  # the gripper base's hover, metres
    RANGE_M = ((-0.12, 0.12), (-0.12, 0.12), (-0.24, 0.10))
    KP = 4000.0  # N/m
    DAMPING = 130.0  # N.s/m, joint damping
    ARMATURE = 0.05  # kg, reflected carriage inertia
    MASS = 0.1  # kg, the carriage body itself
    SIZE = (0.03, 0.03, 0.01)  # half-sizes of its (visual-only) plate
    RGBA = (0.25, 0.25, 0.3, 1.0)
    # The gripper's mount: a half-turn about x, fingers down.
    MOUNT_QUAT = (0.0, 1.0, 0.0, 0.0)

    @classmethod
    def joint(cls, axis: str) -> str:
        return f"{cls.BODY}_{axis}"

    @classmethod
    def drive(cls, axis: str) -> str:
        return f"{cls.joint(axis)}_drive"

    @classmethod
    def reach(cls, axis: int) -> tuple[float, float]:
        """Where the gripper base can go along one axis, in the world."""
        low, high = cls.RANGE_M[axis]
        return (cls.HOME[axis] + low, cls.HOME[axis] + high)


# The finger drive, as the bundle names it (its ctrlrange, 0-0.8, is
# read off the compiled model: open is the bottom, closed the top).
FINGER_DRIVE = f"{GRIPPER_PREFIX}finger_joint_drive"
# What agent_pos is: the carriage's three jointpos, then the finger's.
STATE_WIDTH = len(Carriage.AXES) + 1
FINGER_SENSOR = len(Carriage.AXES)  # finger_joint_pos in sensordata


@dataclass(frozen=True)
class GripperPickSpec:
    """The gripper-pick task as DATA: the cube, where it spawns, what
    counts as picked, the episode length and the sentence it is judged
    under. A variant is `replace(GRIPPER_PICK_SPEC, ...)`, named by
    `Task.stamp`, admitted only by `tasks/acceptance.py`. Changing any
    field changes the stamp: records under the old one are a different
    protocol."""

    cube_half: float = 0.02  # a 40 mm cube; the stroke is 85 mm
    cube_mass: float = 0.05
    cube_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    # Torsional friction on the cube's contacts (condim 4). Measured
    # 2026-09-25: at condim 3 a pinch has no torsional friction, the
    # cube pivots about the line through its two pad contacts as it
    # rises (a 120-degree swing) and one trial of four threw it. See
    # `CubeContact` for how the pair gets it without softening the pads.
    cube_condim: int = 4
    # The spawn band (cube centre, metres, table frame) — inside the
    # carriage's reach with room for the pads on either side.
    cube_spawn_x: tuple[float, float] = (-0.05, 0.05)
    cube_spawn_y: tuple[float, float] = (-0.05, 0.05)
    # The paired starts: the band's corners pulled in by this fraction.
    spawn_inset: float = 0.1
    # The referee: the cube's centre at least this far above where it
    # spawned, for the whole of the episode's last `hold_s`.
    lift_m: float = 0.05
    hold_s: float = 1.0
    steps: int = 3500  # 7 s at 500 Hz
    trials: int = PAIRED_TRIALS
    instruction: str = "pick up the cube and hold it above the table"


GRIPPER_PICK_SPEC = GripperPickSpec()

CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec("front", "front", width=640, height=480),
    CameraSpec("wrist", "wrist", width=640, height=480),
)
CUBE_RGBA = (0.85, 0.35, 0.2, 1.0)


class Stage:
    """The table, the light and the two cameras (visual facts; the
    cameras' poses are what a vision policy on this task would see).
    `front` looks along -x so the pads close across its image; `wrist`
    rides the carriage on the gripper's -x side, looking down at the
    pads."""

    TABLE_HALF = (0.25, 0.25, 0.02)
    TABLE_RGBA = (0.55, 0.5, 0.45, 1.0)
    LIGHT_POS = (0.0, -0.4, 1.2)
    LIGHT_DIR = (0.0, 0.3, -1.0)
    LIGHT_DIFFUSE = (0.8, 0.8, 0.8)
    FRONT_POS = (0.55, 0.0, 0.30)
    FRONT_XYAXES = (0.0, 1.0, 0.0, -0.45, 0.0, 0.89)
    FRONT_FOVY = 55.0
    WRIST_POS = (-0.09, 0.0, -0.04)  # in the carriage's frame
    WRIST_XYAXES = (0.0, -1.0, 0.0, 0.5, 0.0, 0.87)
    WRIST_FOVY = 70.0


class CubeContact:
    """How the cube meets the pads. The bundle's fingertip hulls carry
    priority 1 with their USD contact material (solref 0.004 2, solimp
    0.95 0.99), and a higher-priority geom's condim, friction, solref and
    solimp govern the pair outright — so a plain cube (priority 0) gets
    the pads' condim 3. Two routes to condim 4, both measured 2026-09-25:
    matching the pads' priority MIXES the pair, and the mixed solref/
    solimp let the pads sink 16-23 mm into the cube; outranking them
    with the pads' OWN solref/solimp (read off the bundle at compose
    time, never restated) keeps the pads' stiffness and adds the
    torsion. The second is what the scene does."""

    PAD_GEOM = "left_fingertip_hull"  # the bundle's name, before the prefix
    PRIORITY_ABOVE_PAD = 1


# A cube counts as moved past this displacement (the funnel's second rung).
MOVED_M = 0.003
# The finger is closed on something past this angle (the funnel's jaw rung).
FINGER_CLOSED_RAD = 0.3


def gripper_pick_scene(bundle_xml: Path = BUNDLE_XML, *, spec: GripperPickSpec) -> Any:
    """The composed MjSpec: table, carriage, gripper, cube, cameras."""
    import mujoco  # noqa: PLC0415 - sim extra

    gripper = mujoco.MjSpec.from_file(str(require_bundle_file(bundle_xml)))
    pad = gripper.geom(CubeContact.PAD_GEOM)
    if pad is None:
        raise KeyError(
            f"{bundle_xml} has no geom {CubeContact.PAD_GEOM!r}: gripper-pick "
            "takes the cube's contact material from the pads"
        )
    scene = mujoco.MjSpec()
    scene.modelname = GRIPPER_PICK
    scene.compiler.degree = False  # MjSpec's own default is degrees
    # attach drops the bundle's cone/impratio: the full block, pinned.
    pin_nominal_options(scene)
    set_render_budget(scene)
    add_sky(scene)  # visual only: a camera's background is not black
    scene.worldbody.add_light(
        pos=list(Stage.LIGHT_POS),
        dir=list(Stage.LIGHT_DIR),
        diffuse=list(Stage.LIGHT_DIFFUSE),
    )
    scene.worldbody.add_geom(
        name=TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(Stage.TABLE_HALF),
        pos=[0.0, 0.0, -Stage.TABLE_HALF[2]],  # the top at z = 0
        rgba=list(Stage.TABLE_RGBA),
    )
    carriage = scene.worldbody.add_body(name=Carriage.BODY, pos=list(Carriage.HOME))
    for (axis, direction), limits in zip(Carriage.AXES, Carriage.RANGE_M, strict=True):
        carriage.add_joint(
            name=Carriage.joint(axis),
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=list(direction),
            range=list(limits),
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            damping=Carriage.DAMPING,
            armature=Carriage.ARMATURE,
        )
    carriage.add_geom(
        name=f"{Carriage.BODY}_plate",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(Carriage.SIZE),
        pos=[0.0, 0.0, Carriage.SIZE[2]],
        mass=Carriage.MASS,
        contype=0,
        conaffinity=0,
        rgba=list(Carriage.RGBA),
    )
    carriage.add_camera(
        name="wrist",
        pos=list(Stage.WRIST_POS),
        xyaxes=list(Stage.WRIST_XYAXES),
        fovy=Stage.WRIST_FOVY,
    )
    # Carriage sensors and drives BEFORE the gripper's: sensordata[0:3]
    # and ctrl[0:3] are the carriage, the gripper's own follow.
    for (axis, _), limits in zip(Carriage.AXES, Carriage.RANGE_M, strict=True):
        scene.add_sensor(
            name=f"{Carriage.joint(axis)}_pos",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=Carriage.joint(axis),
        )
        drive = scene.add_actuator(
            name=Carriage.drive(axis),
            target=Carriage.joint(axis),
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=list(limits),
        )
        drive.set_to_position(kp=Carriage.KP)
    frame = carriage.add_frame(quat=list(Carriage.MOUNT_QUAT))
    frame.attach_body(gripper.worldbody.first_body(), GRIPPER_PREFIX, "")
    add_free_box(
        scene,
        CUBE_BODY,
        (0.0, 0.0, spec.cube_half),
        spec.cube_half,
        rgba=CUBE_RGBA,
        mass=spec.cube_mass,
        friction=list(spec.cube_friction),
        condim=spec.cube_condim,
        priority=pad.priority + CubeContact.PRIORITY_ABOVE_PAD,
        solref=list(pad.solref),
        solimp=list(pad.solimp),
    )
    scene.worldbody.add_camera(
        name="front",
        pos=list(Stage.FRONT_POS),
        xyaxes=list(Stage.FRONT_XYAXES),
        fovy=Stage.FRONT_FOVY,
    )
    return scene


@dataclass(frozen=True)
class Layout:
    """Where the referee and the expert read: the cube's position in the
    FULLPHYSICS row, read off the compiled model (not a hand-counted
    offset), and the carriage's and finger's ctrl indices."""

    cube_pos: slice
    carriage_ctrl: tuple[int, int, int]
    finger_ctrl: int
    finger_open: float  # the drive's ctrlrange bottom
    finger_closed: float  # and its top
    nu: int
    timestep: float

    @classmethod
    def of(cls, model: Any) -> Layout:
        import mujoco  # noqa: PLC0415 - sim extra

        from rq_pipeline.physics.backend import FullPhysicsLayout  # noqa: PLC0415

        def ctrl(name: str) -> int:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index < 0:
                raise KeyError(f"no actuator {name!r} in the gripper-pick scene")
            return int(index)

        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
        start = FullPhysicsLayout(model).qpos.start + int(
            model.jnt_qposadr[model.body_jntadr[cube]]
        )
        x, y, z = (ctrl(Carriage.drive(a)) for a, _ in Carriage.AXES)
        finger = ctrl(FINGER_DRIVE)
        return cls(
            cube_pos=slice(start, start + 3),
            carriage_ctrl=(x, y, z),
            finger_ctrl=finger,
            finger_open=float(model.actuator_ctrlrange[finger][0]),
            finger_closed=float(model.actuator_ctrlrange[finger][1]),
            nu=int(model.nu),
            timestep=float(model.opt.timestep),
        )


def refuse_unreachable_band(spec: GripperPickSpec) -> None:
    """A spawn band the carriage cannot put the gripper over, or that
    hangs the cube off the table, is refused by name before anything is
    built — an overlay that moves the band is a new task, and one the
    expert cannot reach is a rejection waiting to happen."""
    for axis, band in ((0, spec.cube_spawn_x), (1, spec.cube_spawn_y)):
        name = "xy"[axis]
        low, high = band
        reach_low, reach_high = Carriage.reach(axis)
        table = Stage.TABLE_HALF[axis] - spec.cube_half
        if low > high:
            raise ValueError(f"cube_spawn_{name} {band} runs backwards")
        if low < reach_low or high > reach_high:
            raise ValueError(
                f"cube_spawn_{name} {band} leaves the carriage's reach "
                f"{(reach_low, reach_high)}"
            )
        if low < -table or high > table:
            raise ValueError(
                f"cube_spawn_{name} {band} hangs the cube off the table "
                f"(its centre must stay within +-{table:.3f} m)"
            )


def hold_steps(spec: GripperPickSpec, timestep: float) -> int:
    return round(spec.hold_s / timestep)


@register(GRIPPER_PICK, rig=RIG)
def build_gripper_pick(
    bundle_xml: Path = BUNDLE_XML, *, spec: GripperPickSpec = GRIPPER_PICK_SPEC
) -> Task:
    """gripper-pick: the cube lifted `lift_m` above its spawn and held
    for the last `hold_s` of the episode. Success reads privileged
    STATE; no sensor carries the cube's pose."""
    import numpy as np  # noqa: PLC0415

    from rq_pipeline.tasks.scene import NominalOptions  # noqa: PLC0415

    refuse_unreachable_band(spec)
    scene = gripper_pick_scene(bundle_xml, spec=spec)
    layout = Layout.of(scene.compile())
    cube = layout.cube_pos
    hold = hold_steps(spec, NominalOptions.TIMESTEP)
    if hold >= spec.steps:
        raise ValueError(
            f"hold_s {spec.hold_s} is {hold} steps, not inside an episode of "
            f"{spec.steps}"
        )

    def perturb(trial: int, home: Any) -> Any:
        initial = home.copy()
        fx, fy = corner_fraction(trial, inset=spec.spawn_inset)
        (x_low, x_high), (y_low, y_high) = spec.cube_spawn_x, spec.cube_spawn_y
        initial[cube.start] = x_low + fx * (x_high - x_low)
        initial[cube.start + 1] = y_low + fy * (y_high - y_low)
        return initial

    def lifted(states: Any, step: Any) -> Any:
        return states[step, cube.start + 2] - states[0, cube.start + 2] > spec.lift_m

    def success(states: Any, sensors: Any) -> bool:
        del sensors
        return bool(np.all(lifted(states, slice(-hold, None))))

    def cube_moved(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return bool(np.linalg.norm(states[step, cube] - states[0, cube]) > MOVED_M)

    def jaw_closed(states: Any, sensors: Any, step: int) -> bool:
        del states
        return bool(sensors[step, FINGER_SENSOR] > FINGER_CLOSED_RAD)

    def cube_lifted(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return bool(lifted(states, step))

    def cube_held(states: Any, sensors: Any, step: int) -> bool:
        del sensors
        return step + 1 >= hold and bool(
            np.all(lifted(states, slice(step + 1 - hold, step + 1)))
        )

    return Task(
        name=GRIPPER_PICK,
        spec=scene,
        cameras=CAMERAS,
        state_width=STATE_WIDTH,
        instruction=spec.instruction,
        bundle_dir=Path(bundle_xml).parent,
        task_spec=spec,
        protocol=EpisodeProtocol(
            trials=spec.trials,
            steps=spec.steps,
            control_interval=CONTROL_INTERVAL,
            perturb=perturb,
            success=success,
            placements=(
                Placement(
                    CUBE_BODY, TABLE_GEOM, x=spec.cube_spawn_x, y=spec.cube_spawn_y
                ),
            ),
            milestones=(
                ("jaw_closed", jaw_closed),
                ("cube_moved", cube_moved),
                ("cube_lifted", cube_lifted),
                ("cube_held", cube_held),
            ),
        ),
    )


# ------------------------------------------------------------- expert --


class PickChoreography:
    """The scripted expert's measured knobs, in one place. Heights are
    of the gripper BASE above the table top. `PAD_REACH_M` is the closed
    fingertip hulls' lowest point below the base, measured on the bundle
    (module docstring) and pinned by test against the compiled scene, so
    a regenerated bundle whose fingers moved fails the test before it
    fails a grasp."""

    PAD_REACH_M = 0.162  # base to closed fingertip tip, along the fingers
    TIP_CLEARANCE_M = 0.006  # the tips' gap over the table at the grasp
    LIFT_M = 0.10  # base rise from the grasp to the carry
    # Phase ends, seconds from the episode's start.
    APPROACH_S = 0.8  # over the cube at the hover height
    DESCEND_S = 1.8  # down to the grasp height
    CLOSE_S = 2.8  # squeeze
    LIFT_S = 4.0  # up to the carry height; hold to the end


def pick_controls(
    layout: Layout, cube_xy: Any, spec: GripperPickSpec, *, close: bool = True
) -> Any:
    """The open-loop plan for a cube at `cube_xy`: `act(step, sensors)`
    returning the full ctrl vector for that physics step's phase.
    `close=False` is the ladder's no-close rung — every beat identical,
    the fingers never commanded shut."""
    import numpy as np  # noqa: PLC0415

    knobs = PickChoreography
    home_z = Carriage.HOME[2]
    grasp_z = knobs.PAD_REACH_M + knobs.TIP_CLEARANCE_M  # the table top is z=0
    dx, dy = float(cube_xy[0]) - Carriage.HOME[0], float(cube_xy[1]) - Carriage.HOME[1]
    down = grasp_z - home_z
    up = down + knobs.LIFT_M
    open_ = layout.finger_open
    squeeze = layout.finger_closed if close else open_
    beats = (
        (knobs.APPROACH_S, (dx, dy, 0.0), open_),
        (knobs.DESCEND_S, (dx, dy, down), open_),
        (knobs.CLOSE_S, (dx, dy, down), squeeze),
        (knobs.LIFT_S, (dx, dy, up), squeeze),
    )

    def act(step: int, sensors: Any) -> Any:
        del sensors
        t = step * layout.timestep
        carriage, finger = next(
            ((c, f) for end, c, f in beats if t < end), beats[-1][1:]
        )
        control = np.zeros(layout.nu)
        control[list(layout.carriage_ctrl)] = carriage
        control[layout.finger_ctrl] = finger
        return control

    return act


def limp_model(model: Any) -> Any:
    """The ladder's limp rung: a copy of the model with every drive cut
    (gain and bias zeroed), so whatever is commanded, no actuator pushes
    — the carriage falls onto the table and the fingers hang."""
    import copy  # noqa: PLC0415

    cut = copy.copy(model)
    cut.actuator_gainprm[:] = 0.0
    cut.actuator_biasprm[:] = 0.0
    return cut


@register_expert(GRIPPER_PICK)
def scripted_gripper_pick(
    model: Any,
    initial_state: Any,
    *,
    spec: GripperPickSpec = GRIPPER_PICK_SPEC,
    close: bool = True,
) -> tuple[Any, Any]:
    """One scripted pick, privileged: the cube's spawn is read off the
    initial state, the carriage goes over it, down, the fingers close,
    the carriage lifts and holds. Returns the per-physics-step
    FULLPHYSICS states and sensors the task's referee judges."""
    import numpy as np  # noqa: PLC0415

    from rq_pipeline.physics.mujoco_backend import Stepper  # noqa: PLC0415

    layout = Layout.of(model)
    cube_xy = np.asarray(initial_state)[layout.cube_pos][:2]
    act = pick_controls(layout, cube_xy, spec, close=close)
    stepper = Stepper(model, initial_state, spec.steps)
    while not stepper.done:
        stepper.advance(act(stepper.step, stepper.sensordata), CONTROL_INTERVAL)
    return stepper.states, stepper.sensors


def limp_gripper_pick(
    model: Any, initial_state: Any, *, spec: GripperPickSpec = GRIPPER_PICK_SPEC
) -> tuple[Any, Any]:
    """The ladder's limp rung: the expert's commands into `limp_model`."""
    return scripted_gripper_pick(limp_model(model), initial_state, spec=spec)


def no_close_gripper_pick(
    model: Any, initial_state: Any, *, spec: GripperPickSpec = GRIPPER_PICK_SPEC
) -> tuple[Any, Any]:
    """The ladder's no-close rung: every beat, the fingers left open."""
    return scripted_gripper_pick(model, initial_state, spec=spec, close=False)


# The acceptance ladder by rung name: the expert must pass every paired
# trial, the two graded failures none (tests/test_tasks_gripper_pick.py,
# tools/show-gripper-pick.py).
EXPERT_RUNG = "pick"
LADDER = {
    EXPERT_RUNG: scripted_gripper_pick,
    "no-close": no_close_gripper_pick,
    "limp": limp_gripper_pick,
}
