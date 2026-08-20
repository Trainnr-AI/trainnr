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

# The arm's own sensor block: six jointpos then six jointvel.
ARM_SENSOR_WIDTH = 12
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
    return scene


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
