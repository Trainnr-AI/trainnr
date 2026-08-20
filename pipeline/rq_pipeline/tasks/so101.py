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
prefixing, and the harness plumbing end-to-end; the ArmnetBench-shaped
manipulation tasks build on exactly this scaffolding.
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
    target: tuple[float, float, float]


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
