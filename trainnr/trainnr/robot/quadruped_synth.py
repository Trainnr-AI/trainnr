"""The synthetic quadruped identifiability check — the legged fit's proof.

Before the method touches a public log it must recover what it was
given (the rule docs/e2e-research/26 set for the servo): a TRUE model
with known armature, damping and Coulomb friction per joint runs a
chirp under a motor-side PD law, the recording is corrupted the way an
encoder and a current sensor corrupt it, and `torque_balance.fit_terms`
must return intervals that cover the truth — and NOT PINNED where the
excitation left a term free. Two postures: the base fixed in the air
(the bench posture, IIT's chirp), and the base floating and shaken by
an external wrench with no floor (the walking posture without contacts,
which exercises the base-state path of the inverse dynamics).

The truth is a per-joint table, not one number repeated: a fit that
returns the same value for every joint has not been tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trainnr.robot.torque_balance import (
    FREE_DOF,
    FREE_QPOS,
    free_joint_of,
)
from trainnr.robots.joint_orders import GO2_MENAGERIE_FEET, anatomy_of
from trainnr.robots.recording import (
    ACCELERATION_COMPONENTS,
    BASE_ACCELERATION,
    BASE_POSE,
    BASE_TWIST,
    BASIS_SIMULATION,
    COLLECTION_SCRIPTED,
    JOINT_COMMAND,
    JOINT_COMMAND_VELOCITY,
    JOINT_EFFORT,
    JOINT_KD,
    JOINT_KP,
    JOINT_POSITION,
    JOINT_VELOCITY,
    POSE_COMPONENTS,
    TWIST_COMPONENTS,
    Channel,
    Recording,
)

FIXED = "fixed"
SHAKEN = "shaken"
ACCELERATION_TRUTH = "joint.acceleration_truth"
POSTURES = (FIXED, SHAKEN)


@dataclass(frozen=True)
class JointTruth:
    armature: float  # kg*m^2
    damping: float  # N*m*s/rad
    frictionloss: float  # N*m


# Distinct per joint class and per leg (a small spread), so the fit is
# tested joint by joint. Values sit where the Go2's declared numbers do.
TRUTH_BY_CLASS: dict[str, JointTruth] = {
    "hip": JointTruth(0.008, 0.40, 0.30),
    "thigh": JointTruth(0.012, 0.80, 0.50),
    "calf": JointTruth(0.020, 1.20, 0.80),
}
LEG_SPREAD = (1.00, 1.10, 0.90, 1.05)  # multiplies each class value per leg
LEGS = GO2_MENAGERIE_FEET


@dataclass(frozen=True)
class Chirp:
    """Per-joint chirp about the home pose: amplitude by joint class,
    frequency sweeping f0 → f1 over the run, phases spread so the legs
    do not move in unison."""

    seconds: float = 20.0
    f0_hz: float = 0.2
    f1_hz: float = 2.5
    amplitude: dict[str, float] | None = None

    def amplitude_of(self, joint_class: str) -> float:
        table = self.amplitude or {"hip": 0.30, "thigh": 0.40, "calf": 0.40}
        return table[joint_class]


@dataclass(frozen=True)
class Servo:
    """The motor-side PD law and the physics/control rates."""

    kp: float = 20.0
    kd: float = 0.5
    control_hz: float = 250.0
    physics_hz: float = 1000.0


@dataclass(frozen=True)
class Corruption:
    """How a Go2-class joint corrupts its own record: a 2^14-count encoder
    quantises position, velocity is the firmware's difference of it, the
    torque is a current estimate with noise."""

    encoder_counts: int = 16384
    torque_noise_nm: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class BaseShake:
    """The external wrench that shakes a floating base in the air: gravity
    is cancelled by a mean upward force, sinusoids on top."""

    force_n: float = 25.0
    torque_nm: float = 4.0
    hz: float = 0.7


def joint_class(name: str) -> str:
    """hip, thigh or calf, from the joint table (never parsed from the name)."""
    return anatomy_of(name)[1]


def truth_table(joints: tuple[str, ...]) -> dict[str, JointTruth]:
    out = {}
    for name in joints:
        leg, kind = anatomy_of(name)
        base = TRUTH_BY_CLASS[kind]
        k = LEG_SPREAD[LEGS.index(leg)]
        out[name] = JointTruth(
            base.armature * k, base.damping * k, base.frictionloss * k
        )
    return out


def hinge_joints(spec: Any) -> tuple[str, ...]:
    import mujoco  # noqa: PLC0415

    return tuple(j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE)


def prepared_spec(model_xml: Path, posture: str) -> tuple[Any, tuple[str, ...]]:
    """The model with a motor per hinge, the free joint removed for the
    fixed posture, all three passive terms cleared (the truth sets them)."""
    import mujoco  # noqa: PLC0415

    if posture not in POSTURES:
        raise ValueError(f"posture must be one of {POSTURES}, got {posture!r}")
    spec = mujoco.MjSpec.from_file(str(model_xml))
    # Runge-Kutta, not the default Euler: Euler integrates joint damping
    # IMPLICITLY, so the velocity a log records differs from the explicit
    # acceleration by dt*damping/inertia per step (~5 % here), and a fit
    # from that log reads armature 14 % high with a perfect residual
    # (measured 2026-09-24). A real robot's physics is continuous; the
    # explicit integrator is the honest stand-in for it.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
    joints = hinge_joints(spec)
    for joint in list(spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE and posture == FIXED:
            spec.delete(joint)
    for name in joints:
        joint = spec.joint(name)
        joint.armature = 0.0
        joint.damping[0] = 0.0
        joint.frictionloss = 0.0
        motor = spec.add_actuator(name=f"{name}_motor", target=name)
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT
        motor.gainprm[0] = 1.0
    return spec, joints


def apply_truth(spec: Any, truth: dict[str, JointTruth]) -> None:
    for name, value in truth.items():
        joint = spec.joint(name)
        joint.armature = value.armature
        joint.damping[0] = value.damping
        joint.frictionloss = value.frictionloss


DEFAULT_CHIRP = Chirp()
DEFAULT_SERVO = Servo()
DEFAULT_CORRUPTION = Corruption()
DEFAULT_SHAKE = BaseShake()


def simulate(  # noqa: PLR0913, PLR0915 - the study's axes, each a dataclass; one rollout
    model_xml: Path,
    *,
    posture: str = FIXED,
    chirp: Chirp = DEFAULT_CHIRP,
    servo: Servo = DEFAULT_SERVO,
    corruption: Corruption | None = DEFAULT_CORRUPTION,
    shake: BaseShake = DEFAULT_SHAKE,
) -> tuple[Recording, dict[str, JointTruth]]:
    """Roll the true model out under the chirp; return the (corrupted)
    recording and the truth it came from."""
    import mujoco  # noqa: PLC0415

    spec, joints = prepared_spec(Path(model_xml), posture)
    truth = truth_table(joints)
    apply_truth(spec, truth)
    model = spec.compile()
    model.opt.timestep = 1.0 / servo.physics_hz
    data = mujoco.MjData(model)
    qpos_adr = np.array(
        [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in joints
        ]
    )
    dof_adr = np.array(
        [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in joints
        ]
    )
    home = np.array([model.qpos0[a] for a in qpos_adr])
    # A standing-like pose inside every joint's range: mid-range for the
    # thigh and calf, zero for the hip, so the chirp never hits a stop.
    for k, name in enumerate(joints):
        lo, hi = model.jnt_range[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        home[k] = 0.0 if joint_class(name) == "hip" else 0.5 * (lo + hi)
    data.qpos[qpos_adr] = home
    # The floating base is the body the free joint moves, wherever the
    # joint sits and whatever the body is called (a name lookup returned -1
    # on a model whose root is not `base_link`, and the shake landed on the
    # last body; review 2026-09-24).
    free = free_joint_of(model)
    base_free = free is not None
    body = int(model.jnt_bodyid[free]) if base_free else -1
    base_q = int(model.jnt_qposadr[free]) if base_free else 0
    base_v = int(model.jnt_dofadr[free]) if base_free else 0
    weight = float(np.sum(model.body_mass)) * float(-model.opt.gravity[2])
    mujoco.mj_forward(model, data)

    steps_per_control = round(servo.physics_hz / servo.control_hz)
    n = int(chirp.seconds * servo.control_hz)
    times = np.arange(n) / servo.control_hz
    phases = np.linspace(0.0, 2 * np.pi, len(joints), endpoint=False)
    amps = np.array([chirp.amplitude_of(joint_class(j)) for j in joints])
    sweep = chirp.f0_hz + (chirp.f1_hz - chirp.f0_hz) * times / chirp.seconds
    phase = 2 * np.pi * np.cumsum(sweep) / servo.control_hz
    targets = home + amps * np.sin(phase[:, None] + phases[None, :])

    position = np.zeros((n, len(joints)))
    velocity = np.zeros_like(position)
    torque = np.zeros_like(position)
    acceleration = np.zeros_like(position)
    base_pose = np.zeros((n, 7))
    base_vel = np.zeros((n, 6))
    base_acc = np.zeros((n, 6))
    for i in range(n):
        # The record is what a robot logs at tick i: the state the torque
        # was computed from and the torque then applied over the tick — a
        # log that stored the post-tick state beside the tick's torque
        # would be one tick misaligned and bias every term.
        q = data.qpos[qpos_adr]
        v = data.qvel[dof_adr]
        tau = servo.kp * (targets[i] - q) - servo.kd * v
        data.ctrl[:] = tau
        if base_free:
            w = 2 * np.pi * shake.hz * times[i]
            data.xfrc_applied[body, :3] = [
                shake.force_n * np.sin(w),
                shake.force_n * np.cos(1.3 * w),
                weight + shake.force_n * np.sin(0.7 * w),
            ]
            data.xfrc_applied[body, 3:] = shake.torque_nm * np.array(
                [np.sin(1.1 * w), np.cos(0.9 * w), np.sin(0.5 * w)]
            )
        mujoco.mj_forward(model, data)
        position[i] = data.qpos[qpos_adr]
        velocity[i] = data.qvel[dof_adr]
        torque[i] = data.actuator_force
        acceleration[i] = data.qacc[dof_adr]
        if base_free:
            base_pose[i] = data.qpos[base_q : base_q + FREE_QPOS]
            base_vel[i] = data.qvel[base_v : base_v + FREE_DOF]
            base_acc[i] = data.qacc[base_v : base_v + FREE_DOF]
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)

    if corruption is not None:
        rng = np.random.default_rng(corruption.seed)
        step = 2 * np.pi / corruption.encoder_counts
        position[...] = np.round(position / step) * step
        velocity[...] = np.gradient(position, 1.0 / servo.control_hz, axis=0)
        torque[...] += rng.normal(0.0, corruption.torque_noise_nm, torque.shape)

    channels = {
        JOINT_POSITION: Channel(JOINT_POSITION, times, position, "rad", joints),
        JOINT_VELOCITY: Channel(JOINT_VELOCITY, times, velocity, "rad/s", joints),
        JOINT_EFFORT: Channel(JOINT_EFFORT, times, torque, "N*m", joints),
        # The integrator's own acceleration at each tick: what no robot
        # records, kept under its own name so a method never mistakes it
        # for a measurement — the study compares against it.
        ACCELERATION_TRUTH: Channel(
            ACCELERATION_TRUTH, times, acceleration, "rad/s^2", joints
        ),
        JOINT_COMMAND: Channel(JOINT_COMMAND, times, targets, "rad", joints),
        JOINT_COMMAND_VELOCITY: Channel(
            JOINT_COMMAND_VELOCITY, times, np.zeros_like(targets), "rad/s", joints
        ),
        JOINT_KP: Channel(
            JOINT_KP, times, np.full_like(targets, servo.kp), "N*m/rad", joints
        ),
        JOINT_KD: Channel(
            JOINT_KD, times, np.full_like(targets, servo.kd), "N*m*s/rad", joints
        ),
    }
    if base_free:
        channels[BASE_POSE] = Channel(
            BASE_POSE,
            times,
            base_pose,
            "m, unit quaternion",
            POSE_COMPONENTS,
        )
        channels[BASE_TWIST] = Channel(
            BASE_TWIST,
            times,
            base_vel,
            "m/s, rad/s",
            TWIST_COMPONENTS,
        )
        channels[BASE_ACCELERATION] = Channel(
            BASE_ACCELERATION,
            times,
            base_acc,
            "m/s^2, rad/s^2",
            ACCELERATION_COMPONENTS,
        )
    recording = Recording(
        source=f"synthetic-{posture}-chirp",
        adapter="synthetic",
        channels=channels,
        census={
            "joints": list(joints),
            "posture": posture,
            "base": "fixed in the air"
            if posture == FIXED
            else "floating, shaken, no floor",
            "truth": {j: vars(t) for j, t in truth.items()},
            "servo": vars(servo),
            "chirp": {
                "seconds": chirp.seconds,
                "f0_hz": chirp.f0_hz,
                "f1_hz": chirp.f1_hz,
            },
            "corruption": vars(corruption) if corruption else None,
        },
        notes=[
            "synthetic: a true model rolled out in MuJoCo; the truth is in the census"
        ],
        collection=COLLECTION_SCRIPTED,
        basis=BASIS_SIMULATION,
    )
    return recording, truth
