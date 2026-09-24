"""The legged identification method: a quadruped's joints from a log.

`legged-joints` fits every hinge joint's armature, viscous damping and
Coulomb friction from a recording of joint position, velocity and
torque (measured effort, or the motor-side PD law reconstructed from
the command and the gains), through `torque_balance`. It reads the
robot's own MJCF from its bundle — the model whose rigid bodies the
torque balance subtracts — and writes a fit record with bootstrap
intervals, pinned verdicts, bound flags, the torque source, the base
handling and the data's basis (own robot, public log, simulation).

What it needs and what it refuses, by name: joint channels whose
component names are the model's joint names; a torque source; for a
floating-base model with a walking log, an orientation and a gyro (an
IMU, or base channels already in MuJoCo's conventions) and foot
contacts, else every sample counts and the anchor says the base was
taken as still. No walking robot's stance legs are fitted: a foot on
the ground carries a contact force the log does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.bundles.bundle import model_file_of
from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.robot.fit_record import write_fit_record
from rq_pipeline.robot.identify import IdentificationResult
from rq_pipeline.robot.methods import method
from rq_pipeline.robot.torque_balance import (
    DEFAULT_BANDWIDTH,
    DEFAULT_BOUNDS,
    DEFAULT_COULOMB,
    DEFAULT_GATE,
    DEFAULT_PLAN,
    DEFAULT_SMOOTHING,
    Bandwidth,
    BootstrapPlan,
    Coulomb,
    JointSamples,
    Smoothing,
    TermBounds,
    TorqueBalance,
    VelocityGate,
    fit_terms,
    tick_aligned,
    units_for,
)
from rq_pipeline.robots.recording import (
    BASE_ACCELERATION,
    BASE_POSE,
    BASE_TWIST,
    FOOT_CONTACT,
    IMU_ANGULAR_VELOCITY,
    IMU_LINEAR_ACCELERATION,
    IMU_ORIENTATION,
    JOINT_COMMAND,
    JOINT_COMMAND_VELOCITY,
    JOINT_EFFORT,
    JOINT_FEEDFORWARD,
    JOINT_KD,
    JOINT_KP,
    JOINT_POSITION,
    JOINT_VELOCITY,
    Recording,
)

NAME = "legged-joints"
GRAVITY = np.array([0.0, 0.0, -9.81])
MIN_HINGES = 3  # fewer is not a leg
MAX_SAMPLES = 60_000  # inverse dynamics per sample; decimate a longer log
# Samples this close to a touchdown or lift-off are out: the filter's
# transient at the balance's bandwidth.
CONTACT_MARGIN_S = 0.04
IN_CONTACT = 0.5  # a contact channel reads 1.0 down, 0.0 in the air
# The torque sources in the order the method takes them: what the motor
# did before what it was asked.
EFFORT = "measured effort"
PD_COMMAND = "pd command"
TORQUE_SOURCES = (EFFORT, PD_COMMAND)


@dataclass(frozen=True)
class LeggedFit:
    """A fit with everything a viewer shows beside the record."""

    balance: TorqueBalance
    samples: JointSamples
    torque_source: str
    base_handling: str
    anchor: str


def _resample(recording: Recording, name: str, times: np.ndarray) -> np.ndarray | None:
    channel = recording.channels.get(name)
    if channel is None:
        return None
    values = np.atleast_2d(channel.values.T).T
    out = np.column_stack(
        [np.interp(times, channel.times, values[:, k]) for k in range(values.shape[1])]
    )
    return out


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def torque_of(recording: Recording, times: np.ndarray) -> tuple[np.ndarray, str]:
    """The joint torque on the recording's clock and the source it came
    from, in TORQUE_SOURCES order; refused by name when there is none."""
    effort = _resample(recording, JOINT_EFFORT, times)
    if effort is not None and np.any(effort != 0.0):
        return effort, EFFORT
    command = _resample(recording, JOINT_COMMAND, times)
    kp = _resample(recording, JOINT_KP, times)
    kd = _resample(recording, JOINT_KD, times)
    if command is not None and kp is not None and kd is not None:
        q = _resample(recording, JOINT_POSITION, times)
        v = _resample(recording, JOINT_VELOCITY, times)
        v_cmd = _resample(recording, JOINT_COMMAND_VELOCITY, times)
        ff = _resample(recording, JOINT_FEEDFORWARD, times)
        assert q is not None and v is not None
        torque = kp * (command - q) + kd * ((v_cmd if v_cmd is not None else 0.0) - v)
        if ff is not None:
            torque = torque + ff
        return torque, PD_COMMAND
    raise ValueError(
        "no torque source: the recording carries neither a non-zero "
        f"{JOINT_EFFORT} nor {JOINT_COMMAND} with {JOINT_KP} and {JOINT_KD}"
    )


def base_of(
    recording: Recording,
    times: np.ndarray,
    model: Any,
    smoothing: Smoothing,
    bandwidth: Bandwidth,
) -> tuple[dict[str, np.ndarray], str]:
    """The floating base's pose, velocity and acceleration on the clock,
    in MuJoCo's free-joint conventions, and the sentence saying where
    they came from. Empty when the model has no free joint."""
    import mujoco  # noqa: PLC0415

    if not any(
        model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(model.njnt)
    ):
        return {}, "the model has no floating base"
    n = len(times)
    pose = _resample(recording, BASE_POSE, times)
    twist = _resample(recording, BASE_TWIST, times)
    accel = _resample(recording, BASE_ACCELERATION, times)
    quat = _resample(recording, IMU_ORIENTATION, times)
    gyro = _resample(recording, IMU_ANGULAR_VELOCITY, times)
    specific = _resample(recording, IMU_LINEAR_ACCELERATION, times)
    home = mujoco.MjData(model).qpos[:7].copy()
    words = []
    if pose is None and quat is not None:
        pose = np.tile(home, (n, 1))
        pose[:, 3:] = quat
        words.append("orientation from the IMU")
    elif pose is not None:
        words.append("pose from the base channel")
    if gyro is not None:
        angular = gyro
        words.append("angular velocity from the IMU gyro")
    elif twist is not None:
        angular = twist[:, 3:]
        words.append("angular velocity from the base channel")
    else:
        angular = None
    if pose is None or angular is None:
        return {}, (
            "the base taken as still: no orientation and gyro in the recording "
            "(a log of a robot in the air)"
        )
    velocity = np.zeros((n, 6))
    velocity[:, 3:] = angular
    if accel is None:
        dt = float(np.median(np.diff(times)))
        rate = 1.0 / dt
        # The gyro's derivative over the smoothing window and the
        # accelerometer as logged: the balance low-passes both sides of
        # the regression once, so nothing is pre-filtered here.
        from scipy.signal import savgol_filter  # noqa: PLC0415

        window = round(smoothing.window_s * rate) | 1
        ang_acc = savgol_filter(
            angular,
            max(window, smoothing.order + 2),
            smoothing.order,
            deriv=1,
            delta=dt,
            axis=0,
        )
        lin = np.zeros((n, 3))
        if specific is not None:
            for i in range(n):
                lin[i] = _quat_to_matrix(pose[i, 3:]) @ specific[i] + GRAVITY
            words.append(
                "linear acceleration from the IMU's specific force plus gravity"
            )
        else:
            words.append("linear acceleration taken as zero")
        accel = np.column_stack([lin, ang_acc])
        words.append(f"angular acceleration from the gyro ({dt * 1e3:.0f} ms ticks)")
    else:
        words.append("acceleration from the base channel")
    # Linear base velocity does not enter the joint torques (Galilean
    # invariance), so it stays zero whatever the log says.
    return {
        "base_pose": pose,
        "base_velocity": velocity,
        "base_acceleration": accel,
    }, "; ".join(words)


def usable_of(
    recording: Recording, times: np.ndarray, joints: tuple[str, ...]
) -> tuple[np.ndarray | None, str]:
    """Per joint, the samples whose leg is in the air, with a margin
    around every contact transition; None when there is no contact
    channel (every sample counts)."""
    contact = recording.channels.get(FOOT_CONTACT)
    if contact is None:
        return None, "every sample counts (no foot-contact channel)"
    feet = tuple(contact.components)
    values = np.column_stack(
        [
            np.interp(times, contact.times, contact.values[:, k])
            for k in range(len(feet))
        ]
    )
    dt = float(np.median(np.diff(times)))
    margin = round(CONTACT_MARGIN_S / dt)
    in_air = values < IN_CONTACT
    for k in range(len(feet)):
        edges = np.flatnonzero(np.diff(values[:, k] >= IN_CONTACT))
        for edge in edges:
            in_air[max(0, edge - margin) : edge + margin + 1, k] = False
    usable = np.ones((len(times), len(joints)), dtype=bool)
    for j, joint in enumerate(joints):
        leg = joint.split("_", 1)[0]
        if leg not in feet:
            raise ValueError(
                f"joint {joint!r} names leg {leg!r}, not one of the contact feet {feet}"
            )
        usable[:, j] = in_air[:, feet.index(leg)]
    share = usable.mean(axis=0)
    return usable, (
        f"swing-phase samples only, {CONTACT_MARGIN_S * 1e3:.0f} ms clear of every "
        f"touchdown and lift-off ({share.min():.0%}-{share.max():.0%} of the log "
        "per joint)"
    )


def prepare(
    recording: Recording, model: Any, smoothing: Smoothing, bandwidth: Bandwidth
) -> tuple[JointSamples, str, str]:
    """The recording as the fitter's samples: joint channels on the
    position clock, tick-aligned derivatives, torque, base, mask."""
    position = recording.channels.get(JOINT_POSITION)
    velocity = recording.channels.get(JOINT_VELOCITY)
    if position is None or velocity is None:
        raise ValueError(f"the recording needs {JOINT_POSITION} and {JOINT_VELOCITY}")
    joints = tuple(position.components)
    if not joints:
        raise ValueError(f"{JOINT_POSITION} names no joints in its components")
    clock = position.times
    torque, source = torque_of(recording, clock)
    times, q_mid, v_mid, acc = tick_aligned(clock, position.values, smoothing=smoothing)
    torque = torque[:-1]
    base, base_words = base_of(recording, times, model, smoothing, bandwidth)
    usable, mask_words = usable_of(recording, times, joints)
    stride = max(1, int(np.ceil(len(times) / MAX_SAMPLES)))
    if stride > 1:
        times, q_mid, v_mid, acc, torque = (
            a[::stride] for a in (times, q_mid, v_mid, acc, torque)
        )
        base = {k: a[::stride] for k, a in base.items()}
        usable = None if usable is None else usable[::stride]
    samples = JointSamples(
        times=times,
        joints=joints,
        position=q_mid,
        velocity=v_mid,
        acceleration=acc,
        torque=torque,
        usable=usable,
        **base,
    )
    handling = f"{base_words}; {mask_words}"
    if stride > 1:
        handling += f"; every {stride}th aligned sample kept ({len(times)} of the log)"
    return samples, source, handling


def anchor_statement(fit: LeggedFit, smoothing: Smoothing, gate: VelocityGate) -> str:
    per_joint = ", ".join(
        f"{j.joint} {j.samples_used} samples explained {j.explained:.1%}"
        for j in fit.balance.joints
    )
    return (
        f"TORQUE BALANCE: tau_motor - tau_rigid regressed per joint on "
        f"[ddq, dq, tanh(dq/{fit.balance.extras['coulomb_knee']:g})] with bounds; "
        f"torque = {fit.torque_source}; "
        f"the rigid torque is MuJoCo's inverse dynamics of the bundle's model "
        f"with armature, damping and frictionloss zeroed and contacts off; "
        f"derivatives from the position (velocity a Savitzky-Golay derivative over "
        f"{smoothing.window_s * 1e3:.0f} ms, acceleration its forward difference over "
        f"the tick, mid-tick state); "
        f"samples slower than {gate.still:g} rad/s left out; both sides low-passed at "
        f"{fit.balance.extras['cutoff_hz']:g} Hz (zero phase); "
        f"base: {fit.base_handling}. "
        f"Intervals: moving-block bootstrap, {fit.balance.extras['replicates']} "
        f"replicates of {fit.balance.extras['block_samples']}-sample blocks. "
        f"Per joint: {per_joint}."
    )


@method(
    NAME,
    doc="A legged robot's joints from a log: armature, damping and Coulomb "
    "friction per hinge by torque balance against the bundle's MJCF, "
    "bootstrap intervals, bound flags.",
)
class LeggedJoints:
    name = NAME
    # Nothing fixed from outside the data: the rigid model is the bundle's.
    anchored = ()

    def __init__(  # noqa: PLR0913 - the balance's knobs, each a named dataclass
        self,
        *,
        bounds: TermBounds = DEFAULT_BOUNDS,
        plan: BootstrapPlan = DEFAULT_PLAN,
        gate: VelocityGate = DEFAULT_GATE,
        smoothing: Smoothing = DEFAULT_SMOOTHING,
        bandwidth: Bandwidth = DEFAULT_BANDWIDTH,
        coulomb: Coulomb = DEFAULT_COULOMB,
    ) -> None:
        self.bounds = bounds
        self.plan = plan
        self.gate = gate
        self.smoothing = smoothing
        self.bandwidth = bandwidth
        self.coulomb = coulomb

    def accepts(  # noqa: PLR0911 - one return per reason the fit cannot run
        self, bundle_dir: Path, recording_dir: Path
    ) -> str | None:
        import mujoco  # noqa: PLC0415

        bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
        model_file = model_file_of(bundle_dir)
        if model_file is None:
            return "the bundle has no MJCF"
        try:
            model = mujoco.MjModel.from_xml_path(str(model_file))
        except ValueError as why:
            return f"the bundle's MJCF does not compile ({why})"
        hinges = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        if len(hinges) < MIN_HINGES:
            return f"the model has {len(hinges)} hinge joints, fewer than {MIN_HINGES}"
        try:
            recording = Recording.read(recording_dir)
        except (OSError, ValueError) as why:
            return f"the recording cannot be read ({why})"
        position = recording.channels.get(JOINT_POSITION)
        if position is None or JOINT_VELOCITY not in recording.channels:
            return f"the recording has no {JOINT_POSITION} and {JOINT_VELOCITY}"
        unknown = [name for name in position.components if name not in hinges]
        if unknown:
            return f"joints {unknown} are not hinges of the model"
        try:
            torque_of(recording, position.times[:2])
        except ValueError as why:
            return str(why)
        return None

    def fit_balance(self, bundle_dir: Path, recording_dir: Path) -> LeggedFit:
        import mujoco  # noqa: PLC0415

        model_file = model_file_of(Path(bundle_dir))
        if model_file is None:
            raise ValueError("the bundle has no MJCF")
        model = mujoco.MjModel.from_xml_path(str(model_file))
        recording = Recording.read(Path(recording_dir))
        samples, source, handling = prepare(
            recording, model, self.smoothing, self.bandwidth
        )
        balance = fit_terms(
            model,
            samples,
            bounds=self.bounds,
            plan=self.plan,
            gate=self.gate,
            bandwidth=self.bandwidth,
            coulomb=self.coulomb,
        )
        fit = LeggedFit(
            balance=balance,
            samples=samples,
            torque_source=source,
            base_handling=handling,
            anchor="",
        )
        return LeggedFit(
            balance=balance,
            samples=samples,
            torque_source=source,
            base_handling=handling,
            anchor=anchor_statement(fit, self.smoothing, self.gate),
        )

    def fit(
        self, bundle_dir: Path, recording_dir: Path, *, write: bool = True
    ) -> tuple[IdentificationResult, Path | None]:
        bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
        fit = self.fit_balance(bundle_dir, recording_dir)
        result = fit.balance.result
        if not write:
            return result, None
        recording = Recording.read(recording_dir)
        path = write_fit_record(
            bundle_dir,
            result,
            robot=bundle_dir.name,
            recording=stamp(recording_dir.name, recording_dir),
            anchor=fit.anchor,
            units=units_for(fit.samples.joints),
            basis=recording.basis,
            provenance={
                "source": recording.source,
                "adapter": recording.adapter,
                "collection": recording.collection,
                "notes": list(recording.notes),
                "torque_source": fit.torque_source,
            },
            metrics={
                j.joint: {
                    "samples": j.samples_used,
                    "rms_before": j.rms_before,
                    "rms_after": j.rms_after,
                    "explained": j.explained,
                }
                for j in fit.balance.joints
            },
        )
        return result, path
