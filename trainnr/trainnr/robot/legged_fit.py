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
component names are the model's joint names, or map onto them through
a declared table (`robots.joint_orders.JOINT_RENAMES`: Unitree's bus
names a motor `FR_hip`, the model `FR_hip_joint`); a torque source; for a
floating-base model with a walking log, an orientation and a gyro (an
IMU, or base channels already in MuJoCo's conventions) and foot
contacts, else every sample counts and the anchor says the base was
taken as still. No walking robot's stance legs are fitted: a foot on
the ground carries a contact force the log does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from trainnr.bundles.bundle import model_file_for_use
from trainnr.bundles.hashing import stamp
from trainnr.robot.fit_record import write_fit_record
from trainnr.robot.identify import IdentificationResult
from trainnr.robot.import_audit import quat_to_matrix
from trainnr.robot.methods import method
from trainnr.robot.torque_balance import (
    DEFAULT_BANDWIDTH,
    DEFAULT_BOUNDS,
    DEFAULT_COULOMB,
    DEFAULT_GATE,
    DEFAULT_PLAN,
    DEFAULT_SMOOTHING,
    FREE_QPOS,
    Bandwidth,
    BootstrapPlan,
    Coulomb,
    JointSamples,
    Smoothing,
    TermBounds,
    TorqueBalance,
    VelocityGate,
    fit_terms,
    free_joint_of,
    savgol_window,
    tick_aligned,
    units_for,
)
from trainnr.robots.joint_orders import anatomy_of, rename_to
from trainnr.robots.recording import (
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
MIN_HINGES = 3  # fewer is not a leg
# The torque sources in the order the method takes them: what the motor
# did before what it was asked.
EFFORT = "measured effort"
PD_COMMAND = "pd command"
TORQUE_SOURCES = (EFFORT, PD_COMMAND)
# Said beside "pd command" when the log carries no command velocity.
NO_COMMAND_VELOCITY = "command velocity absent in the log: taken as zero"


@dataclass(frozen=True)
class Decimation:
    """How a long log is thinned before inverse dynamics. `max_samples`
    is a compute bound (one inverse-dynamics call per sample). The
    thinned rate keeps its Nyquist at least `nyquist_margin` times the
    balance's cutoff, so the zero-phase low-pass runs on what is kept:
    a stride chosen by the budget alone took an hour-long 500 Hz log to
    17 Hz, under the 15 Hz band, and the filter silently did nothing
    while the record said it ran (review 2026-09-24). Where the two
    disagree the band wins and the log keeps more samples than the
    budget. Declared ours; the thinning is every k-th aligned sample."""

    max_samples: int = 60_000
    nyquist_margin: float = 2.0

    def stride(self, samples: int, rate_hz: float, cutoff_hz: float) -> int:
        by_budget = max(1, int(np.ceil(samples / self.max_samples)))
        if cutoff_hz <= 0:
            return by_budget
        by_band = max(1, int(rate_hz // (2.0 * self.nyquist_margin * cutoff_hz)))
        return min(by_budget, by_band)


@dataclass(frozen=True)
class ContactMask:
    """Which samples of a leg count: in the air, `margin_s` clear of every
    touchdown and lift-off. The margin is the balance's filter transient:
    a second-order zero-phase Butterworth at 15 Hz settles to 2 % in about
    two periods of its cutoff, 0.13 s over both passes, ~0.04 s either side
    of an edge (declared ours from the filter; not fitted). `in_contact`
    is the threshold on a contact channel that reads 1.0 down, 0.0 up."""

    margin_s: float = 0.04
    in_contact: float = 0.5


DEFAULT_DECIMATION = Decimation()
DEFAULT_CONTACT = ContactMask()


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


def torque_source_of(recording: Recording) -> str:
    """Which TORQUE_SOURCES entry the recording supplies, decided from its
    CHANNELS over the whole log (a two-sample look refused a log whose
    motors idled at the start, and could pick another source than the fit
    used; review 2026-09-24); refused by name when there is none."""
    effort = recording.channels.get(JOINT_EFFORT)
    if effort is not None and np.any(effort.values != 0.0):
        return EFFORT
    if all(n in recording.channels for n in (JOINT_COMMAND, JOINT_KP, JOINT_KD)):
        return PD_COMMAND
    raise ValueError(
        "no torque source: the recording carries neither a non-zero "
        f"{JOINT_EFFORT} nor {JOINT_COMMAND} with {JOINT_KP} and {JOINT_KD}"
    )


def torque_of(recording: Recording, times: np.ndarray) -> tuple[np.ndarray, str]:
    """The joint torque on the recording's clock and the source it came
    from (`torque_source_of`), said with any assumption the source made."""
    source = torque_source_of(recording)
    if source == EFFORT:
        effort = _resample(recording, JOINT_EFFORT, times)
        assert effort is not None
        return effort, EFFORT
    command = _resample(recording, JOINT_COMMAND, times)
    kp = _resample(recording, JOINT_KP, times)
    kd = _resample(recording, JOINT_KD, times)
    q = _resample(recording, JOINT_POSITION, times)
    v = _resample(recording, JOINT_VELOCITY, times)
    v_cmd = _resample(recording, JOINT_COMMAND_VELOCITY, times)
    ff = _resample(recording, JOINT_FEEDFORWARD, times)
    if command is None or kp is None or kd is None or q is None or v is None:
        raise ValueError(
            f"{PD_COMMAND} needs {JOINT_POSITION} and {JOINT_VELOCITY} beside the "
            "command and the gains"
        )
    said = PD_COMMAND
    if v_cmd is None:
        said = f"{PD_COMMAND} ({NO_COMMAND_VELOCITY})"
    torque = kp * (command - q) + kd * ((v_cmd if v_cmd is not None else 0.0) - v)
    if ff is not None:
        torque = torque + ff
    return torque, said


def base_of(
    recording: Recording,
    times: np.ndarray,
    model: Any,
    smoothing: Smoothing,
) -> tuple[dict[str, np.ndarray], str]:
    """The floating base's pose, velocity and acceleration on the clock,
    in MuJoCo's free-joint conventions, and the sentence saying where
    they came from. Empty when the model has no free joint."""
    import mujoco  # noqa: PLC0415

    free = free_joint_of(model)
    if free is None:
        return {}, "the model has no floating base"
    n = len(times)
    pose = _resample(recording, BASE_POSE, times)
    twist = _resample(recording, BASE_TWIST, times)
    accel = _resample(recording, BASE_ACCELERATION, times)
    quat = _resample(recording, IMU_ORIENTATION, times)
    gyro = _resample(recording, IMU_ANGULAR_VELOCITY, times)
    specific = _resample(recording, IMU_LINEAR_ACCELERATION, times)
    adr = int(model.jnt_qposadr[free])
    home = mujoco.MjData(model).qpos[adr : adr + FREE_QPOS].copy()
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

        ang_acc = savgol_filter(
            angular,
            savgol_window(smoothing, rate),
            smoothing.order,
            deriv=1,
            delta=dt,
            axis=0,
        )
        lin = np.zeros((n, 3))
        gravity = np.asarray(model.opt.gravity, dtype=np.float64)
        if specific is not None:
            for i in range(n):
                lin[i] = quat_to_matrix(pose[i, 3:]) @ specific[i] + gravity
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
    recording: Recording,
    times: np.ndarray,
    joints: tuple[str, ...],
    mask: ContactMask = DEFAULT_CONTACT,
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
    margin = round(mask.margin_s / dt)
    in_air = values < mask.in_contact
    for k in range(len(feet)):
        edges = np.flatnonzero(np.diff(values[:, k] >= mask.in_contact))
        for edge in edges:
            in_air[max(0, edge - margin) : edge + margin + 1, k] = False
    usable = np.ones((len(times), len(joints)), dtype=bool)
    for j, joint in enumerate(joints):
        leg = anatomy_of(joint)[0]
        if leg not in feet:
            raise ValueError(
                f"joint {joint!r} names leg {leg!r}, not one of the contact feet {feet}"
            )
        usable[:, j] = in_air[:, feet.index(leg)]
    share = usable.mean(axis=0)
    return usable, (
        f"swing-phase samples only, {mask.margin_s * 1e3:.0f} ms clear of every "
        f"touchdown and lift-off ({share.min():.0%}-{share.max():.0%} of the log "
        "per joint)"
    )


def hinges_of(model: Any) -> set[str]:
    import mujoco  # noqa: PLC0415

    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    }


def prepare(  # noqa: PLR0913 - the balance's knobs, each a named dataclass
    recording: Recording,
    model: Any,
    smoothing: Smoothing,
    bandwidth: Bandwidth,
    *,
    decimation: Decimation = DEFAULT_DECIMATION,
    mask: ContactMask = DEFAULT_CONTACT,
) -> tuple[JointSamples, str, str]:
    """The recording as the fitter's samples: joint channels on the
    position clock (named as the model's hinges, through a declared rename
    table when the log uses the vendor's motor names), tick-aligned
    derivatives, torque, base, mask."""
    position = recording.channels.get(JOINT_POSITION)
    velocity = recording.channels.get(JOINT_VELOCITY)
    if position is None or velocity is None:
        raise ValueError(f"the recording needs {JOINT_POSITION} and {JOINT_VELOCITY}")
    if not position.components:
        raise ValueError(f"{JOINT_POSITION} names no joints in its components")
    joints, renamed = rename_to(tuple(position.components), hinges_of(model))
    clock = position.times
    torque, source = torque_of(recording, clock)
    times, q_mid, v_mid, acc = tick_aligned(clock, position.values, smoothing=smoothing)
    torque = torque[:-1]
    base, base_words = base_of(recording, times, model, smoothing)
    usable, mask_words = usable_of(recording, times, joints, mask)
    rate = 1.0 / float(np.median(np.diff(times)))
    stride = decimation.stride(len(times), rate, bandwidth.cutoff_hz)
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
    if renamed is not None:
        handling += f"; joint names mapped to the model's by {renamed!r}"
    if stride > 1:
        handling += (
            f"; every {stride}th aligned sample kept ({len(times)} of the log, "
            f"{rate / stride:.0f} Hz, no anti-alias filter before thinning)"
        )
    return samples, source, handling


def _filtered(extras: dict[str, Any]) -> str:
    cutoff = float(extras["cutoff_hz"])
    if cutoff <= 0:
        return "not low-passed"
    return (
        f"both sides low-passed at {cutoff:g} Hz (zero phase) on samples at "
        f"{float(extras['rate_hz']):.0f} Hz"
    )


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
        f"samples slower than {gate.still:g} rad/s left out; "
        f"{_filtered(fit.balance.extras)}; "
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
        decimation: Decimation = DEFAULT_DECIMATION,
        mask: ContactMask = DEFAULT_CONTACT,
    ) -> None:
        self.bounds = bounds
        self.plan = plan
        self.gate = gate
        self.smoothing = smoothing
        self.bandwidth = bandwidth
        self.coulomb = coulomb
        self.decimation = decimation
        self.mask = mask

    def accepts(  # noqa: PLR0911 - one return per reason the fit cannot run
        self, bundle_dir: Path, recording_dir: Path
    ) -> str | None:
        import mujoco  # noqa: PLC0415

        bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
        model_file = model_file_for_use(bundle_dir)
        if model_file is None:
            return "the bundle has no MJCF"
        try:
            model = mujoco.MjModel.from_xml_path(str(model_file))
        except ValueError as why:
            return f"the bundle's MJCF does not compile ({why})"
        hinges = hinges_of(model)
        if len(hinges) < MIN_HINGES:
            return f"the model has {len(hinges)} hinge joints, fewer than {MIN_HINGES}"
        try:
            recording = Recording.read(recording_dir)
        except (OSError, ValueError) as why:
            return f"the recording cannot be read ({why})"
        position = recording.channels.get(JOINT_POSITION)
        if position is None or JOINT_VELOCITY not in recording.channels:
            return f"the recording has no {JOINT_POSITION} and {JOINT_VELOCITY}"
        try:
            rename_to(tuple(position.components), hinges)
            torque_source_of(recording)
        except ValueError as why:
            return str(why)
        return None

    def fit_balance(self, bundle_dir: Path, recording: Recording | Path) -> LeggedFit:
        import mujoco  # noqa: PLC0415

        model_file = model_file_for_use(Path(bundle_dir))
        if model_file is None:
            raise ValueError("the bundle has no MJCF")
        model = mujoco.MjModel.from_xml_path(str(model_file))
        if not isinstance(recording, Recording):
            recording = Recording.read(Path(recording))
        samples, source, handling = prepare(
            recording,
            model,
            self.smoothing,
            self.bandwidth,
            decimation=self.decimation,
            mask=self.mask,
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
        return replace(fit, anchor=anchor_statement(fit, self.smoothing, self.gate))

    def fit(
        self, bundle_dir: Path, recording_dir: Path, *, write: bool = True
    ) -> tuple[IdentificationResult, Path | None]:
        bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
        recording = Recording.read(recording_dir)
        fit = self.fit_balance(bundle_dir, recording)
        result = fit.balance.result
        if not write:
            return result, None
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
