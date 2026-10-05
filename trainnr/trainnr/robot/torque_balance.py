"""Per-joint passive terms from a torque balance — the legged fit's core.

For every joint of a rigid-body model the motor torque splits into what
the rigid body needs and what the joint's own passivity eats:

    tau_motor = tau_rigid(q, dq, ddq)          — MuJoCo's inverse dynamics
              + armature * ddq                 — the rotor's reflected inertia
              + damping * dq                   — viscous friction
              + frictionloss * tanh(dq/knee)   — Coulomb friction, a smooth sign

`tau_rigid` comes from `mj_inverse` on the model with those three terms
zeroed (and contacts off), so the residual `tau_motor - tau_rigid` is
LINEAR in the three unknowns and every joint is a small bounded
least-squares problem — no rollout, no optimizer state, seconds for
minutes of data. The interval is a moving-block bootstrap over time
(blocks longer than the friction correlation time, resampled with
replacement), the verdict the same range-relative rule every fit in
this repo uses (`identify.DEFAULT_PINNED_FRACTION`), and an estimate
sitting on its search bound is flagged, because a bound hit is the
fit saying "the data pushed me here and I could not go further".

Where the model has a floating base, the base's recorded orientation,
angular velocity and accelerations enter the inverse dynamics, so a
walking robot's swing legs can be fitted; which samples count for a
joint (a foot in the air, a velocity outside the stiction band) is a
mask the caller builds from what it recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trainnr.robot.identify import (
    DEFAULT_PINNED_FRACTION,
    IdentificationResult,
    IdentifiedParameter,
)
from trainnr.stats.intervals import DEFAULT_CONFIDENCE

TERMS = ("armature", "damping", "frictionloss")
UNITS = {"armature": "kg*m^2", "damping": "N*m*s/rad", "frictionloss": "N*m"}
# An estimate within this fraction of its range from a bound is AT the
# bound (the actuator bundles' RAIL_TOLERANCE, the same 1 %).
AT_BOUND_FRACTION = 0.01


@dataclass(frozen=True)
class TermBounds:
    """The search box per term, one truth for every quadruped joint.

    The Go2's declared values are armature 0.01-0.02 kg*m^2 and damping
    0.5-2 N*m*s/rad (unitree_rl_mjlab's actuator table, its vendor DR
    tables), Coulomb friction under 1 N*m in every published quadruped
    fit read (2026-09-24). Each box is 2-3 times the largest
    declared value: wide enough that a bound hit is information about
    the data, never a clamp that flatters the fit, and narrow enough that
    the range-relative pinned rule (a tenth of the box) still means
    something — armature within 0.005, damping within 0.3, friction
    within 0.2.
    """

    armature: tuple[float, float] = (0.0, 0.05)
    damping: tuple[float, float] = (0.0, 3.0)
    frictionloss: tuple[float, float] = (0.0, 2.0)

    def of(self, term: str) -> tuple[float, float]:
        return getattr(self, term)


@dataclass(frozen=True)
class BootstrapPlan:
    """Moving-block bootstrap: `replicates` refits over blocks of
    `block_s` seconds drawn with replacement. Half a second is longer
    than any friction transient at walking speeds; 200 replicates put
    the 2.5/97.5 percentiles within a few percent of their limit."""

    replicates: int = 200
    block_s: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class Coulomb:
    """The Coulomb column is `tanh(dq / knee)`, not `sign(dq)`: a smooth
    sign that reaches the full value above the knee. MuJoCo applies
    frictionloss as a soft constraint — measured on a hinge under the
    default impedance (2026-09-24): 69 % of the declared value at 0.2-0.5
    rad/s, 87 % at 0.5-1, 99.8 % above 1 rad/s, which a 0.45 rad/s knee
    reproduces — and a real actuator has its own knee there. A smooth
    column also survives the balance's low-pass: a hard sign's harmonics
    fall out of the regressor and not out of the torque, and the fit read
    friction 16-23 % low with armature 20 % high (synthetic Go2, 6 Hz)."""

    knee: float = 0.45

    def column(self, velocity: np.ndarray) -> np.ndarray:
        return np.tanh(velocity / self.knee)


@dataclass(frozen=True)
class VelocityGate:
    """Samples slower than `still` rad/s are left out of every column:
    below it the friction is in its knee (MuJoCo's soft constraint, a
    real actuator's stiction) and the smooth sign is only a model of it.
    Measured on the synthetic Go2 (2026-09-24): a 0.1 rad/s gate reads
    friction 13 % high and armature 6 % high; 1 rad/s reads every term
    within 3 %. A chirp and a swing leg spend most of their time above
    it."""

    still: float = 1.0


FREE_QPOS = 7  # a free joint's position: xyz and a unit quaternion
FREE_DOF = 6  # its velocity: linear and angular


def free_joint_of(model: Any) -> int | None:
    """The model's one free joint (the floating base), or None for a fixed
    base; more than one is refused by name."""
    import mujoco  # noqa: PLC0415

    free = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free) > 1:
        raise ValueError(f"{len(free)} free joints; a quadruped has one floating base")
    return free[0] if free else None


@dataclass(frozen=True)
class Bandwidth:
    """The filtered regressor: both sides of the torque balance pass the
    same zero-phase low-pass before the regression, so what the rigid
    model cannot hold — foot-impact shocks on the base, the IMU's
    high-frequency content, current-loop noise — falls out of both at
    once while the linear relation between them stays exact. Measured on
    DFKI's Go2 field bag (2026-09-24): unfiltered, the inertial term of
    the rigid torque was 3x the measured swing-leg effort and explained
    0-3 %. Each side is filtered exactly once (a pre-filtered position
    made the inertial term pass twice and biased armature 17 %). On the
    synthetic Go2 15 Hz leaves every term within 3 % of the truth
    (36/36); 6 Hz reads armature 9 % high, because the velocity gate
    selects rows after the filter smeared the slow ones into them. So
    15 Hz, and off when cutoff_hz is 0."""

    cutoff_hz: float = 15.0
    order: int = 2


def lowpass(values: np.ndarray, rate_hz: float, bandwidth: Bandwidth) -> np.ndarray:
    """Zero-phase low-pass at the declared cutoff; the values as they are
    when the cutoff is 0 (off). A cutoff at or above the Nyquist of
    `rate_hz` is refused by name: it used to return the values
    unfiltered while the record said they were filtered (review
    2026-09-24)."""
    from scipy.signal import butter, filtfilt  # noqa: PLC0415

    if bandwidth.cutoff_hz <= 0:
        return values
    if bandwidth.cutoff_hz >= rate_hz / 2:
        raise ValueError(
            f"the balance's {bandwidth.cutoff_hz:g} Hz band needs samples faster "
            f"than {2 * bandwidth.cutoff_hz:g} Hz; these run at {rate_hz:.1f} Hz "
            "(record faster, decimate less, or lower Bandwidth.cutoff_hz)"
        )
    b, a = butter(bandwidth.order, bandwidth.cutoff_hz / (rate_hz / 2))
    return np.asarray(filtfilt(b, a, values, axis=0))


def savgol_window(smoothing: Smoothing, rate_hz: float) -> int:
    """The odd Savitzky-Golay window `smoothing.window_s` holds at
    `rate_hz`; refused by name when it holds too few samples for the
    polynomial (one rule for every caller: one of them used to widen the
    window silently where the other refused)."""
    window = round(smoothing.window_s * rate_hz)
    window += 1 - window % 2  # odd
    if window <= smoothing.order + 1:
        raise ValueError(
            f"the smoothing window ({smoothing.window_s} s) holds {window} samples "
            f"at {rate_hz:.0f} Hz, too few for a cubic — record faster or widen it"
        )
    return window


@dataclass(frozen=True)
class Smoothing:
    """Savitzky-Golay smoothing of the logged velocity before it is
    differenced: `window_s` of samples, a cubic. Off when window_s is 0.
    Measured on the synthetic Go2 with a 2^14-count encoder (2026-09-24):
    unsmoothed, the quantisation noise reads armature 20 % low; 20 ms
    leaves 5 %; 40 ms recovers every term within 1 %."""

    window_s: float = 0.04
    order: int = 3


DEFAULT_SMOOTHING = Smoothing()
DEFAULT_BANDWIDTH = Bandwidth()


def tick_aligned(
    times: np.ndarray,
    position: np.ndarray,
    *,
    smoothing: Smoothing = DEFAULT_SMOOTHING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Position, velocity and acceleration aligned to the torque of each
    tick, from a log that stores the state at the START of tick i and
    holds torque i over the tick.

    Everything derives from the POSITION: the encoder is the precise
    channel, and a log's velocity is the firmware's own difference of it
    (on DFKI's Go2 it reaches 28 rad/s where the position says 6, and
    an acceleration taken from it reads three times the true inertial
    torque, which dilutes every regression coefficient toward zero,
    measured 2026-09-24). The velocity is a Savitzky-Golay derivative of
    the position over the smoothing window, and the acceleration is the
    velocity's change over the tick — a forward difference, because the torque acts over
    [t_i, t_i+1) — with the state at the tick's midpoint. Measured on
    the synthetic Go2 (2026-09-24, clean data, 250 Hz ticks): a centred
    difference at the tick reads armature 10 % high and damping 5 % low
    with a perfect residual; the forward difference with mid-tick
    states reads every term within 1 %. Returns (times, position,
    velocity, acceleration) with one sample fewer than the log.
    """
    from scipy.signal import savgol_filter  # noqa: PLC0415

    times = np.asarray(times, dtype=np.float64)
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("times must be strictly increasing")
    step = float(np.median(dt))
    rate = 1.0 / step
    clean = np.asarray(position, dtype=np.float64)
    window = savgol_window(smoothing, rate)
    velocity = savgol_filter(
        clean, window, smoothing.order, deriv=1, delta=step, axis=0
    )
    acceleration = np.diff(velocity, axis=0) / dt[:, None]
    mid_velocity = 0.5 * (velocity[1:] + velocity[:-1])
    mid_position = 0.5 * (clean[1:] + clean[:-1])
    return times[:-1], mid_position, mid_velocity, acceleration


@dataclass(frozen=True)
class JointSamples:
    """Row-aligned samples for one fit. `usable[i, j]` says whether sample
    i counts for joint j (a swing leg, a moving joint); None means all."""

    times: np.ndarray
    joints: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    torque: np.ndarray
    base_pose: np.ndarray | None = None  # (n, 7): xyz, wxyz — MuJoCo's free joint
    base_velocity: np.ndarray | None = None  # (n, 6): linear world, angular body
    base_acceleration: np.ndarray | None = None  # (n, 6): same frames
    usable: np.ndarray | None = None

    def __post_init__(self) -> None:
        n, nj = len(self.times), len(self.joints)
        for name in ("position", "velocity", "acceleration", "torque"):
            shape = getattr(self, name).shape
            if shape != (n, nj):
                raise ValueError(f"{name} has shape {shape}, expected {(n, nj)}")
        if self.usable is not None and self.usable.shape != (n, nj):
            raise ValueError(
                f"usable has shape {self.usable.shape}, expected {(n, nj)}"
            )

    @property
    def rate_hz(self) -> float:
        span = float(self.times[-1] - self.times[0])
        return (len(self.times) - 1) / span if span > 0 else 0.0


@dataclass(frozen=True)
class JointFit:
    """One joint's fit beside the numbers a reader needs to judge it."""

    joint: str
    estimates: dict[str, float]
    replicates: dict[str, np.ndarray]
    samples_used: int
    rms_before: float  # |tau_motor - tau_rigid| before the terms
    rms_after: float  # the residual the fitted terms leave
    explained: float  # 1 - after^2 / before^2


@dataclass(frozen=True)
class TorqueBalance:
    """Everything the fit computed: the verdicts and, per joint, the
    rigid and modelled torques for a viewer."""

    result: IdentificationResult
    joints: tuple[JointFit, ...]
    rigid_torque: np.ndarray  # (n, nj)
    modelled_torque: np.ndarray  # (n, nj): rigid + fitted terms
    extras: dict[str, Any] = field(default_factory=dict)


def _joint_addresses(
    model: Any, joints: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco  # noqa: PLC0415

    qpos, dof = [], []
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"the model has no joint named {name!r}")
        if model.jnt_type[jid] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"joint {name!r} is not a hinge or a slide")
        qpos.append(int(model.jnt_qposadr[jid]))
        dof.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos), np.asarray(dof)


def _free_base(model: Any) -> tuple[int, int] | None:
    """(qpos address, dof address) of the one free joint, or None."""
    import mujoco  # noqa: PLC0415

    free = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if not free:
        return None
    if len(free) > 1:
        raise ValueError(
            "more than one free joint; a legged fit expects one floating base"
        )
    return int(model.jnt_qposadr[free[0]]), int(model.jnt_dofadr[free[0]])


def rigid_torque(model: Any, samples: JointSamples) -> np.ndarray:
    """`tau_rigid` per sample and joint: inverse dynamics with the three
    passive terms zeroed and contacts disabled, on a copy of the model."""
    import copy  # noqa: PLC0415

    import mujoco  # noqa: PLC0415

    bare = copy.deepcopy(model)  # a copy: the caller's model keeps its terms
    bare.dof_armature[:] = 0.0
    bare.dof_damping[:] = 0.0
    bare.dof_frictionloss[:] = 0.0
    bare.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(bare)
    qpos_adr, dof_adr = _joint_addresses(bare, samples.joints)
    base = _free_base(bare)
    home = mujoco.MjData(bare).qpos.copy()
    n = len(samples.times)
    out = np.zeros((n, len(samples.joints)))
    for i in range(n):
        data.qpos[:] = home
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        if base is not None:
            qadr, dadr = base
            if samples.base_pose is not None:
                data.qpos[qadr : qadr + 7] = samples.base_pose[i]
            if samples.base_velocity is not None:
                data.qvel[dadr : dadr + 6] = samples.base_velocity[i]
            if samples.base_acceleration is not None:
                data.qacc[dadr : dadr + 6] = samples.base_acceleration[i]
        data.qpos[qpos_adr] = samples.position[i]
        data.qvel[dof_adr] = samples.velocity[i]
        data.qacc[dof_adr] = samples.acceleration[i]
        mujoco.mj_inverse(bare, data)
        out[i] = data.qfrc_inverse[dof_adr]
    return out


def _design(
    velocity: np.ndarray, acceleration: np.ndarray, coulomb: Coulomb
) -> np.ndarray:
    return np.column_stack([acceleration, velocity, coulomb.column(velocity)])


def _solve(design: np.ndarray, target: np.ndarray, bounds: TermBounds) -> np.ndarray:
    from scipy.optimize import lsq_linear  # noqa: PLC0415

    lower = [bounds.of(term)[0] for term in TERMS]
    upper = [bounds.of(term)[1] for term in TERMS]
    return lsq_linear(design, target, bounds=(lower, upper)).x


def _blocks(count: int, block: int) -> list[np.ndarray]:
    edges = list(range(0, count, max(1, block)))
    return [np.arange(start, min(start + block, count)) for start in edges]


def _bootstrap(
    design: np.ndarray,
    target: np.ndarray,
    bounds: TermBounds,
    plan: BootstrapPlan,
    block: int,
) -> np.ndarray:
    """Replicate estimates, shape (replicates, 3)."""
    rng = np.random.default_rng(plan.seed)
    blocks = _blocks(len(target), block)
    picks = np.empty((plan.replicates, len(TERMS)))
    for r in range(plan.replicates):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[c] for c in chosen])
        picks[r] = _solve(design[rows], target[rows], bounds)
    return picks


DEFAULT_BOUNDS = TermBounds()
DEFAULT_PLAN = BootstrapPlan()
DEFAULT_GATE = VelocityGate()
DEFAULT_COULOMB = Coulomb()


def fit_terms(  # noqa: PLR0913 - the fit's own knobs, each a named dataclass
    model: Any,
    samples: JointSamples,
    *,
    bounds: TermBounds = DEFAULT_BOUNDS,
    plan: BootstrapPlan = DEFAULT_PLAN,
    gate: VelocityGate = DEFAULT_GATE,
    bandwidth: Bandwidth = DEFAULT_BANDWIDTH,
    coulomb: Coulomb = DEFAULT_COULOMB,
    confidence: float = DEFAULT_CONFIDENCE,
    pinned_fraction: float = DEFAULT_PINNED_FRACTION,
) -> TorqueBalance:
    """Fit armature, damping and frictionloss per joint with bootstrap
    intervals and verdicts. Parameter names are `<joint>.<term>`."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    rigid = rigid_torque(model, samples)
    residual = lowpass(samples.torque - rigid, samples.rate_hz, bandwidth)
    block = max(2, round(plan.block_s * samples.rate_hz))
    low_q, high_q = 100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2
    parameters: list[IdentifiedParameter] = []
    fits: list[JointFit] = []
    modelled = rigid.copy()
    for j, joint in enumerate(samples.joints):
        moving = np.abs(samples.velocity[:, j]) >= gate.still
        usable = moving if samples.usable is None else moving & samples.usable[:, j]
        rows = np.flatnonzero(usable)
        if len(rows) < 2 * block:
            raise ValueError(
                f"joint {joint!r}: {len(rows)} usable samples, fewer than two "
                f"bootstrap blocks of {block} — record longer or loosen the gate"
            )
        columns = lowpass(
            _design(samples.velocity[:, j], samples.acceleration[:, j], coulomb),
            samples.rate_hz,
            bandwidth,
        )
        design = columns[rows]
        target = residual[rows, j]
        estimate = _solve(design, target, bounds)
        picks = _bootstrap(design, target, bounds, plan, block)
        after = target - design @ estimate
        rms_before = float(np.sqrt(np.mean(target**2)))
        rms_after = float(np.sqrt(np.mean(after**2)))
        modelled[:, j] = (
            rigid[:, j]
            + _design(samples.velocity[:, j], samples.acceleration[:, j], coulomb)
            @ estimate
        )
        replicates: dict[str, np.ndarray] = {}
        estimates: dict[str, float] = {}
        for k, term in enumerate(TERMS):
            lo, hi = bounds.of(term)
            allowed = hi - lo
            low, high = np.percentile(picks[:, k], [low_q, high_q])
            half = float(high - low) / 2.0
            value = float(estimate[k])
            replicates[term] = picks[:, k]
            estimates[term] = value
            at_bound = bool(
                value <= lo + AT_BOUND_FRACTION * allowed
                or value >= hi - AT_BOUND_FRACTION * allowed
            )
            # An estimate on its bound has a narrow bootstrap because the
            # box, not the data, holds every replicate there: never pinned.
            parameters.append(
                IdentifiedParameter(
                    name=f"{joint}.{term}",
                    estimate=value,
                    half_width=half,
                    allowed_range=allowed,
                    pinned=bool(half <= pinned_fraction * allowed and not at_bound),
                    at_bound=at_bound,
                )
            )
        fits.append(
            JointFit(
                joint=joint,
                estimates=estimates,
                replicates=replicates,
                samples_used=len(rows),
                rms_before=rms_before,
                rms_after=rms_after,
                explained=float(1.0 - (rms_after / rms_before) ** 2)
                if rms_before
                else 0.0,
            )
        )
    return TorqueBalance(
        result=IdentificationResult(
            parameters=tuple(parameters), confidence=confidence
        ),
        joints=tuple(fits),
        rigid_torque=rigid,
        modelled_torque=modelled,
        extras={
            "block_samples": block,
            "replicates": plan.replicates,
            "cutoff_hz": bandwidth.cutoff_hz,
            "rate_hz": float(samples.rate_hz),
            "coulomb_knee": coulomb.knee,
        },
    )


def units_for(joints: tuple[str, ...]) -> dict[str, str]:
    """The units entry every fit record needs, per `<joint>.<term>`."""
    return {f"{joint}.{term}": UNITS[term] for joint in joints for term in TERMS}
