"""The `servo-bench` identification method (#101): a hobby servo's actuator
model from a bench recording, as an actuator bundle like the library's.

The bench is a bundle of its own: a small MJCF, the servo's hinge driving
a pendulum arm of known mass and length (its inertia and its gravity
torque come from the model), and `bench.json` declaring the servo: its
family, the supply voltage, the P-gain register it ran at and the pwm
ceiling. The recording holds the commanded position (`joint.command`)
and the measured one (`joint.position`) of that joint: a ROS 2 MCAP's
`/joint_commands` and `/joint_states`, or any adapter's equivalent.
Steps and chirps at two or three arm masses are the protocol BAM's own
bench uses (Duclusaud et al., ICRA 2025).

The law is BAM's first tier (`m1`), the one trainnr_mjlab's actuator runs:
the firmware's duty `clip(G * (target - q), +-max_pwm)` drives a DC motor,
`tau = S * duty - B * qd` with `S = kt*vin/R` (the stall torque at full
duty, as trainnr_mjlab's kernel computes it) and the back-EMF
`B = kt^2/R`, against Coulomb friction, the armature and the arm. From
positions alone `B` and the gearbox's viscous friction are one term
(both multiply the velocity): the fit puts it all in the motor and says
so. The arm's known gravity torque anchors the torque scale and
its known inertia the armature; the command delay is fitted too.

The recording is cut in 2 s blocks, each simulated from its own measured
start (multiple shooting), so a block bootstrap gives each parameter an
interval and a pinned verdict; the last block is held out and its error
reported. The bundle lands in the bench's `fits/actuators/<servo>.bundle.json`,
where a walk names it as its actuator (`robots/<bench>/fits/actuators/...`).
One recording's interval is about that recording: the encoder's
quantization biases a run systematically, so two or more recordings (the
cross-run spread `identify_system` writes) are what earn trust.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from trainnr.bundles.bundle import model_file_for_use
from trainnr.bundles.hashing import FITS_DIR, stamp
from trainnr.robot.identify import IdentificationResult, IdentifiedParameter
from trainnr.robot.methods import method
from trainnr.robots.recording import JOINT_COMMAND, JOINT_POSITION, Recording

NAME = "servo-bench"
BENCH_FILE = "bench.json"
# Where the fitted actuator bundle lands in the bench's bundle: a record
# about it, outside its hash (`fits/*`), beside the fit records.
ACTUATORS_DIR = f"{FITS_DIR}/actuators"
BENCH_KEYS = frozenset({"servo", "joint", "vin", "kp", "max_pwm"})
SUBSTEP_S = 0.001  # the integrator's step: 2 ms loses the stick-slip
BLOCK_S = 2.0  # one shooting segment
MIN_BLOCKS = 4  # three to fit and bootstrap, one held out
MIN_SAMPLES, MIN_COMMANDS = 10, 2  # a block with fewer is dropped
REPLICATES = 16
GLOBAL_POPULATION, GLOBAL_GENERATIONS = 12, 60  # the global search's size
CONFIDENCE = 0.95
PINNED_FRACTION = 0.25  # an interval this narrow against its search range
STICK_VELOCITY = 1e-3  # rad/s: slower than this, static friction holds
MAX_SPEED, MAX_ANGLE = 1e3, 20.0  # rad/s, rad: a simulation past these diverged
# The parameters, their units and their search ranges (log-spaced except
# the delay): wide enough for hobby servos of a few hundred grams-cm to a
# few kilograms-cm.
PARAMETERS = (
    ("stall_torque", "N*m", 0.01, 10.0),
    ("duty_per_rad", "1/rad", 0.1, 500.0),
    ("back_emf", "N*m*s/rad", 1e-5, 5.0),
    ("friction_base", "N*m", 1e-5, 2.0),
    ("armature", "kg*m^2", 1e-7, 0.05),
    ("command_delay", "s", 0.0, 0.1),
)


@dataclass(frozen=True)
class Bench:
    """What `bench.json` declares about the servo on the bench."""

    servo: str
    joint: str
    vin: float
    kp: float
    max_pwm: float = 1.0

    @classmethod
    def load(cls, bundle_dir: Path) -> Bench:
        path = Path(bundle_dir) / BENCH_FILE
        raw = json.loads(path.read_text())
        unknown = sorted(set(raw) - BENCH_KEYS)
        if unknown:
            raise ValueError(f"{path}: no field {', '.join(unknown)}")
        missing = sorted({"servo", "joint", "vin", "kp"} - set(raw))
        if missing:
            raise ValueError(f"{path}: needs {', '.join(missing)}")
        return cls(
            servo=str(raw["servo"]),
            joint=str(raw["joint"]),
            vin=float(raw["vin"]),
            kp=float(raw["kp"]),
            max_pwm=float(raw.get("max_pwm", 1.0)),
        )


@dataclass(frozen=True)
class Arm:
    """The bench's rigid part, read from its model: the arm's inertia about
    the hinge (the model's own armature left out) and its gravity torque
    `a * sin(q) + b * cos(q)`, exact for one hinge."""

    inertia: float
    a: float
    b: float

    @classmethod
    def of(cls, model_file: Path, joint: str) -> Arm:
        import mujoco  # noqa: PLC0415 - sim extra

        model = mujoco.MjModel.from_xml_path(str(model_file))
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"the bench model has no hinge {joint!r}")
        if model.nv != 1:
            raise ValueError(
                f"the bench model has {model.nv} degrees of freedom; a servo bench "
                "has one, the servo's hinge"
            )
        dof = model.jnt_dofadr[j]
        model.dof_armature[dof] = 0.0
        data = mujoco.MjData(model)

        def bias(q: float) -> float:
            data.qpos[model.jnt_qposadr[j]] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            return float(data.qfrc_bias[dof])

        mass = np.zeros((1, 1))
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        try:  # MuJoCo 3.3 takes the data itself (the sparse store is gone)
            mujoco.mj_fullM(model, data, mass)
        except TypeError:
            mujoco.mj_fullM(model, mass, data.qM)
        # qfrc_bias is the torque gravity asks of the joint: its negative
        # is the torque gravity applies
        return cls(inertia=float(mass[0, 0]), a=-bias(math.pi / 2), b=-bias(0.0))

    def gravity(self, q: float) -> float:
        return self.a * math.sin(q) + self.b * math.cos(q)


@dataclass(frozen=True)
class Block:
    """One shooting segment: the measured positions at their times, the
    command on the same clock, and the measured start."""

    times: np.ndarray
    measured: np.ndarray
    command_times: np.ndarray
    command: np.ndarray
    start_velocity: float = 0.0


def blocks_of(recording: Recording, joint: str) -> list[Block]:
    """The recording's joint cut in BLOCK_S segments."""
    position = recording.channels[JOINT_POSITION]
    command = recording.channels[JOINT_COMMAND]
    pc = list(position.components).index(joint)
    cc = list(command.components).index(joint)
    t, q = position.times, position.values[:, pc]
    ct, cq = command.times, command.values[:, cc]
    out = []
    start = float(t[0])
    while start + BLOCK_S <= float(t[-1]) + 1e-9:
        keep = (t >= start) & (t < start + BLOCK_S)
        ckeep = (ct >= start - 0.2) & (ct < start + BLOCK_S)
        if keep.sum() >= MIN_SAMPLES and ckeep.sum() >= MIN_COMMANDS:
            first = int(np.flatnonzero(keep)[0])
            # the velocity at the block's start: the central difference
            # around it; the recording's first sample is at rest (a bench
            # recording starts before the first command moves the arm)
            lo, hi = first - 1, min(first + 1, len(t) - 1)
            velocity = float((q[hi] - q[lo]) / (t[hi] - t[lo])) if first > 0 else 0.0
            out.append(
                Block(t[keep] - start, q[keep], ct[ckeep] - start, cq[ckeep], velocity)
            )
        start += BLOCK_S
    return out


def simulate_many(arm: Arm, bench: Bench, xs: np.ndarray, block: Block) -> np.ndarray:
    """The block's positions under each row of parameters `xs` (P, 6) in
    PARAMETERS' order, all at once: (P, samples). From the block's
    measured start; semi-implicit Euler at SUBSTEP_S, Coulomb friction
    holding below STICK_VELOCITY while the drive does not overcome it."""
    xs = np.atleast_2d(xs)
    stall, gain, emf, coulomb, armature, delay = (xs[:, k] for k in range(6))
    inertia = arm.inertia + armature
    steps = math.ceil(float(block.times[-1]) / SUBSTEP_S) + 1
    clock = np.arange(steps) * SUBSTEP_S
    # zero-order hold on the command, each candidate's own delay late
    held = (
        np.searchsorted(
            block.command_times, clock[None, :] - delay[:, None], side="right"
        )
        - 1
    )
    target = block.command[np.clip(held, 0, len(block.command) - 1)]
    q: np.ndarray = np.full(len(xs), float(block.measured[0]))
    qd: np.ndarray = np.full(len(xs), block.start_velocity)
    out = np.empty((len(xs), steps))
    top = bench.max_pwm
    for i in range(steps):
        out[:, i] = q
        duty = np.clip(gain * (target[:, i] - q), -top, top)
        drive = stall * duty - emf * qd + arm.a * np.sin(q) + arm.b * np.cos(q)
        holds = np.abs(drive) <= coulomb
        sign = np.sign(np.where(qd != 0.0, qd, drive))
        moved = qd + (drive - coulomb * sign) / inertia * SUBSTEP_S
        # static friction: a velocity friction would carry through zero
        # stops there when the drive cannot overcome it (else it jitters
        # around zero at the step's size and never sticks)
        stuck = holds & (
            (np.abs(qd) < STICK_VELOCITY) | (np.sign(moved) != np.sign(qd))
        )
        qd = np.where(stuck, 0.0, moved)
        # a candidate too stiff for the step diverges: bounded here, and
        # far from any measurement, so the search leaves it
        qd = np.clip(qd, -MAX_SPEED, MAX_SPEED)
        q = np.clip(q + qd * SUBSTEP_S, -MAX_ANGLE, MAX_ANGLE)
    return np.stack([np.interp(block.times, clock, row) for row in out])


def simulate(arm: Arm, bench: Bench, x: np.ndarray, block: Block) -> np.ndarray:
    """The block's positions under one parameter vector."""
    return simulate_many(arm, bench, np.asarray(x)[None, :], block)[0]


def _residuals(
    arm: Arm, bench: Bench, blocks: list[Block], z: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [simulate(arm, bench, _from_unit(z), b) - b.measured for b in blocks]
    )


# The optimiser works on a unit cube: log-spaced for the scales, linear
# for the delay; the bounds are PARAMETERS'.
def _to_unit(x: np.ndarray) -> np.ndarray:
    out = []
    for v, (name, _, lo, hi) in zip(x, PARAMETERS, strict=True):
        if name == "command_delay":
            out.append((v - lo) / (hi - lo))
        else:
            out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
    return np.clip(np.array(out), 0.0, 1.0)


def _from_unit(z: np.ndarray) -> np.ndarray:
    out = []
    for u, (name, _, lo, hi) in zip(z, PARAMETERS, strict=True):
        if name == "command_delay":
            out.append(lo + u * (hi - lo))
        else:
            out.append(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
    return np.array(out)


def fit_blocks(
    arm: Arm, bench: Bench, blocks: list[Block], start: np.ndarray | None = None
) -> np.ndarray:
    """The parameters for these blocks (PARAMETERS' order): with no `start`,
    the global search (differential evolution over the whole population at
    once: the servo's response has more than one valley), which `polish`
    then refines; with a `start` (a bootstrap replicate), least squares
    from it on the blocks' fitted starts."""
    from scipy.optimize import (  # noqa: PLC0415 - mujoco[sysid] brings it
        differential_evolution,
        least_squares,
    )

    if start is None:

        def cost(zs: np.ndarray) -> np.ndarray:  # zs: (6, P), scipy's vectorised form
            xs = np.array([_from_unit(z) for z in zs.T])
            total = np.zeros(len(xs))
            for b in blocks:
                total += ((simulate_many(arm, bench, xs, b) - b.measured) ** 2).sum(
                    axis=1
                )
            return total

        found = differential_evolution(
            cost,
            [(0.0, 1.0)] * len(PARAMETERS),
            vectorized=True,
            updating="deferred",
            popsize=GLOBAL_POPULATION,
            maxiter=GLOBAL_GENERATIONS,
            tol=1e-8,
            seed=0,
            polish=False,
        )
        return _from_unit(found.x)  # fit_bench polishes it, starts and all
    # a bootstrap replicate: the servo's parameters alone, from the estimate,
    # the blocks' fitted starts kept
    again = least_squares(
        lambda z: _residuals(arm, bench, blocks, z),
        _to_unit(start),
        bounds=(0.0, 1.0),
        diff_step=1e-4,
        max_nfev=100,
    )
    return _from_unit(again.x)


def polish(
    arm: Arm, bench: Bench, blocks: list[Block], x0: np.ndarray
) -> tuple[np.ndarray, list[Block]]:
    """Least squares from `x0` with each block's starting velocity fitted
    too (a nuisance: two quantized samples estimate it poorly, and a wrong
    start biases the servo's parameters). Returns the parameters and the
    blocks with their fitted starts."""
    from scipy.optimize import least_squares  # noqa: PLC0415 - mujoco[sysid] brings it

    n = len(PARAMETERS)

    def unpack(w: np.ndarray) -> tuple[np.ndarray, list[Block]]:
        moved = [
            replace(b, start_velocity=float(v))
            for b, v in zip(blocks, w[n:], strict=True)
        ]
        return _from_unit(w[:n]), moved

    def residuals(w: np.ndarray) -> np.ndarray:
        x, moved = unpack(w)
        return np.concatenate([simulate(arm, bench, x, b) - b.measured for b in moved])

    w0 = np.concatenate([_to_unit(x0), [b.start_velocity for b in blocks]])
    lo = np.concatenate([np.zeros(n), np.full(len(blocks), -MAX_SPEED)])
    hi = np.concatenate([np.ones(n), np.full(len(blocks), MAX_SPEED)])
    found = least_squares(residuals, w0, bounds=(lo, hi), diff_step=1e-4, max_nfev=300)
    return unpack(found.x)


# The finite-difference steps of the Jacobian: a fraction of each scale,
# and two integrator steps of delay (finer would see no change at all).
JACOBIAN_FRACTION = 0.02
JACOBIAN_DELAY_S = 2 * SUBSTEP_S
Z_95 = 1.959964
PAIR = 2  # samples a correlation needs


def asymptotic_half_widths(
    arm: Arm, bench: Bench, x: np.ndarray, blocks: list[Block]
) -> np.ndarray:
    """Each parameter's 95 % half-width from the fit's Jacobian, sigma^2 *
    (J^T J)^-1 with sigma^2 the residual variance; infinite for one the
    data cannot move."""
    base = np.concatenate([simulate(arm, bench, x, b) - b.measured for b in blocks])
    columns = []
    # the blocks' starting velocities are fitted too: their columns stay in
    # the information matrix, so the parameters' intervals carry them
    for k, block in enumerate(blocks):
        bumped = list(blocks)
        bumped[k] = replace(block, start_velocity=block.start_velocity + 0.05)
        shifted = np.concatenate(
            [simulate(arm, bench, x, b) - b.measured for b in bumped]
        )
        columns.append((shifted - base) / 0.05)
    for k, (name, _, _, _) in enumerate(PARAMETERS):
        step = JACOBIAN_DELAY_S if name == "command_delay" else JACOBIAN_FRACTION * x[k]
        moved = x.copy()
        moved[k] += step
        shifted = np.concatenate(
            [simulate(arm, bench, moved, b) - b.measured for b in blocks]
        )
        columns.append((shifted - base) / step)
    jac = np.stack(columns, axis=1)
    dof = max(len(base) - len(x), 1)
    # the residuals of a time series are not independent: the variance is
    # inflated by their lag-1 autocorrelation (n / n_effective)
    rho = float(np.corrcoef(base[:-1], base[1:])[0, 1]) if len(base) > PAIR else 0.0
    rho = min(max(rho, 0.0), 0.99)
    sigma2 = float(base @ base) / dof * (1 + rho) / (1 - rho)
    info = jac.T @ jac
    every = np.full(info.shape[0], np.inf)
    sensitive = np.diag(info) > 0
    if sensitive.any():
        sub = info[np.ix_(sensitive, sensitive)]
        cov = sigma2 * np.linalg.pinv(sub)
        every[sensitive] = Z_95 * np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return every[len(blocks) :]  # the parameters' (the velocity columns come first)


def rmse(arm: Arm, bench: Bench, x: np.ndarray, blocks: list[Block]) -> float:
    errors = np.concatenate([simulate(arm, bench, x, b) - b.measured for b in blocks])
    return float(np.sqrt(np.mean(errors**2)))


@dataclass(frozen=True)
class BenchFit:
    """The point fit, its bootstrap replicates, and its errors."""

    estimate: np.ndarray
    replicates: np.ndarray
    asymptotic: (
        np.ndarray
    )  # each parameter's interval half-width from the fit's Jacobian
    rmse_rad: float
    held_out_rmse_rad: float
    blocks: int

    def result(self) -> IdentificationResult:
        low_pct, high_pct = 100 * (1 - CONFIDENCE) / 2, 100 * (1 + CONFIDENCE) / 2
        parameters = []
        for k, (name, _, lo, hi) in enumerate(PARAMETERS):
            low, high = np.percentile(self.replicates[:, k], [low_pct, high_pct])
            # the wider of the block bootstrap and the asymptotic interval:
            # a replicate started at the estimate can stay there where the
            # cost is flat or stepped (the delay moves in substeps)
            spread = max(high - self.estimate[k], self.estimate[k] - low, 0.0)
            half = float(max(spread, self.asymptotic[k]))
            # a scale is judged on its own size, not its search decade
            allowed = hi - lo if name == "command_delay" else float(self.estimate[k])
            parameters.append(
                IdentifiedParameter(
                    name=name,
                    estimate=float(self.estimate[k]),
                    half_width=half,
                    allowed_range=float(allowed),
                    pinned=half <= PINNED_FRACTION * allowed,
                    at_bound=bool(
                        np.isclose(self.estimate[k], [lo, hi], rtol=1e-3).any()
                    ),
                )
            )
        return IdentificationResult(parameters=tuple(parameters), confidence=CONFIDENCE)


def fit_bench(
    arm: Arm,
    bench: Bench,
    blocks: list[Block],
    *,
    replicates: int = REPLICATES,
    seed: int = 0,
) -> BenchFit:
    """The point fit on every block but the last, a block bootstrap around
    it, and the error on the last block, held out."""
    if len(blocks) < MIN_BLOCKS:
        raise ValueError(
            f"the recording holds {len(blocks)} blocks of {BLOCK_S:g} s; the fit needs "
            f"{MIN_BLOCKS} ({MIN_BLOCKS * BLOCK_S:g} s or more of steps and chirps)"
        )
    train, held_out = blocks[:-1], blocks[-1:]
    estimate = fit_blocks(arm, bench, train)
    estimate, train = polish(arm, bench, train, estimate)
    rng = np.random.default_rng(seed)
    draws = [
        fit_blocks(
            arm,
            bench,
            [train[i] for i in rng.integers(0, len(train), len(train))],
            estimate,
        )
        for _ in range(replicates)
    ]
    return BenchFit(
        estimate=estimate,
        replicates=np.array(draws),
        asymptotic=asymptotic_half_widths(arm, bench, estimate, train),
        rmse_rad=rmse(arm, bench, estimate, train),
        held_out_rmse_rad=rmse(arm, bench, estimate, held_out),
        blocks=len(blocks),
    )


def actuator_bundle(
    bench: Bench, fit: BenchFit, *, bench_stamp: str, recording: str
) -> dict[str, Any]:
    """The fit as an actuator bundle in the library's format (BAM's m1
    params, the firmware it ran under in `context`, the fit's errors and
    intervals), stamped by its content."""
    from trainnr.robot.actuator_bundle import (  # noqa: PLC0415
        SCHEMA,
        _stamp,
        run_checks,
    )

    x = dict(
        zip([p[0] for p in PARAMETERS], (float(v) for v in fit.estimate), strict=True)
    )
    # S = kt*vin/R and B = kt^2/R give kt and R back
    kt = x["back_emf"] * bench.vin / x["stall_torque"]
    resistance = kt * bench.vin / x["stall_torque"]
    result = fit.result()
    bundle: dict[str, Any] = {
        "schema": SCHEMA,
        "params": {
            "model": "m1",
            "actuator": bench.servo,
            "kt": kt,
            "R": resistance,
            "armature": x["armature"],
            "friction_base": x["friction_base"],
            "friction_viscous": 0.0,
            "command_delay": x["command_delay"],
        },
        "provenance": {
            "source": "trainnr servo-bench",
            "method": NAME,
            "bench": bench_stamp,
            "recording": recording,
            "law": "BAM m1 (Duclusaud et al., ICRA 2025): duty = clip(G*(target-q), "
            "+-max_pwm), tau = S*duty - B*qd, Coulomb friction, armature",
            "anchor": "the arm's gravity torque and inertia, from the bench model; "
            "back-EMF and viscous friction are one term from positions, all in kt^2/R",
        },
        "context": {
            "firmware": {
                "vin": bench.vin,
                "kp": bench.kp,
                "error_gain": x["duty_per_rad"] / bench.kp,
                "max_pwm": bench.max_pwm,
                "source": "fitted on the bench (duty per radian of error over kp)",
            }
        },
        "metrics": {
            "position_rmse_rad": round(fit.rmse_rad, 6),
            "held_out_position_rmse_rad": round(fit.held_out_rmse_rad, 6),
            "blocks": fit.blocks,
        },
        "uncertainty": _bundle_intervals(result, bench.vin),
    }
    bundle["checks"] = run_checks(bundle["params"])  # derived, outside the stamp
    bundle["stamp"] = _stamp(bundle)
    return bundle


def _bundle_intervals(result: IdentificationResult, vin: float) -> dict[str, Any]:
    """The fit's intervals on the bundle's own parameters (what a walk's
    `identified` randomization draws from): kt and R bounded from the stall
    torque's and the back-EMF's intervals, conservatively (kt = B*vin/S,
    R = kt*vin/S); the armature, friction and delay as fitted."""
    by = {p.name: p for p in result.parameters}
    s, b = by["stall_torque"], by["back_emf"]
    s_lo, s_hi = max(s.lower, 1e-12), s.upper
    b_lo, b_hi = max(b.lower, 0.0), b.upper
    kt_lo, kt_hi = b_lo * vin / s_hi, b_hi * vin / s_lo
    out: dict[str, Any] = {
        "kt": {"low": kt_lo, "high": kt_hi},
        "R": {"low": kt_lo * vin / s_hi, "high": kt_hi * vin / s_lo},
    }
    for name in ("armature", "friction_base", "command_delay"):
        p = by[name]
        out[name] = {"low": max(p.lower, 0.0), "high": p.upper}
    return out


@method(
    NAME,
    doc="One servo on a bench (bench.json beside a one-hinge pendulum model): "
    "BAM's m1 law from commanded and measured position, as an actuator bundle.",
)
class ServoBench:
    name = NAME
    # torque scale and inertia come from the arm's model, not the data
    anchored = ("arm_gravity", "arm_inertia")

    def accepts(self, bundle_dir: Path, recording_dir: Path) -> str | None:
        bundle_dir = Path(bundle_dir)
        if not (bundle_dir / BENCH_FILE).is_file():
            return f"the bundle declares no servo bench ({BENCH_FILE})"
        try:
            bench = Bench.load(bundle_dir)
            recording = Recording.read(Path(recording_dir))
        except (OSError, ValueError, KeyError) as why:
            return str(why)
        for channel in (JOINT_POSITION, JOINT_COMMAND):
            found = recording.channels.get(channel)
            if found is None:
                return f"the recording has no {channel} channel"
            if bench.joint not in found.components:
                return f"the recording's {channel} has no joint {bench.joint!r}"
        return None

    def fit(
        self, bundle_dir: Path, recording_dir: Path, *, write: bool = True
    ) -> tuple[IdentificationResult, Path | None]:
        from trainnr.robot.fit_record import write_fit_record  # noqa: PLC0415

        bundle_dir, recording_dir = Path(bundle_dir), Path(recording_dir)
        bench = Bench.load(bundle_dir)
        model_file = model_file_for_use(bundle_dir)
        if model_file is None:
            raise FileNotFoundError(f"{bundle_dir}: no bench model")
        arm = Arm.of(model_file, bench.joint)
        recording = Recording.read(recording_dir)
        fit = fit_bench(arm, bench, blocks_of(recording, bench.joint))
        result = fit.result()
        if not write:
            return result, None
        bench_stamp = stamp(bundle_dir.name, bundle_dir)
        recorded = stamp(recording_dir.name, recording_dir)
        bundle = actuator_bundle(
            bench, fit, bench_stamp=bench_stamp, recording=recorded
        )
        # beside the fit records, not among them (`fits/*.json` are records)
        out = bundle_dir / ACTUATORS_DIR / f"{bench.servo}.bundle.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bundle, indent=1) + "\n")
        record = write_fit_record(
            bundle_dir,
            result,
            robot=bench_stamp,
            recording=recorded,
            anchor=ServoBench.anchor_text,
            units={p[0]: p[1] for p in PARAMETERS},
            basis=recording.basis,
            metrics={
                "position_rmse_rad": fit.rmse_rad,
                "held_out_position_rmse_rad": fit.held_out_rmse_rad,
                "actuator_bundle": f"{ACTUATORS_DIR}/{bench.servo}.bundle.json",
                "actuator_stamp": bundle["stamp"],
            },
        )
        return result, record

    anchor_text = (
        "the torque scale from the arm's gravity torque and the armature from the "
        "arm's inertia, both read off the bench model; back-EMF and viscous friction "
        "are one term from positions, fitted as the motor's kt^2/R"
    )
