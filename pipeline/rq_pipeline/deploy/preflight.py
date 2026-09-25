"""Pre-flight safety (docs/77 §10): what stands between an exported policy
and the first tick on a robot. Three parts, each in the runtime so the
plain MuJoCo runtime and Unitree's stack over DDS both get them.

1. **Checks before the first tick**, each a refusal by name with its
   number: the policy's input and output widths against the manifest
   (unitree_rl_mjlab #25: a 160-wide observation fed to a 154-wide
   policy, found only on the robot); the joint names and order against
   the scene; the gains the manifest trained under against the scene's
   actuators and against the YAML Unitree's controller will read; the
   targets and the torques a dry rollout of the gate's own held twists
   commands, against the joints' ranges and the actuators' force ranges
   with a declared margin; the compute per control tick against the
   period; the robot's reported state (upright, at rest, temperatures,
   battery) before handover.
2. **A staged ramp-in**: from damping, the commanded target is blended
   from the measured pose to the policy's over a declared window, so the
   first tick never slams (unitree_rl_lab #147, mjlab #729).
3. **A soft stop**: on any watchdog (the SDK's own thresholds, a table
   below with its source) or an operator stop, the gains are blended to
   damping over a declared window rather than zeroed.

The record `preflight.json` lands beside the manifest: every check with
its measured number and its limit, the ramp and the stop measured, and
the basis of the state it read (a simulation, or a simulation stand-in
for the robot when the state came from Unitree's simulator over DDS).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rq_pipeline.bundles.basis import BASIS_OWN, BASIS_SIMULATION
from rq_pipeline.deploy import stills
from rq_pipeline.deploy.gate import (
    DEFAULT_SEED,
    DRAW_NOW,
    hold_twists,
)
from rq_pipeline.deploy.manifest import Key, Manifest, load_manifest, read_gates
from rq_pipeline.deploy.poses import PoseTrack, poses_file
from rq_pipeline.deploy.runtime import GRAVITY_DOWN, rotate_inverse
from rq_pipeline.deploy.runtimes import DEFAULT_RUNTIME

if TYPE_CHECKING:
    from rq_pipeline.deploy.runtime import Runtime

PREFLIGHT_FILE = "preflight.json"
PREFLIGHT_SCHEMA = "trainnr-preflight/1"
# The Deployments card's key for the verdict; the Studio shows it first
# after the gate's word (`model.rs::PREFLIGHT_KEY`, pinned by test).
PREFLIGHT_SUMMARY_KEY = "pre-flight"
# The operator's stop, from any process: a file beside the manifest the
# guarded loop reads every tick (`stop_deployment` writes it).
STOP_FILE = "STOP"
STILL_FILE = "preflight-handover.png"
STREAM = "preflight"
# The replay's segments in the pose file beside the stream (`deploy.poses`):
# the ramp each way, each stop, and theirs; the Studio's viewport lists them.
RAMP_SEGMENT = "ramp in · {variant}"
STOP_SEGMENT = "stop · {variant}"
THEIR_STOP_SEGMENT = "stop · their Passive"
POSES_KEY = "poses"  # the record's pointer to the pose file, relative to the folder
POSES_TRACK = "_poses"  # the in-memory track, written beside the record, never in it
DIGITS = 4


# -- declared numbers --------------------------------------------------------------


@dataclass(frozen=True)
class Margins:
    """The declared numbers a pre-flight judges by; each is a stated
    choice, recorded in every record, never a hidden one."""

    # a commanded target may sit this far past a joint's range (a PD target
    # past the stop is how a policy presses a foot down; far past it is a
    # scale or an offset copied wrong)
    target_past_range_rad: float = 0.35
    # the torque a PD target demands may reach this multiple of the force
    # range before it counts as a demand the motor cannot meet
    torque_demand_ratio: float = 2.0
    # and may sit at the force range this share of the ticks at most
    saturated_share: float = 0.25
    # gains agree when within this relative difference
    gain_rel_tol: float = 1e-6
    # compute per control tick, at the 99th percentile, within this share
    # of the period (the rest is the bus and the motors)
    compute_share: float = 0.5
    # the robot is at rest before handover: every joint slower than this
    at_rest_rad_s: float = 0.5
    # the dry rollout: the gate's own held twists, the first this many
    dry_trials: int = 4


MARGINS = Margins()
# Where the margins come from: ours, declared, each reasoned above; none is
# a vendor's number (the watchdogs below are Unitree's, cited).
MARGINS_SOURCE = "declared (ours): pipeline/rq_pipeline/deploy/preflight.py Margins"


@dataclass(frozen=True)
class Transitions:
    """The declared windows of the ramp-in and the soft stop. The manifest
    may carry its own (`control.ramp_in_s`, `control.soft_stop_s`); these
    are the defaults."""

    ramp_in_s: float = 1.0
    soft_stop_s: float = 1.0
    # the damping a soft stop ends in: Unitree's Passive state's kd for the
    # Go2 (their `config.yaml`, Passive: kd 3 on every joint)
    damping_kd: float = 3.0


TRANSITIONS = Transitions()
TRANSITIONS_SOURCE = (
    "windows declared (ours); damping_kd is Unitree's Passive state's kd "
    "(unitree_rl_mjlab deploy/robots/go2/config/config.yaml, Passive)"
)
RAMP_KEY = "ramp_in_s"
STOP_KEY = "soft_stop_s"
DAMPING_KEY = "damping_kd"


def transitions_of(manifest: Manifest) -> Transitions:
    """The manifest's own windows and damping (`control.ramp_in_s`,
    `control.soft_stop_s`, `control.damping_kd`), else the defaults."""
    control = manifest.raw.get(Key.CONTROL) or {}
    return Transitions(
        ramp_in_s=float(control.get(RAMP_KEY, TRANSITIONS.ramp_in_s)),
        soft_stop_s=float(control.get(STOP_KEY, TRANSITIONS.soft_stop_s)),
        damping_kd=float(control.get(DAMPING_KEY, TRANSITIONS.damping_kd)),
    )


# -- the watchdogs: Unitree's own thresholds ------------------------------------

# The numbers are the defaults of the SDK's protect functions for its
# humanoids (the `unitree_hg` robots, H2 and G1); the Go2 SDK ships none, so
# they are applied to the Go2 as the vendor's nearest stated limits, and
# its one temperature per motor is read against the casing limit.
SDK_COMMIT = "9754cd1"
SDK_SOURCE = (
    f"unitree_sdk2 @ {SDK_COMMIT}: include/unitree/robot/h2/common/terminations.hpp "
    "(the same in g1/), the H2/G1 defaults, read 2026-09-24; shipped as examples "
    "that print, and no deploy repo wires them"
)
# The link: the SDK measures `now - GetLastDataAvailableTime()`; their
# Python SDK exposes no last-data time, so the link is enforced by the read
# timeout itself (`dds_runtime.STATE_TIMEOUT_MS`, the same 1000 ms): a
# silent bus raises, it is never measured as an age.
LINK_BY_TIMEOUT = "enforced by the read timeout (dds_runtime.STATE_TIMEOUT_MS)"


@dataclass(frozen=True)
class Watchdog:
    """One protective limit a running deployment is stopped soft on.
    `reading` names the `Health` field it judges; `trips_below` for a
    limit a value must stay above (the battery); `enforced_by` when it is
    not read off the state at all."""

    name: str
    limit: float
    unit: str
    sdk_function: str
    describe: str
    reading: str | None
    trips_below: bool = False
    enforced_by: str | None = None


WATCHDOGS: dict[str, Watchdog] = {
    w.name: w
    for w in (
        Watchdog(
            "tilt",
            1.0,
            "rad",
            "bad_orientation",
            "the body's down from gravity",
            "tilt_rad",
        ),
        Watchdog(
            "joint_velocity",
            10.0,
            "rad/s",
            "joint_vel_out_of_limit",
            "any joint",
            "max_joint_velocity",
        ),
        Watchdog(
            "angular_velocity",
            6.0,
            "rad/s",
            "ang_vel_out_of_limit",
            "the gyro",
            "max_angular_velocity",
        ),
        Watchdog(
            "winding_temperature",
            120.0,
            "degC",
            "motor_winding_overheat",
            "any motor's winding",
            "max_winding_temperature",
        ),
        Watchdog(
            "casing_temperature",
            85.0,
            "degC",
            "motor_casing_overheat",
            "any motor",
            "max_casing_temperature",
        ),
        Watchdog(
            "battery",
            20.0,
            "%",
            "low_battery",
            "state of charge (below trips)",
            "battery_percent",
            trips_below=True,
        ),
        Watchdog(
            "link_lost",
            1000.0,
            "ms",
            "lost_connection",
            "no LowState for",
            None,
            enforced_by=LINK_BY_TIMEOUT,
        ),
    )
}
OPERATOR_STOP = "operator stop"


@dataclass(frozen=True)
class Health:
    """What the robot reports, as far as its runtime can see. None: the
    runtime does not report it (plain MuJoCo has no battery; Unitree's
    simulator publishes no temperatures)."""

    tilt_rad: float
    max_joint_velocity: float
    max_angular_velocity: float
    max_winding_temperature: float | None = None
    max_casing_temperature: float | None = None
    battery_percent: float | None = None

    def value(self, watchdog: str) -> float | None:
        reading = WATCHDOGS[watchdog].reading
        return None if reading is None else getattr(self, reading)


def tripped(health: Health) -> list[str]:
    """The watchdogs this reading trips, by name."""
    out: list[str] = []
    for w in WATCHDOGS.values():
        v = health.value(w.name)
        if v is None:
            continue
        if (v < w.limit) if w.trips_below else (v > w.limit):
            out.append(w.name)
    return out


def tilt_of(quat_wxyz: np.ndarray) -> float:
    """The SDK's `bad_orientation` angle: acos(-g_b·z)."""
    gravity_b = rotate_inverse(np.asarray(quat_wxyz, dtype=np.float64), GRAVITY_DOWN)
    return float(np.arccos(np.clip(-gravity_b[2], -1.0, 1.0)))


def health_of_runtime(runtime: Runtime) -> Health:
    """The plain MuJoCo runtime's robot, read the SDK's way; no
    temperatures, no battery, no link (it is a library)."""
    return Health(
        tilt_rad=tilt_of(runtime.quat),
        max_joint_velocity=float(np.max(np.abs(runtime.data.qvel[runtime.joint_qvel]))),
        max_angular_velocity=float(np.max(np.abs(runtime.base_angular_velocity()))),
    )


def health_of_lowstate(reading: dict[str, Any], *, stand_in: bool) -> Health:
    """A DDS runtime's reading (`DdsRuntime.health`) read the SDK's way.
    On a stand-in (their simulator), temperatures and battery that read
    exactly zero are what the simulator does not publish: unreported, not
    a cold empty robot; on a robot they are read as they are."""
    temps = np.asarray(reading.get("motor_temperature", []), dtype=float)
    soc = float(reading.get("battery_percent", 0.0))
    silent_temps = stand_in and (temps.size == 0 or not np.any(temps))
    return Health(
        tilt_rad=tilt_of(np.asarray(reading["quaternion"], dtype=float)),
        max_joint_velocity=float(np.max(np.abs(reading["motor_speed"]))),
        max_angular_velocity=float(np.max(np.abs(reading["gyroscope"]))),
        max_casing_temperature=None if silent_temps else float(np.max(temps)),
        battery_percent=None if (stand_in and soc == 0.0) else soc,
    )


DDS_WALK_S = 2.0
DDS_WATCH_S = 2.0
DDS_POLL_MS = 1000
WALK_END = "the walk's end"
# The walk a stop is measured from: forward at this speed, clipped into
# the manifest's forward range (`walk_command`); one number, spelled once.
WALK_SPEED_MPS = 0.5
STALE_STOP = (
    "{file} is already there (an earlier stop, {reason!r}): a run that "
    "starts on it stops at its first tick; read it, delete it, then run again"
)


def walk_command(manifest: Manifest) -> np.ndarray:
    """The held command a stop is measured from: WALK_SPEED_MPS forward,
    inside the manifest's own forward range."""
    lo, hi = manifest.commands.lin_vel_x
    return np.array([float(np.clip(WALK_SPEED_MPS, lo, hi)), 0.0, 0.0], np.float32)


def refuse_stale_stop(deployment_dir: Path) -> None:
    """A STOP file left by an earlier `stop_deployment` would end the next
    run at its first tick and read as a stop: refused by name."""
    path = Path(deployment_dir) / STOP_FILE
    if path.is_file():
        try:
            reason = json.loads(path.read_text(encoding="utf-8")).get("reason", "")
        except ValueError:
            reason = "unreadable"
        raise ValueError(STALE_STOP.format(file=path, reason=reason))


def measure_dds_stop(
    runtime: Any,
    command: np.ndarray,
    *,
    stand_in: bool,
    stop_file: Path | None = None,
    poses: PoseTrack | None = None,
) -> dict[str, Any]:
    """Unitree's own stop through their controller: the policy walking at
    a held command, then their Passive chord (the pad is the hook);
    watched from the bus. Their controller owns the gains, so this is
    their damping at once, measured beside ours. An operator's STOP file
    (`request_stop`) ends the walk early, and so do the SDK's watchdogs
    read off their state; the record says how many ticks it walked. The
    Passive chord is sent whatever happens after the handover (a silent
    bus, a health that cannot be read): the robot is never left in their
    velocity mode. With `poses`, the walk and the watch are kept at the
    control rate for the Studio's replay."""
    runtime.command = command.astype(np.float32)
    if poses is not None:
        poses.begin(THEIR_STOP_SEGMENT)
    reason = WALK_END
    planned = round(DDS_WALK_S / runtime.step_dt)
    walked = 0
    try:
        for _ in range(planned):
            runtime.observe()
            if stop_file is not None and stop_file.is_file():
                reason = OPERATOR_STOP
                break
            trips = tripped(health_of_lowstate(runtime.health(), stand_in=stand_in))
            if trips:
                reason = trips[0]
                break
            runtime.apply(np.zeros(0, dtype=np.float32))
            walked += 1
            if poses is not None:
                poses.add(runtime.pose())
    finally:
        runtime.stop()
    began = time.monotonic()
    kept = began  # the watch reads the bus as fast as it comes; poses at the tick
    fall, spin, height = 0.0, 0.0, float("nan")
    while time.monotonic() - began < DDS_WATCH_S:
        _quat, velocity_w = runtime.bus.latest(DDS_POLL_MS)
        if poses is not None and time.monotonic() >= kept:
            poses.add(runtime.pose())
            kept += runtime.step_dt
        reading = runtime.health()
        fall = max(fall, float(max(0.0, -velocity_w[2])))
        spin = max(spin, float(np.max(np.abs(reading["motor_speed"]))))
        height = float(runtime.bus.pose()[0][2])
    return {
        "describe": "their Passive chord (LT + B): kp 0, kd 3 in one step, "
        "their controller's own",
        "while": f"walking at {float(command[0]):g} m/s through their controller "
        f"for {walked} of {planned} ticks ({walked * runtime.step_dt:g} of "
        f"{DDS_WALK_S:g} s), watched {DDS_WATCH_S:g} s",
        "ticks_walked": walked,
        "ticks_planned": planned,
        "stopped_by": reason,
        "max_body_fall_mps": round(fall, DIGITS),
        "max_joint_speed_rad_s": round(spin, DIGITS),
        "end_height_m": round(height, DIGITS),
    }


# -- the checks ------------------------------------------------------------------


@dataclass
class Check:
    """One pre-flight check: what was measured against what limit."""

    name: str
    passed: bool | None  # None: not measurable here, said why
    measured: str
    limit: str
    detail: str = ""


@dataclass
class Context:
    """What the checks read: the manifest, an opened runtime (plain MuJoCo
    always, for the dry rollout), the robot's health as reported by the
    runtime that will drive it, and where that reading came from."""

    manifest: Manifest
    runtime: Runtime
    health: Health
    state_from: str
    margins: Margins = MARGINS
    seed: int = DEFAULT_SEED
    # where the dry rollout's twists come from, as the record says it
    twists_from: str = ""
    dry: DryRollout | None = None


CheckFn = Callable[[Context], Check]


def _policy_widths(runtime: Runtime) -> tuple[int | None, int | None]:
    def width(shape: Iterable[Any]) -> int | None:
        last = list(shape)[-1] if shape else None
        return int(last) if isinstance(last, int) else None

    return (
        width(runtime.session.get_inputs()[0].shape),
        width(runtime.session.get_outputs()[0].shape),
    )


def check_policy_widths(ctx: Context) -> Check:
    """The ONNX graph's own input and output widths against the manifest's
    observation terms and joints (unitree_rl_mjlab #25)."""
    m = ctx.manifest
    wanted_in = sum(o.width for o in m.observations)
    wanted_out = len(m.joints.policy_order)
    got_in, got_out = _policy_widths(ctx.runtime)
    ok = got_in == wanted_in and got_out == wanted_out
    return Check(
        "policy widths",
        ok,
        f"policy takes {got_in}, gives {got_out}",
        f"manifest's observations sum to {wanted_in}, {wanted_out} joints",
        ""
        if ok
        else "the observation the robot will send is not the one it trained on",
    )


def check_joint_order(ctx: Context) -> Check:
    """Every policy joint exists in the scene, and each action lands on the
    actuator driving that joint (the order the field mis-copies)."""
    model = ctx.runtime.model
    joints = ctx.manifest.joints
    wrong: list[str] = []
    for i, name in enumerate(joints.policy_order):
        ctrl = joints.action_to_ctrl[i]
        try:
            jid = model.joint(name).id
        except KeyError:
            wrong.append(f"{name}: not in the scene")
            continue
        if ctrl >= model.nu or int(model.actuator_trnid[ctrl, 0]) != jid:
            driven = (
                model.joint(int(model.actuator_trnid[ctrl, 0])).name
                if ctrl < model.nu
                else "nothing"
            )
            wrong.append(f"{name}: action {i} drives {driven}")
    n = len(joints.policy_order)
    return Check(
        "joint order",
        not wrong,
        f"{n - len(wrong)}/{n} actions land on their joint",
        "every action on the actuator of its own joint",
        "; ".join(wrong[:4]),
    )


def _scene_gains(runtime: Runtime) -> tuple[np.ndarray, np.ndarray]:
    """kp and kd of each policy joint's actuator, from an affine `general`
    (gain kp, bias [0, -kp, -kd]) as the exporter writes it."""
    ctrl = list(runtime.manifest.joints.action_to_ctrl)
    model = runtime.model
    kp = model.actuator_gainprm[ctrl, 0].astype(float)
    kd = -model.actuator_biasprm[ctrl, 2].astype(float)
    return kp, kd


def _unitree_yaml_gains(manifest: Manifest) -> tuple[list[float], list[float]] | None:
    """The gains in the YAML Unitree's controller reads, when the
    deployment was staged for it (`deploy.unitree_stage`)."""
    from rq_pipeline.deploy.unitree_stage import (  # noqa: PLC0415
        POLICY_VERSION,
        STAGE_DIR,
    )
    from rq_pipeline.deploy.unitree_yaml import UNITREE_DEPLOY_FILE  # noqa: PLC0415

    path = (
        manifest.root
        / STAGE_DIR
        / "config"
        / "policy"
        / "velocity"
        / POLICY_VERSION
        / "params"
        / UNITREE_DEPLOY_FILE
    )
    if not path.is_file():
        return None
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("stiffness") or []), list(raw.get("damping") or [])


def check_gains(ctx: Context) -> Check:
    """The gains the policy trained under (the manifest's) against the
    scene's actuators and, when staged, the YAML their controller reads."""
    joints = ctx.manifest.joints
    if not joints.stiffness:
        return Check(
            "gains", None, "unrecorded", "the manifest's gains", "none exported"
        )
    want_kp = np.asarray(joints.stiffness, float)
    want_kd = np.asarray(joints.damping, float)
    tol = ctx.margins.gain_rel_tol
    kp, kd = _scene_gains(ctx.runtime)
    off: list[str] = []

    def compare(where: str, got_kp: np.ndarray, got_kd: np.ndarray) -> None:
        for name, want, got in (("kp", want_kp, got_kp), ("kd", want_kd, got_kd)):
            if got.shape != want.shape:
                off.append(f"{where} {name}: {got.size} values for {want.size} joints")
                continue
            bad = np.flatnonzero(
                np.abs(got - want) > tol * np.maximum(1.0, np.abs(want))
            )
            if bad.size:
                i = int(bad[0])
                off.append(
                    f"{where} {name} of {joints.policy_order[i]}: {got[i]:g}, "
                    f"trained {want[i]:g}"
                )

    compare("scene", kp, kd)
    staged = _unitree_yaml_gains(ctx.manifest)
    if staged is not None:
        compare(
            "their YAML", np.asarray(staged[0], float), np.asarray(staged[1], float)
        )
    where = "scene and their YAML" if staged is not None else "scene"
    return Check(
        "gains",
        not off,
        f"{len(off)} disagreements ({where})",
        f"equal to the trained kp and kd within {tol:g}",
        "; ".join(off[:4]),
    )


@dataclass
class DryRollout:
    """What the policy commanded over the gate's own held twists, in plain
    MuJoCo: per tick and joint, the target, how far past the joint's range
    it sat, the torque the PD target demanded (unclipped) over the force
    range, whether the actuator sat at its range; and the compute per
    tick."""

    ticks: int = 0
    worst_past_range: float = 0.0
    worst_past_joint: str = ""
    worst_demand_ratio: float = 0.0
    worst_demand_joint: str = ""
    saturated_ticks: int = 0
    compute_s: list[float] = field(default_factory=list)
    successes: int = 0
    trials: int = 0
    unlimited: list[str] = field(default_factory=list)  # joints with no force range


class _Recording:
    """The plain runtime with every tick's commanded targets and demanded
    torques kept (the gate's protocol calls through it unchanged)."""

    def __init__(self, runtime: Runtime, out: DryRollout) -> None:
        self._rt = runtime
        self._out = out
        model = runtime.model
        ctrl = list(runtime.action_to_ctrl)
        joint_ids = [int(model.actuator_trnid[c, 0]) for c in ctrl]
        self._names = list(runtime.manifest.joints.policy_order)
        self._ctrl = ctrl
        self._range = model.jnt_range[joint_ids].astype(float)
        self._limited = model.jnt_limited[joint_ids].astype(bool)
        # an actuator with no force range has no demand to judge: its range
        # is infinite, its ratio 0, and the torque check says so
        limited = model.actuator_forcelimited[ctrl].astype(bool)
        self._force = np.where(
            limited, np.abs(model.actuator_forcerange[ctrl, 1]), np.inf
        ).astype(float)
        self._unlimited = [self._names[i] for i in np.flatnonzero(~limited)]
        self._kp, self._kd = _scene_gains(runtime)
        self._t0 = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rt, name)

    @property
    def command(self) -> np.ndarray:
        return self._rt.command

    @command.setter
    def command(self, value: np.ndarray) -> None:
        self._rt.command = value

    def observe(self) -> np.ndarray:
        self._t0 = time.perf_counter()
        return self._rt.observe()

    def apply(self, action: np.ndarray) -> None:
        rt = self._rt
        target = rt.target_of(action)
        q = rt.data.qpos[rt.joint_qpos]
        dq = rt.data.qvel[rt.joint_qvel]
        self._out.compute_s.append(time.perf_counter() - self._t0)
        lo, hi = self._range[:, 0], self._range[:, 1]
        past = np.where(
            self._limited, np.maximum(lo - target, target - hi).clip(min=0.0), 0.0
        )
        i = int(np.argmax(past))
        if past[i] > self._out.worst_past_range:
            self._out.worst_past_range = float(past[i])
            self._out.worst_past_joint = self._names[i]
        demand = np.abs(self._kp * (target - q) - self._kd * dq)
        ratio = demand / np.maximum(self._force, 1e-9)
        j = int(np.argmax(ratio))
        if ratio[j] > self._out.worst_demand_ratio:
            self._out.worst_demand_ratio = float(ratio[j])
            self._out.worst_demand_joint = self._names[j]
        if np.any(ratio >= 1.0):
            self._out.saturated_ticks += 1
        self._out.ticks += 1
        self._out.unlimited = self._unlimited
        rt.apply(action)


def dry_rollout(ctx: Context) -> DryRollout:
    """The gate's own held twists, a few, in plain MuJoCo, every command
    kept (computed once and shared by the checks that read it)."""
    if ctx.dry is not None:
        return ctx.dry
    out = DryRollout()
    driver = _Recording(ctx.runtime, out)
    results = hold_twists(
        ctx.manifest, driver, {}, ctx.margins.dry_trials, ctx.seed, None, []
    )
    out.trials = len(results)
    out.successes = sum(r.success for r in results)
    ctx.dry = out
    return out


def check_targets(ctx: Context) -> Check:
    dry = dry_rollout(ctx)
    limit = ctx.margins.target_past_range_rad
    ok = dry.worst_past_range <= limit
    return Check(
        "targets within the joints' ranges",
        ok,
        f"worst {dry.worst_past_range:.3f} rad past range "
        f"({dry.worst_past_joint or '-'}) over {dry.ticks} ticks",
        f"at most {limit:g} rad past a joint's range",
        f"dry rollout: {dry.trials} held twists ({ctx.twists_from}), "
        f"{dry.successes} tracked",
    )


def check_torques(ctx: Context) -> Check:
    dry = dry_rollout(ctx)
    m = ctx.margins
    if len(dry.unlimited) == len(ctx.manifest.joints.policy_order):
        return Check(
            "torques within the actuators' ranges",
            None,
            "not measurable",
            "the actuators' force ranges",
            "no actuator declares a force range (forcelimited off)",
        )
    share = dry.saturated_ticks / max(dry.ticks, 1)
    ok = dry.worst_demand_ratio <= m.torque_demand_ratio and share <= m.saturated_share
    return Check(
        "torques within the actuators' ranges",
        ok,
        f"peak demand {dry.worst_demand_ratio:.2f}x the force range "
        f"({dry.worst_demand_joint or '-'}); at the range {share:.1%} of ticks",
        f"peak at most {m.torque_demand_ratio:g}x, at the range at most "
        f"{m.saturated_share:.0%} of ticks",
    )


def check_control_rate(ctx: Context) -> Check:
    dry = dry_rollout(ctx)
    period = ctx.manifest.control.step_dt
    p99 = float(np.percentile(dry.compute_s, 99)) if dry.compute_s else float("nan")
    budget = ctx.margins.compute_share * period
    return Check(
        "control rate",
        bool(p99 <= budget),
        f"observe + infer p99 {p99 * 1e3:.2f} ms per tick",
        f"within {ctx.margins.compute_share:.0%} of the {period * 1e3:g} ms period",
        "measured on this machine; the robot's computer is measured on the day",
    )


def check_robot_state(ctx: Context) -> Check:
    """Upright, at rest, and nothing the watchdogs would trip, as the
    runtime that will drive the robot reports it."""
    h = ctx.health
    trips = tripped(h)
    moving = h.max_joint_velocity > ctx.margins.at_rest_rad_s
    unreported = [
        w.name
        for w in WATCHDOGS.values()
        if h.value(w.name) is None and w.name != "link_lost"
    ]
    parts = [
        f"tilt {h.tilt_rad:.3f} rad",
        f"joints {h.max_joint_velocity:.3f} rad/s",
        f"gyro {h.max_angular_velocity:.3f} rad/s",
    ]
    if h.max_casing_temperature is not None:
        parts.append(f"motors {h.max_casing_temperature:g} degC")
    if h.battery_percent is not None:
        parts.append(f"battery {h.battery_percent:g} %")
    detail = f"read from {ctx.state_from}"
    if unreported:
        detail += f"; not reported there: {', '.join(unreported)}"
    if trips:
        detail += f"; trips {', '.join(trips)}"
    return Check(
        "robot state before handover",
        not trips and not moving,
        ", ".join(parts),
        f"the SDK's watchdogs untripped; every joint under "
        f"{ctx.margins.at_rest_rad_s:g} rad/s",
        detail,
    )


@dataclass(frozen=True)
class CheckSpec:
    """A check as the table knows it: whether it drives the policy (the
    dry rollout), which it must not do after a structural refusal."""

    fn: CheckFn
    drives_policy: bool = False


# One table, in the order a person would read them; a check is added here.
CHECKS: dict[str, CheckSpec] = {
    "policy widths": CheckSpec(check_policy_widths),
    "joint order": CheckSpec(check_joint_order),
    "gains": CheckSpec(check_gains),
    "targets within the joints' ranges": CheckSpec(check_targets, drives_policy=True),
    "torques within the actuators' ranges": CheckSpec(
        check_torques, drives_policy=True
    ),
    "control rate": CheckSpec(check_control_rate, drives_policy=True),
    "robot state before handover": CheckSpec(check_robot_state),
}
NOT_RUN = "not run: {first} refused first, and driving the policy would be unsafe"


# Told as each step of a pre-flight ends: (steps done, steps in all,
# the line) - the Studio's Running now panel (`mcp_jobs.Tracker.progress`)
# shows "check 3 of 9: gains passed" without the pre-flight knowing it.
StepProgress = Callable[[int, int, str], None]
# The steps after the checks when a pre-flight measures: the ramp-in and
# the stops.
MEASURED_STEPS = ("ramp-in", "stops")


def run_checks(
    ctx: Context, on_step: StepProgress | None = None, steps: int = 0
) -> list[Check]:
    """Every check in table order; a check that drives the policy is not
    run once a structural one refused (a policy fed the wrong vector does
    nothing meaningful, in simulation or on a robot). `on_step` is told
    each check's verdict as it lands, as step n of `steps` (the checks'
    own count when 0)."""
    total = steps or len(CHECKS)
    out: list[Check] = []
    for name, spec in CHECKS.items():
        refused = next((c.name for c in out if c.passed is False), None)
        if spec.drives_policy and refused is not None:
            out.append(Check(name, None, "not run", "", NOT_RUN.format(first=refused)))
        else:
            try:
                out.append(spec.fn(ctx))
            except (ValueError, IndexError, KeyError, RuntimeError) as why:
                out.append(Check(name, False, "could not run", "", str(why)))
        if on_step is not None:
            last = out[-1]
            on_step(len(out), total, f"{last.name}: {CHECK_MARKS[last.passed]}")
    return out


# A check's verdict in one word: the stream, the tool's lines and the
# drawer all read this.
CHECK_MARKS: dict[bool | None, str] = {
    True: "passed",
    False: "REFUSED",
    None: "not measured",
}


def verdict_word(checks: list[Check]) -> str:
    """ "passed 7/7" or "refused: <first failing check>"."""
    judged = [c for c in checks if c.passed is not None]
    failed = [c for c in judged if not c.passed]
    if failed:
        return f"refused: {failed[0].name} ({failed[0].measured})"
    return f"passed {len(judged)}/{len(judged)}"


# -- the ramp-in and the soft stop ---------------------------------------------------


def blend(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend, alpha clipped to [0, 1]."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - a) * np.asarray(start, float) + a * np.asarray(end, float)


class Guarded:
    """The plain runtime behind the ramp-in, the watchdogs and the soft
    stop. `apply` blends the policy's target from the pose measured at
    handover over the ramp window; every tick reads the watchdogs and
    the operator's stop file; once stopping, the gains blend to damping
    over the stop window and the target holds the pose at the stop."""

    def __init__(  # noqa: PLR0913 - the guard's own knobs, each named
        self,
        runtime: Runtime,
        transitions: Transitions,
        *,
        ramp: bool = True,
        soft: bool = True,
        end_kd: float | None = None,
        stop_file: Path | None = None,
        poses: PoseTrack | None = None,
    ) -> None:
        self.rt = runtime
        self.poses = poses
        self.tr = transitions
        self.ramp = ramp
        self.soft = soft
        self.end_kd = transitions.damping_kd if end_kd is None else end_kd
        self.stop_file = stop_file
        self.kp0, self.kd0 = _scene_gains(runtime)
        self._ctrl = list(runtime.action_to_ctrl)
        self.start_pose: np.ndarray | None = None
        self.stopped_at: int | None = None
        self.stop_reason: str | None = None
        self.hold: np.ndarray | None = None
        self.ticks = 0
        self.log: list[dict[str, Any]] = []

    @property
    def ramp_ticks(self) -> int:
        return max(1, round(self.tr.ramp_in_s / self.rt.step_dt))

    @property
    def stop_ticks(self) -> int:
        return max(1, round(self.tr.soft_stop_s / self.rt.step_dt))

    def handover(self) -> None:
        """The pose the ramp starts from: where the robot is, measured; the
        held target is that pose (what damping holds), so the first tick's
        step is measured from where the motors were told to be."""
        self.start_pose = self.rt.data.qpos[self.rt.joint_qpos].copy()
        for i, c in enumerate(self._ctrl):
            self.rt.data.ctrl[c] = self.start_pose[i]
        self.ticks = 0

    def stop(self, reason: str = OPERATOR_STOP) -> None:
        if self.stopped_at is None:
            self.stopped_at = self.ticks
            self.stop_reason = reason
            self.hold = self.rt.data.qpos[self.rt.joint_qpos].copy()

    def _set_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        model = self.rt.model
        model.actuator_gainprm[self._ctrl, 0] = kp
        model.actuator_biasprm[self._ctrl, 1] = -kp
        model.actuator_biasprm[self._ctrl, 2] = -kd

    def _watch(self) -> None:
        if self.stopped_at is not None:
            return
        if self.stop_file is not None and self.stop_file.is_file():
            self.stop(OPERATOR_STOP)
            return
        trips = tripped(health_of_runtime(self.rt))
        if trips:
            self.stop(trips[0])

    def target_of(self, action: np.ndarray) -> np.ndarray:
        rt = self.rt
        policy = rt.target_of(action)
        if self.stopped_at is not None and self.hold is not None:
            return self.hold
        if self.ramp and self.start_pose is not None and self.ticks < self.ramp_ticks:
            return blend(self.start_pose, policy, (self.ticks + 1) / self.ramp_ticks)
        return policy

    def apply(self, action: np.ndarray) -> None:
        """One guarded tick: watchdogs, gains, target, `decimation` steps."""
        rt = self.rt
        self._watch()
        if self.stopped_at is not None:
            since = self.ticks - self.stopped_at + 1
            alpha = since / self.stop_ticks if self.soft else 1.0
            kd_end = np.full_like(self.kd0, self.end_kd)
            self._set_gains(
                blend(self.kp0, np.zeros_like(self.kp0), alpha),
                blend(self.kd0, kd_end, alpha),
            )
        target = self.target_of(action)
        prev = rt.data.ctrl[self._ctrl].copy()
        force_before = rt.data.actuator_force[self._ctrl].copy()
        rt.step(target, action)
        if self.poses is not None:
            self.poses.add(rt.pose())
        self.log.append(
            {
                "tick": self.ticks,
                "target_step": float(np.max(np.abs(target - prev))),
                "force_step": float(
                    np.max(np.abs(rt.data.actuator_force[self._ctrl] - force_before))
                ),
                "kp": float(np.mean(rt.model.actuator_gainprm[self._ctrl, 0])),
                "kd": float(np.mean(-rt.model.actuator_biasprm[self._ctrl, 2])),
                "height": float(rt.base_position()[2]),
                "fall_speed": float(max(0.0, -rt.base_linear_velocity_w()[2])),
                "joint_speed": float(np.max(np.abs(rt.data.qvel[rt.joint_qvel]))),
                "stopping": self.stopped_at is not None,
            }
        )
        self.ticks += 1

    def restore(self) -> None:
        self._set_gains(self.kp0, self.kd0)


# The pose a Go2 lies in before its controller stands it up: Unitree's
# FixStand first key pose (their config.yaml, qs[1]) in THEIR order
# (FR, FL, RR, RL); the ramp is measured from it, not from the home pose.
LYING_POSE_SDK = (0.0, 1.36, -2.65) * 4
# Where the body is dropped from to settle (lying, the Go2's base sits at
# about 0.1 m) and how long it settles in damping before the handover.
LYING_DROP_HEIGHT_M = 0.12
LYING_SETTLE_S = 1.0
FROM_LYING = "lying (Unitree's FixStand first key pose), in damping"
FROM_HOME = "the manifest's home pose, in damping (no lying pose for this robot)"


def _lying_pose(runtime: Runtime) -> tuple[np.ndarray, str]:
    """The pose the handover is measured from, and which it is: the Go2's
    lying pose in the policy's order when the manifest maps the SDK's
    motors, else the home pose, said so in the record."""
    order = runtime.manifest.joints.sdk_order_map
    sdk = np.asarray(LYING_POSE_SDK, float)
    if order is None or len(order) != sdk.size:
        return runtime.default_pos.copy(), FROM_HOME
    return sdk[list(order)], FROM_LYING


def _lie_down(runtime: Runtime, transitions: Transitions) -> str:
    """Reset, then place the joints in the lying pose and let the body
    settle on the floor in damping, the state a controller hands over
    from; returns which pose it was."""
    import mujoco  # noqa: PLC0415

    runtime.reset()
    pose, which = _lying_pose(runtime)
    runtime.data.qpos[runtime.joint_qpos] = pose
    runtime.set_base_height(LYING_DROP_HEIGHT_M)
    runtime.data.qvel[:] = 0.0
    ctrl = list(runtime.action_to_ctrl)
    kp0, kd0 = _scene_gains(runtime)
    model = runtime.model
    model.actuator_gainprm[ctrl, 0] = 0.0
    model.actuator_biasprm[ctrl, 1] = 0.0
    model.actuator_biasprm[ctrl, 2] = -transitions.damping_kd
    for _ in range(round(LYING_SETTLE_S / model.opt.timestep)):
        mujoco.mj_step(model, runtime.data)
    model.actuator_gainprm[ctrl, 0] = kp0
    model.actuator_biasprm[ctrl, 1] = -kp0
    model.actuator_biasprm[ctrl, 2] = -kd0
    return which


RAMP_EPISODE_S = 3.0
# The ramp measured both ways; the labels the record, the drawer and the
# stream all read.
WITH_RAMP = "with ramp"
WITHOUT_RAMP = "without ramp"
RAMP_VARIANTS = ((WITH_RAMP, True), (WITHOUT_RAMP, False))
STOOD_HEIGHT_M = 0.2  # the base above this at the end: it stood (lying is ~0.1)
KP_OFF = 1e-9  # a stiffness this small is damping
STOP_AFTER_S = 2.0
STOP_EPISODE_S = 4.0


def measure_ramp(
    runtime: Runtime, transitions: Transitions, *, poses: PoseTrack | None = None
) -> dict[str, Any]:
    """Handover from lying in damping, standing command, with the ramp and
    without: the largest target step in one tick, the largest torque step,
    and whether the robot stood. With `poses`, each variant's ticks are
    kept for the Studio's replay."""
    out: dict[str, Any] = {"window_s": transitions.ramp_in_s}
    for label, ramp in RAMP_VARIANTS:
        out["from"] = _lie_down(runtime, transitions)
        if poses is not None:
            poses.begin(RAMP_SEGMENT.format(variant=label))
        guarded = Guarded(runtime, transitions, ramp=ramp, poses=poses)
        guarded.handover()
        runtime.command = np.zeros(3, dtype=np.float32)
        ticks = round(RAMP_EPISODE_S / runtime.step_dt)
        for _ in range(ticks):
            guarded.apply(runtime.act(runtime.observe()))
        window = guarded.log[: guarded.ramp_ticks]
        out[label] = {
            "max_target_step_rad": round(max(r["target_step"] for r in window), DIGITS),
            "first_tick_target_step_rad": round(window[0]["target_step"], DIGITS),
            "max_force_step_nm": round(max(r["force_step"] for r in window), DIGITS),
            "stood": bool(
                runtime.base_position()[2] > STOOD_HEIGHT_M and not runtime.fell_over()
            ),
            "height_m": round(float(runtime.base_position()[2]), DIGITS),
            "log": guarded.log,
        }
    return out


@dataclass(frozen=True)
class StopVariant:
    """One way to take the motors away: blended or at once, ending in
    which damping."""

    name: str
    soft: bool
    end_kd: float | None  # None: the declared damping
    describe: str


STOP_VARIANTS: tuple[StopVariant, ...] = (
    StopVariant(
        "soft", True, None, "gains blended to damping over the window, pose held"
    ),
    StopVariant(
        "damping at once", False, None, "kp to 0 and kd to damping in one tick"
    ),
    StopVariant(
        "zeroed",
        False,
        0.0,
        "kp and kd to 0 in one tick: the limp motor unitree_rl_lab #147 reports",
    ),
)


def measure_stop(
    runtime: Runtime, transitions: Transitions, *, poses: PoseTrack | None = None
) -> dict[str, Any]:
    """Walking at a held command, an operator stop at STOP_AFTER_S, each
    way of `STOP_VARIANTS`: the largest torque step in one tick, the time
    to damping, the body's fastest fall and the joints' fastest spin after
    the stop, the height it ends at. With `poses`, each variant's ticks
    are kept for the Studio's replay."""
    out: dict[str, Any] = {
        "window_s": transitions.soft_stop_s,
        "ends_in": f"kp 0, kd {transitions.damping_kd:g} (Unitree's Passive)",
    }
    command = walk_command(runtime.manifest)
    out["while"] = (
        f"walking at {float(command[0]):g} m/s, stopped at {STOP_AFTER_S:g} s, "
        f"watched {STOP_EPISODE_S - STOP_AFTER_S:g} s"
    )
    for variant in STOP_VARIANTS:
        runtime.reset()
        if poses is not None:
            poses.begin(STOP_SEGMENT.format(variant=variant.name))
        guarded = Guarded(
            runtime,
            transitions,
            soft=variant.soft,
            end_kd=variant.end_kd,
            poses=poses,
        )
        guarded.handover()
        guarded.ticks = guarded.ramp_ticks  # already walking: no ramp here
        runtime.command = command
        stop_at = round(STOP_AFTER_S / runtime.step_dt)
        for t in range(round(STOP_EPISODE_S / runtime.step_dt)):
            if t == stop_at:
                guarded.stop(OPERATOR_STOP)
            guarded.apply(runtime.act(runtime.observe()))
        after = [r for r in guarded.log if r["stopping"]]
        damped = next((i for i, r in enumerate(after) if r["kp"] <= KP_OFF), len(after))
        out[variant.name] = {
            "describe": variant.describe,
            "max_force_step_nm": round(max(r["force_step"] for r in after), DIGITS),
            "seconds_to_damping": round((damped + 1) * runtime.step_dt, DIGITS),
            "max_body_fall_mps": round(max(r["fall_speed"] for r in after), DIGITS),
            "max_joint_speed_rad_s": round(
                max(r["joint_speed"] for r in after), DIGITS
            ),
            "end_height_m": round(after[-1]["height"], DIGITS),
            "log": guarded.log,
        }
        guarded.restore()
    return out


# -- the record ------------------------------------------------------------------


# Where the dry rollout's twists come from, as the record words it.
TWISTS_OF_GATE = "the {runtime} gate's first {k}, its seed {seed}"
TWISTS_NO_GATE = (
    "the gate's draw at seed {seed}; no {runtime} gate record yet, so nothing "
    "pairs them"
)
SEED_MISMATCH = (
    "the {runtime} gate of {name} ran at seed {theirs}; a pre-flight at seed "
    "{ours} would not dry-run the gate's twists: leave the seed unset"
)
STATE_OF_SIMULATION = "plain MuJoCo (a simulation)"


def gate_twists(
    deployment_dir: Path, seed: int | None, runtime: str = DEFAULT_RUNTIME
) -> tuple[int, str]:
    """The seed of the dry rollout and how the record words it: the
    plain gate's own seed when it has a record (refused by name when that
    record was drawn by the old count-dependent draw, or when `seed` is
    given and differs), else the gate's default."""
    gates = read_gates(deployment_dir)
    record = gates.get(runtime)
    name = Path(deployment_dir).name
    if record is None:
        used = DEFAULT_SEED if seed is None else int(seed)
        return used, TWISTS_NO_GATE.format(seed=used, runtime=runtime)
    from rq_pipeline.deploy.gate import require_same_draw  # noqa: PLC0415

    require_same_draw(record, f"the {runtime} gate of {name}")
    theirs = int((record.get("protocol") or {}).get("seed", DEFAULT_SEED))
    if seed is not None and int(seed) != theirs:
        raise ValueError(
            SEED_MISMATCH.format(runtime=runtime, name=name, theirs=theirs, ours=seed)
        )
    return theirs, TWISTS_OF_GATE.format(
        runtime=runtime, k=MARGINS.dry_trials, seed=theirs
    )


def preflight(  # noqa: PLR0913 - the pre-flight's own knobs, each named
    deployment_dir: Path,
    *,
    assets_dir: Path | None,
    health: Health | None = None,
    state_from: str = STATE_OF_SIMULATION,
    basis: str = BASIS_SIMULATION,
    stand_in: bool = False,
    margins: Margins = MARGINS,
    seed: int | None = None,
    measure: bool = True,
    manifest: Manifest | None = None,
    on_step: StepProgress | None = None,
) -> dict[str, Any]:
    """Every check, then (when `measure`) the ramp-in and the soft stop
    measured in plain MuJoCo; the record written beside the manifest.
    `health` is the robot's state as the driving runtime reports it (the
    DDS stand-in's), else plain MuJoCo's own at rest; `basis` and
    `stand_in` say whose state it was (`bundles.basis`), never a default
    the caller forgot. The dry rollout runs the plain gate's first twists
    at the gate's own seed (`gate_twists`). `manifest` overrides the one
    on disk (a provoked refusal under test)."""
    from rq_pipeline.deploy.runtime import open_runtime  # noqa: PLC0415

    manifest = manifest or load_manifest(deployment_dir)
    used_seed, twists_from = gate_twists(deployment_dir, seed)
    runtime = open_runtime(manifest, assets_dir=assets_dir)
    ctx = Context(
        manifest=manifest,
        runtime=runtime,
        health=health if health is not None else health_of_runtime(runtime),
        state_from=state_from,
        margins=margins,
        seed=used_seed,
        twists_from=twists_from,
    )
    steps = len(CHECKS) + (len(MEASURED_STEPS) if measure else 0)
    checks = run_checks(ctx, on_step, steps)
    passed = all(c.passed is not False for c in checks)
    transitions = transitions_of(manifest)
    record: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "deployment": manifest.raw.get(Key.STAMP_OF),
        "policy": manifest.raw.get(Key.POLICY),
        "basis": basis,
        "stand_in": stand_in,
        "state_from": state_from,
        "checks": [asdict(c) for c in checks],
        "passed": passed,
        "verdict": verdict_word(checks),
        "margins": asdict(margins),
        "margins_source": MARGINS_SOURCE,
        "transitions": asdict(transitions),
        "transitions_source": TRANSITIONS_SOURCE,
        "dry_rollout": {"seed": used_seed, "twists": twists_from, "draw": DRAW_NOW},
        "watchdogs": {
            "source": SDK_SOURCE,
            "limits": [asdict(w) for w in WATCHDOGS.values()],
        },
        "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if measure and passed:
        poses = PoseTrack(manifest.control.step_dt, tuple(manifest.joints.policy_order))
        ramp = measure_ramp(runtime, transitions, poses=poses)
        if on_step is not None:
            on_step(len(CHECKS) + 1, steps, f"{MEASURED_STEPS[0]} measured")
        stop = measure_stop(runtime, transitions, poses=poses)
        if on_step is not None:
            on_step(steps, steps, f"{MEASURED_STEPS[1]} measured")
        record["ramp_in"] = _without_logs(ramp)
        record["soft_stop"] = _without_logs(stop)
        record["_logs"] = {"ramp_in": ramp, "soft_stop": stop}
        record[POSES_TRACK] = poses  # a live runtime's own stop adds to it
    return record


def _without_logs(block: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (
            {kk: vv for kk, vv in v.items() if kk != "log"}
            if isinstance(v, dict)
            else v
        )
        for k, v in block.items()
    }


def write_record(deployment_dir: Path, record: dict[str, Any]) -> Path:
    """The record beside the manifest; its poses (when the ramp and the
    stops ran) beside the saved stream, for the Studio's replay."""
    from rq_pipeline.viz import viewer_file  # noqa: PLC0415

    body = {k: v for k, v in record.items() if not k.startswith("_")}
    track = record.get(POSES_TRACK)
    if track is not None and len(track):
        saved = track.save(poses_file(viewer_file(deployment_dir, STREAM)))
        body[POSES_KEY] = saved.relative_to(deployment_dir).as_posix()
    out = Path(deployment_dir) / PREFLIGHT_FILE
    staging = out.with_suffix(".json.tmp")
    staging.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    staging.replace(out)
    return out


def read_preflight(folder: Path) -> dict[str, Any] | None:
    path = Path(folder) / PREFLIGHT_FILE
    if not path.is_file():
        return None
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError(
            f"{path}: schema {raw.get('schema')!r}, this reader speaks "
            f"{PREFLIGHT_SCHEMA!r}"
        )
    return raw


# How the card qualifies a verdict by the state it read; only the
# operator's own robot goes unqualified, and a basis this table does not
# know is named as it is.
ON_STAND_IN = " on a simulation stand-in"
BASIS_QUALIFIER = {BASIS_OWN: "", BASIS_SIMULATION: " in simulation"}
RAMP_DID_NOT_STAND = "; the ramp-in did not stand"


def card_line(record: dict[str, Any]) -> str:
    """What the Deployments card says: the verdict and on what state - a
    simulation stand-in for the robot (Unitree's simulator over DDS), the
    plain simulation, or the robot itself - and a ramp-in that did not
    stand the robot up, which the seven checks do not judge."""
    basis = str(record.get("basis", "unrecorded"))
    if record.get("stand_in"):
        where = ON_STAND_IN
    else:
        where = BASIS_QUALIFIER.get(basis, f" on {basis}")
    ramp = (record.get("ramp_in") or {}).get(WITH_RAMP) or {}
    fell = RAMP_DID_NOT_STAND if ramp and not ramp.get("stood", True) else ""
    return f"{record.get('verdict', 'unrecorded')}{where}{fell}"


def request_stop(deployment_dir: Path, reason: str = OPERATOR_STOP) -> Path:
    """The operator's stop: a file the guarded loop reads every tick."""
    out = Path(deployment_dir) / STOP_FILE
    out.write_text(
        json.dumps(
            {
                "reason": reason,
                "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return out


# -- seen ------------------------------------------------------------------------


def stream(record: dict[str, Any], folder: Path, *, rr: Any = None) -> Path:
    """The ramp and the stop as series: per tick, the largest target step
    and torque step, the mean kp and kd, the body's height; with and
    without, soft and hard, side by side. Saved as the deployment's
    `preflight` stream and sent to a listening Studio."""
    from rq_pipeline.viz import open_stream, viewer_file  # noqa: PLC0415

    file = viewer_file(folder, STREAM)
    file.parent.mkdir(parents=True, exist_ok=True)
    rr = open_stream(
        f"rq-{STREAM}-{Path(folder).name}", file=file, rr=rr, on_term=False
    )
    logs = record.get("_logs") or {}
    for phase, block in logs.items():
        for variant, v in block.items():
            if not isinstance(v, dict) or "log" not in v:
                continue
            for row in v["log"]:
                rr.set_time("tick", sequence=int(row["tick"]))
                base = f"{STREAM}/{phase}/{variant.replace(' ', '_')}"
                for key in ("target_step", "force_step", "kp", "kd", "height"):
                    rr.log(f"{base}/{key}", rr.Scalars(float(row[key])))
    lines = [f"# {Path(folder).name}: pre-flight", "", f"**{card_line(record)}**", ""]
    for c in record.get("checks", []):
        mark = CHECK_MARKS[c["passed"]]
        lines.append(f"- {mark} · {c['name']}: {c['measured']} (limit: {c['limit']})")
    rr.log(f"{STREAM}/record", rr.TextDocument("\n".join(lines)), static=True)
    return file


def still_at_handover(
    deployment_dir: Path, *, assets_dir: Path | None, manifest: Manifest | None = None
) -> dict[str, Any]:
    """The robot at the end of the ramp-in window, from lying: the moment
    the policy has the robot alone."""
    try:
        from rq_pipeline.deploy.runtime import open_runtime  # noqa: PLC0415
    except ImportError as missing:
        return {"unrendered": str(missing)}
    manifest = manifest or load_manifest(deployment_dir)
    runtime = open_runtime(manifest, assets_dir=assets_dir)
    transitions = transitions_of(manifest)
    _lie_down(runtime, transitions)
    guarded = Guarded(runtime, transitions)
    guarded.handover()
    runtime.command = np.zeros(3, dtype=np.float32)
    for _ in range(guarded.ramp_ticks):
        guarded.apply(runtime.act(runtime.observe()))
    try:
        with stills.renderer(runtime.model, stills.HANDOVER_CAMERA) as render:
            render(
                runtime.data, runtime.base_position(), Path(deployment_dir) / STILL_FILE
            )
    except (RuntimeError, OSError, ValueError, ImportError) as why:
        return {"unrendered": f"no renderer: {why}"}
    return {"file": STILL_FILE, "tick": guarded.ramp_ticks}
