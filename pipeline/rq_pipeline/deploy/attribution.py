"""Attribution: which parameter would break this policy first (docs/77
§9). A gate that PASSED is re-run with one dynamics knob turned at a
time, up a ladder of the field's plausible deployment deviations from
the smallest; the knob's **cliff** is the first rung where the tracked
rate's exact lower bound falls below the certificate's own lower bound;
the knobs ranked by the rung they fall at are the answer to "it walked
in simulation and fell on the robot: what is most likely wrong".

Every knob turns the plain MuJoCo runtime's model or its loop and
nothing else: the policy, the manifest, the commands and the judge are
the gate's. A gate that did not pass has no cliff to find and is refused
by name; a staged scene walks another protocol and is refused too; the
DDS runtime is another process's model and cannot be turned from here.

The record, `attribution.json` beside the manifest, keeps every rung
that ran (a rung past the cliff is not run: the ladders are monotone by
construction), the certificate it was judged against, the ranking and
the one-line sensitivity the Deployments card shows.
"""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rq_pipeline.deploy import stills
from rq_pipeline.deploy.gate import (
    CI_DIGITS,
    DEFAULT_TOLERANCE,
    DRAW_KEY,
    DRAW_NOW,
    draw_commands,
    hold_twists,
    require_same_draw,
)
from rq_pipeline.deploy.manifest import Key, Manifest, load_manifest, read_gates
from rq_pipeline.deploy.runtimes import DEFAULT_RUNTIME, Opener, runtime_spec
from rq_pipeline.evaluate.tracking import TrackingOutcome
from rq_pipeline.stats.intervals import clopper_pearson

if TYPE_CHECKING:
    from rq_pipeline.deploy.runtime import Runtime

ATTRIBUTION_FILE = "attribution.json"
ATTRIBUTION_SCHEMA = "trainnr-attribution/1"
CLIFF_RULE = (
    "a rung is past the cliff when the exact 95 % lower bound of its tracked "
    "rate falls below the certificate's exact 95 % lower bound; the ladder "
    "stops there"
)
STILL_FILE = "attribution-cliff.png"
PUSH_PERIOD_S = 4.0
NOT_PASSED = (
    "attribution needs a passing gate; the {runtime} gate of {name} read "
    "{word} ({successes}/{trials}): fix the gate first, then ask what would "
    "break it"
)
NO_CERTIFICATE = (
    "attribution needs the certificate the gate was judged against; {name} "
    "cites no evaluation"
)
NO_GATE = "attribution needs a passing gate; {name} has no {runtime} gate record"
STAGED = (
    "attribution runs the plane's protocol (held twists); {name} is staged on "
    "a scene and walks a course, another protocol"
)
OTHER_RUNTIME = (
    "attribution turns the plain MuJoCo runtime's own model; the {runtime} "
    "runtime is another process's and cannot be turned from here"
)
SURVIVED = "survived the ladder"
NO_INTERVAL = (
    "attribution judges every rung against the certificate's exact lower "
    "bound; the evaluation {name} cites carries no ci95 (and {missing}): "
    "re-run the evaluation"
)
PROTOCOL_MISMATCH = (
    "the {runtime} gate of {name} ran {field} {theirs}; an attribution at "
    "{field} {ours} would not run the gate's trials: leave {field} unset"
)


# -- the knobs ------------------------------------------------------------------


@dataclass(frozen=True)
class Knob:
    """One dynamics deviation a deployment may meet, and the ladder of
    sizes it is tried at, the smallest plausible first (the field's own
    deployment mistakes and randomization ranges set the rungs; each
    ladder is monotone in severity, which is what lets the sweep stop at
    the cliff)."""

    name: str
    unit: str
    ladder: tuple[float, ...]
    describe: str
    source: str = ""  # where the rungs come from, as the record states it

    def label(self, level: float) -> str:
        return f"{self.name} {level:g} {self.unit}".rstrip()


Applier = Callable[["Runtime", float, int], "Runtime"]
"""Turn a knob on an opened runtime to `level` (seeded by the third
argument where the knob draws); returns the runtime to drive, which may
be a wrapper around the one given."""


def _world_friction(runtime: Runtime, level: float, _seed: int) -> Runtime:
    """Every geom's sliding friction scaled: the contact's coefficient is
    the pair's max (or the priority's), so scaling the floor alone would
    stop at the feet's own value."""
    runtime.model.geom_friction[:, 0] *= level
    return runtime


def _base_body(runtime: Runtime) -> int:
    import mujoco  # noqa: PLC0415

    model = runtime.model
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if free.size == 0:
        raise ValueError("the scene has no floating base (no free joint)")
    return int(model.jnt_bodyid[free[0]])


def _payload(runtime: Runtime, level: float, _seed: int) -> Runtime:
    """`level` kilograms on the base, its inertia grown in proportion
    (a payload where the reference randomizes base mass)."""
    import mujoco  # noqa: PLC0415

    body = _base_body(runtime)
    model = runtime.model
    before = float(model.body_mass[body])
    model.body_mass[body] = before + level
    model.body_inertia[body] *= (before + level) / before
    mujoco.mj_setConst(model, runtime.data)
    return runtime


def _kp_scale(runtime: Runtime, level: float, _seed: int) -> Runtime:
    """The position servos' stiffness scaled (an affine `general`
    actuator: gain and the position bias together)."""
    model = runtime.model
    model.actuator_gainprm[:, 0] *= level
    model.actuator_biasprm[:, 1] *= level
    return runtime


def _kd_scale(runtime: Runtime, level: float, _seed: int) -> Runtime:
    """The position servos' damping scaled (the velocity bias)."""
    runtime.model.actuator_biasprm[:, 2] *= level
    return runtime


def _tilt(runtime: Runtime, level: float, _seed: int) -> Runtime:
    """Gravity pitched by `level` degrees: the slope the robot believes
    is flat, seen by the policy only through projected gravity; the
    model's own gravity's magnitude, turned."""
    theta = np.radians(level)
    g = float(np.linalg.norm(runtime.model.opt.gravity))
    runtime.model.opt.gravity[:] = (g * np.sin(theta), 0.0, -g * np.cos(theta))
    return runtime


@dataclass
class Delayed:
    """The runtime with its actions applied `ticks` control steps late:
    the policy observes its own last action (as in training); the robot
    receives an older one (as over a slow link)."""

    inner: Runtime
    ticks: int
    queue: deque[np.ndarray] = field(default_factory=deque)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def command(self) -> np.ndarray:
        return self.inner.command

    @command.setter
    def command(self, value: np.ndarray) -> None:
        self.inner.command = value

    def reset(self) -> None:
        self.inner.reset()
        self.queue.clear()

    def apply(self, action: np.ndarray) -> None:
        self.queue.append(np.asarray(action, dtype=np.float32))
        late = (
            self.queue.popleft()
            if len(self.queue) > self.ticks
            else np.zeros_like(self.queue[-1])
        )
        self.inner.apply(late)
        self.inner.last_action = np.asarray(action, dtype=np.float32)


def _latency(runtime: Runtime, level: float, _seed: int) -> Runtime:
    return Delayed(runtime, int(level))


@dataclass
class Noisy:
    """The runtime whose named observation term carries seeded gaussian
    noise of a set standard deviation (an encoder that reads worse than
    the one trained against)."""

    inner: Runtime
    term: str
    std: float
    rng: np.random.Generator
    lo: int = field(init=False)
    hi: int = field(init=False)

    def __post_init__(self) -> None:
        offset = 0
        for term in self.inner.manifest.observations:
            if term.name == self.term:
                self.lo, self.hi = offset, offset + term.width
                return
            offset += term.width
        raise ValueError(f"no observation term {self.term!r} in the manifest")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def command(self) -> np.ndarray:
        return self.inner.command

    @command.setter
    def command(self, value: np.ndarray) -> None:
        self.inner.command = value

    def observe(self) -> np.ndarray:
        obs = self.inner.observe()
        obs[self.lo : self.hi] += self.rng.normal(0.0, self.std, self.hi - self.lo)
        return obs.astype(np.float32)


JOINT_POS_TERM = "joint_pos"


def _joint_pos_noise(runtime: Runtime, level: float, seed: int) -> Runtime:
    return Noisy(runtime, JOINT_POS_TERM, level, np.random.default_rng(seed))


@dataclass
class Pushed:
    """The runtime whose base is shoved every `period` seconds by a
    velocity of `speed` in a seeded horizontal direction (the reference's
    push event, applied to the deployed policy)."""

    inner: Runtime
    speed: float
    rng: np.random.Generator
    period_ticks: int

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def command(self) -> np.ndarray:
        return self.inner.command

    @command.setter
    def command(self, value: np.ndarray) -> None:
        self.inner.command = value

    def apply(self, action: np.ndarray) -> None:
        ticks = self.inner.ticks
        if ticks and ticks % self.period_ticks == 0:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            base = self.inner.base_qvel
            self.inner.data.qvel[base] += self.speed * np.cos(angle)
            self.inner.data.qvel[base + 1] += self.speed * np.sin(angle)
        self.inner.apply(action)


def _push(runtime: Runtime, level: float, seed: int) -> Runtime:
    period = max(1, round(PUSH_PERIOD_S / runtime.step_dt))
    return Pushed(runtime, level, np.random.default_rng(seed), period)


# The knobs, one table: the sweep, the door's docstring, the drawer and the
# record all read it. Each ladder's source is stated in the record: the
# reference's DR tables (unitree_rl_mjlab src/tasks/velocity/
# velocity_env_cfg.py, read 2026-09-24), the deploy configs that went wrong
# in the field's trackers (docs/e2e-research/78 §1), our own findings, and
# where none exists, "declared (ours)".
REFERENCE_DR = (
    "unitree_rl_mjlab velocity_env_cfg.py DR table (friction (0.3, 1.6), "
    "base CoM, pushes 5-6 s), read 2026-09-24"
)
KNOBS: tuple[Knob, ...] = (
    Knob(
        "latency",
        "ticks",
        (1, 2, 3, 4),
        "actions applied this many control ticks late",
        "the walk's own budget finding (walk-latency-budget-2026-09-05); "
        "rungs declared (ours)",
    ),
    Knob(
        "friction",
        "x",
        (0.8, 0.6, 0.4, 0.3, 0.2),
        "every geom's sliding friction scaled",
        REFERENCE_DR + "; rungs below its floor declared (ours)",
    ),
    Knob(
        "payload",
        "kg",
        (1, 2, 4, 6, 8),
        "mass added to the base, inertia in proportion",
        "declared (ours): bracketing the field's base-mass randomization, "
        "carried as a payload",
    ),
    Knob(
        "kp",
        "x",
        (0.8, 0.6, 0.4, 0.3),
        "the servos' stiffness scaled",
        "hand-written deploy configs gone wrong (unitree_rl_mjlab #32, #57; "
        "docs/e2e-research/78 §1); rungs declared (ours)",
    ),
    Knob(
        "kd",
        "x",
        (2, 4, 8, 16),
        "the servos' damping scaled",
        "hand-written deploy configs gone wrong (unitree_rl_mjlab #32, #57; "
        "docs/e2e-research/78 §1); rungs declared (ours)",
    ),
    Knob(
        "joint_pos_noise",
        "rad",
        (0.01, 0.02, 0.05, 0.1),
        "gaussian noise on the joint position term",
        "the training's own observation noise (unitree_rl_mjlab "
        "velocity_env_cfg.py joint_pos Unoise ±0.01, read 2026-09-24); rungs "
        "past it declared (ours)",
    ),
    Knob(
        "tilt",
        "deg",
        (3, 5, 8, 12, 15),
        "gravity pitched: a slope believed flat",
        "declared (ours): the slopes a plane-trained policy meets outdoors",
    ),
    Knob(
        "push",
        "m/s",
        (0.5, 1.0, 1.5, 2.0),
        f"the base shoved every {PUSH_PERIOD_S:g} s",
        REFERENCE_DR + "; speeds declared (ours)",
    ),
)
APPLIERS: dict[str, Applier] = {
    "latency": _latency,
    "friction": _world_friction,
    "payload": _payload,
    "kp": _kp_scale,
    "kd": _kd_scale,
    "joint_pos_noise": _joint_pos_noise,
    "tilt": _tilt,
    "push": _push,
}


# The untouched runtime under the sweep's own draw: the record's baseline,
# and the check that this trial count reproduces the gate (a draw of two
# trials holds a command a draw of twenty does not, 2026-09-24).
NOMINAL = Knob(
    "nominal", "", (1.0,), "untouched: the gate's own protocol", "the gate itself"
)
APPLIERS[NOMINAL.name] = lambda runtime, _level, _seed: runtime
BASELINE_BELOW = (
    "at {trials} trials the untouched runtime reads {successes}/{trials} "
    "[{lo}, {hi}], under the certificate's lower bound {lower}: the passing "
    "gate does not reproduce here, so no cliff can be told from it"
)


def knob_names() -> tuple[str, ...]:
    return tuple(k.name for k in KNOBS)


def knob(name: str) -> Knob:
    for k in KNOBS:
        if k.name == name:
            return k
    raise ValueError(f"unknown knob {name!r}; one of {', '.join(knob_names())}")


# -- the sweep -----------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One rung's gate: the count and its exact interval."""

    level: float
    successes: int
    trials: int
    ci95: tuple[float, float]
    mean_err_ratio: float

    @property
    def rate(self) -> float:
        return self.successes / max(self.trials, 1)

    def row(self, tolerance_floor: float) -> dict[str, Any]:
        return asdict(self) | {
            "ci95": list(self.ci95),  # JSON has no tuples: the record is plain
            "rate": round(self.rate, CI_DIGITS),
            # the gate's own rule beside the cliff's, so a reader sees both
            "within_tolerance": self.rate >= tolerance_floor,
        }


@dataclass(frozen=True)
class Sweep:
    """The gate's protocol, carried to every worker."""

    deployment_dir: Path
    assets_dir: Path | None
    trials: int
    seed: int
    certificate_lower: float
    runtime: str = DEFAULT_RUNTIME


def gate_rung(
    sweep: Sweep,
    knob_: Knob,
    level: float,
    *,
    open: Opener | None = None,
    appliers: dict[str, Applier] | None = None,
) -> Rung:
    """One rung: a fresh runtime, the knob turned, the gate's held-twist
    trials at the gate's own seed and count (`Sweep`, read from the gate's
    record), judged the gate's way."""
    manifest = load_manifest(sweep.deployment_dir)
    opener = open if open is not None else runtime_spec(sweep.runtime).open()
    turned = (appliers or APPLIERS)[knob_.name](
        opener(manifest, assets_dir=sweep.assets_dir), level, sweep.seed
    )
    try:
        protocol: dict[str, Any] = {}
        results: list[TrackingOutcome] = hold_twists(
            manifest, turned, protocol, sweep.trials, sweep.seed, None, []
        )
    finally:
        close = getattr(turned, "close", None)
        if close is not None:
            close()
    k = sum(t.success for t in results)
    lo, hi = clopper_pearson(k, sweep.trials)
    return Rung(
        level=float(level),
        successes=k,
        trials=sweep.trials,
        ci95=(round(lo, CI_DIGITS), round(hi, CI_DIGITS)),
        mean_err_ratio=round(float(np.mean([t.err_ratio for t in results])), CI_DIGITS),
    )


def climb(
    sweep: Sweep,
    knob_: Knob,
    *,
    open: Opener | None = None,
    appliers: dict[str, Applier] | None = None,
) -> list[Rung]:
    """The knob's rungs from the smallest, stopping at the first past
    the cliff (the ladders are monotone, so a higher rung tells nothing
    new)."""
    rungs: list[Rung] = []
    for level in knob_.ladder:
        rung = gate_rung(sweep, knob_, level, open=open, appliers=appliers)
        rungs.append(rung)
        if rung.ci95[0] < sweep.certificate_lower:
            break
    return rungs


def _climb_in_worker(args: tuple[Sweep, Knob]) -> tuple[str, list[Rung]]:
    sweep, knob_ = args
    return knob_.name, climb(sweep, knob_)


def cliff_of(rungs: Iterable[Rung], certificate_lower: float) -> Rung | None:
    """The first rung whose lower bound is under the certificate's."""
    for rung in rungs:
        if rung.ci95[0] < certificate_lower:
            return rung
    return None


def rank(
    climbed: dict[str, list[Rung]], certificate_lower: float
) -> list[tuple[str, Rung | None]]:
    """Knobs by the rung they fall at (its index in the ladder, then the
    rate there); the ones that survived their ladder last, in the order
    they were climbed."""
    climbed_order = list(climbed)

    def key(item: tuple[str, list[Rung]]) -> tuple[int, int, float, int]:
        name, rungs = item
        cliff = cliff_of(rungs, certificate_lower)
        order = climbed_order.index(name)
        if cliff is None:
            return (1, 0, 0.0, order)
        return (0, rungs.index(cliff), cliff.rate, order)

    return [
        (name, cliff_of(rungs, certificate_lower))
        for name, rungs in sorted(climbed.items(), key=key)
    ]


def sensitivity_line(
    ranking: list[tuple[str, Rung | None]], knobs: tuple[Knob, ...] = KNOBS
) -> str:
    """The card's one line: the first two knobs that fell, at their
    rung; or the word when none did."""
    by_name = {k.name: k for k in knobs}
    fallen = [(name, cliff) for name, cliff in ranking if cliff is not None]
    if not fallen:
        return SURVIVED
    parts = [by_name[name].label(cliff.level) for name, cliff in fallen[:2]]
    return "most sensitive to " + ", then ".join(parts)


# -- the door's work -------------------------------------------------------------


def require_passing_gate(
    deployment_dir: Path, runtime: str, certificate: dict[str, Any] | None
) -> tuple[Manifest, dict[str, Any], dict[str, Any]]:
    """The manifest, the passing gate's record and the certificate it was
    judged against; every other case refused by name."""
    name = Path(deployment_dir).name
    if runtime != DEFAULT_RUNTIME:
        raise ValueError(OTHER_RUNTIME.format(runtime=runtime))
    manifest = load_manifest(deployment_dir)
    if (manifest.raw.get(Key.SCENE) or {}).get("scene"):
        raise ValueError(STAGED.format(name=name))
    if not certificate:
        raise ValueError(NO_CERTIFICATE.format(name=name))
    gates = read_gates(deployment_dir)
    if runtime not in gates:
        raise ValueError(NO_GATE.format(name=name, runtime=runtime))
    base = gates[runtime]
    require_same_draw(base, f"the {runtime} gate of {name}")
    if (base.get("verdict") or {}).get("passed") is not True:
        from rq_pipeline.deploy.manifest import gate_word  # noqa: PLC0415

        raise ValueError(
            NOT_PASSED.format(
                runtime=runtime,
                name=name,
                word=gate_word(base),
                successes=base.get("successes"),
                trials=base.get("trials"),
            )
        )
    return manifest, base, certificate


def gate_protocol(
    base: dict[str, Any], name: str, runtime: str, **asked: int | None
) -> dict[str, int]:
    """The trials and seed the passing gate ran at; one asked for that
    differs is refused by name (a rung would not run the gate's trials)."""
    protocol = base.get("protocol") or {}
    out = {
        "trials": int(base.get("trials") or protocol.get("trials")),
        "seed": int(protocol["seed"]),
    }
    for field_, ours in asked.items():
        if ours is not None and int(ours) != out[field_]:
            raise ValueError(
                PROTOCOL_MISMATCH.format(
                    runtime=runtime,
                    name=name,
                    field=field_,
                    theirs=out[field_],
                    ours=ours,
                )
            )
    return out


def certificate_bound(cited: dict[str, Any], name: str) -> tuple[float, float]:
    """The certificate's exact lower bound and its rate, or a refusal by
    name when either is missing (a lower bound of 0 would find no cliff)."""
    missing = [k for k in ("ci95", "successes", "trials") if not cited.get(k)]
    if "ci95" in missing or not cited.get("trials"):
        raise ValueError(NO_INTERVAL.format(name=name, missing=", ".join(missing)))
    return (
        float(cited["ci95"][0]),
        float(cited.get("successes") or 0) / float(cited["trials"]),
    )


# Told as each knob's ladder is climbed: (knobs climbed, knobs in all,
# that knob's name, its rungs) - how a sweep says "knob 3 of 8: latency"
# to the Studio's Running now panel (`mcp_jobs.Tracker.progress`).
KnobProgress = Callable[[int, int, str, "list[Rung]"], None]


def attribute(  # noqa: PLR0913 - the sweep's own knobs, each named
    deployment_dir: Path,
    *,
    assets_dir: Path | None,
    certificate: dict[str, Any] | None,
    runtime: str = DEFAULT_RUNTIME,
    trials: int | None = None,
    seed: int | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    knobs: tuple[Knob, ...] = KNOBS,
    workers: int | None = None,
    open: Opener | None = None,
    appliers: dict[str, Applier] | None = None,
    on_knob: KnobProgress | None = None,
) -> dict[str, Any]:
    """Climb every knob's ladder (in `workers` spawned processes, one per
    knob; in this process when 0, or when a fake `open`/`appliers` is
    injected), rank, write `attribution.json` beside the manifest and
    return it. `trials` and `seed` are the passing gate's own (read from
    its record); a value given that differs is refused by name."""
    name = Path(deployment_dir).name
    manifest, base, cited = require_passing_gate(deployment_dir, runtime, certificate)
    lower, cert_rate = certificate_bound(cited, name)
    protocol = gate_protocol(base, name, runtime, trials=trials, seed=seed)
    trials, seed = protocol["trials"], protocol["seed"]
    sweep = Sweep(
        deployment_dir=Path(deployment_dir),
        assets_dir=assets_dir,
        trials=trials,
        seed=seed,
        certificate_lower=lower,
        runtime=runtime,
    )
    in_process = workers == 0 or open is not None or appliers is not None
    count = 0 if in_process else min(len(knobs), workers or os.cpu_count() or 1)
    baseline = gate_rung(
        sweep, NOMINAL, NOMINAL.ladder[0], open=open, appliers=appliers
    )
    if baseline.ci95[0] < lower:
        raise ValueError(
            BASELINE_BELOW.format(
                trials=trials,
                successes=baseline.successes,
                lo=baseline.ci95[0],
                hi=baseline.ci95[1],
                lower=lower,
            )
        )
    climbed: dict[str, list[Rung]] = {}

    def done(knob_name: str, rungs: list[Rung]) -> None:
        climbed[knob_name] = rungs
        if on_knob is not None:
            on_knob(len(climbed), len(knobs), knob_name, rungs)

    if in_process:
        for k in knobs:
            done(k.name, climb(sweep, k, open=open, appliers=appliers))
    else:
        import multiprocessing  # noqa: PLC0415

        with ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            # as each ladder lands, not in knob order: the panel moves live
            futures = [pool.submit(_climb_in_worker, (sweep, k)) for k in knobs]
            for future in as_completed(futures):
                done(*future.result())
        climbed = {k.name: climbed[k.name] for k in knobs}  # the knobs' own order
    ranking = rank(climbed, lower)
    floor = cert_rate - tolerance
    record: dict[str, Any] = {
        "schema": ATTRIBUTION_SCHEMA,
        "deployment": manifest.raw.get(Key.STAMP_OF),
        "policy": manifest.raw.get(Key.POLICY),
        "runtime": runtime,
        "base_gate": {
            "successes": base.get("successes"),
            "trials": base.get("trials"),
            "ci95": base.get("ci95"),
            "judged": base.get("judged"),
        },
        "certificate": {
            "stamp": manifest.raw.get(Key.CERTIFICATE),
            "successes": cited.get("successes"),
            "trials": cited.get("trials"),
            "ci95": cited.get("ci95"),
            "lower": lower,
        },
        "protocol": {
            "trials": trials,
            "seed": seed,
            DRAW_KEY: DRAW_NOW,
            "tolerance": tolerance,
            "rule": CLIFF_RULE,
            "workers": count,
        },
        "baseline": baseline.row(floor),
        "knobs": [_knob_record(k, climbed[k.name], lower, floor) for k in knobs],
        "ranking": [
            {"name": name, "cliff": None if c is None else c.level}
            for name, c in ranking
        ],
        "sensitivity": sensitivity_line(ranking, knobs),
        "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    write_attribution(deployment_dir, record)
    return record


def write_attribution(deployment_dir: Path, record: dict[str, Any]) -> Path:
    """The record beside the manifest, atomically (a half-written file
    once read would fail the whole project's index)."""
    out = Path(deployment_dir) / ATTRIBUTION_FILE
    staging = out.with_suffix(".json.tmp")
    staging.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    staging.replace(out)
    return out


def _knob_record(
    knob_: Knob, rungs: list[Rung], lower: float, floor: float
) -> dict[str, Any]:
    cliff = cliff_of(rungs, lower)
    return {
        "name": knob_.name,
        "unit": knob_.unit,
        "describe": knob_.describe,
        "source": knob_.source,
        "ladder": list(knob_.ladder),
        "rungs": [r.row(floor) for r in rungs],
        "cliff": (
            None
            if cliff is None
            else {"level": cliff.level, "rung": rungs.index(cliff) + 1}
        ),
    }


def read_attribution(folder: Path) -> dict[str, Any] | None:
    path = Path(folder) / ATTRIBUTION_FILE
    if not path.is_file():
        return None
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != ATTRIBUTION_SCHEMA:
        raise ValueError(
            f"{path}: schema {raw.get('schema')!r}, this reader speaks "
            f"{ATTRIBUTION_SCHEMA!r}"
        )
    return raw


# -- seen ------------------------------------------------------------------------

STREAM = "attribution"
UNRENDERED = "unrendered: {why}"


def still_at_cliff(
    deployment_dir: Path,
    record: dict[str, Any],
    *,
    assets_dir: Path | None,
    open: Opener | None = None,
    appliers: dict[str, Applier] | None = None,
) -> dict[str, Any]:
    """One picture of the fall: the top-ranked knob at its cliff, the
    sweep's own commands replayed until a trial fails, the frame at the
    fall (or the last frame of the first untracked trial) saved beside
    the record as `attribution-cliff.png`. Honest when nothing fell or
    nothing can render: the reason, no file. The runtime and the renderer
    are closed on every way out."""
    fallen = [r for r in record.get("ranking", []) if r.get("cliff") is not None]
    if not fallen:
        return {"unrendered": UNRENDERED.format(why=SURVIVED)}
    name, level = str(fallen[0]["name"]), float(fallen[0]["cliff"])
    protocol = record.get("protocol") or {}
    trials, seed = int(protocol.get("trials", 0)), int(protocol.get("seed", 0))
    try:
        from rq_pipeline.deploy.ticks import Ticks  # noqa: PLC0415
    except ImportError as missing:
        return {"unrendered": UNRENDERED.format(why=str(missing))}
    manifest = load_manifest(deployment_dir)
    opener = open if open is not None else runtime_spec(DEFAULT_RUNTIME).open()
    turned = (appliers or APPLIERS)[name](
        opener(manifest, assets_dir=assets_dir), level, seed
    )
    out = Path(deployment_dir) / STILL_FILE
    try:
        with stills.renderer(turned.model, stills.CLIFF_CAMERA) as render:
            first_untracked: tuple[int, int] | None = None
            for i, command in enumerate(draw_commands(manifest, trials, seed)):
                turned.reset()
                meter = Ticks(manifest.control.step_dt)
                for tick in range(manifest.control.episode_ticks):
                    if meter.tick(turned, command):
                        render(turned.data, turned.base_position(), out)
                        return {
                            "file": STILL_FILE,
                            "knob": name,
                            "level": level,
                            "trial": i,
                            "tick": tick,
                            "fell": True,
                        }
                if (
                    first_untracked is None
                    and not TrackingOutcome(**meter.outcome()).tracked
                ):
                    first_untracked = (i, meter.steps)
                    render(turned.data, turned.base_position(), out)
        if first_untracked is not None:
            return {
                "file": STILL_FILE,
                "knob": name,
                "level": level,
                "trial": first_untracked[0],
                "tick": first_untracked[1],
                "fell": False,
            }
        return {"unrendered": UNRENDERED.format(why="no trial failed on replay")}
    except (RuntimeError, OSError, ValueError, ImportError) as why:
        return {"unrendered": UNRENDERED.format(why=f"no renderer: {why}")}
    finally:
        close = getattr(turned, "close", None)
        if close is not None:
            close()


# The series of one knob's ladder, spelled once (the stream here and the
# presenter's replay both write them).
SERIES = ("rate", "lower", "upper", "certificate_lower")


def ladder_series(rung: dict[str, Any], certificate_lower: float) -> dict[str, float]:
    """One rung as the series values `SERIES` names."""
    return dict(
        zip(
            SERIES,
            (
                float(rung["rate"]),
                float(rung["ci95"][0]),
                float(rung["ci95"][1]),
                certificate_lower,
            ),
            strict=True,
        )
    )


def stream(record: dict[str, Any], folder: Path, *, rr: Any = None) -> Path:
    """The record as the Studio sees it: per knob, the tracked rate and
    its interval up the ladder on a `rung` timeline, the certificate's
    lower bound beside them, the ranking as a document. Saved as the
    deployment's `attribution` stream; sent to a listening Studio."""
    from rq_pipeline.viz import open_stream, viewer_file  # noqa: PLC0415

    file = viewer_file(folder, STREAM)
    file.parent.mkdir(parents=True, exist_ok=True)
    name = Path(folder).name
    rr = open_stream(f"rq-{STREAM}-{name}", file=file, rr=rr, on_term=False)
    lower = float((record.get("certificate") or {}).get("lower", 0.0))
    for entry in record.get("knobs", []):
        knob_name = entry["name"]
        for i, rung in enumerate(entry.get("rungs", []), start=1):
            rr.set_time("rung", sequence=i)
            for series, value in ladder_series(rung, lower).items():
                rr.log(f"{STREAM}/{knob_name}/{series}", rr.Scalars(value))
    units = {k["name"]: k.get("unit", "") for k in record.get("knobs", [])}
    lines = [f"# {name}: what would break it first", "", record.get("sensitivity", "")]
    for r in record.get("ranking", []):
        cliff = r.get("cliff")
        lines.append(
            f"- {r['name']}: "
            + (
                SURVIVED
                if cliff is None
                else f"cliff at {cliff:g} {units.get(r['name'], '')}".rstrip()
            )
        )
    rr.log(f"{STREAM}/ranking", rr.TextDocument("\n".join(lines)), static=True)
    return file
