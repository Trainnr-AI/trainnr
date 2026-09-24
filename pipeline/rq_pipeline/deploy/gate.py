"""The sim-to-sim gate (docs/76 A6): the exported policy, driven only
through its manifest by a named runtime (`deploy.runtimes`), judged the
way the evaluation judged the torch policy — survived the episode and
tracked the commanded velocity (`evaluate.tracking`) — over seeded held
commands; the exact interval is compared with the evaluation the
deployment cites, within a stated tolerance. Each runtime's record
lands beside the manifest under its own name.

A deployment staged on a captured scene (`scenes.stage`) is judged along
the scene's course instead (`deploy.course`): the manifest chooses, and
the record's protocol block names every field that differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.deploy.course import (
    COMMANDS_ALONG,
    Course,
    Steering,
    course_criterion_text,
    draw_speeds,
    run_course_trial,
)
from rq_pipeline.deploy.manifest import GATE_SCHEMA, Key, Manifest, load_manifest
from rq_pipeline.deploy.mirror import GateMirror
from rq_pipeline.deploy.runtimes import (
    DEFAULT_RUNTIME,
    GateRuntime,
    Opener,
    runtime_spec,
)
from rq_pipeline.deploy.ticks import Ticks
from rq_pipeline.evaluate.tracking import (
    ERR_FLOOR_MPS,
    ERR_RATIO_BOUND,
    TrackingOutcome,
    criterion_text,
)
from rq_pipeline.stats.intervals import clopper_pearson
from rq_pipeline.viz import viewer_file

# The gate passes when the exported policy's rate is within this much of
# the evaluation's; a stated number, never a hidden one.
DEFAULT_TOLERANCE = 0.10
DEFAULT_TRIALS = 20
DEFAULT_SEED = 1000
CI_DIGITS = 4
COMMANDS_DRAWN = "held per episode, drawn in the manifest's twist ranges"


@dataclass(frozen=True)
class Trial(TrackingOutcome):
    """One held command's episode, judged under the shared rule."""

    command: list[float]


# The saved stream of a gate, per runtime, inside its deployment.
GATE_STREAM = "gate"


def run_trial(  # noqa: PLR0913 - the trial's own knobs, each named
    manifest: Manifest,
    runtime: GateRuntime,
    command: np.ndarray,
    *,
    mirror: GateMirror | None = None,
    index: int = 0,
    contacts: list[np.ndarray] | None = None,
) -> Trial:
    """One episode at a held command, the manifest's length and rate;
    with a `mirror`, every tick's pose goes to the Studio; with a
    `contacts` list, every tick's contact points are appended to it."""
    runtime.reset()
    meter = Ticks(manifest.control.step_dt, mirror=mirror, contacts=contacts)
    if mirror is not None:
        mirror.trial(index, command)
    for _ in range(manifest.control.episode_ticks):
        if meter.tick(runtime, command):
            break
    return Trial(command=[float(c) for c in command], **meter.outcome())


# How the gate draws its trials, versioned in every record's protocol (the
# `draw` field). Version 1 drew each axis as one vector the length of the
# trial count, so trial i of a 2-trial gate was not trial i of a 20-trial
# one; version 2 draws each trial from (seed, trial), so trial i is the
# same command at any count and a gate is paired with its attribution and
# its pre-flight by index (the review of 2026-09-24). A record without
# the field was written before it: version 1, never reinterpreted.
DRAW_KEY = "draw"
DRAW_BY_COUNT = "by-count/1"
DRAW_PER_TRIAL = "per-trial/2"
DRAW_NOW = DRAW_PER_TRIAL
MIXED_DRAWS = (
    "{what} was drawn {theirs}, this code draws {ours}: trial i is not the same "
    "command under both; re-run the gate (gate_deployment) before comparing"
)


def trial_rng(seed: int, trial: int) -> np.random.Generator:
    """The generator of one trial: its own stream of (seed, trial), the
    same whatever the trial count."""
    return np.random.default_rng([int(seed), int(trial)])


def draw_commands(manifest: Manifest, trials: int, seed: int) -> np.ndarray:
    """Seeded held commands inside the manifest's ranges (heading off),
    one per trial from (seed, trial): the first k are the same at any
    count. A manifest without ranges was refused by the loader."""
    commands = manifest.commands
    lo_hi = np.array(
        (commands.lin_vel_x, commands.lin_vel_y, commands.ang_vel_z), dtype=np.float64
    )
    # one generator per trial, every axis from it (a generator per axis
    # drew the same number three times: a trial's axes moved together)
    return np.array(
        [trial_rng(seed, i).uniform(lo_hi[:, 0], lo_hi[:, 1]) for i in range(trials)],
        dtype=np.float64,
    ).reshape(trials, len(lo_hi))


def draw_of(record: dict[str, Any]) -> str:
    """The draw a gate record was made under; one written before the
    field is the count-dependent draw."""
    return str((record.get("protocol") or {}).get(DRAW_KEY, DRAW_BY_COUNT))


def require_same_draw(record: dict[str, Any], what: str) -> None:
    """Refuse, by name, a record whose trials this code would not redraw."""
    theirs = draw_of(record)
    if theirs != DRAW_NOW:
        raise ValueError(MIXED_DRAWS.format(what=what, theirs=theirs, ours=DRAW_NOW))


def hold_twists(  # noqa: PLR0913, PLR0917 - the gate's shape, positional inside the gate
    manifest: Manifest,
    driver: GateRuntime,
    protocol: dict[str, Any],
    trials: int,
    seed: int,
    mirror: GateMirror | None,
    contacts: list[np.ndarray],
) -> list[TrackingOutcome]:
    """The plane's protocol: seeded held twists, the evaluation's own.
    Public because the attribution sweep (`deploy.attribution`) drives a
    turned runtime through the very same trials."""
    commands = draw_commands(manifest, trials, seed)
    protocol["commands"] = COMMANDS_DRAWN
    limit = driver.command_limit
    if limit is not None:  # a gamepad's sticks stop at 1.0: say so, and clip
        commands = np.clip(commands, -float(limit), float(limit))
        protocol["command_limit"] = float(limit)
        protocol["commands"] = (
            f"{COMMANDS_DRAWN}, clipped to ±{limit:g} (the runtime's envelope)"
        )
    return [
        run_trial(manifest, driver, c, mirror=mirror, index=i, contacts=contacts)
        for i, c in enumerate(commands)
    ]


def _walk_course(  # noqa: PLR0913, PLR0917 - the gate's shape, positional inside the gate
    manifest: Manifest,
    driver: GateRuntime,
    course: Course,
    protocol: dict[str, Any],
    trials: int,
    seed: int,
    mirror: GateMirror | None,
    contacts: list[np.ndarray],
) -> list[TrackingOutcome]:
    """A staged scene's protocol: along its course (`deploy.course`),
    judged by arrival; the record says so in every field that differs."""
    limit = driver.command_limit
    steering = Steering.of_manifest(manifest, limit=limit)
    speeds = draw_speeds(manifest, trials, seed, limit=limit)
    protocol["commands"] = COMMANDS_ALONG
    protocol["criterion"] = course_criterion_text()
    protocol["course"] = course.describe()
    protocol["steer"] = steering.describe()
    if limit is not None:
        protocol["command_limit"] = float(limit)
    if mirror is not None:
        mirror.course(course.path)
    return [
        run_course_trial(
            manifest,
            driver,
            course,
            steering,
            float(v),
            mirror=mirror,
            index=i,
            contacts=contacts,
        )
        for i, v in enumerate(speeds)
    ]


def gate(  # noqa: PLR0913 - the gate's own knobs, each named
    deployment_dir: Path,
    *,
    assets_dir: Path | None,
    runtime: str = DEFAULT_RUNTIME,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    tolerance: float = DEFAULT_TOLERANCE,
    certificate: dict[str, Any] | None = None,
    open: Opener | None = None,
    narrate: bool = False,
    scene_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the gate under the named runtime and write its record beside
    the manifest; returns the record. `open` replaces the registry's
    opener (a fake runtime under test); the judge, the draw, the interval
    and the tolerance rule are the same whatever drives the policy.
    `narrate` mirrors every trial into the Studio (`deploy/mirror.py`),
    over the captured scene's splat when the deployment stands on one
    (`scene_dir`, a staged deployment; `scenes.stage`)."""
    spec = runtime_spec(runtime)
    manifest = load_manifest(deployment_dir)
    opener = open if open is not None else spec.open()
    driver = opener(manifest, assets_dir=assets_dir)
    try:
        mirror = (
            GateMirror.open(
                manifest,
                spec.name,
                # The gate's picture, saved inside the deployment (docs/76 §10.5).
                file=viewer_file(deployment_dir, f"{GATE_STREAM}-{spec.name}"),
                scene_dir=scene_dir,
                name=deployment_dir.name,
            )
            if narrate
            else None
        )
        protocol: dict[str, Any] = {
            "trials": trials,
            "seed": seed,
            DRAW_KEY: DRAW_NOW,
            "criterion": criterion_text(),
            "err_ratio_bound": ERR_RATIO_BOUND,
            "err_floor_mps": ERR_FLOOR_MPS,
            "runtime": spec.description,
            "instrument": driver.instrument,
        }
        contacts: list[np.ndarray] = []
        course = Course.of_manifest(manifest)
        results: list[TrackingOutcome] = (
            _walk_course(
                manifest, driver, course, protocol, trials, seed, mirror, contacts
            )
            if course is not None
            else hold_twists(manifest, driver, protocol, trials, seed, mirror, contacts)
        )
        k = sum(t.success for t in results)
        lo, hi = clopper_pearson(k, trials)
        record: dict[str, Any] = {
            "schema": GATE_SCHEMA,
            "runtime": spec.name,
            "deployment": manifest.raw.get(Key.STAMP_OF),
            "policy": manifest.raw.get(Key.POLICY),
            "protocol": protocol,
            "successes": k,
            "trials": trials,
            "ci95": [round(lo, CI_DIGITS), round(hi, CI_DIGITS)],
            "records": [t.row() for t in results],
            "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        record["verdict"] = _verdict(
            k, trials, tolerance, certificate, along_course=course is not None
        )
        record["contacts"] = _contacts_record(
            deployment_dir,
            spec.name,
            contacts,
            scene_dir,
            seen=driver.contact_points() is not None,
        )
        if certificate:
            record["certificate"] = {
                "stamp": manifest.raw.get(Key.CERTIFICATE),
                "successes": certificate.get("successes"),
                "trials": certificate.get("trials"),
                "ci95": certificate.get("ci95"),
            }
        staging = Path(deployment_dir) / (spec.record_file + ".tmp")
        staging.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        staging.replace(Path(deployment_dir) / spec.record_file)
        return record
    finally:
        # a runtime that holds a pad or a bus lets go (the DDS one recentres its sticks)
        close = getattr(driver, "close", None)
        if close is not None:
            close()


OTHER_PROTOCOL = (
    "the cited evaluation judged held twists on the trained plane; a course gate "
    "on a captured scene is another protocol, so it reports its rate and judges "
    "nothing against it"
)


def _verdict(
    k: int,
    trials: int,
    tolerance: float,
    certificate: dict[str, Any] | None,
    *,
    along_course: bool = False,
) -> dict[str, Any]:
    """The tolerance rule against the cited evaluation, or a report that
    judges nothing when none was cited - or when the gate walked a
    course the evaluation never did."""
    if not certificate:
        return {
            "passed": None,
            "rule": "no evaluation cited: the gate reports its rate and judges nothing",
        }
    if along_course:
        return {"passed": None, "rule": OTHER_PROTOCOL}
    ck, cn = certificate.get("successes"), certificate.get("trials")
    cert_rate = (ck / cn) if (ck is not None and cn) else None
    rate = k / trials
    return {
        "passed": cert_rate is not None and rate >= cert_rate - tolerance,
        "tolerance": tolerance,
        "rule": "the gate's success rate is at least the evaluation's minus the "
        "tolerance",
        "gate_rate": rate,
        "certificate_rate": cert_rate,
    }


# Where the task touched: the sites' file beside the record, and the
# scene's gap within this radius of them (docs/78 §4.1).
CONTACTS_FILE = "contacts-{runtime}.npy"
CONTACT_SITE_RADIUS_M = 0.10
CONTACTS_UNSEEN = "unrecorded: this runtime cannot see its contacts"


def _contacts_record(
    deployment_dir: Path,
    runtime_name: str,
    contacts: list[np.ndarray],
    scene_dir: Path | None,
    *,
    seen: bool,
) -> dict[str, Any]:
    """The contact sites saved as a point file and, on a captured scene,
    the gap measured at them; honest when there were none to see."""
    if not seen:
        return {"points": CONTACTS_UNSEEN}
    points = (
        np.concatenate(contacts, axis=0) if contacts else np.zeros((0, 3), np.float64)
    )
    name = CONTACTS_FILE.format(runtime=runtime_name)
    np.save(Path(deployment_dir) / name, points.astype(np.float32))
    out: dict[str, Any] = {"file": name, "points": int(points.shape[0])}
    if scene_dir is None or not points.shape[0]:
        return out
    from dataclasses import asdict  # noqa: PLC0415

    from rq_pipeline.scenes import gap as gap_audit  # noqa: PLC0415
    from rq_pipeline.scenes.record import PROXY_FILE, SPLAT_FILE  # noqa: PLC0415
    from rq_pipeline.scenes.splat import read_ply  # noqa: PLC0415

    try:
        site_gap = gap_audit.measure(
            read_ply(Path(scene_dir) / SPLAT_FILE),
            Path(scene_dir) / PROXY_FILE,
            sites=points,
            radius_m=CONTACT_SITE_RADIUS_M,
        )
    except ImportError as missing:
        site_gap = gap_audit.unmeasured(str(missing))
    out["site_gap"] = asdict(site_gap) | {"radius_m": CONTACT_SITE_RADIUS_M}
    return out
