"""The sim-to-sim gate (docs/76 A6): the exported policy, driven only
through its manifest by a named runtime (`deploy.runtimes`), judged the
way the evaluation judged the torch policy — survived the episode and
tracked the commanded velocity (`evaluate.tracking`) — over seeded held
commands; the exact interval is compared with the evaluation the
deployment cites, within a stated tolerance. Each runtime's record
lands beside the manifest under its own name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.deploy.manifest import GATE_SCHEMA, Key, Manifest, load_manifest
from rq_pipeline.deploy.mirror import GateMirror
from rq_pipeline.deploy.runtimes import (
    DEFAULT_RUNTIME,
    GateRuntime,
    Opener,
    runtime_spec,
)
from rq_pipeline.evaluate.tracking import (
    ERR_FLOOR_MPS,
    ERR_RATIO_BOUND,
    TrackingOutcome,
    criterion_text,
)
from rq_pipeline.stats.intervals import clopper_pearson

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


def run_trial(
    manifest: Manifest,
    runtime: GateRuntime,
    command: np.ndarray,
    *,
    mirror: GateMirror | None = None,
    index: int = 0,
) -> Trial:
    """One episode at a held command, the manifest's length and rate;
    with a `mirror`, every tick's pose goes to the Studio."""
    runtime.reset()
    runtime.command = command.astype(np.float32)
    err_sum = cmd_sum = 0.0
    fell = False
    steps = 0
    pose = getattr(runtime, "pose", None) if mirror is not None else None
    if mirror is not None:
        mirror.trial(index, command)
    for _ in range(manifest.control.episode_ticks):
        obs = runtime.observe()
        runtime.apply(runtime.act(obs))
        v = runtime.base_velocity_b()
        if pose is not None:
            mirror.tick(manifest.control.step_dt, pose(), command, v)
        err_sum += float(np.linalg.norm(v[:2] - command[:2]))
        cmd_sum += float(np.linalg.norm(command[:2]))
        steps += 1
        if runtime.fell_over():
            fell = True
            break
    return Trial(
        command=[float(c) for c in command],
        steps=steps,
        fell=fell,
        mean_err=err_sum / max(steps, 1),
        mean_cmd=cmd_sum / max(steps, 1),
    )


def draw_commands(manifest: Manifest, trials: int, seed: int) -> np.ndarray:
    """Seeded held commands inside the manifest's ranges (heading off);
    a manifest without ranges was refused by the loader."""
    commands = manifest.commands
    rng = np.random.default_rng(seed)
    lo_hi = (commands.lin_vel_x, commands.lin_vel_y, commands.ang_vel_z)
    return np.stack([rng.uniform(lo, hi, size=trials) for lo, hi in lo_hi], axis=1)


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
) -> dict[str, Any]:
    """Run the gate under the named runtime and write its record beside
    the manifest; returns the record. `open` replaces the registry's
    opener (a fake runtime under test); the judge, the draw, the interval
    and the tolerance rule are the same whatever drives the policy.
    `narrate` mirrors every trial into the Studio (`deploy/mirror.py`)."""
    spec = runtime_spec(runtime)
    manifest = load_manifest(deployment_dir)
    opener = open if open is not None else spec.open()
    driver = opener(manifest, assets_dir=assets_dir)
    mirror = GateMirror.open(manifest, spec.name) if narrate else None
    commands = draw_commands(manifest, trials, seed)
    protocol: dict[str, Any] = {
        "trials": trials,
        "seed": seed,
        "commands": COMMANDS_DRAWN,
        "criterion": criterion_text(),
        "err_ratio_bound": ERR_RATIO_BOUND,
        "err_floor_mps": ERR_FLOOR_MPS,
        "runtime": spec.description,
        "instrument": driver.instrument,
    }
    limit = driver.command_limit
    if limit is not None:  # a gamepad's sticks stop at 1.0: say so, and clip
        commands = np.clip(commands, -float(limit), float(limit))
        protocol["command_limit"] = float(limit)
        protocol["commands"] = (
            f"{COMMANDS_DRAWN}, clipped to ±{limit:g} (the runtime's envelope)"
        )
    results = [
        run_trial(manifest, driver, c, mirror=mirror, index=i)
        for i, c in enumerate(commands)
    ]
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
        "records": [
            asdict(t) | {"err_ratio": t.err_ratio, "success": t.success}
            for t in results
        ],
        "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    record["verdict"] = _verdict(k, trials, tolerance, certificate)
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


def _verdict(
    k: int, trials: int, tolerance: float, certificate: dict[str, Any] | None
) -> dict[str, Any]:
    """The tolerance rule against the cited evaluation, or a report that
    judges nothing when none was cited."""
    if not certificate:
        return {
            "passed": None,
            "rule": "no evaluation cited: the gate reports its rate and judges nothing",
        }
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
