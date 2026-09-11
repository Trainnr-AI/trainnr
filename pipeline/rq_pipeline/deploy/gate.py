"""The sim-to-sim gate (docs/76 A6): the exported policy, driven only
through its manifest by the plain-MuJoCo runtime, judged the way the
certificate judged the torch policy — survived the episode and tracked
the commanded velocity — over seeded held commands; the exact interval
is compared with the certificate the deployment cites, within a stated
tolerance. `gate.json` lands beside the manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.deploy.manifest import GATE_FILE, GATE_SCHEMA, Manifest, load_manifest
from rq_pipeline.deploy.runtime import open_runtime
from rq_pipeline.stats.intervals import clopper_pearson

# The certificate's criterion (rq_mjlab.walk_verdict, read 2026-09-11):
# survived, and the tracking error closed at least half the gap standing
# still would leave, the denominator floored at a slow command.
ERR_RATIO_BOUND = 0.5
ERR_FLOOR_MPS = 0.1
# The gate passes when the exported policy's rate is within this much of
# the certificate's; a stated number, never a hidden one.
DEFAULT_TOLERANCE = 0.10
DEFAULT_TRIALS = 20


@dataclass(frozen=True)
class Trial:
    command: list[float]
    steps: int
    fell: bool
    mean_err: float
    mean_cmd: float

    @property
    def err_ratio(self) -> float:
        return self.mean_err / max(self.mean_cmd, ERR_FLOOR_MPS)

    @property
    def success(self) -> bool:
        return (not self.fell) and self.err_ratio < ERR_RATIO_BOUND


def run_trial(manifest: Manifest, runtime: Any, command: np.ndarray) -> Trial:
    """One episode at a held command, the manifest's length and rate."""
    control = manifest.control
    ticks = round(float(control["episode_length_s"]) * float(control["control_hz"]))
    runtime.reset()
    runtime.command = command.astype(np.float32)
    err_sum = cmd_sum = 0.0
    fell = False
    steps = 0
    for _ in range(ticks):
        obs = runtime.observe()
        runtime.apply(runtime.act(obs))
        v = runtime.base_velocity_b()
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
    """Seeded held commands inside the manifest's ranges (heading off)."""
    ranges = manifest.raw.get("commands", {}).get("twist", {})
    rng = np.random.default_rng(seed)
    lo_hi = [
        ranges.get(k, [-0.5, 0.5]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    ]
    return np.stack([rng.uniform(lo, hi, size=trials) for lo, hi in lo_hi], axis=1)


def gate(  # noqa: PLR0913 - the gate's own knobs, each named
    deployment_dir: Path,
    *,
    assets_dir: Path,
    trials: int = DEFAULT_TRIALS,
    seed: int = 1000,
    tolerance: float = DEFAULT_TOLERANCE,
    certificate: dict[str, Any] | None = None,
    open: Any = open_runtime,
) -> dict[str, Any]:
    """Run the gate and write `gate.json`; returns the record. `open`
    builds the runtime the trials drive - plain MuJoCo by default; any
    object answering the same seven calls (reset, observe, act, apply,
    base_velocity_b, fell_over, command) judges under the same rule,
    so Unitree's own simulator and controller are one argument away."""
    manifest = load_manifest(deployment_dir)
    runtime = open(manifest, assets_dir=assets_dir)
    commands = draw_commands(manifest, trials, seed)
    results = [run_trial(manifest, runtime, c) for c in commands]
    k = sum(t.success for t in results)
    lo, hi = clopper_pearson(k, trials)
    record: dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "deployment": manifest.raw.get("stamp_of"),
        "policy": manifest.raw.get("policy"),
        "protocol": {
            "trials": trials,
            "seed": seed,
            "commands": "held per episode, drawn in the manifest's twist ranges",
            "criterion": f"survived and err_ratio<{ERR_RATIO_BOUND}",
            "err_floor_mps": ERR_FLOOR_MPS,
            "runtime": "plain MuJoCo + onnxruntime, driven by the manifest alone",
            "instrument": f"mujoco-{_mujoco_version()}",
        },
        "successes": k,
        "trials": trials,
        "ci95": [round(lo, 4), round(hi, 4)],
        "records": [
            asdict(t) | {"err_ratio": t.err_ratio, "success": t.success}
            for t in results
        ],
        "judged": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if certificate:
        ck, cn = certificate.get("successes"), certificate.get("trials")
        cert_rate = (ck / cn) if (ck is not None and cn) else None
        record["certificate"] = {
            "stamp": manifest.raw.get("certificate"),
            "successes": ck,
            "trials": cn,
            "ci95": certificate.get("ci95"),
        }
        rate = k / trials
        passed = cert_rate is not None and rate >= cert_rate - tolerance
        record["verdict"] = {
            "passed": passed,
            "tolerance": tolerance,
            "rule": (
                "the gate's success rate is at least the certificate's minus the "
                "tolerance"
            ),
            "gate_rate": rate,
            "certificate_rate": cert_rate,
        }
    else:
        record["verdict"] = {
            "passed": None,
            "rule": (
                "no certificate cited: the gate reports its rate and judges nothing"
            ),
        }
    staging = Path(deployment_dir) / (GATE_FILE + ".tmp")
    staging.write_text(json.dumps(record, indent=1) + "\n")
    staging.replace(Path(deployment_dir) / GATE_FILE)
    return record


def _mujoco_version() -> str:
    import mujoco  # noqa: PLC0415

    return mujoco.__version__
