"""A fitted interval for a BAM actuator, from Rhoban's public bench logs.

    python3 tools/bam-bootstrap.py --bam <clone of Rhoban/bam at v1.0.2> \\
        --processed <dir of bam.process output> --actuator xl330 --model m6 \\
        --out <dir> [--replicates 100] [--trials 5000] [--workers 8] [--seed 7]

BAM publishes point fits and its raw logs (docs/e2e-research/72). This
tool turns the logs into an interval: a BLOCK bootstrap over the bench
design — the logs are grouped by (kp, mass, length), 60 blocks of 5-6
trajectories for the XL330, and each replicate resamples the files
WITHIN every block with replacement, so every replicate sees the whole
design — then one `bam.fit` per replicate (their optimiser, their
objective, our trial budget), and the 2.5th-97.5th percentile of each
parameter across replicates. Every replicate's parameter vector and
the fit's own MAE are kept; the summary carries the zip's sha256, BAM's
commit, the trial budget and the sampler, so the interval is a record,
not a number.

Resumable: a replicate whose params file exists is not refit. The
interval measures log-sampling variability on Rhoban's single unit and
bench, not unit-to-unit or temperature spread — the summary says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

BLOCK_KEYS = ("kp", "mass", "length")
LOW_PCT, HIGH_PCT = 2.5, 97.5
# The header fields bam.process keeps on each log; the design key.
SUMMARY = "bootstrap.json"
EMPTY_JSON = 2  # bytes: bam.fit writes "{}" first and fills it at the end


def blocks(processed: Path) -> dict[tuple[Any, ...], list[Path]]:
    grouped: dict[tuple[Any, ...], list[Path]] = {}
    for path in sorted(processed.glob("*.json")):
        header = json.loads(path.read_text())
        key = tuple(header.get(k) for k in BLOCK_KEYS)
        grouped.setdefault(key, []).append(path)
    if not grouped:
        raise FileNotFoundError(f"no processed logs under {processed}")
    return grouped


def resample(
    design: dict[tuple[Any, ...], list[Path]], rng: random.Random, into: Path
) -> int:
    """One replicate's log directory: within each block, draw as many
    files as the block holds, with replacement; duplicates get distinct
    names so bam's glob sees them all."""
    into.mkdir(parents=True, exist_ok=True)
    n = 0
    for _key, files in sorted(design.items()):
        for i in range(len(files)):
            src = rng.choice(files)
            shutil.copyfile(src, into / f"{src.stem}__b{i}.json")
            n += 1
    return n


def fit(  # noqa: PLR0913 - the fit's knobs, named
    bam: Path, logdir: Path, out: Path, *, actuator: str, model: str, trials: int
) -> None:
    python = bam / ".venv" / "bin" / "python"
    command = [
        str(python),
        "-m",
        "bam.fit",
        "--actuator",
        actuator,
        "--model",
        model,
        "--logdir",
        str(logdir),
        "--output",
        str(out),
        "--trials",
        str(trials),
        "--workers",
        "1",
    ]
    with (out.with_suffix(".log")).open("w") as log:
        subprocess.run(
            command, cwd=bam, check=True, stdout=log, stderr=subprocess.STDOUT
        )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(
    fits: list[dict[str, Any]], *, point: dict[str, Any] | None
) -> dict[str, Any]:
    names = [k for k, v in fits[0].items() if isinstance(v, float)]
    interval: dict[str, Any] = {}
    for name in names:
        values = [float(f[name]) for f in fits]
        interval[name] = {
            "low": percentile(values, LOW_PCT),
            "high": percentile(values, HIGH_PCT),
            "median": statistics.median(values),
            "point": None if point is None else point.get(name),
        }
    return interval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--raw-zip", type=Path, default=None, help="for its sha256")
    parser.add_argument("--actuator", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    design = blocks(args.processed)
    sizes = sorted({len(v) for v in design.values()})
    print(f"{len(design)} blocks by {BLOCK_KEYS}, {sizes} files per block")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    jobs = []
    for k in range(args.replicates):
        rep_rng = random.Random(rng.random())
        logdir = args.out / f"rep{k:03d}" / "logs"
        params = args.out / f"rep{k:03d}" / "params.json"
        if params.is_file() and params.stat().st_size > EMPTY_JSON:
            continue
        if not logdir.is_dir():
            resample(design, rep_rng, logdir)
        jobs.append((logdir, params))
    print(f"{args.replicates - len(jobs)} replicates already fit; {len(jobs)} to run")

    def run(job: tuple[Path, Path]) -> None:
        logdir, params = job
        fit(
            args.bam,
            logdir,
            params,
            actuator=args.actuator,
            model=args.model,
            trials=args.trials,
        )
        print(f"fit -> {params}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, jobs))

    fits = []
    for k in range(args.replicates):
        params = args.out / f"rep{k:03d}" / "params.json"
        if params.is_file():
            data = json.loads(params.read_text())
            if data:
                fits.append(data)
    shipped = args.bam / "bam" / "params" / args.actuator / f"{args.model}.json"
    point = json.loads(shipped.read_text()) if shipped.is_file() else None
    bam_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=args.bam,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    summary = {
        "actuator": args.actuator,
        "model": args.model,
        "replicates": len(fits),
        "trials_per_fit": args.trials,
        "sampler": "optuna CmaEsSampler(restart_strategy='bipop'), bam.fit defaults",
        "bootstrap": (
            f"block bootstrap within {BLOCK_KEYS}; {len(design)} blocks; "
            f"seed {args.seed}"
        ),
        "bam_commit": bam_commit,
        "raw_zip_sha256": sha256(args.raw_zip) if args.raw_zip else None,
        "processed_logs": len(list(args.processed.glob("*.json"))),
        "interval": summarize(fits, point=point),
        "caveats": [
            "The interval is log-sampling variability on Rhoban's single unit and "
            "bench (their rig's q_offset and command_delay included), not "
            "unit-to-unit or temperature spread.",
            "Each replicate is one optimiser run at the stated trial budget; optimiser "
            "variance is inside the interval, not separated from it.",
            "The data licence is unstated on the HF bucket; the logs are not "
            "redistributed here, only the fitted numbers.",
        ],
    }
    (args.out / SUMMARY).write_text(json.dumps(summary, indent=1))
    for name, iv in summary["interval"].items():
        pt = "" if iv["point"] is None else f"  shipped {iv['point']:.5g}"
        print(
            f"{name:32s} [{iv['low']:.5g}, {iv['high']:.5g}]  "
            f"median {iv['median']:.5g}{pt}"
        )
    print(f"summary -> {args.out / SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
