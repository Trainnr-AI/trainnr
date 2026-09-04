"""Fold the walk C1 arms' certificates into one finding and its figure.

    python3 tools/walk-c1-fold.py <study_root> [--arms point,narrow,wide]

Each arm (tools/walk-c1-arm.sh) leaves two certificates under
`<root>/<arm>/train/verdict/`: `walk-verdict-at-fit-cuda.json` (judged
at the bundle's point fit, the shared world) and `walk-verdict-cuda.json`
(judged under law DR drawn from ±0.10 around the fit — walk_verdict's
LAW_DR_SPAN, the SAME span for every arm, not each arm's own; it was
called "own DR" here until 2026-09-04). The at-fit certificates are the
study's arms; the ±0.10 ones ride along under `under_span_0.10`. Comparisons are
paired on the certificate's shared seed and folded with the same
`main_effect` every study uses; the finding lands in `docs/findings/`
with its figure, like every other study.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from _lab import REPO, bootstrap

bootstrap()

from rq_pipeline.evaluate.figures import render  # noqa: E402
from rq_pipeline.evaluate.findings import Finding, repo_commit, today  # noqa: E402
from rq_pipeline.evaluate.records import read_records  # noqa: E402
from rq_pipeline.stats.effects import main_effect  # noqa: E402

AT_FIT = "walk-verdict-at-fit-cuda.json"
OWN_DR = "walk-verdict-cuda.json"
AT_FIT_RECORDS = "records-at-fit-cuda.jsonl"
DEFAULT_ARMS = ("point", "narrow", "wide")
COMPARE = (("wide", "narrow"), ("point", "narrow"), ("wide", "point"))
PROTOCOL = "docs/66 §8 (C1 on the walk: tools/walk-c1-arm.sh, judged at the fit)"
ROBOT = "microduck@14b51b63c52e"
ACTUATOR = "xl330-m6@3ae1b8c2156b (point fit, no intervals)"
RECIPE = "g3_agent, 8000 iterations, 4096 envs"


REPLICATE_SEP = "#"  # "narrow#2": replicate 2 of the narrow arm


def replicate_dirs(root: Path, arm: str) -> list[Path]:
    """The arm's runs: `<root>/<arm>` (run 1) and every `<root>/<arm>#k`,
    in replicate order. An arm with no directory at all is refused."""
    found = sorted(
        (p for p in root.glob(f"{arm}{REPLICATE_SEP}*") if p.is_dir()),
        key=lambda p: int(p.name.split(REPLICATE_SEP, 1)[1]),
    )
    first = root / arm
    dirs = ([first] if first.is_dir() else []) + found
    if not dirs:
        raise FileNotFoundError(f"arm {arm!r} has no run under {root}")
    return dirs


def certificate(root: Path, arm: str, name: str) -> dict[str, Any]:
    """One run's certificate; `arm` may be a replicate dir name."""
    path = root / arm / "train" / "verdict" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"arm {arm!r} has no certificate {name} in {path.parent}"
        )
    return json.loads(path.read_text())


def outcomes(root: Path, arm: str) -> list[bool]:
    """Every replicate's at-fit outcomes of one arm, concatenated in
    replicate order (trial k of each run started identically)."""
    pooled: list[bool] = []
    for run in replicate_dirs(root, arm):
        rows = read_records(run / "train" / "verdict" / AT_FIT_RECORDS)
        pooled += [r.success for r in sorted(rows, key=lambda r: r.trial)]
    return pooled


def fold(
    root: Path, arms: tuple[str, ...], *, alpha: float, delta: float
) -> dict[str, Any]:
    """The at-fit certificates as arms, the own-DR ones beside them, and
    the paired effects between arms."""
    from rq_pipeline.stats.intervals import clopper_pearson  # noqa: PLC0415

    result: dict[str, Any] = {
        "arms": {},
        "under_span_0.10": {},
        "replicates": {},
        "effects": [],
    }
    for arm in arms:
        runs = replicate_dirs(root, arm)
        fits = [certificate(root, run.name, AT_FIT) for run in runs]
        # Each replicate as its own row (the spread IS the finding), and
        # the arm as their pool with an exact interval on the pooled count.
        result["replicates"][arm] = [
            {
                "run": run.name,
                "seed": fit["identity"].get("seed"),
                "successes": fit["successes"],
                "trials": fit["trials"],
                "ci95": fit["ci95"],
                "funnel": fit["funnel"],
            }
            for run, fit in zip(runs, fits, strict=True)
        ]
        successes = sum(f["successes"] for f in fits)
        trials = sum(f["trials"] for f in fits)
        low, high = clopper_pearson(successes, trials)
        result["arms"][arm] = {
            "successes": successes,
            "trials": trials,
            "ci95": [round(low, 4), round(high, 4)],
            "runs": len(fits),
            "per_run": [f["successes"] for f in fits],
            "funnel": {
                "survived": sum(f["funnel"]["survived"] for f in fits),
                "tracked": sum(f["funnel"]["tracked"] for f in fits),
            },
            "median_err_ratio": sorted(f["median_err_ratio"] for f in fits)[
                len(fits) // 2
            ],
            "trained_dr_basis": fits[0]["identity"].get("trained_dr_basis"),
            "judged_at": fits[0]["protocol"].get("judged_at"),
            "instrument": sorted({f["instrument"] for f in fits}),
        }
        owns = []
        for run in runs:
            with contextlib.suppress(FileNotFoundError):
                owns.append(certificate(root, run.name, OWN_DR))
        result["under_span_0.10"][arm] = (
            {
                "successes": sum(o["successes"] for o in owns),
                "trials": sum(o["trials"] for o in owns),
                "per_run": [o["successes"] for o in owns],
                "dr_basis": owns[0]["protocol"].get("dr_basis"),
            }
            if owns
            else None
        )
    for a, b in COMPARE:
        if a in arms and b in arms:
            effect = main_effect(
                f"{a} vs {b}",
                a,
                b,
                outcomes(root, a),
                outcomes(root, b),
                alpha=alpha,
                delta=delta,
            )
            result["effects"].append(
                {
                    "a": a,
                    "b": b,
                    "difference": effect.difference,
                    "p": effect.p_value,
                    "verdict": effect.verdict,
                }
            )
    result["alpha"], result["delta"] = alpha, delta
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument("--id", default="walk-c1")
    args = parser.parse_args()
    arms = tuple(args.arms.split(","))
    outcome = fold(args.root, arms, alpha=args.alpha, delta=args.delta)
    for arm, r in outcome["arms"].items():
        own = outcome["under_span_0.10"][arm]
        own_text = f"; under ±0.10 {own['successes']}/{own['trials']}" if own else ""
        print(
            f"{arm}: at the fit {r['successes']}/{r['trials']} CI95 {r['ci95']} "
            f"over {r['runs']} run(s) {r['per_run']} (trained {r['trained_dr_basis']})"
            f"{own_text}"
        )
    for e in outcome["effects"]:
        print(
            f"{e['a']} vs {e['b']}: diff {e['difference']:+.3f}, "
            f"p {e['p']:.3g}, {e['verdict']}"
        )
    first = next(iter(outcome["arms"].values()))
    record = Finding(
        id=f"{args.id}-{today()}",
        claim=(
            "C1 on the walk: the microduck teacher trained under NO law DR (the "
            "bundle's point fit), the declared ±0.10 span, and the folklore ±0.30 "
            "span — same G3 recipe — each certified AT THE FIT (no law DR, pushes "
            f"on) on {first['trials']} matched trials; own-DR certificates ride along."
        ),
        date=today(),
        repo_commit=repo_commit(REPO),
        argv=sys.argv,
        instrument=", ".join(
            sorted({i for r in outcome["arms"].values() for i in r["instrument"]})
        ),
        outcome=outcome,
        inputs={"robot": ROBOT, "actuator": ACTUATOR, "recipe": RECIPE},
        artifacts={
            f"{run.name}.at_fit": str(run / "train" / "verdict" / AT_FIT)
            for arm in arms
            for run in replicate_dirs(args.root, arm)
        },
        protocol=PROTOCOL,
        caveats=(
            "No bundle in the store carries intervals: the arms are point / declared "
            "±0.10 / declared ±0.30 around BAM's point fit — the identified-INTERVAL "
            "arm needs a fit with intervals (the bench chapter).",
            "Replicates are pooled per arm (trial k of each run starts identically); "
            "per-run counts are on the record so the between-run spread is visible.",
        ),
    )
    figures = render(record, REPO)
    artifacts = {**record.artifacts, **{f"figure.{k}": v for k, v in figures.items()}}
    record = Finding(**{**record.__dict__, "artifacts": artifacts})
    path = record.path(REPO)
    record.write(path)
    print(f"finding -> {path}\nfigure -> {figures['svg']} (+pdf, png, csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
