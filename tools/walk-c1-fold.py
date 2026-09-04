"""Fold the walk C1 arms' certificates into one finding and its figure.

    python3 tools/walk-c1-fold.py <study_root> [--arms point,narrow,wide]

Each arm (tools/walk-c1-arm.sh) leaves two certificates under
`<root>/<arm>/train/verdict/`: `walk-verdict-at-fit-cuda.json` (judged
at the bundle's point fit, the shared world) and `walk-verdict-cuda.json`
(judged under the arm's own DR). The at-fit certificates are the study's
arms; the own-DR ones ride along as the robustness view. Comparisons are
paired on the certificate's shared seed and folded with the same
`main_effect` every study uses; the finding lands in `docs/findings/`
with its figure, like every other study.
"""

from __future__ import annotations

import argparse
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


def certificate(root: Path, arm: str, name: str) -> dict[str, Any]:
    path = root / arm / "train" / "verdict" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"arm {arm!r} has no certificate {name} in {path.parent}"
        )
    return json.loads(path.read_text())


def outcomes(root: Path, arm: str) -> list[bool]:
    rows = read_records(root / arm / "train" / "verdict" / AT_FIT_RECORDS)
    return [r.success for r in sorted(rows, key=lambda r: r.trial)]


def fold(
    root: Path, arms: tuple[str, ...], *, alpha: float, delta: float
) -> dict[str, Any]:
    """The at-fit certificates as arms, the own-DR ones beside them, and
    the paired effects between arms."""
    result: dict[str, Any] = {"arms": {}, "own_dr": {}, "effects": []}
    for arm in arms:
        fit = certificate(root, arm, AT_FIT)
        result["arms"][arm] = {
            "successes": fit["successes"],
            "trials": fit["trials"],
            "ci95": fit["ci95"],
            "funnel": fit["funnel"],
            "median_err_ratio": fit["median_err_ratio"],
            "trained_dr_basis": fit["identity"].get("trained_dr_basis"),
            "judged_at": fit["protocol"].get("judged_at"),
            "instrument": [fit["instrument"]],
        }
        try:
            own = certificate(root, arm, OWN_DR)
        except FileNotFoundError:
            result["own_dr"][arm] = None
        else:
            result["own_dr"][arm] = {
                "successes": own["successes"],
                "trials": own["trials"],
                "ci95": own["ci95"],
                "funnel": own["funnel"],
                "dr_basis": own["protocol"].get("dr_basis"),
            }
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
        own = outcome["own_dr"][arm]
        own_text = f"; under own DR {own['successes']}/{own['trials']}" if own else ""
        print(
            f"{arm}: at the fit {r['successes']}/{r['trials']} CI95 {r['ci95']} "
            f"(trained {r['trained_dr_basis']}){own_text}"
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
            f"{arm}.at_fit": str(args.root / arm / "train" / "verdict" / AT_FIT)
            for arm in arms
        },
        protocol=PROTOCOL,
        caveats=(
            "No bundle in the store carries intervals: the arms are point / declared "
            "±0.10 / declared ±0.30 around BAM's point fit — the identified-INTERVAL "
            "arm needs a fit with intervals (the bench chapter).",
            "One training run per arm: the lift's cliff curve showed per-run variance "
            "can exceed the effect; replicate before quoting a difference.",
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
