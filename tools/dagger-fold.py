"""Fold a DAgger round into a finding: base student vs round student.

    python3 tools/dagger-fold.py VERDICT_DIR BASE_STAMP ROUND_STAMP [--round-dir DIR]

Both students were certified on the SAME 40 matched trials
(`walk_verdict --student`, seed 1000), so their certificates sit side
by side in the teacher's `verdict/` directory as
`walk-verdict-student-cuda.json` rows keyed by policy stamp. This
tool reads the two, folds the paired effect with the same
`main_effect` every study uses, and writes the finding with the
round's datasheet (how many student-driven episodes the referee kept)
and its union's provenance beside the numbers.
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

RECORDS = "records-student-cuda.jsonl"
PROTOCOL = "docs/66 §6 (DAgger on the walk: tools/walk-dagger-round.sh)"


def rows_for(verdict_dir: Path, policy: str) -> list[Any]:
    rows = [r for r in read_records(verdict_dir / RECORDS) if r.policy == policy]
    if not rows:
        known = sorted({r.policy for r in read_records(verdict_dir / RECORDS)})
        raise KeyError(f"no rows for policy {policy!r}; known: {known}")
    # The latest certificate for that policy: rows are appended per run,
    # so take the last complete block of trials.
    trials = max(r.trial for r in rows) + 1
    return sorted(rows[-trials:], key=lambda r: r.trial)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("verdict_dir", type=Path)
    parser.add_argument("base", help="the base student's policy stamp")
    parser.add_argument("round", help="the round student's policy stamp")
    parser.add_argument("--round-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.15)
    args = parser.parse_args()

    from rq_pipeline.stats.intervals import clopper_pearson  # noqa: PLC0415

    arms: dict[str, Any] = {}
    outcomes: dict[str, list[bool]] = {}
    for name, stamp in (("base", args.base), ("dagger-1", args.round)):
        rows = rows_for(args.verdict_dir, stamp)
        successes = sum(r.success for r in rows)
        low, high = clopper_pearson(successes, len(rows))
        arms[name] = {
            "policy": stamp,
            "successes": successes,
            "trials": len(rows),
            "ci95": [round(low, 4), round(high, 4)],
            "funnel": {
                "survived": sum(r.events[0]["passed"] for r in rows),
                "tracked": sum(r.events[1]["passed"] for r in rows),
            },
            "instrument": sorted({r.instrument for r in rows}),
        }
        outcomes[name] = [r.success for r in rows]
        print(f"{name}: {successes}/{len(rows)} CI95 [{low:.3f}, {high:.3f}] ({stamp})")
    effect = main_effect(
        "base vs dagger-1",
        "base",
        "dagger-1",
        outcomes["base"],
        outcomes["dagger-1"],
        alpha=args.alpha,
        delta=args.delta,
    )
    print(
        f"base vs dagger-1: diff {effect.difference:+.3f}, "
        f"p {effect.p_value:.3g}, {effect.verdict}"
    )

    inputs: dict[str, Any] = {"teacher_verdict_dir": str(args.verdict_dir)}
    artifacts: dict[str, str] = {"records": str(args.verdict_dir / RECORDS)}
    if args.round_dir is not None:
        sheet = args.round_dir / "demos" / "datasheet.md"
        if sheet.is_file():
            inputs["round_datasheet"] = sheet.read_text()[:600]
            artifacts["round_datasheet"] = str(sheet)
        union = args.round_dir / "dataset" / "provenance.json"
        if union.is_file():
            prov = json.loads(union.read_text())
            inputs["union"] = {
                "sources": prov.get("sources"),
                "episodes": prov.get("episodes"),
            }
            artifacts["union_provenance"] = str(union)
    record = Finding(
        id=f"walk-dagger-round-1-{today()}",
        claim=(
            "DAgger round 1 on the walk: the base student drove 120 episodes, the "
            "teacher labeled every state, the referee kept the passes; a new student "
            "trained on the union of the base dataset and the relabeled batch, "
            "certified on the same 40 matched trials as the base — base "
            f"{arms['base']['successes']}/40 vs round-1 "
            f"{arms['dagger-1']['successes']}/40."
        ),
        date=today(),
        repo_commit=repo_commit(REPO),
        argv=sys.argv,
        instrument=", ".join(
            sorted({i for a in arms.values() for i in a["instrument"]})
        ),
        outcome={
            "arms": arms,
            "effects": [
                {
                    "a": "base",
                    "b": "dagger-1",
                    "difference": effect.difference,
                    "p": effect.p_value,
                    "verdict": effect.verdict,
                }
            ],
            "alpha": args.alpha,
            "delta": args.delta,
        },
        inputs=inputs,
        artifacts=artifacts,
        protocol=PROTOCOL,
        caveats=(
            "One round, one training run per student, 40 trials each: a direction, "
            "replicate before a sentence.",
            "The round student trained 20k steps on 360 episodes; the base trained "
            "60k on 240 — steps per episode differ.",
        ),
    )
    figures = render(record, REPO)
    record = Finding(
        **{
            **record.__dict__,
            "artifacts": {
                **record.artifacts,
                **{f"figure.{k}": v for k, v in figures.items()},
            },
        }
    )
    path = record.path(REPO)
    record.write(path)
    print(f"finding -> {path}\nfigure -> {figures['svg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
