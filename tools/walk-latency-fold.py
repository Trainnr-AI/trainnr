"""Fold the latency-budget certificates into one finding and its figure.

    python3 tools/walk-latency-fold.py <verdict_dir> [--latencies 0,1,2,4,8]
        [--controls 3:0,1:1] [--trials 40] [--date YYYY-MM-DD]

docs/e2e-research/71 E1: the same student certified under an
inference budget of n control ticks (`walk_verdict --latency n`, the
scheduler's emulation of SmoothRL's timed loop), 40 matched trials on
the shared seed, from the row files walk_verdict leaves under
`<verdict_dir>`: `records-student-cuda.jsonl` for n = 0 and
`records-student-latency-<n>-cuda.jsonl` otherwise; the teacher's
`records-cuda.jsonl` beside them. Controls are `horizon:latency` pairs
read from the same files by their protocol fields. The newest
`--trials` rows of each file are the run.

The question it answers: how much of a chunk policy's certificate
survives the latency a real robot adds — the mismatch SmoothRL fine-
tunes against, measured with intervals instead of asserted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _lab import REPO, bootstrap

bootstrap()

from rq_pipeline.evaluate import figures as fx  # noqa: E402
from rq_pipeline.evaluate.findings import Finding, repo_commit, today  # noqa: E402
from rq_pipeline.stats.effects import main_effect  # noqa: E402
from rq_pipeline.stats.intervals import clopper_pearson  # noqa: E402

PROTOCOL = (
    "docs/e2e-research/71 §4 E1 (walk_verdict --latency n; the scheduler's timed loop)"
)
INSTRUMENT_OF = "instrument"


def rows_of(
    path: Path, *, horizon: int, latency: int, trials: int
) -> list[dict[str, Any]]:
    """The newest `trials` rows of the file that ran at this horizon and latency."""
    if not path.is_file():
        return []
    picked = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        student = (row.get("protocol") or {}).get("student") or {}
        if (
            student.get("horizon", horizon) == horizon
            and student.get("latency", 0) == latency
        ):
            picked.append(row)
    return picked[-trials:]


def arm(rows: list[dict[str, Any]], **facts: Any) -> dict[str, Any]:
    successes = sum(bool(r["success"]) for r in rows)
    trials = len(rows)
    events = [{e["name"]: e["passed"] for e in r["events"]} for r in rows]
    jerks = [r["variations"]["rms_jerk"] for r in rows if "rms_jerk" in r["variations"]]
    return {
        "successes": successes,
        "trials": trials,
        "ci95": list(clopper_pearson(successes, trials)),
        "survived": sum(e.get("survived", False) for e in events),
        "tracked": sum(e.get("tracked", False) for e in events),
        "median_steps": statistics.median(r["steps"] for r in rows),
        "median_rms_jerk": round(statistics.median(jerks), 1) if jerks else None,
        "policy": rows[0]["policy"],
        **facts,
    }


def student_file(verdict_dir: Path, latency: int) -> Path:
    name = (
        "records-student-cuda.jsonl"
        if latency == 0
        else f"records-student-latency-{latency}-cuda.jsonl"
    )
    return verdict_dir / name


def fold(  # noqa: PLR0913 - the fold's knobs, named
    verdict_dir: Path,
    latencies: list[int],
    controls: list[tuple[int, int]],
    *,
    trials: int,
    alpha: float,
    delta: float,
    horizon: int = 2,
) -> tuple[dict[str, Any], str]:
    arms: dict[str, Any] = {}
    outcomes: dict[str, list[bool]] = {}
    teacher = rows_of(
        verdict_dir / "records-cuda.jsonl", horizon=0, latency=0, trials=trials
    )
    if teacher:
        arms["teacher"] = arm(teacher, latency=None, horizon=None)
    for n in latencies:
        rows = rows_of(
            student_file(verdict_dir, n), horizon=horizon, latency=n, trials=trials
        )
        if rows:
            arms[f"latency-{n}"] = arm(rows, latency=n, horizon=horizon)
            outcomes[f"latency-{n}"] = [bool(r["success"]) for r in rows]
    for h, n in controls:
        rows = rows_of(
            student_file(verdict_dir, n), horizon=h, latency=n, trials=trials
        )
        if rows:
            arms[f"horizon-{h}-latency-{n}"] = arm(rows, latency=n, horizon=h)
    effects = []
    base = outcomes.get(f"latency-{latencies[0]}")
    for name, values in outcomes.items():
        if base is not None and name != f"latency-{latencies[0]}":
            effects.append(
                asdict(
                    main_effect(
                        name,
                        f"latency-{latencies[0]}",
                        name,
                        base,
                        values,
                        alpha=alpha,
                        delta=delta,
                    )
                )
            )
    instrument = (
        next(iter(arms.values()))["policy"] and (teacher or rows)[0][INSTRUMENT_OF]
    )
    return {
        "arms": arms,
        "effects": effects,
        "latencies": latencies,
        "controls": controls,
    }, instrument


def table(outcome: dict[str, Any]) -> str:
    cells = []
    for name, a in outcome["arms"].items():
        low, high = a["ci95"]
        cells.append(f"{name} {a['successes']}/{a['trials']} [{low:.3f}, {high:.3f}]")
    return "; ".join(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("verdict_dir", type=Path)
    parser.add_argument("--latencies", default="0,1,2,4,8")
    parser.add_argument("--controls", default="3:0,1:1", help="horizon:latency pairs")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    latencies = [int(x) for x in args.latencies.split(",")]
    controls = [
        tuple(int(v) for v in c.split(":")) for c in args.controls.split(",") if c
    ]
    outcome, instrument = fold(
        args.verdict_dir,
        latencies,
        [(h, n) for h, n in controls],
        trials=args.trials,
        alpha=args.alpha,
        delta=args.delta,
    )
    date = args.date or today()
    record = Finding(
        id=f"walk-latency-budget-{date}",
        claim=(
            "The campaign 3 vision student certified under an inference budget of "
            "n control ticks (SmoothRL's timed loop, emulated in the scheduler), "
            "40 matched trials on the shared seed - " + table(outcome) + "."
        ),
        date=date,
        repo_commit=repo_commit(REPO),
        argv=sys.argv,
        instrument=instrument,
        outcome=outcome,
        inputs={
            "verdict_dir": str(args.verdict_dir),
            "student": "runs/campaign-3/student (student-last@c39a1b54c06b)",
        },
        artifacts={
            name: str(student_file(args.verdict_dir, a["latency"]))
            for name, a in outcome["arms"].items()
            if a["latency"] is not None
        },
        protocol=PROTOCOL,
        caveats=(
            "One student, one box; the budget is emulated in the scheduler (the "
            "chunk in flight keeps supplying its own later rows, one request in "
            "flight at a time), the first chunk taking over at once.",
            "Zero latency executes rows [0, h) of each chunk; a budget n executes "
            "rows [n, n + h), or [n, 2n) once n exceeds h: the controls separate "
            "staleness from row depth.",
            "Smoothness is the RMS jerk of the issued targets per trial (docs/71 E0).",
        ),
    )
    figure = fx.render(record, REPO)
    record = Finding(
        **{
            **record.__dict__,
            "artifacts": {
                **record.artifacts,
                **{f"figure.{k}": v for k, v in figure.items()},
            },
        }
    )
    path = record.path(REPO)
    record.write(path)
    print(table(outcome))
    for e in outcome["effects"]:
        print(f"{e['key']}: {e['difference']:+.3f} p {e['p_value']:.3g} {e['verdict']}")
    print(f"finding -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
