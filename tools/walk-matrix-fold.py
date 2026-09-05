"""Fold the walk mismatch matrix into one finding and its figure.

    python3 tools/walk-matrix-fold.py <study_root> [--arms point,narrow,wide]
        [--scales 0.7,0.8,0.9,1.1,1.2,1.3] [--span 0.30] [--date YYYY-MM-DD]

Every C1 policy (tools/walk-c1-arm.sh; replicates pooled per arm as in
tools/walk-c1-fold.py) judged in each of these worlds, 40 matched
trials each, from the certificates tools/walk-mismatch-matrix.sh
leaves under `<root>/<arm>[#k]/train/verdict/`:

  pinned at fit x s  `records-at-x<s>-at-fit-cuda.jsonl`  (s in --scales)
  at the fit          `records-at-fit-cuda.jsonl`           (s = 1.0)
  drawn from ±0.10    `records-cuda.jsonl`                  (LAW_DR_SPAN)
  drawn from ±<span>  `records-under-pm<span>-cuda.jsonl`

The question it answers: how wide to randomize around a measured
actuator, as a function of how wrong the measurement is — the cost of
over-randomizing (the wide arm at the fit) against the cost of
under-randomizing (the point arm far from it). Effects are paired on
the shared certificate seed with the same `main_effect` every study
uses; the figure draws the curve per arm with exact intervals and the
span columns beside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _lab import REPO, bootstrap

bootstrap()

from rq_pipeline.evaluate import figures as fx  # noqa: E402
from rq_pipeline.evaluate.findings import Finding, repo_commit, today  # noqa: E402
from rq_pipeline.evaluate.records import read_records  # noqa: E402
from rq_pipeline.stats.effects import main_effect  # noqa: E402
from rq_pipeline.stats.intervals import clopper_pearson  # noqa: E402

DEFAULT_ARMS = ("point", "narrow", "wide")
DEFAULT_SCALES = (0.7, 0.8, 0.9, 1.1, 1.2, 1.3)
DEFAULT_SPAN = 0.30
NARROW_SPAN = 0.10  # walk_verdict's LAW_DR_SPAN: the plain certificate's draws
REPLICATE_SEP = "#"
PROTOCOL = (
    "docs/70 §9 (the mismatch matrix: tools/walk-mismatch-matrix.sh, "
    "every C1 policy judged pinned at fit x s and under drawn spans)"
)
ARM_SPAN = {
    "point": "none",
    "narrow": f"±{NARROW_SPAN:g}",
    "wide": f"±{DEFAULT_SPAN:g}",
    "point-refit": "none (refit bundle)",
    "identified": "identified set (refit bundle)",
}


def replicate_dirs(root: Path, arm: str) -> list[Path]:
    found = sorted(
        (p for p in root.glob(f"{arm}{REPLICATE_SEP}*") if p.is_dir()),
        key=lambda p: int(p.name.split(REPLICATE_SEP, 1)[1]),
    )
    first = root / arm
    dirs = ([first] if first.is_dir() else []) + found
    if not dirs:
        raise FileNotFoundError(f"arm {arm!r} has no run under {root}")
    return dirs


AXIS = "all"  # set from --param: which mismatch axis the pinned cells moved


def records_name(condition: str, span: float) -> str:
    """The rows file for a condition: 'x<s>' (pinned at fit x s on the
    AXIS), 'fit', 'pm<narrow>' (the plain certificate), 'pm<span>'."""
    if condition == "fit":
        return "records-at-fit-cuda.jsonl"
    if condition.startswith("x"):
        tag = "" if AXIS == "all" else f"{AXIS}-"
        return f"records-at-{condition}-{tag}at-fit-cuda.jsonl"
    if condition == f"pm{NARROW_SPAN:g}":
        return "records-cuda.jsonl"
    if condition == f"pm{span:g}":
        return f"records-under-pm{span:g}-cuda.jsonl"
    raise ValueError(f"unknown condition {condition!r}")


def last_certificate(path: Path) -> list[Any]:
    """The last complete block of trials in a rows file (rows are
    appended per certificate run)."""
    rows = read_records(path)
    if not rows:
        raise ValueError(f"{path} holds no rows")
    trials = max(r.trial for r in rows) + 1
    return sorted(rows[-trials:], key=lambda r: r.trial)


def fold(  # noqa: PLR0913 - the study's knobs, named
    root: Path,
    arms: tuple[str, ...],
    conditions: list[str],
    *,
    span: float,
    alpha: float,
    delta: float,
) -> dict[str, Any]:
    outcomes: dict[str, dict[str, list[bool]]] = {}
    result: dict[str, Any] = {
        "conditions": conditions,
        "arms": {},
        "effects": [],
        "alpha": alpha,
        "delta": delta,
        "missing": [],
    }
    for arm in arms:
        runs = replicate_dirs(root, arm)
        result["arms"][arm] = {
            "trained_under": ARM_SPAN.get(arm, arm),
            "runs": len(runs),
        }
        outcomes[arm] = {}
        for cond in conditions:
            pooled: list[bool] = []
            per_run: list[int] = []
            for run in runs:
                path = run / "train" / "verdict" / records_name(cond, span)
                if not path.is_file():
                    result["missing"].append(f"{run.name}:{cond}")
                    continue
                rows = last_certificate(path)
                pooled += [r.success for r in rows]
                per_run.append(sum(r.success for r in rows))
            if not pooled:
                continue
            low, high = clopper_pearson(sum(pooled), len(pooled))
            result["arms"][arm][cond] = {
                "successes": sum(pooled),
                "trials": len(pooled),
                "per_run": per_run,
                "ci95": [round(low, 4), round(high, 4)],
            }
            outcomes[arm][cond] = pooled
    declared = [
        (a, b)
        for a, b in (("wide", "point"), ("wide", "narrow"), ("point", "narrow"))
        if a in arms and b in arms
    ]
    pairs = declared or [(a, b) for i, a in enumerate(arms) for b in arms[i + 1 :]]
    for cond in conditions:
        for a, b in pairs:
            if cond in outcomes.get(a, {}) and cond in outcomes.get(b, {}):
                if len(outcomes[a][cond]) != len(outcomes[b][cond]):
                    continue
                e = main_effect(
                    f"{a} vs {b} @ {cond}",
                    a,
                    b,
                    outcomes[a][cond],
                    outcomes[b][cond],
                    alpha=alpha,
                    delta=delta,
                )
                result["effects"].append(
                    {
                        "condition": cond,
                        "a": a,
                        "b": b,
                        "difference": e.difference,
                        "p": e.p_value,
                        "verdict": e.verdict,
                    }
                )
    return result


def _span_points(
    ax: Any, cells: dict[str, Any], columns: list[str], *, offset: float, color: str
) -> None:
    """One arm's drawn-span cells as dots with exact intervals, offset
    so the three arms sit side by side in each column."""
    for j, col in enumerate(columns):
        cell = cells.get(col)
        if not cell:
            continue
        rate = cell["successes"] / cell["trials"]
        ax.errorbar(
            [j + offset],
            [rate],
            yerr=[[rate - cell["ci95"][0]], [cell["ci95"][1] - rate]],
            color=color,
            marker="o",
            markersize=fx.MARKER_PT,
            capsize=2,
            linestyle="none",
        )


def _frame(fig: Any, ax: Any, finding: Finding) -> None:
    """The house frame for a nested-arms finding (figures._style reads
    flat arms): y in [0, 1], the claim as title, recessive grid, the
    provenance footer."""
    import textwrap  # noqa: PLC0415

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("success rate (exact 95% interval)", color=fx.INK)
    ax.set_title(
        textwrap.fill(finding.claim[: fx.TITLE_CHARS] + "…", fx.TITLE_WRAP),
        fontsize=9,
        color=fx.INK,
        loc="left",
    )
    ax.grid(True, axis="y", color=fx.GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(fx.GRID)
    ax.tick_params(colors=fx.INK_SECONDARY, labelsize=8)
    footer = (
        f"{finding.id} · commit {finding.repo_commit} · {finding.instrument} · "
        f"40 paired trials per run, replicates pooled, exact 95% intervals · "
        f"{finding.protocol}"
    )
    fig.text(0.01, 0.01, footer, fontsize=6, color=fx.INK_SECONDARY, ha="left")


def draw(finding: Finding, scales: tuple[float, ...], span: float) -> dict[str, str]:
    """The curve per arm over the pinned scale (1.0 = the fit) and the
    drawn-span columns beside it; the house style and footer."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    arms = finding.outcome["arms"]
    xs_scale = sorted((*scales, 1.0))
    columns = [f"pm{NARROW_SPAN:g}", f"pm{span:g}"]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(8.0, 3.8), dpi=fx.DPI, gridspec_kw={"width_ratios": (3, 1)}
    )
    fig.patch.set_facecolor(fx.SURFACE)
    for a in (ax, ax2):
        a.set_facecolor(fx.SURFACE)
    for i, cells in enumerate(arms.values()):
        color = fx.SERIES[i % len(fx.SERIES)]
        pts = []
        for s in xs_scale:
            key = "fit" if s == 1.0 else f"x{s:g}"
            cell = cells.get(key)
            if cell:
                pts.append((s, cell))
        if pts:
            ax.errorbar(
                [p[0] for p in pts],
                [p[1]["successes"] / p[1]["trials"] for p in pts],
                yerr=[
                    [p[1]["successes"] / p[1]["trials"] - p[1]["ci95"][0] for p in pts],
                    [p[1]["ci95"][1] - p[1]["successes"] / p[1]["trials"] for p in pts],
                ],
                color=color,
                linewidth=fx.LINE_PT,
                marker="o",
                markersize=fx.MARKER_PT * 0.8,
                capsize=2,
                label=f"trained {cells['trained_under']}",
            )
        _span_points(ax2, cells, columns, offset=(i - 1) * 0.18, color=color)
    _frame(fig, ax, finding)
    ax.set_xlabel(
        "judged pinned at fit x s (1.0 = the measured actuator)", color=fx.INK
    )
    ax.axvline(1.0, color=fx.GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xticks(range(len(columns)))
    ax2.set_xticklabels([f"drawn ±{NARROW_SPAN:g}", f"drawn ±{span:g}"], fontsize=8)
    ax2.set_xlim(-0.6, len(columns) - 0.4)
    ax2.grid(True, axis="y", color=fx.GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color(fx.GRID)
    ax2.tick_params(colors=fx.INK_SECONDARY, labelsize=8)
    ax2.set_yticklabels([])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_dir = REPO / fx.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for fmt in fx.FORMATS:
        path = out_dir / f"{finding.id}.{fmt}"
        fig.savefig(path, format=fmt, facecolor=fx.SURFACE)
        paths[fmt] = str(path.relative_to(REPO))
    plt.close(fig)
    csv_path = out_dir / f"{finding.id}.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "arm,trained_under,condition,successes,trials,ci_low,ci_high,per_run\n"
        )
        for arm, cells in arms.items():
            for cond, cell in cells.items():
                if isinstance(cell, dict):
                    handle.write(
                        f"{arm},{cells['trained_under']},{cond},{cell['successes']},"
                        f"{cell['trials']},{cell['ci95'][0]},{cell['ci95'][1]},"
                        f"{'/'.join(str(k) for k in cell['per_run'])}\n"
                    )
    paths["csv"] = str(csv_path.relative_to(REPO))
    return paths


def table(outcome: dict[str, Any]) -> str:
    """Every cell as k/n, one line per arm — the sentence the numbers
    gate can trace."""
    lines = []
    for arm, cells in outcome["arms"].items():
        parts = [
            f"{cond} {cell['successes']}/{cell['trials']}"
            for cond, cell in cells.items()
            if isinstance(cell, dict)
        ]
        lines.append(f"{arm} (trained {cells['trained_under']}): " + ", ".join(parts))
    return "; ".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--scales", default=",".join(f"{s:g}" for s in DEFAULT_SCALES))
    parser.add_argument("--span", type=float, default=DEFAULT_SPAN)
    parser.add_argument("--date", default=None)
    parser.add_argument(
        "--id",
        default="walk-mismatch-matrix",
        help="the record id's stem (the refit arms' matrix names its bundle)",
    )
    parser.add_argument(
        "--param",
        default="all",
        help="the mismatch axis the pinned cells moved (walk_verdict --judge-param); "
        "'all' is the diagonal, the default matrix",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.15)
    args = parser.parse_args()

    global AXIS  # noqa: PLW0603 - one run, one axis
    AXIS = args.param
    arms = tuple(args.arms.split(","))
    scales = tuple(float(s) for s in args.scales.split(","))
    conditions = (
        [f"x{s:g}" for s in sorted(scales) if s < 1.0]
        + ["fit"]
        + [f"x{s:g}" for s in sorted(scales) if s > 1.0]
        + [f"pm{NARROW_SPAN:g}", f"pm{args.span:g}"]
    )
    outcome = fold(
        args.root, arms, conditions, span=args.span, alpha=args.alpha, delta=args.delta
    )
    if outcome["missing"]:
        print(f"missing certificates: {outcome['missing']}", file=sys.stderr)
    date = args.date or today()
    instrument = "mjlab-1.6.0+mujoco-3.11.0+warp-1.17.0+cuda"
    axis_id = "" if AXIS == "all" else f"-{AXIS}"
    moved = "every law parameter" if AXIS == "all" else f"only {AXIS}"
    record = Finding(
        id=f"{args.id}{axis_id}-{date}",
        claim=(
            "The mismatch matrix on the walk: the C1 policies "
            f"{', '.join(arms)} (replicates pooled) judged "
            f"pinned at the fit scaled by s ({moved} moved) and under drawn spans, "
            "40 matched trials per run — " + table(outcome) + "."
        ),
        date=date,
        repo_commit=repo_commit(REPO),
        argv=sys.argv,
        instrument=instrument,
        outcome=outcome,
        inputs={
            "study_root": str(args.root),
            "policies": "the walk C1 checkpoints (record walk-c1-2026-09-04)",
            "scales": list(scales),
            "spans": [NARROW_SPAN, args.span],
        },
        artifacts={
            f"{run.name}.{cond}": str(
                run / "train" / "verdict" / records_name(cond, args.span)
            )
            for arm in arms
            for run in replicate_dirs(args.root, arm)
            for cond in conditions
            if (run / "train" / "verdict" / records_name(cond, args.span)).is_file()
        },
        protocol=PROTOCOL,
        caveats=(
            (
                "Pinned scale s multiplies EVERY law parameter of the actuator model "
                "by s (walk_verdict --judge-at-scale); a real mismatch moves them "
                "separately — the per-axis matrices (--judge-param) are the slices."
                if AXIS == "all"
                else f"Pinned scale s multiplies only the {AXIS} axis of the actuator "
                "model by s; every other law parameter sits at the fit."
            ),
            "One robot, one gait, one recipe; the policies are the C1 runs "
            "(three per arm).",
            "The fit is a point estimate: 'how wrong the measurement is' is simulated, "
            "not measured — paper 2's bench says how wrong it actually was.",
        ),
    )
    figure = draw(record, scales, args.span)
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
    for e in outcome["effects"]:
        print(
            f"{e['condition']}: {e['a']} vs {e['b']} {e['difference']:+.3f} "
            f"p {e['p']:.3g} {e['verdict']}"
        )
    print(f"finding -> {path}\nfigure -> {figure['svg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
