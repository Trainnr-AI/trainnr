"""The paper's figures, drawn from finding records — never by hand.

A figure here is a FUNCTION of a `Finding`: the arms' successes, trials
and exact intervals become dots with interval whiskers (a curve over
episodes when every arm declares a distinct count, a categorical row
otherwise), and the provenance rides on the figure itself — finding
id, commit, instrument, trials, protocol — so a figure separated from
its caption still says where it came from. Saved as SVG (the paper),
PDF (LaTeX), PNG (the README), and a CSV of exactly the numbers
plotted, under `docs/figures/`, TRACKED beside the record.

Marks follow the charting method (thin marks, ink-coloured text, a
single hue for a single series, fixed categorical order, no dual
axes); colours are the reference palette's first slots, validated
2026-09-04.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from rq_pipeline.evaluate.findings import Finding

FIGURES_DIR = Path("docs") / "figures"
FORMATS = ("svg", "pdf", "png")

# The reference palette's categorical slots (light surface), in fixed
# order: blue, orange, aqua. Validated with the palette script.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

MARKER_PT = 7.0
MIN_CURVE_ARMS = 3  # fewer points is a row, not a curve
TITLE_CHARS = 110  # the claim, trimmed to one title line
LINE_PT = 1.6
DPI = 200


def arm_rows(finding: Finding) -> list[dict[str, Any]]:
    """One row per arm, in the finding's order: name, episodes (when the
    arm declares one), successes, trials, rate, ci_low, ci_high."""
    arms = finding.outcome.get("arms", {})
    rows = []
    for name, arm in arms.items():
        low, high = arm["ci95"]
        rows.append(
            {
                "arm": name,
                "episodes": arm.get("episodes"),
                "successes": int(arm["successes"]),
                "trials": int(arm["trials"]),
                "rate": arm["successes"] / arm["trials"],
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    if not rows:
        raise ValueError(f"finding {finding.id} has no arms to plot")
    return rows


def is_curve(rows: list[dict[str, Any]]) -> bool:
    """A curve when every arm declares a distinct episode count and
    there are at least three of them; otherwise a categorical row."""
    counts = [r["episodes"] for r in rows]
    return (
        len(rows) >= MIN_CURVE_ARMS
        and all(c is not None for c in counts)
        and len(set(counts)) == len(counts)
    )


def footer(finding: Finding) -> str:
    trials = sorted({r["trials"] for r in arm_rows(finding)})
    n = "/".join(str(t) for t in trials)
    return (
        f"{finding.id} · commit {finding.repo_commit} · {finding.instrument} · "
        f"{n} paired trials per arm, exact 95% intervals · {finding.protocol}"
    )


def render(finding: Finding, root: Path) -> dict[str, str]:
    """Write the figure in every format plus its CSV; returns the paths
    (relative to `root`) keyed by format — the finding's artifacts."""
    import matplotlib  # noqa: PLC0415 - the train extra

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = arm_rows(finding)
    out_dir = Path(root) / FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = is_curve(rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [r["episodes"] for r in rows] if curve else list(range(len(rows)))
    ys = [r["rate"] for r in rows]
    lower = [r["rate"] - r["ci_low"] for r in rows]
    upper = [r["ci_high"] - r["rate"] for r in rows]
    if curve:
        # One series, one hue, no legend: the title names it.
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            color=SERIES[0],
            linewidth=LINE_PT,
            marker="o",
            markersize=MARKER_PT,
            capsize=3,
            elinewidth=1.0,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("demonstrations pressed (referee-gated episodes)", color=INK)
    else:
        for i, (x, y, lo, hi) in enumerate(zip(xs, ys, lower, upper, strict=True)):
            ax.errorbar(
                [x],
                [y],
                yerr=[[lo], [hi]],
                color=SERIES[i % len(SERIES)],
                marker="o",
                markersize=MARKER_PT,
                capsize=3,
                elinewidth=1.0,
                linestyle="none",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([r["arm"] for r in rows], color=INK)
        ax.set_xlim(-0.6, len(rows) - 0.4)
    for x, r in zip(xs, rows, strict=True):
        ax.annotate(
            f"{r['successes']}/{r['trials']}",
            (x, r["ci_high"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("success rate (exact 95% interval)", color=INK)
    ax.set_title(
        finding.claim[:TITLE_CHARS] + ("…" if len(finding.claim) > TITLE_CHARS else ""),
        fontsize=9,
        color=INK,
        loc="left",
    )
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    fig.text(0.01, 0.01, footer(finding), fontsize=6, color=INK_SECONDARY, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    written: dict[str, str] = {}
    for fmt in FORMATS:
        path = out_dir / f"{finding.id}.{fmt}"
        fig.savefig(path, format=fmt, dpi=DPI, facecolor=SURFACE)
        written[fmt] = str(path.relative_to(root))
    plt.close(fig)

    csv_path = out_dir / f"{finding.id}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written["csv"] = str(csv_path.relative_to(root))
    return written
