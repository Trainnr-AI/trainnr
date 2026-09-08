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
import re
import textwrap
from pathlib import Path
from typing import Any

from rq_pipeline.evaluate.findings import Finding

FIGURES_DIR = Path("docs") / "figures"
FORMATS = ("svg", "pdf", "png")

# The reference paper's look (arXiv 2603.19312, adopted 2026-09-08):
# pastel fills with a darker edge, black hairline whiskers, the count
# printed on the bar. Blue is the default arm; red marks the measured
# or identified arm the text argues for; grey a control. SERIES keeps
# the line figures' hues in the same family.
FILL_BLUE, EDGE_BLUE = "#a9c4e4", "#5b8fd0"
FILL_RED, EDGE_RED = "#f4a6a6", "#d9534f"
FILL_GREEN, EDGE_GREEN = "#b9dcc5", "#5fae7c"
FILL_GREY, EDGE_GREY = "#dcdcdc", "#8a8a8a"
SERIES = (EDGE_BLUE, EDGE_RED, EDGE_GREEN)
FILLS = (FILL_BLUE, FILL_RED, FILL_GREEN)
HIGHLIGHT_ARMS = ("narrow", "identified", "±0.1", "±10")
CONTROL_ARMS = ("random", "control")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

MARKER_PT = 4.0
MIN_CURVE_ARMS = 3  # fewer points is a row, not a curve
MANY_ARMS = 6  # more arms than this: a wider canvas, slanted labels
TITLE_CHARS = 110  # the claim, trimmed
TITLE_HEAD_CHARS = 90  # a claim's lead clause is the title when it fits
TITLE_WRAP = 44  # characters per title line at the panel width
TITLE_ACRONYMS = frozenset({"PPO", "ACT", "BAM", "RSS", "DR", "GPU", "CPU", "MJX"})
INTERVAL_TITLE_WRAP = 44  # the interval figure's axes start right of its labels
LINE_PT = 1.1
DPI = 220
PANEL = (2.9, 2.15)  # one panel, inches: three fit the 5.5 in text width side by side
BAR_WIDTH = 0.62
FOOTER_WRAP = 88  # characters per footer line at the panel width
# Display names for arms whose record names are the study's shorthand;
# the paper's tables use these words (the record keeps its own).
ARM_LABELS = {
    "point": "none",
    "narrow": "±10 %",
    "wide": "±30 %",
    "point-refit": "none (refit)",
    "identified": "bootstrap set",
    "teacher-delay-1": "teacher +1 tick",
    "teacher-delay-2": "teacher +2",
    "teacher-delay-4": "teacher +4",
    "latency-0": "student 0",
    "latency-1": "student +1",
    "latency-2": "student +2",
    "latency-4": "student +4",
    "latency-8": "student +8",
    "horizon-3-latency-0": "student, 3 rows",
    "horizon-1-latency-1": "student, 1 row +1",
}
LABEL_INSIDE_ABOVE = 0.45  # a bar taller than this carries its count inside


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
                "group": arm.get("replicate_of", name),
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
    """A curve when every declared arm (replicates pooled) has a
    distinct episode count and there are at least three of them;
    otherwise a categorical row."""
    groups: dict[str, Any] = {}
    for r in rows:
        groups.setdefault(r["group"], r["episodes"])
    counts = list(groups.values())
    return (
        len(counts) >= MIN_CURVE_ARMS
        and all(c is not None for c in counts)
        and len(set(counts)) == len(counts)
    )


def footer(finding: Finding) -> str:
    """The provenance line every figure carries: record id, commit,
    instrument, what the trials were, the protocol."""
    outcome = finding.outcome
    if "conditions" in outcome:
        trials = sorted({r["trials"] for r in matrix_rows(finding)})
        what = (
            "/".join(str(t) for t in trials)
            + " paired trials per cell, exact 95% intervals"
        )
    elif "arms" in outcome:
        trials = sorted({r["trials"] for r in arm_rows(finding)})
        what = (
            "/".join(str(t) for t in trials)
            + " paired trials per arm, exact 95% intervals"
        )
    else:
        refits = outcome.get("replicates", "?")
        what = f"{refits} bootstrap refits, 2.5-97.5 % of the replicates"
    return (
        f"{finding.id} · commit {finding.repo_commit} · {finding.instrument} · "
        f"{what} · {finding.protocol}"
    )


def title_of(finding: Finding) -> str:
    """A title that names the study, not the whole claim: the claim up
    to its first colon or dash when that is short, else its first
    TITLE_CHARS characters; the record id on a second line so a reader
    can find the record from the figure alone (figure pass, 2026-09-06)."""
    claim = finding.claim.replace("WITHDRAWN: ", "")
    head = claim
    for sep in (": ", " — "):
        if sep in claim and len(claim.split(sep, 1)[0]) <= TITLE_HEAD_CHARS:
            head = claim.split(sep, 1)[0]
            break
    if len(head) > TITLE_CHARS:
        head = head[:TITLE_CHARS] + "…"
    status = " (WITHDRAWN)" if finding.claim.startswith("WITHDRAWN") else ""
    return textwrap.fill(head, TITLE_WRAP) + f"\n{finding.id}{status}"


def arm_colours(arm: str) -> tuple[str, str]:
    """Fill and edge for an arm: red for the measured or identified arm
    the text argues for, grey for a control, blue otherwise."""
    name = arm.lower()
    if any(k in name for k in HIGHLIGHT_ARMS):
        return FILL_RED, EDGE_RED
    if any(k in name for k in CONTROL_ARMS):
        return FILL_GREY, EDGE_GREY
    return FILL_BLUE, EDGE_BLUE


def panel_title(finding: Finding) -> str:
    """The study's name only, as the reference titles each panel with
    the environment's name; the record id rides in the footer. The
    claim up to its first colon, dash or parenthesis, at most two
    lines at the panel width."""
    claim = finding.claim.replace("WITHDRAWN: ", "")
    head = re.split(r": | — | - ", claim, maxsplit=1)[0].strip()
    # A claim's shouted phrase ("NEAR THE EXPERT'S EDGE") reads as a
    # title in sentence case; acronyms keep their capitals.
    head = re.sub(
        r"\b([A-Z]{3,}(?:'S)?)\b(?![-\d])",
        lambda m: m.group(1) if m.group(1) in TITLE_ACRONYMS else m.group(1).lower(),
        head,
    )
    if len(head) > 2 * TITLE_WRAP:
        head = head[: 2 * TITLE_WRAP - 1].rstrip() + "…"
    return textwrap.fill(head, TITLE_WRAP)


def _style(fig: Any, ax: Any, finding: Finding, *, title: str | None = None) -> None:
    """The frame every success-rate figure shares: y in [0, 1] with
    room for the counts above the whiskers, and the frame of `_frame`."""
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("success rate (\u2191)", color=INK, fontsize=7)
    _frame(fig, ax, finding, grid_axis="y", title=title)


def _frame(
    fig: Any, ax: Any, finding: Finding, *, grid_axis: str, title: str | None = None
) -> None:
    """A centred panel title, hairline grid and spines, small ticks,
    the provenance footer in the margin."""
    ax.set_title(title or panel_title(finding), fontsize=7.5, color=INK, pad=4)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
        ax.spines[side].set_linewidth(0.5)
    ax.tick_params(colors=INK_SECONDARY, labelsize=6.5, length=2, width=0.5)
    fig.tight_layout()


def _count_label(group: list[dict[str, Any]]) -> str:
    trials = sorted({r["trials"] for r in group})
    if len(group) == 1:
        return f"{group[0]['successes']}/{group[0]['trials']}"
    return "/".join(str(r["successes"]) for r in group) + f" of {trials[0]}"


def render(  # noqa: PLR0915 - one drawing, each statement a mark
    finding: Finding, root: Path, *, title: str | None = None
) -> dict[str, str]:
    """Write the figure in every format plus its CSV; returns the paths
    (relative to `root`) keyed by format — the finding's artifacts.
    A categorical row of arms is drawn as bars with the exact interval
    as a whisker and the count on the bar; a curve over episode counts
    as a thin line through the means with a dot per run."""
    import matplotlib  # noqa: PLC0415 - the train extra

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = arm_rows(finding)
    out_dir = Path(root) / FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = is_curve(rows)

    many = not curve and len(rows) > MANY_ARMS
    width = PANEL[0] + (0.32 * (len(rows) - MANY_ARMS) if many else 0.0)
    fig, ax = plt.subplots(figsize=(width, PANEL[1]), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [r["episodes"] for r in rows] if curve else list(range(len(rows)))
    ys = [r["rate"] for r in rows]
    lower = [r["rate"] - r["ci_low"] for r in rows]
    upper = [r["ci_high"] - r["rate"] for r in rows]
    if curve:
        ax.errorbar(
            xs,
            ys,
            yerr=[lower, upper],
            color=EDGE_BLUE,
            linestyle="none",
            marker="o",
            markersize=MARKER_PT,
            capsize=2,
            elinewidth=0.7,
        )
        by_x: dict[Any, list[float]] = {}
        for x, y in zip(xs, ys, strict=True):
            by_x.setdefault(x, []).append(y)
        line_x = sorted(by_x)
        ax.plot(
            line_x,
            [sum(by_x[x]) / len(by_x[x]) for x in line_x],
            color=EDGE_BLUE,
            linewidth=LINE_PT,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(set(xs)))
        ax.set_xticklabels([str(x) for x in sorted(set(xs))])
        ax.set_xlabel("demonstrations pressed (referee-gated)", color=INK, fontsize=7)
    else:
        for x, r, lo, hi in zip(xs, rows, lower, upper, strict=True):
            fill, edge = arm_colours(r["arm"])
            ax.bar(
                x, r["rate"], width=BAR_WIDTH, color=fill, edgecolor=edge, linewidth=0.7
            )
            ax.errorbar(
                [x],
                [r["rate"]],
                yerr=[[lo], [hi]],
                color=INK,
                capsize=2,
                elinewidth=0.7,
                linestyle="none",
                zorder=3,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [ARM_LABELS.get(r["arm"], r["arm"]) for r in rows],
            color=INK,
            rotation=30 if many else 0,
            ha="right" if many else "center",
        )
        ax.set_xlim(-0.6, len(rows) - 0.4)
    # The count on every bar or point: replicates that share an x get
    # one label listing each run's count over the shared trial count.
    shared: dict[Any, list[dict[str, Any]]] = {}
    for x, r in zip(xs, rows, strict=True):
        shared.setdefault(x, []).append(r)
    for x, group in [] if curve else shared.items():
        inside = len(group) == 1 and group[0]["rate"] > LABEL_INSIDE_ABOVE
        y = group[0]["rate"] / 2 if inside else max(r["ci_high"] for r in group)
        ax.annotate(
            _count_label(group),
            (x, y),
            textcoords="offset points",
            xytext=(0, 0 if inside else 3),
            ha="center",
            va="center" if inside else "bottom",
            fontsize=6.2,
            fontweight="bold" if inside else "normal",
            color=INK if inside else INK_SECONDARY,
        )
    _style(fig, ax, finding, title=title)

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


def interval_rows(finding: Finding) -> list[dict[str, Any]]:
    """One row per parameter of a bootstrap-interval record: the 2.5 %,
    median and 97.5 % values, the shipped (published) value, whether the
    bench identifies it, and the three as ratios to the shipped value -
    what the figure draws. Parameters with a non-positive value anywhere
    cannot sit on the log axis and are kept in the CSV with `plotted`
    false."""
    interval = finding.outcome.get("interval", {})
    if not interval:
        raise ValueError(f"finding {finding.id} carries no interval to plot")
    unidentified = set(finding.outcome.get("unidentified", []))
    rows = []
    for name, v in interval.items():
        values = (
            float(v["low"]),
            float(v["median"]),
            float(v["high"]),
            float(v["shipped"]),
        )
        positive = all(x > 0 for x in values)
        low, median, high, shipped = values
        rows.append(
            {
                "parameter": name,
                "low": low,
                "median": median,
                "high": high,
                "shipped": shipped,
                "identified": name not in unidentified,
                "plotted": positive,
                "rel_low": low / shipped if positive else None,
                "rel_median": median / shipped if positive else None,
                "rel_high": high / shipped if positive else None,
            }
        )
    halfwidth = finding.outcome.get("relative_halfwidth", {})
    rows.sort(
        key=lambda r: (
            not r["identified"],
            halfwidth.get(r["parameter"], 0.0),
            r["parameter"],
        )
    )
    return rows


def render_interval(
    finding: Finding, root: Path, *, title: str | None = None
) -> dict[str, str]:
    """The bootstrap-interval figure: one horizontal whisker per
    parameter on a log axis of value / published fit, the published fit
    at 1; identified parameters in the first hue, the ones the bench
    does not pin in the secondary ink. Same formats and CSV as
    `render`."""
    import matplotlib  # noqa: PLC0415 - the train extra

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = interval_rows(finding)
    shown = [r for r in rows if r["plotted"]]
    out_dir = Path(root) / FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 0.22 * len(shown) + 1.3), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = list(range(len(shown)))[::-1]
    for y, r in zip(ys, shown, strict=True):
        colour = EDGE_BLUE if r["identified"] else INK_SECONDARY
        ax.plot([r["rel_low"], r["rel_high"]], [y, y], color=colour, linewidth=LINE_PT)
        ax.plot(
            [r["rel_median"]], [y], color=colour, marker="o", markersize=MARKER_PT - 2
        )
        ax.annotate(
            f"{r['rel_low']:.2f}-{r['rel_high']:.2f}×",  # noqa: RUF001 - a ratio
            (r["rel_high"], y),
            textcoords="offset points",
            xytext=(5, -3),
            fontsize=7,
            color=INK_SECONDARY,
        )
    ax.axvline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            r["parameter"] + ("" if r["identified"] else " (not identified)")
            for r in shown
        ],
        fontsize=8,
    )
    ax.set_xlabel(
        "95 % bootstrap interval / published fit (dotted line = 1)", color=INK
    )
    refits = finding.outcome.get("replicates", "?")
    _frame(
        fig,
        ax,
        finding,
        grid_axis="x",
        # Wrapped narrower than a success-rate title: the long parameter
        # names push these axes to the right half of the canvas.
        title=title
        or textwrap.fill(
            f"Bootstrap intervals of the fit ({refits} refits), "
            "as ratios to the published parameters",
            INTERVAL_TITLE_WRAP,
        ),
    )

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


def matrix_rows(finding: Finding) -> list[dict[str, Any]]:
    """One row per (arm, condition) of a mismatch-matrix record: the
    scale s for a pinned condition ("x0.7", "fit" = 1.0), the span for
    a drawn one ("pm0.3"), successes, trials and the exact interval."""
    conditions = finding.outcome.get("conditions", [])
    arms = finding.outcome.get("arms", {})
    rows = []
    for arm, cells in arms.items():
        for cond in conditions:
            cell = cells.get(cond)
            if not isinstance(cell, dict) or "successes" not in cell:
                continue
            if cond == "fit":
                scale, span = 1.0, None
            elif cond.startswith("x"):
                scale, span = float(cond[1:]), None
            elif cond.startswith("pm"):
                scale, span = None, float(cond[2:])
            else:
                continue
            low, high = cell["ci95"]
            rows.append(
                {
                    "arm": arm,
                    "condition": cond,
                    "scale": scale,
                    "span": span,
                    "successes": int(cell["successes"]),
                    "trials": int(cell["trials"]),
                    "rate": cell["successes"] / cell["trials"],
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    if not rows:
        raise ValueError(f"finding {finding.id} has no matrix cells to plot")
    return rows


def render_matrix(  # noqa: PLR0915 - two panels, each statement a mark
    finding: Finding, root: Path, *, title: str | None = None
) -> dict[str, str]:
    """The mismatch matrix: per arm a thin line over the pinned scale
    with the exact interval as a shaded band (the reference's curves),
    and the drawn-span conditions as bars beside it."""
    import matplotlib  # noqa: PLC0415 - the train extra

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = matrix_rows(finding)
    out_dir = Path(root) / FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    spans = sorted({r["span"] for r in rows if r["span"] is not None})
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.7),
        dpi=DPI,
        gridspec_kw={"width_ratios": (4.2, 1.0 + 0.6 * len(spans)), "wspace": 0.08},
        sharey=True,
    )
    fig.patch.set_facecolor(SURFACE)
    for i, arm in enumerate(arms):
        line_colour = SERIES[i % len(SERIES)]
        band_colour = FILLS[i % len(FILLS)]
        pinned = sorted(
            (r for r in rows if r["arm"] == arm and r["scale"] is not None),
            key=lambda r: r["scale"],
        )
        xs = [r["scale"] for r in pinned]
        ax.plot(
            xs,
            [r["rate"] for r in pinned],
            color=line_colour,
            linewidth=LINE_PT,
            marker="o",
            markersize=MARKER_PT - 1,
            label=ARM_LABELS.get(arm, arm),
        )
        ax.fill_between(
            xs,
            [r["ci_low"] for r in pinned],
            [r["ci_high"] for r in pinned],
            color=band_colour,
            alpha=0.45,
            linewidth=0,
        )
        for j, span in enumerate(spans):
            cell = next(
                (r for r in rows if r["arm"] == arm and r["span"] == span), None
            )
            if cell is None:
                continue
            x = j * (len(arms) + 1) + i
            ax2.bar(
                x,
                cell["rate"],
                width=0.8,
                color=band_colour,
                edgecolor=line_colour,
                linewidth=0.7,
            )
            ax2.errorbar(
                [x],
                [cell["rate"]],
                yerr=[
                    [cell["rate"] - cell["ci_low"]],
                    [cell["ci_high"] - cell["rate"]],
                ],
                color=INK,
                capsize=1.5,
                elinewidth=0.6,
                linestyle="none",
                zorder=3,
            )
    ax.axvline(1.0, color=INK_SECONDARY, linewidth=0.5, linestyle=":")
    ax.set_xlabel("judged pinned at fit × s", color=INK, fontsize=7)  # noqa: RUF001
    ax.set_xticks(sorted({r["scale"] for r in rows if r["scale"] is not None}))
    ax.legend(
        fontsize=6,
        frameon=False,
        loc="lower right",
        title="trained under",
        title_fontsize=6,
    )
    ax2.set_xticks(
        [j * (len(arms) + 1) + (len(arms) - 1) / 2 for j in range(len(spans))]
    )
    ax2.set_xticklabels([f"drawn ±{sp:g}" for sp in spans], fontsize=6.5)
    ax2.set_xlabel("judged under a span", color=INK, fontsize=7)
    for axis in (ax, ax2):
        axis.set_facecolor(SURFACE)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(INK_SECONDARY)
            axis.spines[side].set_linewidth(0.5)
        axis.tick_params(colors=INK_SECONDARY, labelsize=6.5, length=2, width=0.5)
    ax.set_ylabel("success rate (\u2191)", color=INK, fontsize=7)
    fig.suptitle(title or panel_title(finding), fontsize=7.5, color=INK, y=0.99)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.88, wspace=0.08)
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


def render_any(
    finding: Finding, root: Path, *, title: str | None = None
) -> dict[str, str]:
    """Dispatch by the record's shape: arms → `render`, an interval →
    `render_interval`."""
    if "conditions" in finding.outcome:
        return render_matrix(finding, root, title=title)
    if "arms" in finding.outcome:
        return render(finding, root, title=title)
    if "interval" in finding.outcome:
        return render_interval(finding, root, title=title)
    raise ValueError(f"finding {finding.id} has neither arms nor an interval to draw")
