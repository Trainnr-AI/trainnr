"""The datasheet: what a batch says about itself before anyone asks.

The report a dataset consumer reads before trusting (docs/e2e-research/
60 §3) — kept episodes and the honest bound on the keep rate, whose
stamps produced them (task, expert, instrument), the dynamics draws
with their per-parameter spread, the BASIS every draw came from
(identified interval vs caller-declared span — A3's string, recorded
so it cannot be retold), and a warnings section that says out loud what
most datasets hide: mixed stamps, unrecorded provenance, mixed bases.
Nobody in the studied ecosystems ships one (Arena's dataset provenance
is a seed — 59 §5).

Reads BOTH sidecar schemas: the press's `EpisodeManifest` and the
kitting task's legacy `Manifest` (whose damping/gain scales normalize
into a dynamics dict, and whose `dr_span` becomes its basis). A legacy
manifest's absent stamps are reported as unrecorded, never invented.

Stdlib only, deliberately — a datasheet must render on an auditor's
laptop; `statistics` does the arithmetic.
"""

from __future__ import annotations

import itertools
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from trainnr.collect.kitting_export import (
    UNSTAMPED_EXPERT,
    DemoLayout,
    episode_dirs,
)
from trainnr.collect.shards import MergedRate, merge_records, read_shard_records

DATASHEET_FILE = "datasheet.md"
# Two different absences, two different names: a LEGACY sidecar never
# had the field, while a go-forward one left it empty — reporting the
# second as "legacy" named the wrong artifact (review 2026-09-01).
UNRECORDED = "unrecorded (legacy manifest)"
UNSTATED = "unstated (manifest left it empty)"


@dataclass(frozen=True)
class DynamicsSpread:
    """One parameter's draws across the batch."""

    low: float
    mean: float
    high: float

    @classmethod
    def of(cls, values: list[float]) -> DynamicsSpread:
        return cls(min(values), statistics.fmean(values), max(values))


@dataclass(frozen=True)
class DatasheetSummary:
    """Everything the rendered page states, as data — so tests assert
    facts, not markdown."""

    episodes: int
    max_attempt: int
    tasks: tuple[str, ...]
    experts: tuple[str, ...]
    instruments: tuple[str, ...]
    bases: tuple[str, ...]
    dynamics: dict[str, DynamicsSpread]
    retries_total: int
    # How many distinct seeds shipped into this directory: N generators
    # with disjoint episode ranges fill ONE batch in parallel, and each
    # restarts its own attempt counter.
    shards: int = 1
    warnings: tuple[str, ...] = field(default=())
    # The merge (collect/shards.py): shard records sum every attempt,
    # kept or not, so a sharded batch states an EXACT keep rate where
    # a single run can only bound it.
    merged: MergedRate | None = None
    # Visual draws (lighting, camera pose — a note), spread per
    # scalar knob or per vector component; empty for batches pressed
    # without visual DR.
    visuals: dict[str, DynamicsSpread] = field(default_factory=dict)
    visual_bases: tuple[str, ...] = ()

    @property
    def keep_rate_bound(self) -> float:
        """kept/max_attempt — an UPPER bound on the true keep rate: the
        batch records each kept episode's attempt number, not the
        attempts after the last keep. Meaningless across shards (each
        restarts its counter), so `render` states it only for one."""
        return self.episodes / self.max_attempt if self.max_attempt else 0.0


def _normalize(raw: dict) -> dict:
    """Either sidecar schema → one shape; absences named, not invented."""
    if "dynamics" in raw:  # the press's EpisodeManifest
        return {
            "task": raw.get("task", UNRECORDED),
            "expert": raw.get("expert", UNRECORDED),
            "instrument": raw.get("instrument", UNRECORDED),
            "seed": raw.get("seed"),
            "dynamics": raw["dynamics"],
            "basis": raw.get("dynamics_basis") or UNSTATED,
            "attempt": raw["attempt"],
            "retries": raw.get("retries", []),
            "visuals": raw.get("visuals") or {},
            "visual_basis": raw.get("visual_basis") or "",
        }
    # The kitting task's legacy Manifest: scales into a dynamics dict,
    # its declared span into a basis.
    return {
        "task": UNRECORDED,
        "expert": raw.get("expert", UNRECORDED),
        "instrument": UNRECORDED,
        "seed": raw.get("seed"),
        "dynamics": {
            "damping": raw["damping_scale"],
            "gain": raw["gain_scale"],
        },
        "basis": f"caller-declared span ±{raw['dr_span']:g} (legacy kitting manifest)",
        "attempt": raw["attempt"],
        "retries": raw.get("retries", []),
        "visuals": {},
        "visual_basis": "",
    }


def _fold_visuals(
    normalized: list[dict],
) -> tuple[dict[str, list[float]], tuple[str, ...]]:
    """Visual draws across the batch — vector draws (a camera offset)
    spread per component — and the bases they were drawn under."""
    visual_draws: dict[str, list[float]] = {}
    for episode in normalized:
        for key, value in episode["visuals"].items():
            if isinstance(value, (list, tuple)):
                for i, component in enumerate(value):
                    visual_draws.setdefault(f"{key}[{i}]", []).append(float(component))
            else:
                visual_draws.setdefault(key, []).append(float(value))
    # the basis is named with or without draws: a captured scene's splat
    # is a visual basis nothing was drawn from (docs/78 E3)
    bases = tuple(sorted({e["visual_basis"] for e in normalized if e["visual_basis"]}))
    return visual_draws, bases


def summarize(demos_dir: str | Path) -> DatasheetSummary:
    """Fold every episode's sidecar into the batch's summary.

    Takes str or Path: the one-liner in the data-press skill passed a
    string and crashed in `episode_dirs` — normalising in only one of
    the two entry points is the bug (review 2026-09-01)."""
    demos_dir = Path(demos_dir)
    # An episode directory without its manifest yet belongs to a shard
    # still writing it (two presses fill one directory, a note D4):
    # not kept yet, not counted - a crash here took a shard down with
    # it (2026-09-03). A manifest that exists but does not parse still
    # raises: that is corruption, not timing.
    manifests = [
        ep / DemoLayout.MANIFEST_FILE
        for ep in episode_dirs(demos_dir)
        if (ep / DemoLayout.MANIFEST_FILE).exists()
    ]
    normalized = [
        _normalize(json.loads(path.read_text(encoding="utf-8"))) for path in manifests
    ]
    draws: dict[str, list[float]] = {}
    for episode in normalized:
        for param, value in episode["dynamics"].items():
            draws.setdefault(param, []).append(float(value))
    visual_draws, visual_bases = _fold_visuals(normalized)

    def distinct(key: str) -> tuple[str, ...]:
        return tuple(sorted({episode[key] for episode in normalized}))

    tasks, experts = distinct("task"), distinct("expert")
    instruments, bases = distinct("instrument"), distinct("basis")
    # Shards: distinct seeds AND attempt-counter restarts. Seeds alone
    # collapse when N runs share the default seed with disjoint episode
    # ranges (the documented sharding pattern), and legacy manifests may
    # carry no seed at all — but each shard restarts its attempt counter,
    # so a DECREASE in attempt number across the episode order is a
    # shard boundary regardless (second review, 2026-09-01).
    seeds = {episode["seed"] for episode in normalized if episode["seed"] is not None}
    restarts = 1 + sum(
        1
        for previous, current in itertools.pairwise(normalized)
        if current["attempt"] < previous["attempt"]
    )
    shards = max(len(seeds), restarts)
    warnings = [
        f"mixed {name} stamps: {values}"
        for name, values in (
            ("task", tasks),
            ("expert", experts),
            ("instrument", instruments),
        )
        if len(values) > 1
    ]
    if len(bases) > 1:
        warnings.append(f"mixed dynamics bases: {bases}")
    if len(visual_bases) > 1:
        warnings.append(f"mixed visual bases: {visual_bases}")
    if visual_draws and any(not e["visuals"] for e in normalized):
        warnings.append("visual draws on some episodes only (mixed batch)")
    for name, values in (
        ("task", tasks),
        ("instrument", instruments),
        ("expert", experts),
    ):
        for absence in (UNRECORDED, UNSTATED):
            if absence in values:
                warnings.append(f"{name} stamp {absence} on some episodes")
    if UNSTAMPED_EXPERT in experts:
        warnings.append("expert unstamped on some episodes (pre-2026-08-27 batch)")
    records = read_shard_records(demos_dir)
    merged = merge_records(records) if records else None
    if merged is not None and merged.kept != len(normalized):
        warnings.append(
            f"shard records count {merged.kept} kept episodes, the directory "
            f"holds {len(normalized)}: a shard is missing or was re-pressed"
        )
    if shards > 1 and merged is None:
        # Attempt counters restart per press run, so a directory filled
        # by N shards has no single denominator — the bound would read
        # "400%" (review 2026-09-01). The merge's shard records fix that.
        warnings.append(
            f"{shards} shards (distinct seeds) shipped into one directory "
            "without shard records: the keep-rate bound is per-shard and "
            "not meaningful here"
        )

    return DatasheetSummary(
        episodes=len(normalized),
        max_attempt=max(episode["attempt"] for episode in normalized),
        tasks=tasks,
        experts=experts,
        instruments=instruments,
        bases=bases,
        dynamics={param: DynamicsSpread.of(vals) for param, vals in draws.items()},
        retries_total=sum(len(episode["retries"]) for episode in normalized),
        shards=shards,
        warnings=tuple(warnings),
        merged=merged,
        visuals={key: DynamicsSpread.of(vals) for key, vals in visual_draws.items()},
        visual_bases=visual_bases,
    )


def _keep_rate_line(summary: DatasheetSummary) -> str:
    if summary.merged is not None:
        m = summary.merged
        return (
            f"- keep rate **{m.rate:.0%}** exactly: {m.kept} kept of {m.attempts}"
            f" attempts across {m.shards} shards (merged shard records)"
        )
    if summary.shards == 1:
        return (
            f"- highest attempt number recorded: {summary.max_attempt}"
            f" (success rate ≤ {summary.keep_rate_bound:.0%} — attempts after the"
            " last success are not recorded)"
        )
    return (
        f"- {summary.shards} shards in this directory: no single"
        " keep-rate bound (each shard restarts its attempt counter)"
    )


def render(summary: DatasheetSummary) -> str:
    """The summary as a markdown page."""
    lines = [
        "# Datasheet",
        "",
        "Every episode in this dataset passed its environment's success",
        "criterion; the counts below are of SUCCESSFUL episodes only.",
        "",
        f"- successful episodes: **{summary.episodes}**",
        _keep_rate_line(summary),
        f"- scripted-policy retries across the dataset: {summary.retries_total}",
        "",
        "## Versions",
        "",
        f"- environment: {', '.join(summary.tasks)}",
        f"- scripted policy: {', '.join(summary.experts)}",
        f"- simulator build: {', '.join(summary.instruments)}",
        "",
        "## Domain randomization draws",
        "",
        f"Range: {', '.join(summary.bases)}",
        "",
        "| parameter | low | mean | high |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {param} | {s.low:.4g} | {s.mean:.4g} | {s.high:.4g} |"
        for param, s in sorted(summary.dynamics.items())
    ]
    if summary.visual_bases and not summary.visuals:
        lines += ["", "## Visuals", "", f"Basis: {', '.join(summary.visual_bases)}"]
    if summary.visuals:
        lines += [
            "",
            "## Visual draws",
            "",
            f"Range: {', '.join(summary.visual_bases)}",
            "",
            "| knob | low | mean | high |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {key} | {s.low:.4g} | {s.mean:.4g} | {s.high:.4g} |"
            for key, s in sorted(summary.visuals.items())
        ]
    if summary.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in summary.warnings]
    return "\n".join(lines) + "\n"


def write_datasheet(demos_dir: str | Path) -> Path:
    path = Path(demos_dir) / DATASHEET_FILE
    path.write_text(render(summarize(demos_dir)), encoding="utf-8")
    return path
