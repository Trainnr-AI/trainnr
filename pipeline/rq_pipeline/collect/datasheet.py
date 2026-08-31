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

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from rq_pipeline.collect.kitting_export import (
    UNSTAMPED_EXPERT,
    DemoLayout,
    episode_dirs,
)

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
    }


def summarize(demos_dir: str | Path) -> DatasheetSummary:
    """Fold every episode's sidecar into the batch's summary.

    Takes str or Path: the one-liner in the data-press skill passed a
    string and crashed in `episode_dirs` — normalising in only one of
    the two entry points is the bug (review 2026-09-01)."""
    demos_dir = Path(demos_dir)
    normalized = [
        _normalize(json.loads((ep / DemoLayout.MANIFEST_FILE).read_text()))
        for ep in episode_dirs(demos_dir)
    ]
    draws: dict[str, list[float]] = {}
    for episode in normalized:
        for param, value in episode["dynamics"].items():
            draws.setdefault(param, []).append(float(value))

    def distinct(key: str) -> tuple[str, ...]:
        return tuple(sorted({episode[key] for episode in normalized}))

    tasks, experts = distinct("task"), distinct("expert")
    instruments, bases = distinct("instrument"), distinct("basis")
    shards = len({episode["seed"] for episode in normalized})
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
    if shards > 1:
        # Attempt counters restart per press run, so a directory filled
        # by N shards has no single denominator — the bound would read
        # "400%" (review 2026-09-01).
        warnings.append(
            f"{shards} shards (distinct seeds) shipped into one directory: "
            "the keep-rate bound is per-shard and not meaningful here"
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
    )


def render(summary: DatasheetSummary) -> str:
    """The summary as a markdown page."""
    lines = [
        "# Datasheet",
        "",
        "Every episode in this batch passed its task's own referee; the",
        "counts below are of SUCCESSFUL episodes only.",
        "",
        f"- episodes kept: **{summary.episodes}**",
        (
            f"- highest attempt number recorded: {summary.max_attempt}"
            f" (keep rate ≤ {summary.keep_rate_bound:.0%} — attempts after the"
            " last keep are not recorded)"
            if summary.shards == 1
            else f"- {summary.shards} shards in this directory: no single"
            " keep-rate bound (each shard restarts its attempt counter)"
        ),
        f"- expert retries across the batch: {summary.retries_total}",
        "",
        "## Stamps",
        "",
        f"- task: {', '.join(summary.tasks)}",
        f"- expert: {', '.join(summary.experts)}",
        f"- instrument: {', '.join(summary.instruments)}",
        "",
        "## Dynamics draws",
        "",
        f"Basis: {', '.join(summary.bases)}",
        "",
        "| parameter | low | mean | high |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {param} | {s.low:.4g} | {s.mean:.4g} | {s.high:.4g} |"
        for param, s in sorted(summary.dynamics.items())
    ]
    if summary.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in summary.warnings]
    return "\n".join(lines) + "\n"


def write_datasheet(demos_dir: str | Path) -> Path:
    path = Path(demos_dir) / DATASHEET_FILE
    path.write_text(render(summarize(demos_dir)))
    return path
