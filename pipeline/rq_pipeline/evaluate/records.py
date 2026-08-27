"""The per-trial record: one JSON line per episode, and the fold that
turns lines into counts.

Written 2026-08-26 (docs/32 §4.3, docs/30 §7). Five of the six Arena
reports converged on this artifact: the funnel, the sensitivity table,
the placement verdicts and the executed horizon are all fields of the
row the runner throws away today. LeRobot's `eval_info.json` keeps a
success list in episode order and no seed (0.6.1); openpi and GR00T
print; so the row is ours to write, in LeRobot's shape widened with the
identity a certificate needs — the stamped source, the instrument
(backend + version), the protocol, and the trial index that pairs
policies.

`SimScore` lives here because it IS the fold: counts, never a rate, so
the trial count survives into every downstream statistic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rq_pipeline.bundles.hashing import fields_hash, require_stamp
from rq_pipeline.protocol import protocol_fields

__all__ = [
    "EpisodeRecord",
    "SimScore",
    "append_records",
    "disagreements",
    "fold",
    "from_success_list",
    "funnel",
    "milestones",
    "passes",
    "protocol_fields",
    "protocol_hash",
    "read_records",
]


@dataclass(frozen=True)
class SimScore:
    """One policy's showing in simulation, as counts — never just a rate,
    so the trial count survives into every downstream statistic."""

    name: str
    successes: int
    trials: int

    @property
    def score(self) -> float:
        return self.successes / self.trials


@dataclass(frozen=True)
class EpisodeRecord:
    """One trial of one policy under one protocol on one stamped source.

    `trial` is the pairing key: policy A's trial 7 and policy B's trial 7
    started identically. `seed` is what the runner passed (LeRobot's
    `seed + i`), None when the harness drove the trial by index. `steps`
    is physics steps executed. `events` is reserved for milestones
    (docs/30 §7 row 39) — empty until they land.
    """

    source: str  # the task/scene as name@hash
    policy: str
    trial: int
    success: bool
    steps: int
    instrument: str  # physics/backend.py::instrument_stamp, e.g. "mujoco-3.11.0+x86_64"
    protocol: Mapping[str, Any]  # the scalar protocol fields
    seed: int | None = None
    events: tuple[Mapping[str, Any], ...] = ()
    variations: Mapping[str, Any] = field(default_factory=dict)  # key -> drawn value
    # body -> check -> passed, for the protocol's declared placements
    # (physics/placement.py); empty when the protocol declares none.
    placement: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        require_stamp(self.source)
        if self.trial < 0 or self.steps <= 0:
            raise ValueError(
                f"trial must be >= 0 and steps > 0, got {self.trial}, {self.steps}"
            )

    @property
    def protocol_hash(self) -> str:
        return protocol_hash(self.protocol)


def protocol_hash(fields: Mapping[str, Any]) -> str:
    """Twelve hex digits over the protocol's scalar fields (and its
    variation space, when the env adds one): the seed of every draw, so
    two sweeps that differ in any declared knob draw different values."""
    return fields_hash(fields)


def milestones(records: Sequence[EpisodeRecord]) -> list[str]:
    """The milestone names the records were judged by — read from the
    records, never restated (a fourth copy of the chain, in a tool,
    silently mislabelled a funnel; 2026-08-28)."""
    if not records:
        raise ValueError("no records to read milestones from")
    return list(records[0].protocol.get("milestones", []))


def funnel(records: Sequence[EpisodeRecord]) -> dict[str, list[int]]:
    """Per policy, how many trials reached each milestone: the reading
    that separates "did nothing" from "grasped, then dropped". Stage
    names come from the records' protocol and must agree across them."""
    if not records:
        raise ValueError("no records to funnel")
    stages = milestones(records)
    counts: dict[str, list[int]] = {}
    for record in records:
        if list(record.protocol.get("milestones", [])) != stages:
            raise ValueError(
                f"record {record.policy!r}/{record.trial} has milestones "
                f"{record.protocol.get('milestones')}, expected {stages}"
            )
        reached = counts.setdefault(record.policy, [0] * len(stages))
        for event in record.events:
            reached[int(event["index"])] += 1
    return counts


def disagreements(records: Sequence[EpisodeRecord]) -> tuple[EpisodeRecord, ...]:
    """Records whose verdict and chain disagree: success with the chain
    incomplete means the referee is looser than the milestones (a
    referee-bug detector); failure with the chain complete means the
    hold window rejected a late drop. Either is worth a look; neither
    changes a count."""
    flagged = []
    for record in records:
        chain = record.protocol.get("milestones", [])
        if not chain:
            continue
        if record.success != (len(record.events) == len(chain)):
            flagged.append(record)
    return tuple(flagged)


def append_records(path: Path, records: Iterable[EpisodeRecord]) -> None:
    """Append rows as JSON lines. NaN is refused, not written."""
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), allow_nan=False) + "\n")


def read_records(path: Path) -> tuple[EpisodeRecord, ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["events"] = tuple(row.get("events", ()))
                rows.append(EpisodeRecord(**row))
    return tuple(rows)


def passes(records: Sequence[EpisodeRecord]) -> tuple[tuple[EpisodeRecord, ...], ...]:
    """A file that several evaluations appended to, split back into them.

    LeRobot's trainer evaluates at every checkpoint and our env appends
    every episode to ONE file under one policy name, so trial 0 recurs
    once per checkpoint — and `fold` rightly refuses a repeated
    (policy, trial). A pass ends where a (policy, trial) is seen again;
    each pass folds on its own (measured 2026-08-27: the first cloud
    run's fold crashed on its own in-loop file).
    """
    out: list[list[EpisodeRecord]] = []
    seen: set[tuple[str, int]] = set()
    current: list[EpisodeRecord] = []
    for record in records:
        key = (record.policy, record.trial)
        if key in seen:
            out.append(current)
            current, seen = [], set()
        seen.add(key)
        current.append(record)
    if current:
        out.append(current)
    return tuple(tuple(chunk) for chunk in out)


def fold(records: Sequence[EpisodeRecord]) -> tuple[SimScore, ...]:
    """Records → one `SimScore` per policy, in first-seen order.

    Refuses what would silently break pairing: a (policy, trial) seen
    twice, or policies whose trial sets differ — a hole is an error with
    a named culprit, never a dropped row (Arena and LeRobot both let an
    unscored episode leave the denominator; docs/30 §7).
    """
    trials_by_policy: dict[str, dict[int, bool]] = {}
    for record in records:
        seen = trials_by_policy.setdefault(record.policy, {})
        if record.trial in seen:
            raise ValueError(
                f"policy {record.policy!r} has two records for trial {record.trial}"
            )
        seen[record.trial] = record.success
    if not trials_by_policy:
        raise ValueError("no records to fold")
    reference_name, reference = next(iter(trials_by_policy.items()))
    for name, seen in trials_by_policy.items():
        if set(seen) != set(reference):
            raise ValueError(
                f"policies are not paired: {name!r} ran trials "
                f"{sorted(seen)} while {reference_name!r} ran {sorted(reference)}"
            )
    return tuple(
        SimScore(name=name, successes=sum(seen.values()), trials=len(seen))
        for name, seen in trials_by_policy.items()
    )


def from_success_list(  # noqa: PLR0913 - every argument is identity the list lacks
    successes: Sequence[bool],
    *,
    policy: str,
    source: str,
    instrument: str,
    protocol: Mapping[str, Any],
    start_seed: int,
    trials: int,
    steps: int,
) -> tuple[EpisodeRecord, ...]:
    """An ordered success list from a runner that seeded episode i with
    `start_seed + i` (LeRobot's convention; `envs/lerobot_info.py` reads
    its file) → records with seed and pairing trial recovered from the
    order. Events and draws are unknown to such a runner and stay empty."""
    return tuple(
        EpisodeRecord(
            source=source,
            policy=policy,
            trial=(start_seed + index) % trials,
            success=bool(success),
            steps=steps,
            instrument=instrument,
            protocol=dict(protocol),
            seed=start_seed + index,
        )
        for index, success in enumerate(successes)
    )
