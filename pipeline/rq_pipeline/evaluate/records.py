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
    instrument: str  # backend and version, e.g. "mujoco-3.11.0"
    protocol: Mapping[str, Any]  # the scalar protocol fields
    seed: int | None = None
    events: tuple[Mapping[str, Any], ...] = ()
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if "@" not in self.source:
            raise ValueError(f"source must be a name@hash stamp, got {self.source!r}")
        if self.trial < 0 or self.steps <= 0:
            raise ValueError(
                f"trial must be >= 0 and steps > 0, got {self.trial}, {self.steps}"
            )


def protocol_fields(protocol: Any) -> dict[str, Any]:
    """The scalar half of an `EpisodeProtocol` — what a row can carry."""
    return {
        "trials": protocol.trials,
        "steps": protocol.steps,
        "control_interval": protocol.control_interval,
        "home": protocol.home,
    }


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


def from_eval_info(  # noqa: PLR0913 - every argument is identity the file lacks
    eval_info: Mapping[str, Any],
    *,
    policy: str,
    source: str,
    instrument: str,
    protocol: Mapping[str, Any],
    start_seed: int,
    trials: int,
    steps: int,
) -> tuple[EpisodeRecord, ...]:
    """LeRobot 0.6.1's `eval_info.json` → records.

    The file carries `per_task[0].metrics.successes` as a list in episode
    order and no seed; `lerobot-eval` seeded episode i with
    `start_seed + i` and our env started trial `seed % trials`, so both
    are recovered from the order. One task per file is assumed and
    checked.
    """
    per_task = eval_info["per_task"]
    if len(per_task) != 1:
        raise ValueError(f"expected one task in eval_info, got {len(per_task)}")
    successes = per_task[0]["metrics"]["successes"]
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
