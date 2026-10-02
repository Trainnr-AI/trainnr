"""LeRobot's `eval_info.json` → our records (standard library only).

LeRobot 0.6.1 writes `per_task[0].metrics.successes` as a list in
episode order and no per-episode seed; `lerobot-eval` seeded episode i
with `start_seed + i` and our env started trial `seed % trials`, so both
are recovered from the order. The env writes our own row when given
`--env.record_to`; this reader is for runs that did not ask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainnr.evaluate.records import EpisodeRecord, from_success_list

PER_TASK = "per_task"
METRICS = "metrics"
SUCCESSES = "successes"


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
    per_task = eval_info[PER_TASK]
    if len(per_task) != 1:
        raise ValueError(f"expected one task in eval_info, got {len(per_task)}")
    return from_success_list(
        per_task[0][METRICS][SUCCESSES],
        policy=policy,
        source=source,
        instrument=instrument,
        protocol=protocol,
        start_seed=start_seed,
        trials=trials,
        steps=steps,
    )
