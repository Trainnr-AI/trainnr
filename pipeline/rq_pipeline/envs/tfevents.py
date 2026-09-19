"""The trainer's own record: the TensorBoard event file rsl_rl writes
into every run folder, read as it is.

It holds what the console never prints - every reward term per
iteration, the curriculum's state, the policy's action noise, the
collection and learning times - and the Studio parsed console text
instead until 2026-09-11 (the operator's rule: additive over the
libraries; consume what they write). The console log stays the
fallback for a run that predates the file or trained elsewhere, and
the source of the two facts the file lacks: the planned iteration
count and the environment count.

Column names: the five the console parser named keep their names
(`reward`, `episode_length`, `steps_per_second`, `value_loss`,
`entropy`) so every card and view reads the same record; the rest are
the tag lower-cased with its group (`reward/track_linear_velocity`,
`loss/surrogate`, `perf/collection_time`, `curriculum/…`).
"""

from __future__ import annotations

from pathlib import Path

from rq_pipeline.envs.rsl_rl_log import (
    COL_ENTROPY,
    COL_EPISODE_LENGTH,
    COL_ITERATION,
    COL_REWARD,
    COL_STEPS_PER_SECOND,
    COL_VALUE_LOSS,
    MAX_POINTS,
    TrainingRecord,
)

EVENTS_GLOB = "events.out.tfevents.*"
# rsl_rl's tag -> the console parser's column of the same quantity.
NAMED = {
    "Train/mean_reward": COL_REWARD,
    "Train/mean_episode_length": COL_EPISODE_LENGTH,
    "Perf/total_fps": COL_STEPS_PER_SECOND,
    "Loss/value": COL_VALUE_LOSS,
    "Loss/entropy": COL_ENTROPY,
}
GROUPS = {"Episode_Reward": COL_REWARD}  # every other group keeps its own name
TRAINER = "rsl_rl (tensorboard)"
SKIP_SUFFIX = "/time"  # rsl_rl's wall-clock twins of the step series


def events_file(folder: Path) -> Path | None:
    """The newest event file in a run folder, or None."""
    found = sorted(Path(folder).glob(EVENTS_GLOB), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def column_name(tag: str) -> str | None:
    """`Episode_Reward/pose` -> `reward/pose`; `Train/mean_reward` ->
    `reward`; None for a tag the record does not keep."""
    if tag in NAMED:
        return NAMED[tag]
    if tag.endswith(SKIP_SUFFIX):
        return None
    group, _, name = tag.partition("/")
    if not name:
        return tag.lower()
    return f"{GROUPS.get(group, group.lower())}/{name.lower()}"


def read_scalars(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Every scalar series in the file: tag -> [(step, value)], steps
    ascending. TensorBoard's own reader, so the format is theirs."""
    from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
        EventAccumulator,
    )

    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    return {
        tag: sorted((int(e.step), float(e.value)) for e in acc.Scalars(tag))
        for tag in acc.Tags().get("scalars", [])
    }


def record_from_scalars(
    scalars: dict[str, list[tuple[int, float]]],
    *,
    facts: TrainingRecord | None = None,
    max_points: int = MAX_POINTS,
) -> TrainingRecord | None:
    """The training record from the scalar series, sampled down to
    `max_points` rows like the console record; `facts` carries what the
    file lacks (planned iterations, envs, device, wall seconds) - the
    planned count stays unrecorded without them: the file holds the
    iterations run, never the number asked for."""
    columns_by_tag = {t: c for t in scalars if (c := column_name(t))}
    if not columns_by_tag or COL_REWARD not in columns_by_tag.values():
        return None
    by_step: dict[int, dict[str, float]] = {}
    for tag, column in columns_by_tag.items():
        for step, value in scalars[tag]:
            by_step.setdefault(step, {})[column] = value
    steps = sorted(by_step)
    named = [c for c in NAMED.values() if c in columns_by_tag.values()]
    others = sorted(c for c in set(columns_by_tag.values()) - set(named))
    columns = [COL_ITERATION, *named, *others]
    record = TrainingRecord(**(facts.to_json() if facts else {}))
    record.trainer = TRAINER
    record.iterations_logged = len(steps)
    rewards = [by_step[s][COL_REWARD] for s in steps if COL_REWARD in by_step[s]]
    record.best_reward = max(rewards) if rewards else None
    record.final = {c: v for c, v in by_step[steps[-1]].items()}
    record.columns = columns
    stride = max(1, -(-len(steps) // max_points))
    sampled = steps[::stride]
    if sampled[-1] != steps[-1]:
        sampled.append(steps[-1])
    record.curve = [
        [float(s), *[by_step[s].get(c) for c in columns[1:]]] for s in sampled
    ]
    return record


def record_from_events(
    path: Path, *, facts: TrainingRecord | None = None
) -> TrainingRecord | None:
    return record_from_scalars(read_scalars(path), facts=facts)


def curve_groups(columns: list[str]) -> dict[str, list[str]]:
    """How a viewer lays the columns out: the named totals one panel
    each, every `group/name` column in its group's panel."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        if c == COL_ITERATION:
            continue
        group = c.split("/", 1)[0] + " terms" if "/" in c else c
        groups.setdefault(group, []).append(c)
    return groups


__all__: list[str] = [
    "column_name",
    "curve_groups",
    "events_file",
    "read_scalars",
    "record_from_events",
    "record_from_scalars",
]
