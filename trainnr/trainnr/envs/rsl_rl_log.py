"""Read an rsl_rl console log (the PPO trainer mjlab and Isaac Lab use)
into a training record: what the run was and how its reward moved.

rsl_rl prints one block per learning iteration — a banner line
`Learning iteration i/N`, then `key: value` lines (`Mean reward`,
`Mean episode length`, `Steps per second`, …), then `Iteration time` and
`Time elapsed`. Nothing here is specific to a robot or a task; any rsl_rl
run reads the same way.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = "trainnr-training/1"
# The record beside a run folder (`project/live`, `project/importer` write
# it; the index, the drawers, the previews and the presenter read it).
TRAINING_FILE = "training.json"
ITERATION_LINE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
FIELD_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z _/]*?):\s*([-+0-9.eE]+)\s*s?\s*$")
ENVS_LINE = re.compile(r"(\d+) envs on (\S+),\s*(\d+) iterations")
ELAPSED_LINE = re.compile(r"Time elapsed:\s*(?:(\d+) days?, )?(\d+):(\d\d):(\d\d)")
# The record's column names, spelled once: the x-axis and the five
# quantities the console prints; the event file's other series take
# `group/name` (`envs/tfevents.column_name`).
COL_ITERATION = "iteration"
COL_TOTAL_STEPS = "total_steps"  # environment steps, RL's usual x-axis
COL_REWARD = "reward"
COL_EPISODE_LENGTH = "episode_length"
COL_STEPS_PER_SECOND = "steps_per_second"
COL_VALUE_LOSS = "value_loss"
COL_ENTROPY = "entropy"
# The per-iteration numbers the record keeps, by their name in the log.
CURVE_FIELDS = {
    "Total steps": COL_TOTAL_STEPS,
    "Mean reward": COL_REWARD,
    "Mean episode length": COL_EPISODE_LENGTH,
    "Steps per second": COL_STEPS_PER_SECOND,
    "Mean value loss": COL_VALUE_LOSS,
    "Mean entropy loss": COL_ENTROPY,
}
MAX_POINTS = 400  # the record samples a long run down to this many rows

# trainnr_mjlab's walk tools around the trainer (walk_train, walk_verdict),
# their conventions named once for whoever reads a run folder.
TRAIN_LOG = "train.log"  # walk_train tees its console here
DONE_MARK = "] done"  # walk_train's last line: "[g3] done - checkpoints in ..."
VERDICT_DIR = "verdict"  # walk_verdict writes beside the checkpoint it judged
VERDICT_PREFIX = "walk-verdict-"
VERDICT_GLOB = f"{VERDICT_PREFIX}*.json"
STATUS_RUNNING, STATUS_DONE = "running", "done"


@dataclass
class TrainingRecord:
    """A run's training facts and its sampled curve."""

    schema: str = SCHEMA
    trainer: str = "rsl_rl"
    iterations: int | None = None  # the run's planned count
    iterations_logged: int = 0
    envs: int | None = None
    device: str | None = None
    wall_seconds: int | None = None
    final: dict[str, float] = field(default_factory=dict)
    best_reward: float | None = None
    columns: list[str] = field(default_factory=list)
    curve: list[list[float | None]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def parse_rsl_rl_log(
    text: str, *, max_points: int = MAX_POINTS
) -> TrainingRecord | None:
    """The record, or None when the text holds no learning iteration."""
    record = TrainingRecord()
    rows: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in text.splitlines():
        m = ENVS_LINE.search(line)
        if m and record.envs is None:
            record.envs, record.device = int(m.group(1)), m.group(2)
            record.iterations = int(m.group(3))
            continue
        m = ITERATION_LINE.search(line)
        if m:
            current = {COL_ITERATION: float(m.group(1))}
            rows.append(current)
            record.iterations = record.iterations or int(m.group(2))
            continue
        if current is None:
            continue
        m = ELAPSED_LINE.search(line)
        if m:
            days, h, mi, s = (int(g or 0) for g in m.groups())
            record.wall_seconds = days * 86400 + h * 3600 + mi * 60 + s
            continue
        m = FIELD_LINE.match(line)
        if m and m.group(1).strip() in CURVE_FIELDS:
            try:
                current[CURVE_FIELDS[m.group(1).strip()]] = float(m.group(2))
            except ValueError:
                continue
    if not rows:
        return None
    record.iterations_logged = len(rows)
    columns = [
        COL_ITERATION,
        *[c for c in CURVE_FIELDS.values() if any(c in r for r in rows)],
    ]
    record.columns = columns
    rewards = [r[COL_REWARD] for r in rows if COL_REWARD in r]
    record.best_reward = max(rewards) if rewards else None
    record.final = {c: rows[-1][c] for c in columns[1:] if c in rows[-1]}
    step = max(1, -(-len(rows) // max_points))
    sampled = rows[::step]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    # A missing cell is None (a log cut mid-iteration): JSON has no NaN.
    record.curve = [[r.get(c) for c in columns] for r in sampled]
    return record
