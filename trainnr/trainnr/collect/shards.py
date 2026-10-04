"""Sharding a press: N runs with disjoint episode ranges and distinct
seeds fill ONE batch directory, and a merge restores what sharding
took away - the keep rate (a note, D4).

The datasheet once REFUSED to state a keep-rate bound for a sharded
directory (right: each press restarts its attempt counter, so the
manifests alone have no common denominator). Each shard now leaves a
`ShardRecord` beside the episodes - its seed, range, wanted, kept and
its TOTAL attempts, which the press knows exactly - and the merge sums
them: an exact keep rate across shards, better than the single-run
upper bound. Planning and merging are pure and tested; running shards
is the tools' job (a subprocess per shard, the documented pattern).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trainnr.bundles.json_record import JsonRecord
from trainnr.collect.kitting_export import DemoLayout

if TYPE_CHECKING:
    from trainnr.collect.press import DemoBatch


@dataclass(frozen=True)
class ShardSpec:
    """One shard's slice of the batch."""

    index: int
    first_episode: int
    episodes: int
    seed: int

    @property
    def last_episode(self) -> int:
        return self.first_episode + self.episodes - 1


def plan_shards(
    episodes: int, shards: int, seed: int, first_episode: int = 0
) -> list[ShardSpec]:
    """Balanced, disjoint, contiguous ranges; seed + index per shard so
    two shards never draw the same stream."""
    if shards < 1:
        raise ValueError(f"shards must be >= 1, got {shards}")
    if episodes < shards:
        raise ValueError(
            f"{shards} shards for {episodes} episodes: some would be empty"
        )
    base, extra = divmod(episodes, shards)
    plan: list[ShardSpec] = []
    start = first_episode
    for index in range(shards):
        count = base + (1 if index < extra else 0)
        plan.append(ShardSpec(index, start, count, seed + index))
        start += count
    return plan


@dataclass(frozen=True)
class ShardRecord(JsonRecord):
    """What one press run left behind: the accounting the merge sums."""

    index: int
    seed: int
    first_episode: int
    wanted: int
    kept: int
    attempts: int
    expert: str
    task: str

    @classmethod
    def of(cls, spec: ShardSpec, batch: DemoBatch) -> ShardRecord:
        return cls(
            index=spec.index,
            seed=spec.seed,
            first_episode=batch.first_episode,
            wanted=batch.wanted,
            kept=batch.kept,
            attempts=batch.attempts,
            expert=batch.expert,
            task=batch.task,
        )

    def write_to(self, demos_dir: Path) -> Path:
        return self.write(
            Path(demos_dir) / DemoLayout.SHARD_FILE.format(index=self.index)
        )


def read_shard_records(demos_dir: Path) -> list[ShardRecord]:
    """Every shard record in the batch, by index; empty when the batch
    was pressed in one run."""
    paths = sorted(Path(demos_dir).glob(DemoLayout.SHARD_GLOB))
    return sorted((ShardRecord.read(path) for path in paths), key=lambda r: r.index)


@dataclass(frozen=True)
class MergedRate:
    """The keep rate across shards, exact: every shard counted every
    attempt, kept or not."""

    kept: int
    attempts: int
    shards: int

    @property
    def rate(self) -> float:
        return self.kept / self.attempts if self.attempts else 0.0


def merge_records(records: Iterable[ShardRecord]) -> MergedRate:
    rows = list(records)
    return MergedRate(
        kept=sum(r.kept for r in rows),
        attempts=sum(r.attempts for r in rows),
        shards=len(rows),
    )


def run_shards(
    plan: Sequence[ShardSpec],
    run_one: Callable[[ShardSpec], Any],
    *,
    parallel: int,
) -> list[Any]:
    """`run_one` per shard, at most `parallel` at a time, results in
    plan order. The caller decides what a shard IS (a subprocess of
    the tool, usually - its own renderer, its own Studio recording)."""
    if parallel < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel}")
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(run_one, plan))


def record_shard(demos_dir: Path, spec: ShardSpec, batch: Any) -> Path:
    """A child press leaves its record beside the episodes."""
    return ShardRecord.of(spec, batch).write_to(demos_dir)


def run_sharded_tool(  # noqa: PLR0913 - the orchestration's knobs, named
    argv: Sequence[str],
    out: Path,
    plan: Sequence[ShardSpec],
    *,
    parallel: int,
    say: Callable[[str], None] = print,
    python: str | None = None,
) -> int:
    """The tool re-runs ITSELF once per shard (each child: its own
    renderer, its own Studio recording, its own shard record), then
    the datasheet is rewritten over the merged records. Returns the
    exit code: 0 only when every shard completed."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from trainnr.collect.datasheet import write_datasheet  # noqa: PLC0415

    interpreter = python or sys.executable

    def run_one(spec: ShardSpec) -> int:
        child = [interpreter, *shard_argv(argv, spec, ShardFlags())]
        span = f"{spec.first_episode}..{spec.last_episode}"
        say(f"shard {spec.index}: episodes {span}, seed {spec.seed}")
        return subprocess.run(child, check=False).returncode

    codes = run_shards(plan, run_one, parallel=parallel)
    merged = merge_records(read_shard_records(out))
    say(
        f"merged {merged.shards} shards: {merged.kept} kept of "
        f"{merged.attempts} attempts (keep rate {merged.rate:.0%})"
    )
    if merged.kept:
        say(f"datasheet -> {write_datasheet(out)}")
    failed = [spec.index for spec, code in zip(plan, codes, strict=True) if code]
    if failed:
        say(f"shards failed: {failed}")
        return 1
    return 0


def shard_argv(argv: Sequence[str], spec: ShardSpec, flags: ShardFlags) -> list[str]:
    """The tool's own command line, re-aimed at one shard: the range,
    the seed and the shard index appended (later flags win in argparse),
    the sharding flags themselves stripped so the child presses once."""
    kept: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in (flags.shards, flags.parallel):
            skip = True
            continue
        if arg.startswith((f"{flags.shards}=", f"{flags.parallel}=")):
            continue
        kept.append(arg)
    return [
        *kept,
        flags.first_episode,
        str(spec.first_episode),
        flags.episodes,
        str(spec.episodes),
        flags.seed,
        str(spec.seed),
        flags.index,
        str(spec.index),
    ]


@dataclass(frozen=True)
class ShardFlags:
    """The flag spellings a tool uses, so `shard_argv` stays generic."""

    shards: str = "--shards"
    parallel: str = "--parallel"
    first_episode: str = "--first-episode"
    episodes: str = "--episodes"
    seed: str = "--seed"
    index: str = "--shard-index"
