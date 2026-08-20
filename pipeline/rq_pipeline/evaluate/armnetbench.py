"""ArmnetBench's real rollouts as Gate A's real side.

The committed artifact `data/armnetbench-v01-so101-counts.json` holds
per-(task, policy) episode counts aggregated from the third-party
release (arXiv:2607.24481; HF `armnet/armnetbench_v01_robometer`,
config so101, revision v1.0, Apache-2.0) — the aggregation recipe and
date live in the file's own provenance block, and the raw episodes are
one `load_dataset` away if the aggregation is ever doubted.

This is the data Paper 2 correlates against: real success counts for 7
policies on 8 single-arm SO-101 tasks, labelled by the benchmark's
operators, not by the policy authors — the whole reason the corpus
trusts it (author self-reports diverge 5x; docs/e2e-research/20).

Label semantics are the release's, kept discrete: `successful`,
`suboptimal`, `failure`. The default here is STRICT — suboptimal is not
success — and callers who want the lenient reading must say so in code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

QUALITY_LABELS = ("successful", "suboptimal", "failure")


@dataclass(frozen=True)
class LabelCounts:
    """One policy's episodes on one task, by the release's three labels."""

    successful: int
    suboptimal: int
    failure: int

    def __post_init__(self) -> None:
        for label in QUALITY_LABELS:
            if getattr(self, label) < 0:
                raise ValueError(f"negative count for {label!r}")

    @property
    def trials(self) -> int:
        return self.successful + self.suboptimal + self.failure


@dataclass(frozen=True)
class RealBenchmark:
    """The loaded artifact: provenance plus per-task per-policy counts."""

    provenance: Mapping[str, str]
    tasks: Mapping[str, Mapping[str, LabelCounts]]

    @property
    def total_episodes(self) -> int:
        return sum(
            counts.trials
            for policies in self.tasks.values()
            for counts in policies.values()
        )


def load_benchmark(path: Path) -> RealBenchmark:
    """Read and validate the committed counts artifact.

    Unknown quality labels are an error — a fourth label appearing in a
    future re-aggregation must break loudly here, not be silently folded
    into someone's idea of success.
    """
    raw = json.loads(Path(path).read_text())
    missing = {"provenance", "counts"} - set(raw)
    if missing:
        raise ValueError(f"{path} is missing top-level keys: {sorted(missing)}")
    tasks: dict[str, Mapping[str, LabelCounts]] = {}
    for task, policies in raw["counts"].items():
        per_policy: dict[str, LabelCounts] = {}
        for policy, labels in policies.items():
            unknown = set(labels) - set(QUALITY_LABELS)
            if unknown:
                raise ValueError(
                    f"{task}/{policy}: unknown quality labels {sorted(unknown)} — "
                    f"known: {list(QUALITY_LABELS)}"
                )
            per_policy[policy] = LabelCounts(
                successful=labels.get("successful", 0),
                suboptimal=labels.get("suboptimal", 0),
                failure=labels.get("failure", 0),
            )
        tasks[task] = MappingProxyType(per_policy)
    return RealBenchmark(
        provenance=MappingProxyType(dict(raw["provenance"])),
        tasks=MappingProxyType(tasks),
    )


def real_outcomes(
    benchmark: RealBenchmark,
    task: str,
    *,
    count_suboptimal_as_success: bool = False,
) -> dict[str, tuple[int, int]]:
    """One task's (successes, trials) per policy — `join_with_real`'s shape.

    Strict by default: `suboptimal` counts toward trials but not toward
    successes. Flipping that is a legitimate sensitivity analysis for
    Paper 2, which is why it is a named argument and not a constant.
    """
    if task not in benchmark.tasks:
        raise ValueError(
            f"unknown task {task!r} — benchmark has {sorted(benchmark.tasks)}"
        )
    outcomes = {}
    for policy, counts in benchmark.tasks[task].items():
        successes = counts.successful + (
            counts.suboptimal if count_suboptimal_as_success else 0
        )
        outcomes[policy] = (successes, counts.trials)
    return outcomes
