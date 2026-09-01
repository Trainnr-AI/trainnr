"""Pooling rank correlations across tasks — Paper 2's headline statistic.

One task's correlation says "the simulator ranks like reality *here*";
the claim worth publishing spans every informative task. This module
pools per-task Spearman correlations the meta-analytic way — Fisher-z,
inverse-variance weights — and refuses to let the pooled number hide
disagreement:

- **Cochran's Q** tests whether the per-task correlations are even
  consistent with one underlying value. When Q is significant, the
  pooled rho is an average over genuinely different regimes, and the
  summary says so instead of leading with it.
- **Fisher's method** combines the per-task exact permutation p-values
  (all tasks n <= 8, else no combined p is claimed) into one answer to
  "could the across-task agreement be luck?" — the discrete per-task
  p's make it conservative, which is the right direction to be wrong.

The per-task discreteness cap applies inside each task's z (same rule as
`fisher_rank_ci`): n policies cannot resolve a correlation finer than
one adjacent swap, pooled or not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import atanh, log, sqrt, tanh

from rq_pipeline.stats.intervals import (
    DEFAULT_CONFIDENCE,
    chi_squared_survival,
    normal_quantile,
)
from rq_pipeline.stats.ranking import (
    EXACT_ENUMERATION_LIMIT,
    FISHER_DOF,
    MINIMUM_POLICIES,
    SPEARMAN_Z_INFLATION,
    exact_spearman_p,
    max_resolvable_rho,
    spearman,
)

# A pooled claim over one task is the single-task claim wearing a costume.
MINIMUM_TASKS = 2


@dataclass(frozen=True)
class TaskCorrelation:
    """One task's contribution, kept for the audit trail."""

    task: str
    rho: float
    policy_count: int
    exact_p: float | None


@dataclass(frozen=True)
class PooledCorrelation:
    """The cross-task result: pooled effect, its interval, and the two
    honesty statistics that police it."""

    rho: float
    lower: float
    upper: float
    confidence: float
    tasks: tuple[TaskCorrelation, ...]
    heterogeneity_q: float
    heterogeneity_p: float
    combined_exact_p: float | None

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    def summary(self) -> str:
        head = (
            f"pooled rho {self.rho:.3f} "
            f"[{self.lower:.3f}, {self.upper:.3f}] at {self.confidence:.0%} "
            f"over {self.task_count} tasks"
        )
        if self.combined_exact_p is not None:
            head += f", combined exact p={self.combined_exact_p:.2e}"
        # The heterogeneity verdict leads when it should:
        if self.heterogeneity_p < 1.0 - self.confidence:
            head = (
                f"HETEROGENEOUS (Q={self.heterogeneity_q:.1f}, "
                f"p={self.heterogeneity_p:.3g}): tasks disagree about the "
                "correlation — report per-task results, not this pooled "
                "average. " + head
            )
        return head


def pool_rank_correlations(
    tasks: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    confidence: float = DEFAULT_CONFIDENCE,
) -> PooledCorrelation:
    """Pool per-task (sim_scores, real_scores) into one certificate-grade
    cross-task correlation.

    Rank-dead tasks (identical real scores — e.g. every policy failed)
    must be excluded by the CALLER, explicitly: `spearman` raises on
    them, and silently dropping a task inside this function would let a
    pooled claim quietly shrink its own denominator.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if len(tasks) < MINIMUM_TASKS:
        raise ValueError(
            f"need at least {MINIMUM_TASKS} tasks to pool, got {len(tasks)}"
        )

    per_task: list[TaskCorrelation] = []
    weighted_z = 0.0
    total_weight = 0.0
    z_values: list[tuple[float, float]] = []
    for name, (sim_scores, real_scores) in tasks.items():
        count = len(sim_scores)
        if count < MINIMUM_POLICIES:
            raise ValueError(
                f"task {name!r} has {count} policies; pooling needs at least "
                f"{MINIMUM_POLICIES} per task"
            )
        rho = spearman(sim_scores, real_scores)
        resolvable = max_resolvable_rho(count)
        z_score = atanh(max(-resolvable, min(resolvable, rho)))
        weight = (count - FISHER_DOF) / (SPEARMAN_Z_INFLATION**2)
        weighted_z += weight * z_score
        total_weight += weight
        z_values.append((z_score, weight))
        exact_p = (
            exact_spearman_p(sim_scores, real_scores)[0]
            if count <= EXACT_ENUMERATION_LIMIT
            else None
        )
        per_task.append(
            TaskCorrelation(task=name, rho=rho, policy_count=count, exact_p=exact_p)
        )

    pooled_z = weighted_z / total_weight
    half_width = normal_quantile(1.0 - (1.0 - confidence) / 2.0) / sqrt(total_weight)

    heterogeneity_q = sum(
        weight * (z_score - pooled_z) ** 2 for z_score, weight in z_values
    )
    heterogeneity_p = chi_squared_survival(heterogeneity_q, len(per_task) - 1)

    exact_ps = [entry.exact_p for entry in per_task]
    known_ps = [p for p in exact_ps if p is not None]
    if len(known_ps) == len(exact_ps):
        fisher_statistic = -2.0 * sum(log(p) for p in known_ps)
        combined_exact_p = chi_squared_survival(fisher_statistic, 2 * len(known_ps))
    else:
        combined_exact_p = None

    return PooledCorrelation(
        rho=tanh(pooled_z),
        lower=tanh(pooled_z - half_width),
        upper=tanh(pooled_z + half_width),
        confidence=confidence,
        tasks=tuple(per_task),
        heterogeneity_q=heterogeneity_q,
        heterogeneity_p=heterogeneity_p,
        combined_exact_p=combined_exact_p,
    )
