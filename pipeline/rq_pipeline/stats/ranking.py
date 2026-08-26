"""Gate A statistics: does the simulator rank policies the way reality does?

The sample size for a rank correlation is the number of POLICIES — the
rollouts tighten each point and do nothing about how many points there
are. At r = 0.92 with five policies the 95% interval spans roughly
[0.2, 1.0], which cannot tell an instrument from noise. Every function
here therefore reports or enforces n, and the certificate quantity is
`top_pick_probability` — the number a purchase decision actually turns on.
(docs/e2e-research/30-the-pipeline.md, stage ③.)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import permutations
from math import atanh, sqrt, tanh

from rq_pipeline.stats.intervals import (
    DEFAULT_CONFIDENCE,
    check_counts,
    normal_quantile,
)

# A correlation needs 3 points to be defined at all, a comparison needs
# 2 — and an INTERVAL over rankings needs 4, because the Fisher standard
# error has sqrt(n - 3) in its denominator.
_MINIMUM_FOR_CORRELATION = 3
_MINIMUM_FOR_COMPARISON = 2
# Fisher's z for Spearman has sqrt(n - FISHER_DOF) in its denominator, so a
# comparison needs one policy more than that to have any resolution at all.
FISHER_DOF = 3
MINIMUM_POLICIES = FISHER_DOF + 1

# Spearman's Fisher-z standard error carries a 1.03 adjustment
# (Caruso & Cliff) relative to Pearson's 1.0.
SPEARMAN_Z_INFLATION = 1.03

# Up to 8 policies the full permutation distribution (8! = 40,320
# orderings) is enumerable, so small-n significance needs no
# approximation at all. Beyond that, Fisher-z carries the certificate.
EXACT_ENUMERATION_LIMIT = 8

# Guards float round-off when comparing a permuted rho against the
# observed one — never a statistical fudge, only arithmetic slack.
_RHO_COMPARISON_SLACK = 1e-12

# atanh diverges at |r| = 1, and any fixed clamp would let the clamp
# constant, not the data, set the bound at perfect observed agreement.
# The honest cap is the DISCRETENESS of rankings: n policies cannot
# resolve a correlation finer than one adjacent rank swap, so observed
# rho is capped at the second-most-extreme achievable value,
# 1 - 12/(n(n^2-1)) — 0.9 at n=5, 0.993 at n=12. The pole guard below
# only matters once n is large enough that the discreteness cap
# approaches it.
_RHO_POLE_GUARD = 0.9999


def max_resolvable_rho(policy_count: int) -> float:
    one_swap = 1.0 - 12.0 / (policy_count * (policy_count**2 - 1))
    return min(one_swap, _RHO_POLE_GUARD)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        tied_rank = (start + end) / 2.0 + 1.0
        for position in range(start, end + 1):
            ranks[order[position]] = tied_rank
        start = end + 1
    return ranks


def spearman(sim_scores: Sequence[float], real_scores: Sequence[float]) -> float:
    """Rank correlation with average ranks for ties."""
    if len(sim_scores) != len(real_scores):
        raise ValueError(
            f"score lists differ in length: {len(sim_scores)} vs {len(real_scores)}"
        )
    if len(sim_scores) < _MINIMUM_FOR_CORRELATION:
        raise ValueError(
            f"need at least {_MINIMUM_FOR_CORRELATION} policies, got {len(sim_scores)}"
        )
    sim_ranks = _average_ranks(sim_scores)
    real_ranks = _average_ranks(real_scores)
    count = len(sim_ranks)
    sim_mean = sum(sim_ranks) / count
    real_mean = sum(real_ranks) / count
    covariance = sum(
        (s - sim_mean) * (r - real_mean)
        for s, r in zip(sim_ranks, real_ranks, strict=True)
    )
    sim_variance = sum((s - sim_mean) ** 2 for s in sim_ranks)
    real_variance = sum((r - real_mean) ** 2 for r in real_ranks)
    if sim_variance == 0.0 or real_variance == 0.0:
        raise ValueError(
            "rank variance is zero — a correlation over identical scores is "
            "uninformative by construction; widen the policy set instead"
        )
    return covariance / sqrt(sim_variance * real_variance)


def fisher_rank_ci(
    sim_scores: Sequence[float],
    real_scores: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float, int]:
    """Fisher-z interval for Spearman r over n policies — the gate statistic.

    Returns (lower, upper, n_policies). The gate decision uses `lower`,
    never the point estimate, and n_policies goes on the certificate
    beside the interval. This, not the bootstrap, is the gate: the
    percentile bootstrap cannot widen past the observed data — a
    perfectly monotone pair resamples to r = 1 forever — which this
    package's own test suite caught on its first run.
    """
    correlation = spearman(sim_scores, real_scores)
    policy_count = len(sim_scores)
    if policy_count < MINIMUM_POLICIES:
        raise ValueError(
            f"need at least {MINIMUM_POLICIES} policies, got {policy_count}"
        )
    resolvable = max_resolvable_rho(policy_count)
    clamped = max(-resolvable, min(resolvable, correlation))
    z_score = atanh(clamped)
    half_width = (
        normal_quantile(1.0 - (1.0 - confidence) / 2.0)
        * SPEARMAN_Z_INFLATION
        / sqrt(policy_count - FISHER_DOF)
    )
    return (
        tanh(z_score - half_width),
        tanh(z_score + half_width),
        policy_count,
    )


def _rank_correlation(sim_ranks: Sequence[float], real_ranks: Sequence[float]) -> float:
    """Pearson over already-computed ranks; variances known non-zero."""
    count = len(sim_ranks)
    sim_mean = sum(sim_ranks) / count
    real_mean = sum(real_ranks) / count
    covariance = sum(
        (s - sim_mean) * (r - real_mean)
        for s, r in zip(sim_ranks, real_ranks, strict=True)
    )
    sim_variance = sum((s - sim_mean) ** 2 for s in sim_ranks)
    real_variance = sum((r - real_mean) ** 2 for r in real_ranks)
    return covariance / sqrt(sim_variance * real_variance)


def exact_spearman_p(
    sim_scores: Sequence[float],
    real_scores: Sequence[float],
) -> tuple[float, int]:
    """One-sided exact permutation p-value for positive rank association.

    Under the null of no association, every pairing of the two observed
    rank vectors is equally likely; enumerating all n! of them (n <= 8)
    gives P(rho >= observed) with no approximation, no Fisher transform,
    no discreteness cap. Ties are handled by conditioning on the observed
    average-rank vectors. The smallest attainable p is 1/n! — at n = 5 a
    perfect ranking earns exactly 1/120, which is the honest version of
    what the bootstrap mis-reported as certainty.

    Returns (p_value, policy_count). This answers "could the agreement
    be luck?"; `fisher_rank_ci`'s lower bound answers "how strong is it
    at worst?" — the gate needs the second, the certificate carries both.
    """
    observed = spearman(sim_scores, real_scores)  # validates lengths and n >= 3
    policy_count = len(sim_scores)
    if policy_count > EXACT_ENUMERATION_LIMIT:
        raise ValueError(
            f"exact enumeration is for n <= {EXACT_ENUMERATION_LIMIT} policies, "
            f"got {policy_count} — use fisher_rank_ci there"
        )
    sim_ranks = _average_ranks(sim_scores)
    real_ranks = _average_ranks(real_scores)
    at_least_as_extreme = 0
    total = 0
    for permuted in permutations(sim_ranks):
        total += 1
        if _rank_correlation(permuted, real_ranks) >= observed - _RHO_COMPARISON_SLACK:
            at_least_as_extreme += 1
    return at_least_as_extreme / total, policy_count


def bootstrap_rank_ci(
    sim_scores: Sequence[float],
    real_scores: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, int]:
    """Percentile interval for Spearman r, resampling over POLICIES.

    Diagnostic only — never the gate. A percentile bootstrap cannot
    express uncertainty beyond the observed data: resampling a perfectly
    monotone pair yields r = 1 in every draw, so at small n with high
    observed agreement this interval is overconfident exactly where
    honesty matters most. Use `fisher_rank_ci` for the certificate and
    this to spot asymmetry or influential single policies.
    """
    if len(sim_scores) < MINIMUM_POLICIES:
        raise ValueError(
            f"need at least {MINIMUM_POLICIES} policies for a bootstrap, "
            f"got {len(sim_scores)}"
        )
    rng = random.Random(seed)
    count = len(sim_scores)
    estimates: list[float] = []
    while len(estimates) < resamples:
        indices = [rng.randrange(count) for _ in range(count)]
        resampled_sim = [sim_scores[i] for i in indices]
        resampled_real = [real_scores[i] for i in indices]
        try:
            estimates.append(spearman(resampled_sim, resampled_real))
        except ValueError:
            # Degenerate draw (all one policy, or tied ranks throughout):
            # carries no ranking information, so it is discarded rather
            # than counted as agreement or disagreement.
            continue
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = int(tail * len(estimates))
    upper_index = min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))
    return estimates[lower_index], estimates[upper_index], count


def top_pick_probability(
    sim_scores: Sequence[float],
    real_successes: Sequence[int],
    real_trials: Sequence[int],
    samples: int = 10_000,
    seed: int = 0,
) -> float:
    """P(the policy the simulator ranks first is truly best on the robot).

    Jeffreys Beta(k + 1/2, n - k + 1/2) posterior per policy, Monte Carlo
    over the joint draw. Interpretable at any n, degrades gracefully as
    policies get close, and is the number a deployment decision turns on.
    """
    if not len(sim_scores) == len(real_successes) == len(real_trials):
        raise ValueError("per-policy lists differ in length")
    if len(sim_scores) < _MINIMUM_FOR_COMPARISON:
        raise ValueError(
            f"need at least {_MINIMUM_FOR_COMPARISON} policies, got {len(sim_scores)}"
        )
    for successes, trials in zip(real_successes, real_trials, strict=True):
        check_counts("real", successes, trials)
    sim_best = max(range(len(sim_scores)), key=lambda i: sim_scores[i])
    rng = random.Random(seed)
    wins = 0
    for _ in range(samples):
        draws = [
            rng.betavariate(k + 0.5, n - k + 0.5)
            for k, n in zip(real_successes, real_trials, strict=True)
        ]
        if max(range(len(draws)), key=lambda i: draws[i]) == sim_best:
            wins += 1
    return wins / samples
