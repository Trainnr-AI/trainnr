"""Exact power for the paired study's own test — sizing BEFORE spending.

docs/e2e-research/62 §3's rule: the real C1 run's trial count comes
from this table before any GPU hour is priced. The test is the one the
verdict actually uses — `effects.fisher_exact`, two-sided, at the
protocol's declared alpha — so the power is the power of OUR verdict,
not of a textbook approximation. Everything here is standard library
on purpose: the sizing recomputes on an auditor's laptop, the same
rule as the statistics it sizes (a stdlib-pure core).
"""

from __future__ import annotations

from math import comb

from trainnr.stats.effects import fisher_exact


def _binomial_pmf(successes: int, trials: int, rate: float) -> float:
    return (
        comb(trials, successes) * rate**successes * (1.0 - rate) ** (trials - successes)
    )


def fisher_power(
    rate_a: float,
    rate_b: float,
    trials: int,
    *,
    alpha: float,
) -> float:
    """The probability that Fisher's exact test rejects at `alpha` when
    the two sides' true success rates are `rate_a` and `rate_b`, with
    `trials` paired trials per side. Exact: every (successes_a,
    successes_b) outcome enumerated, its p-value computed by the same
    `fisher_exact` the verdict uses."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for name, rate in (("rate_a", rate_a), ("rate_b", rate_b)):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {rate}")
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    power = 0.0
    pmf_a = [_binomial_pmf(k, trials, rate_a) for k in range(trials + 1)]
    pmf_b = [_binomial_pmf(k, trials, rate_b) for k in range(trials + 1)]
    for sa in range(trials + 1):
        for sb in range(trials + 1):
            if fisher_exact(sa, trials - sa, sb, trials - sb) < alpha:
                power += pmf_a[sa] * pmf_b[sb]
    return power


def trials_needed(
    rate_a: float,
    rate_b: float,
    *,
    alpha: float,
    power: float,
    limit: int = 200,
) -> int:
    """The smallest per-side trial count whose exact power reaches
    `power` — scanned upward (exact-test power is sawtoothed, so the
    first crossing is the answer, not a bisection). Raises when `limit`
    is reached: an effect that small needs a bigger experiment than
    this harness should promise."""
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    for trials in range(2, limit + 1):
        if fisher_power(rate_a, rate_b, trials, alpha=alpha) >= power:
            return trials
    raise ValueError(
        f"detecting {rate_a} vs {rate_b} at alpha={alpha} needs more than "
        f"{limit} trials per side - declare a bigger detectable gap instead"
    )
