"""Main effects per factor, exact at small n — the sensitivity table.

Arena asks the right question (given the outcome, where were the
factors?) and answers it with a neural posterior fitted on ten episodes,
reporting no effect size and no interval (docs/e2e-research/41). For a
balanced, paired sweep the honest answer is a 2x2 table per factor:
successes and trials on each side of the factor's midpoint (or per
label), Clopper-Pearson on both, the rate difference, and Fisher's
exact test — the hypergeometric from the `lgamma` this package already
uses. Three verdicts, declared thresholds, no fit:

- SENSITIVE: the exact p is below `alpha`;
- INSENSITIVE: every difference the two intervals allow lies inside
  ±`delta` — the factor cannot move the rate by more than `delta`;
- UNRESOLVED: neither — the honest answer at n = 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, lgamma

from rq_pipeline.stats.intervals import DEFAULT_CONFIDENCE, clopper_pearson

_ONE_TOLERANCE = 1e-9  # a p summed over every table may overshoot 1 by noise
_TIE_TOLERANCE = 1e-7  # tables with the observed probability count as ties


class Verdict:
    """A factor's reading, under thresholds the protocol declared."""

    SENSITIVE = "SENSITIVE"  # exact p below alpha
    INSENSITIVE = "INSENSITIVE"  # both intervals bound the difference inside ±delta
    UNRESOLVED = "UNRESOLVED"  # neither — the honest answer at n = 4


class SplitLabel:
    """The two sides of a continuous factor, split at its midpoint."""

    LOW = "<midpoint"
    HIGH = ">=midpoint"


def _log_choose(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the table [[a, b], [c, d]] — rows are
    the two factor sides, columns success/failure. Two-sided by summing
    every table with probability no greater than the observed one (the
    convention R and scipy use)."""
    for value in (a, b, c, d):
        if value < 0:
            raise ValueError(f"counts must be non-negative, got {(a, b, c, d)}")
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    if total == 0:
        return 1.0

    def log_probability(x: int) -> float:
        return (
            _log_choose(row1, x)
            + _log_choose(row2, col1 - x)
            - _log_choose(total, col1)
        )

    observed = log_probability(a)
    low, high = max(0, col1 - row2), min(row1, col1)
    tolerance = _TIE_TOLERANCE * abs(observed)
    p = 0.0
    for x in range(low, high + 1):
        lp = log_probability(x)
        if lp <= observed + tolerance:
            p += exp(lp)
    # Summing every table can overshoot 1 by floating noise; snap it.
    return 1.0 if p > 1.0 - _ONE_TOLERANCE else p


@dataclass(frozen=True)
class FactorEffect:
    """One factor's main effect: the two sides as counts, both intervals,
    the difference (high side minus low side, or label B minus A), the
    exact p and the verdict."""

    key: str
    side_a: str
    side_b: str
    successes_a: int
    trials_a: int
    successes_b: int
    trials_b: int
    interval_a: tuple[float, float]
    interval_b: tuple[float, float]
    difference: float
    p_value: float
    verdict: str

    @property
    def rate_a(self) -> float:
        return self.successes_a / self.trials_a

    @property
    def rate_b(self) -> float:
        return self.successes_b / self.trials_b


def main_effect(  # noqa: PLR0913 - the two sides and two thresholds, all required
    key: str,
    side_a: str,
    side_b: str,
    outcomes_a: Sequence[bool],
    outcomes_b: Sequence[bool],
    *,
    alpha: float,
    delta: float,
    confidence: float = DEFAULT_CONFIDENCE,
) -> FactorEffect:
    """The 2x2 reading for one factor from its two sides' outcomes.
    `alpha` and `delta` have no defaults: they are the protocol's claim."""
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError(f"alpha and delta must be in (0, 1), got {alpha}, {delta}")
    if not outcomes_a or not outcomes_b:
        raise ValueError(f"factor {key!r}: both sides need trials")
    sa, na = sum(map(bool, outcomes_a)), len(outcomes_a)
    sb, nb = sum(map(bool, outcomes_b)), len(outcomes_b)
    interval_a = clopper_pearson(sa, na, confidence)
    interval_b = clopper_pearson(sb, nb, confidence)
    p = fisher_exact(sa, na - sa, sb, nb - sb)
    # The widest difference the two intervals allow, either direction.
    spread = max(abs(interval_b[1] - interval_a[0]), abs(interval_b[0] - interval_a[1]))
    if p < alpha:
        verdict = Verdict.SENSITIVE
    elif spread <= delta:
        verdict = Verdict.INSENSITIVE
    else:
        verdict = Verdict.UNRESOLVED
    return FactorEffect(
        key=key,
        side_a=side_a,
        side_b=side_b,
        successes_a=sa,
        trials_a=na,
        successes_b=sb,
        trials_b=nb,
        interval_a=interval_a,
        interval_b=interval_b,
        difference=sb / nb - sa / na,
        p_value=p,
        verdict=verdict,
    )


def split_continuous(
    values: Sequence[float], outcomes: Sequence[bool], midpoint: float
) -> tuple[list[bool], list[bool]]:
    """Outcomes below the midpoint and at/above it — the split Arena's own
    tests use to judge a planted effect."""
    below = [o for v, o in zip(values, outcomes, strict=True) if v < midpoint]
    above = [o for v, o in zip(values, outcomes, strict=True) if v >= midpoint]
    return below, above


def format_table(effects: Sequence[FactorEffect]) -> str:
    """The certificate's sensitivity table, one factor per line:
    `camera offset: SENSITIVE (1/8 vs 7/8, p = 0.005)` is what a customer
    reads, not a density curve."""
    lines = []
    for e in effects:
        lines.append(
            f"{e.key}: {e.verdict} ({e.side_a} {e.successes_a}/{e.trials_a} "
            f"[{e.interval_a[0]:.2f}, {e.interval_a[1]:.2f}] vs {e.side_b} "
            f"{e.successes_b}/{e.trials_b} [{e.interval_b[0]:.2f}, "
            f"{e.interval_b[1]:.2f}], diff {e.difference:+.2f}, p = {e.p_value:.3f})"
        )
    return "\n".join(lines)
