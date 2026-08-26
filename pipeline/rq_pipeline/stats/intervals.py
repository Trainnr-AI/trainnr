"""Exact binomial confidence intervals, stdlib only.

Clopper-Pearson needs the beta quantile, which normally means scipy. The
~40 lines of continued fraction below buy independence from it: a success
rate on a signed evaluation report must be recomputable on any Python,
including the CI container and the customer's auditor's laptop.
"""

from __future__ import annotations

from math import erf, exp, lgamma, log, sqrt

# Lentz's algorithm bottoms out well above float underflow; iteration cap
# is generous — the fraction converges in tens of steps for all (a, b, x)
# reachable from a binomial interval.
_MAX_ITERATIONS = 300
_EPSILON = 3e-12
_TINY = 1e-300


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    ab_sum, a_plus, a_minus = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - ab_sum * x / a_plus
    d = 1.0 / (d if abs(d) >= _TINY else _TINY)
    result = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        even_step = m * (b - m) * x / ((a_minus + m2) * (a + m2))
        d = 1.0 + even_step * d
        d = 1.0 / (d if abs(d) >= _TINY else _TINY)
        c = 1.0 + even_step / (c if abs(c) >= _TINY else _TINY)
        result *= d * c
        odd_step = -(a + m) * (ab_sum + m) * x / ((a + m2) * (a_plus + m2))
        d = 1.0 + odd_step * d
        d = 1.0 / (d if abs(d) >= _TINY else _TINY)
        c = 1.0 + odd_step / (c if abs(c) >= _TINY else _TINY)
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return result


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the CDF of the Beta(a, b) distribution at x."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"shape parameters must be positive, got a={a} b={b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_prefactor = (
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    )
    prefactor = exp(log_prefactor)
    # The fraction converges fastest on the side of the distribution's bulk;
    # use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) to stay on that side.
    if x < (a + 1.0) / (a + b + 2.0):
        return prefactor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - prefactor * _beta_continued_fraction(b, a, 1.0 - x) / b


DEFAULT_CONFIDENCE = 0.95  # the report-level confidence, one decision
_BISECTION_STEPS = 200  # halvings of [0, 1] or [-10, 10]: far past double precision


def check_counts(label: str, successes: int, trials: int) -> None:
    """Counts are valid or refused — the one rule for every side, every
    tier: intervals, ranking, pooling and the certificate call this."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError(f"invalid {label} counts: {successes}/{trials}")


def _beta_quantile(probability: float, a: float, b: float) -> float:
    """Inverse of the Beta CDF by bisection — monotone, so always safe."""
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2.0
        if regularized_incomplete_beta(a, b, mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def clopper_pearson(
    successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Exact two-sided interval for a binomial success rate.

    Reference vector from the field: 63/70 successes at 95% gives
    (0.805, 0.959) — the worked example in NVIDIA's evaluation
    methodology post, reproduced in this package's tests.
    """
    check_counts("binomial", successes, trials)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if successes == 0
        else _beta_quantile(tail, successes, trials - successes + 1)
    )
    upper = (
        1.0
        if successes == trials
        else _beta_quantile(1.0 - tail, successes + 1, trials - successes)
    )
    return lower, upper


def wilson(
    successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """Wilson score interval — the cheap cross-check on Clopper-Pearson.

    Slightly narrower than the exact interval by construction; the tests
    use that known relationship to catch a broken quantile.
    """
    check_counts("binomial", successes, trials)
    # Normal quantile via beta-quantile of the symmetric case would be
    # circular; use the Acklam-style rational approximation's simple cousin:
    # for the confidences used in reports (0.9, 0.95, 0.99) a bisection on
    # the error function is exact enough and stdlib-only.
    z = normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    margin = (
        z
        * sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def normal_quantile(probability: float) -> float:
    low, high = -10.0, 10.0
    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2.0
        if (1.0 + erf(mid / sqrt(2.0))) / 2.0 < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def regularized_lower_gamma(shape: float, x: float) -> float:
    """P(s, x) — the CDF of a Gamma(shape, 1) at x, stdlib only.

    Series expansion below the bulk (x < s + 1), Lentz continued fraction
    above it — the same split every reference implementation uses,
    because each converges fast only on its own side.
    """
    if shape <= 0.0:
        raise ValueError(f"shape must be positive, got {shape}")
    if x < 0.0:
        raise ValueError(f"x must be non-negative, got {x}")
    if x == 0.0:
        return 0.0
    log_prefactor = shape * log(x) - x - lgamma(shape)
    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        numerator = shape
        for _ in range(_MAX_ITERATIONS):
            numerator += 1.0
            term *= x / numerator
            total += term
            if abs(term) < abs(total) * _EPSILON:
                break
        return min(1.0, exp(log_prefactor) * total)
    # Continued fraction for Q(s, x); P = 1 - Q.
    b = x + 1.0 - shape
    c = 1.0 / _TINY
    d = 1.0 / (b if abs(b) >= _TINY else _TINY)
    fraction = d
    for iteration in range(1, _MAX_ITERATIONS + 1):
        coefficient = -iteration * (iteration - shape)
        b += 2.0
        d = coefficient * d + b
        d = 1.0 / (d if abs(d) >= _TINY else _TINY)
        c = b + coefficient / (c if abs(c) >= _TINY else _TINY)
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return max(0.0, 1.0 - exp(log_prefactor) * fraction)


def chi_squared_survival(statistic: float, degrees_of_freedom: int) -> float:
    """P(X >= statistic) for a chi-squared distribution.

    Used by the pooling layer twice: Cochran's Q heterogeneity test and
    Fisher's method for combining per-task exact p-values.
    """
    if degrees_of_freedom <= 0:
        raise ValueError(
            f"degrees of freedom must be positive, got {degrees_of_freedom}"
        )
    if statistic < 0.0:
        raise ValueError(f"statistic must be non-negative, got {statistic}")
    return 1.0 - regularized_lower_gamma(degrees_of_freedom / 2.0, statistic / 2.0)
