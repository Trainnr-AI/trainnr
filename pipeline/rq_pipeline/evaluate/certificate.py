"""The certificate: what the pipeline exists to produce.

A certificate binds together everything a reader needs to trust or
reproduce an evaluation: which robot bundle, which scene bundle, every
policy's real interval, the rank-correlation interval with its n, and
the top-pick probability. The gate decision is taken on the Fisher
LOWER bound — never the point estimate — and the threshold is a
required argument because no published basis for a default exists
(docs/e2e-research/30-the-pipeline.md, stage ③).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from rq_pipeline.bundles.hashing import require_stamp
from rq_pipeline.stats.intervals import (
    DEFAULT_CONFIDENCE,
    check_counts,
    clopper_pearson,
)
from rq_pipeline.stats.ranking import (
    EXACT_ENUMERATION_LIMIT,
    exact_spearman_p,
    fisher_rank_ci,
    top_pick_probability,
)

# A certificate with no simulated side at all (real trials only).
REAL_ONLY_INSTRUMENT = "real-only"


@dataclass(frozen=True)
class PolicyOutcome:
    """One policy's raw showing: what the simulator scored, what the
    robot did — both as counts. Until 2026-08-26 the sim side arrived
    as a rate and the certificate could not state its own sim n; the
    audit that day (docs/32 §6) caught the rule "counts, never rates"
    broken at exactly the join it existed for.
    """

    name: str
    sim_successes: int
    sim_trials: int
    real_successes: int
    real_trials: int

    def __post_init__(self) -> None:
        check_counts("sim", self.sim_successes, self.sim_trials)
        check_counts("real", self.real_successes, self.real_trials)

    @property
    def sim_score(self) -> float:
        return self.sim_successes / self.sim_trials


@dataclass(frozen=True)
class PolicyResult:
    """One policy's showing, sim and real, with the real side intervalled."""

    name: str
    sim_successes: int
    sim_trials: int
    real_successes: int
    real_trials: int
    real_lower: float
    real_upper: float

    @property
    def sim_score(self) -> float:
        return self.sim_successes / self.sim_trials


@dataclass(frozen=True)
class Certificate:
    """A complete, self-describing evaluation result.

    `gate_passed` is the one bit a customer asks about; everything else
    exists so that bit can be audited.
    """

    robot_bundle: str
    scene_bundle: str
    # The simulator IS part of the instrument: a certificate produced by a
    # different engine or version is a different instrument and needs its
    # own Gate A run — so the field carries name AND version
    # (physics/backend.py::instrument_stamp, e.g. "mujoco-3.11.0+x86_64").
    instrument: str
    confidence: float
    policies: tuple[PolicyResult, ...]
    rank_lower: float
    rank_upper: float
    policy_count: int
    # Exact permutation p-value for the observed rank agreement, present
    # whenever n is small enough to enumerate (n <= 8) — exactly the
    # regime where asymptotic statements deserve the least trust. None
    # means n was large enough that Fisher-z carries the claim alone.
    exact_p_value: float | None
    top_pick: float
    gate_threshold: float
    gate_passed: bool
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def summary(self) -> str:
        verdict = "PASS" if self.gate_passed else "FAIL"
        significance = (
            f", exact p={self.exact_p_value:.4f}"
            if self.exact_p_value is not None
            else ""
        )
        return (
            f"{verdict}: rank lower bound {self.rank_lower:.3f} "
            f"(threshold {self.gate_threshold:.2f}, n={self.policy_count} "
            f"policies{significance}), top-pick {self.top_pick:.2f} — "
            f"{self.robot_bundle} in {self.scene_bundle}"
        )


def certify(  # noqa: PLR0913 - keyword-only args, each part of the artifact's identity
    robot_bundle: str,
    scene_bundle: str,
    outcomes: Sequence[PolicyOutcome],
    *,
    gate_threshold: float,
    instrument: str = REAL_ONLY_INSTRUMENT,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Certificate:
    """Assemble a certificate from paired sim and real outcomes.

    Bundle identities must already be `name@hash` stamps — an evaluation
    against an unstamped artifact is not reproducible and is refused.
    """
    require_stamp(robot_bundle, "robot bundle identity")
    require_stamp(scene_bundle, "scene bundle identity")
    if not 0.0 < gate_threshold < 1.0:
        raise ValueError(f"gate threshold must be in (0, 1), got {gate_threshold}")

    sim_scores = [outcome.sim_score for outcome in outcomes]
    real_successes = [outcome.real_successes for outcome in outcomes]
    real_trials = [outcome.real_trials for outcome in outcomes]
    real_rates = [
        successes / trials
        for successes, trials in zip(real_successes, real_trials, strict=True)
    ]
    rank_lower, rank_upper, policy_count = fisher_rank_ci(
        sim_scores, real_rates, confidence
    )
    exact_p_value = (
        exact_spearman_p(sim_scores, real_rates)[0]
        if policy_count <= EXACT_ENUMERATION_LIMIT
        else None
    )
    top_pick = top_pick_probability(sim_scores, real_successes, real_trials)

    policies = tuple(
        PolicyResult(
            name=outcome.name,
            sim_successes=outcome.sim_successes,
            sim_trials=outcome.sim_trials,
            real_successes=outcome.real_successes,
            real_trials=outcome.real_trials,
            real_lower=lower,
            real_upper=upper,
        )
        for outcome, (lower, upper) in zip(
            outcomes,
            (
                clopper_pearson(outcome.real_successes, outcome.real_trials, confidence)
                for outcome in outcomes
            ),
            strict=True,
        )
    )

    return Certificate(
        robot_bundle=robot_bundle,
        scene_bundle=scene_bundle,
        instrument=instrument,
        confidence=confidence,
        policies=policies,
        rank_lower=rank_lower,
        rank_upper=rank_upper,
        policy_count=policy_count,
        exact_p_value=exact_p_value,
        top_pick=top_pick,
        gate_threshold=gate_threshold,
        gate_passed=rank_lower >= gate_threshold,
    )
