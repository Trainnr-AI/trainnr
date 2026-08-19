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

from rq_pipeline.stats.intervals import clopper_pearson
from rq_pipeline.stats.ranking import fisher_rank_ci, top_pick_probability


@dataclass(frozen=True)
class PolicyOutcome:
    """One policy's raw showing: what the simulator scored, what the robot did."""

    name: str
    sim_score: float
    real_successes: int
    real_trials: int


@dataclass(frozen=True)
class PolicyResult:
    """One policy's showing, sim and real, with the real side intervalled."""

    name: str
    sim_score: float
    real_successes: int
    real_trials: int
    real_lower: float
    real_upper: float


@dataclass(frozen=True)
class Certificate:
    """A complete, self-describing evaluation result.

    `gate_passed` is the one bit a customer asks about; everything else
    exists so that bit can be audited.
    """

    robot_bundle: str
    scene_bundle: str
    # The simulator IS part of the instrument: a certificate produced by a
    # different backend (or version) is a different instrument and needs
    # its own Gate A run.
    physics_backend: str
    confidence: float
    policies: tuple[PolicyResult, ...]
    rank_lower: float
    rank_upper: float
    policy_count: int
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
        return (
            f"{verdict}: rank lower bound {self.rank_lower:.3f} "
            f"(threshold {self.gate_threshold:.2f}, n={self.policy_count} "
            f"policies), top-pick {self.top_pick:.2f} — "
            f"{self.robot_bundle} in {self.scene_bundle}"
        )


def certify(  # noqa: PLR0913 - keyword-only args, each part of the artifact's identity
    robot_bundle: str,
    scene_bundle: str,
    outcomes: Sequence[PolicyOutcome],
    *,
    gate_threshold: float,
    physics_backend: str = "real-only",
    confidence: float = 0.95,
) -> Certificate:
    """Assemble a certificate from paired sim and real outcomes.

    Bundle identities must already be `name@hash` stamps — an evaluation
    against an unstamped artifact is not reproducible and is refused.
    """
    for bundle in (robot_bundle, scene_bundle):
        if "@" not in bundle:
            raise ValueError(
                f"bundle identity must be name@hash, got {bundle!r} — "
                "stamp it with rq_pipeline.bundles.stamp first"
            )
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
    top_pick = top_pick_probability(sim_scores, real_successes, real_trials)

    policies = tuple(
        PolicyResult(
            name=outcome.name,
            sim_score=outcome.sim_score,
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
        physics_backend=physics_backend,
        confidence=confidence,
        policies=policies,
        rank_lower=rank_lower,
        rank_upper=rank_upper,
        policy_count=policy_count,
        top_pick=top_pick,
        gate_threshold=gate_threshold,
        gate_passed=rank_lower >= gate_threshold,
    )
