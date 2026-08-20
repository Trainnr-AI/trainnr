"""The evaluation harness: policies + a robot model → sim scores → Gate A.

The missing link docs/22 named: `certify()` existed but nothing produced
its sim side. This module runs every policy through the SAME episode
protocol — same trial count, same per-trial initial-state perturbations
(paired trials: policy A's trial 7 starts exactly where policy B's trial
7 starts), same success predicate — and emits one `SimScore` per policy.
`join_with_real` then binds those to real-robot outcomes by policy name,
refusing any mismatch, and the result feeds `certify()` unchanged.

Two rules inherited from the rest of the pipeline:

- **The census gate runs before any episode.** A model whose import
  silently dropped its actuators or sensors scores every policy 0% with
  no error — the exact silent-import failure ② forbids. Refused up front.
- **Policies observe sensors, never privileged state** — the simulator
  is an instrument, and a policy that reads `qpos` directly in sim has
  no real-robot equivalent to correlate against.

Paper 2 note: this harness IS the sim side of the rank-correlation
experiment. Run it twice — nominal dynamics versus identified dynamics —
by loading two different robot bundles into the backend; nothing here
changes between the runs, which is the point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rq_pipeline.evaluate.certificate import PolicyOutcome
from rq_pipeline.robot.model_checks import assert_model_alive


@dataclass(frozen=True)
class SimPolicy:
    """A named closed-loop controller: (step_index, sensordata) → controls."""

    name: str
    act: Callable[[int, Any], Any]


@dataclass(frozen=True)
class EpisodeProtocol:
    """The episode recipe every policy is scored under, identically.

    `perturb(trial_index, home_state) -> initial_state` varies the start;
    it receives the trial index (not an RNG) so trials are paired across
    policies and the whole evaluation is deterministic by construction.
    `success(states, sensors) -> bool` judges one episode.
    """

    trials: int
    steps: int
    control_interval: int
    perturb: Callable[[int, Any], Any]
    success: Callable[[Any, Any], bool]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("trials", self.trials),
            ("steps", self.steps),
            ("control_interval", self.control_interval),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")


@dataclass(frozen=True)
class SimScore:
    """One policy's showing in simulation, as counts — never just a rate,
    so the trial count survives into every downstream statistic."""

    name: str
    successes: int
    trials: int

    @property
    def score(self) -> float:
        return self.successes / self.trials


def evaluate_policies(
    backend: Any,
    policies: Sequence[SimPolicy],
    protocol: EpisodeProtocol,
    *,
    source: str,
) -> tuple[SimScore, ...]:
    """Score every policy under the identical protocol; census-gated."""
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate policy names: {sorted(names)}")
    if "@" not in source:
        raise ValueError(
            f"source must be a name@hash stamp, got {source!r} — the same "
            "rule certify() enforces, applied before episodes are spent"
        )
    counts = backend.counts()
    assert_model_alive(counts.actuators, counts.sensors, counts.geoms, source=source)
    home = backend.default_initial_state()
    scores = []
    for policy in policies:
        successes = 0
        for trial in range(protocol.trials):
            states, sensors = backend.closed_loop_rollout(
                protocol.perturb(trial, home),
                policy.act,
                protocol.steps,
                protocol.control_interval,
            )
            if protocol.success(states, sensors):
                successes += 1
        scores.append(
            SimScore(name=policy.name, successes=successes, trials=protocol.trials)
        )
    return tuple(scores)


def join_with_real(
    sim_scores: Sequence[SimScore],
    real_outcomes: Mapping[str, tuple[int, int]],
) -> tuple[PolicyOutcome, ...]:
    """Bind sim scores to real (successes, trials) by policy name.

    The join must be exact in both directions: a policy evaluated in sim
    but never on the robot — or vice versa — is a hole in the
    correlation, and holes are refused, not dropped.
    """
    sim_names = {score.name for score in sim_scores}
    real_names = set(real_outcomes)
    if sim_names != real_names:
        raise ValueError(
            "sim and real policy sets differ — "
            f"sim-only: {sorted(sim_names - real_names)}, "
            f"real-only: {sorted(real_names - sim_names)}"
        )
    outcomes = []
    for score in sim_scores:
        successes, trials = real_outcomes[score.name]
        if trials <= 0 or not 0 <= successes <= trials:
            raise ValueError(
                f"invalid real counts for {score.name!r}: {successes}/{trials}"
            )
        outcomes.append(
            PolicyOutcome(
                name=score.name,
                sim_score=score.score,
                real_successes=successes,
                real_trials=trials,
            )
        )
    return tuple(outcomes)
