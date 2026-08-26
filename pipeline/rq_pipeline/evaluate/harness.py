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
from pathlib import Path
from typing import Any

from rq_pipeline.evaluate.certificate import PolicyOutcome
from rq_pipeline.evaluate.records import (
    EpisodeRecord,
    SimScore,
    append_records,
    fold,
    protocol_fields,
)
from rq_pipeline.physics.backend import PhysicsBackend
from rq_pipeline.robot.model_checks import assert_model_alive

__all__ = [
    "EpisodeProtocol",
    "SimPolicy",
    "SimScore",
    "evaluate_policies",
    "home_state",
    "join_with_real",
    "score_policies",
]


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

    `home` names the keyframe every trial starts from (before
    `perturb`); None means the model's reset state. Where an episode
    starts is a fact about the protocol, declared here, never inherited
    from the backend — the ALOHA rig's reset state has both arms
    straight up on a path that jams the grippers together at over 1 kN,
    while the SO-101 tasks were tuned from theirs.
    """

    trials: int
    steps: int
    control_interval: int
    perturb: Callable[[int, Any], Any]
    success: Callable[[Any, Any], bool]
    home: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("trials", self.trials),
            ("steps", self.steps),
            ("control_interval", self.control_interval),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")


def home_state(backend: PhysicsBackend, protocol: EpisodeProtocol) -> Any:
    """The protocol's declared start: its keyframe, else the model's reset."""
    if protocol.home is None:
        return backend.default_initial_state()
    return backend.keyframe_state(protocol.home)


def score_policies(  # noqa: PLR0913 - the skeleton carries both harnesses' knobs
    backend: PhysicsBackend,
    policies: Sequence[Any],
    protocol: EpisodeProtocol,
    *,
    source: str,
    run_episode: Callable[[Any, Any], tuple[Any, Any]],
    gate_cameras: bool = False,
    record_to: Path | None = None,
) -> tuple[SimScore, ...]:
    """The scoring skeleton both harnesses share: refuse duplicate
    names and unstamped sources, census-gate the model, then run every
    policy through the identical paired trials. `run_episode(policy,
    initial_state) -> (states, sensors)` is the only thing that differs
    between observing sensors and observing pixels — so it is the only
    thing callers supply. Vision callers set `gate_cameras` because a
    camera-less model would score their policies 0% silently.

    Every trial becomes an `EpisodeRecord`; the scores are the fold over
    them (records.py), and `record_to` appends the rows as trials finish
    — a crash at trial 9 leaves 8 lines, not nothing.
    """
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate policy names: {sorted(names)}")
    if "@" not in source:
        raise ValueError(
            f"source must be a name@hash stamp, got {source!r} — the same "
            "rule certify() enforces, applied before episodes are spent"
        )
    counts = backend.counts()
    assert_model_alive(
        counts.actuators,
        counts.sensors,
        counts.geoms,
        source=source,
        cameras=counts.cameras if gate_cameras else None,
    )
    home = home_state(backend, protocol)
    instrument = getattr(backend, "instrument", backend.name)
    fields = protocol_fields(protocol)
    records: list[EpisodeRecord] = []
    for policy in policies:
        for trial in range(protocol.trials):
            states, sensors = run_episode(policy, protocol.perturb(trial, home))
            record = EpisodeRecord(
                source=source,
                policy=policy.name,
                trial=trial,
                success=bool(protocol.success(states, sensors)),
                steps=len(states),
                instrument=instrument,
                protocol=fields,
            )
            records.append(record)
            if record_to is not None:
                append_records(record_to, [record])
    return fold(records)


def evaluate_policies(
    backend: PhysicsBackend,
    policies: Sequence[SimPolicy],
    protocol: EpisodeProtocol,
    *,
    source: str,
) -> tuple[SimScore, ...]:
    """Score every policy under the identical protocol; census-gated."""

    def run_episode(policy: SimPolicy, initial: Any) -> tuple[Any, Any]:
        return backend.closed_loop_rollout(
            initial, policy.act, protocol.steps, protocol.control_interval
        )

    return score_policies(
        backend, policies, protocol, source=source, run_episode=run_episode
    )


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
                sim_successes=score.successes,
                sim_trials=score.trials,
                real_successes=successes,
                real_trials=trials,
            )
        )
    return tuple(outcomes)
