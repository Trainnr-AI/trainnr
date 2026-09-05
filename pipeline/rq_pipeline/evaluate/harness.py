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
from typing import Any, Protocol, runtime_checkable

from rq_pipeline.bundles.hashing import require_stamp
from rq_pipeline.evaluate.certificate import PolicyOutcome
from rq_pipeline.evaluate.records import (
    EpisodeRecord,
    SimScore,
    append_records,
    fold,
)
from rq_pipeline.protocol import (
    EpisodeProtocol,
    Milestone,
    events_for,
    protocol_fields,
)
from rq_pipeline.robot.model_checks import assert_model_alive

__all__ = [
    "Engine",
    "EpisodeProtocol",
    "Milestone",
    "SimPolicy",
    "SimScore",
    "evaluate_policies",
    "events_for",
    "home_state",
    "join_with_real",
    "require_observables",
    "run_sensor_episode",
    "score_policies",
    "sensor_episode",
]


@runtime_checkable
class Stepper(Protocol):
    """One episode's stepping loop, as the harness drives it: `advance`
    holds a control for some physics steps, `sensordata` is what the
    policy sees now, `states`/`sensors` are the rows the referees read,
    `extras` is anything the engine exposes beyond the row (a deformable
    engine's particles), keyed as `EpisodeProtocol.observables` names.
    Shapes are the engine's: the CPU stepper is one world (`control
    (nu,)`, `states (steps, width)`), the GPU stepper a batch (`(nbatch,
    nu)`, `(nbatch, steps, width)`); `run_sensor_episode` is the
    single-world loop and drives the CPU engine."""

    step: int
    states: Any
    sensors: Any
    extras: Mapping[str, Any]

    @property
    def done(self) -> bool: ...

    @property
    def sensordata(self) -> Any: ...

    def advance(self, control: Any, substeps: int) -> None: ...


@runtime_checkable
class Engine(Protocol):
    """What the harness asks of a physics engine — declared here, where
    it is consumed, so `evaluate` names no engine. `MuJoCoBackend`
    satisfies it structurally (asserted in tests); a second engine
    implements these seven, and is admitted through the gauntlet
    (tests/test_mjx_backend.py). `observables` names what the engine
    can expose beyond the FULLPHYSICS row; a protocol that needs more
    is refused before a trial is spent (`require_observables`). The
    gate is built; the delivery — `Stepper.extras` reaching a referee —
    lands with the first engine that exposes anything, so it is designed
    against a real consumer."""

    @property
    def instrument(self) -> str: ...  # physics/backend.py::instrument_stamp

    @property
    def observables(self) -> frozenset[str]: ...

    def counts(self) -> Any: ...  # the census: actuators, sensors, geoms, cameras

    def default_initial_state(self) -> Any: ...

    def keyframe_state(self, name: str) -> Any: ...

    def stepper(self, initial_state: Any, steps: int) -> Any: ...

    def validate_start(
        self, state: Any, placements: Sequence[Any], *, trial: int
    ) -> Any: ...  # physics/placement.py: verdicts, or a refusal naming the culprit


@dataclass(frozen=True)
class SimPolicy:
    """A named closed-loop controller: (step_index, sensordata) → controls."""

    name: str
    act: Callable[[int, Any], Any]


def home_state(backend: Engine, protocol: EpisodeProtocol) -> Any:
    """The protocol's declared start: its keyframe, else the model's reset."""
    if protocol.home is None:
        return backend.default_initial_state()
    return backend.keyframe_state(protocol.home)


def run_sensor_episode(
    backend: Engine,
    act: Callable[[int, Any], Any],
    initial_state: Any,
    *,
    steps: int,
    control_interval: int,
) -> tuple[Any, Any]:
    """One episode of a SENSOR policy — scripted experts and the dry
    runs' ablations: `act(step, sensordata) -> nu controls`, held for
    `control_interval` physics steps. The policy sees a copy of the
    sensors (it cannot write into the simulator) and what it commands
    is what the actuators get, exactly like the wire. Pixel policies go
    through the gymnasium env instead (rq_pipeline.envs); both are the
    same `Stepper`."""
    states, sensors, _ = sensor_episode(
        backend, act, initial_state, steps=steps, control_interval=control_interval
    )
    return states, sensors


def sensor_episode(
    backend: Engine,
    act: Callable[[int, Any], Any],
    initial_state: Any,
    *,
    steps: int,
    control_interval: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """`run_sensor_episode` plus what the certificate columns need: the
    smoothness of the issued controls (docs/e2e-research/71 E0), keyed
    like every other `variations` entry, empty when the engine does not
    state its timestep."""
    from rq_pipeline.evaluate.smoothness import (  # noqa: PLC0415
        Smoothness,
        control_period,
    )

    stepper = backend.stepper(initial_state, steps)
    controls = []
    while not stepper.done:
        control = act(stepper.step, stepper.sensordata)
        controls.append(control)
        stepper.advance(control, control_interval)
    period = control_period(stepper, control_interval)
    columns = Smoothness.of(controls, dt=period).columns() if period else {}
    return stepper.states, stepper.sensors, columns


def require_observables(backend: Engine, protocol: EpisodeProtocol) -> None:
    """A protocol that needs what the engine cannot expose is refused
    here, naming both — before any episode is spent."""
    missing = set(protocol.observables) - set(backend.observables)
    if missing:
        raise ValueError(
            f"protocol needs observables {sorted(missing)} that engine "
            f"{backend.instrument} does not expose "
            f"(it exposes {sorted(backend.observables) or 'none'})"
        )


def score_policies(  # noqa: PLR0913 - the skeleton carries both harnesses' knobs
    backend: Engine,
    policies: Sequence[Any],
    protocol: EpisodeProtocol,
    *,
    source: str,
    run_episode: Callable[[Any, Any], tuple[Any, ...]],
    gate_cameras: bool = False,
    record_to: Path | None = None,
) -> tuple[SimScore, ...]:
    """The scoring skeleton both harnesses share: refuse duplicate
    names and unstamped sources, census-gate the model, then run every
    policy through the identical paired trials. `run_episode(policy,
    initial_state) -> (states, sensors)` is the only thing that differs
    between observing sensors and observing pixels — so it is the only
    thing callers supply; a third element, when returned, is the row's
    `variations` (the smoothness columns). Vision callers set `gate_cameras` because a
    camera-less model would score their policies 0% silently.

    Every trial becomes an `EpisodeRecord`; the scores are the fold over
    them (records.py), and `record_to` appends the rows as trials finish
    — a crash at trial 9 leaves 8 lines, not nothing.
    """
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate policy names: {sorted(names)}")
    require_stamp(source)  # the same rule certify() enforces, before episodes are spent
    counts = backend.counts()
    assert_model_alive(
        counts.actuators,
        counts.sensors,
        counts.geoms,
        source=source,
        cameras=counts.cameras if gate_cameras else None,
    )
    require_observables(backend, protocol)
    home = home_state(backend, protocol)
    instrument = backend.instrument
    fields = protocol_fields(protocol)
    # Every trial's start exists and is admissible before any episode is
    # spent (docs/42 §3 A4): a protocol that cannot place trial 3 fails
    # here, with the culprit named, not after trials 0-2 ran.
    starts = [protocol.perturb(trial, home) for trial in range(protocol.trials)]
    placement = [
        backend.validate_start(start, protocol.placements, trial=trial)
        for trial, start in enumerate(starts)
    ]
    records: list[EpisodeRecord] = []
    for policy in policies:
        for trial, start in enumerate(starts):
            states, sensors, *extra = run_episode(policy, start)
            record = EpisodeRecord(
                source=source,
                policy=policy.name,
                trial=trial,
                success=bool(protocol.success(states, sensors)),
                steps=len(states),
                instrument=instrument,
                protocol=fields,
                events=events_for(protocol, states, sensors),
                variations=dict(extra[0]) if extra else {},
                placement=placement[trial],
            )
            records.append(record)
            if record_to is not None:
                append_records(record_to, [record])
    return fold(records)


def evaluate_policies(
    backend: Engine,
    policies: Sequence[SimPolicy],
    protocol: EpisodeProtocol,
    *,
    source: str,
    record_to: Path | None = None,
) -> tuple[SimScore, ...]:
    """Score every sensor policy under the identical protocol; census-gated."""

    def run_episode(policy: SimPolicy, initial: Any) -> tuple[Any, ...]:
        return sensor_episode(
            backend,
            policy.act,
            initial,
            steps=protocol.steps,
            control_interval=protocol.control_interval,
        )

    return score_policies(
        backend,
        policies,
        protocol,
        source=source,
        run_episode=run_episode,
        record_to=record_to,
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
        # Counts are validated once, by PolicyOutcome itself.
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
