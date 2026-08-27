"""Task acceptance: the referee reviews the agent (docs/30 §3.6 rule 1,
docs/e2e-research/44 §3 item 2).

A task is accepted only when it is demonstrably a test: the scripted
EXPERT — the choreographer that made the demonstrations — passes the
task's own referee on every paired trial under the protocol (placement
gate, milestones, the lot), and the do-nothing FLOOR passes none. A
task the expert cannot do is rejected with the funnel as the reason
("moved 4/4, lifted 4/4, placed 0/4"); a task the floor passes is
rejected as no test at all. Arena accepts a spec when it loads; its own
docs say that "only proves a spec is admissible, not that it is the
environment you asked for". Nothing here grades its own output: the
expert is the critic, and the certificate machinery does the counting.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.evaluate.harness import SimPolicy, home_state, run_sensor_episode
from rq_pipeline.evaluate.records import EpisodeRecord, append_records, funnel
from rq_pipeline.physics.mujoco_backend import MuJoCoBackend
from rq_pipeline.protocol import events_for, protocol_fields
from rq_pipeline.tasks.task import Task

# expert(model, initial_state) -> (states, sensors, ...): the privileged
# choreographer; it may refuse a trial (a RuntimeError names why).
Expert = Callable[..., tuple]
EXPERT_NAME = "expert"
FLOOR_NAME = "floor"


@dataclass(frozen=True)
class Acceptance:
    """The verdict and everything it rests on."""

    task: str  # the task's stamp when it has a spec, else its name
    accepted: bool
    reasons: tuple[str, ...]  # empty when accepted
    expert_successes: int
    floor_successes: int
    trials: int
    refusals: tuple[str, ...]  # trials the expert could not even attempt
    funnel: Mapping[str, list[int]]
    records: tuple[EpisodeRecord, ...]

    def __str__(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        stages = self.funnel.get(EXPERT_NAME, [])
        lines = [
            f"{verdict}: {self.task}",
            f"  expert {self.expert_successes}/{self.trials}, "
            f"floor {self.floor_successes}/{self.trials}",
        ]
        if stages:
            names = list(self.records[0].protocol.get("milestones", []))
            lines.append(
                "  funnel "
                + " · ".join(
                    f"{n} {c}/{self.trials}" for n, c in zip(names, stages, strict=True)
                )
            )
        lines.extend(f"  - {reason}" for reason in self.reasons)
        return "\n".join(lines)


def hold_home(backend: MuJoCoBackend, protocol: Any) -> SimPolicy:
    """The floor: hold the home keyframe's control forever — the policy
    that does nothing, which a real task must not reward."""
    import mujoco  # noqa: PLC0415 - sim extra

    model = backend.model
    if protocol.home is not None:
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, protocol.home)
        control = list(model.key_ctrl[key])
    else:
        control = [0.0] * model.nu
    return SimPolicy(FLOOR_NAME, lambda _step, _sensors: control)


def accept(  # noqa: PLR0913 - the loop's knobs, each named
    task: Task,
    expert: Expert,
    *,
    source: str,
    floor: SimPolicy | None = None,
    record_to: Path | None = None,
    backend: MuJoCoBackend | None = None,
) -> Acceptance:
    """Run the expert and the floor through the task's protocol, paired
    trial for paired trial, and decide. `source` is the bundle stamp the
    records cite (`require_stamp` applies); `record_to` keeps the rows."""
    backend = backend or MuJoCoBackend()
    if not backend.loaded:
        backend.load_spec(task.spec)
    protocol = task.protocol
    model = backend.model
    home = home_state(backend, protocol)
    floor = floor or hold_home(backend, protocol)
    fields = protocol_fields(protocol)
    starts = [protocol.perturb(trial, home) for trial in range(protocol.trials)]
    placement = [
        backend.validate_start(start, protocol.placements, trial=trial)
        for trial, start in enumerate(starts)
    ]
    records: list[EpisodeRecord] = []
    refusals: list[str] = []

    def record(name: str, trial: int, states: Any, sensors: Any) -> None:
        row = EpisodeRecord(
            source=source,
            policy=name,
            trial=trial,
            success=bool(protocol.success(states, sensors)),
            steps=len(states),
            instrument=backend.instrument,
            protocol=fields,
            events=events_for(protocol, states, sensors),
            placement=placement[trial],
        )
        records.append(row)
        if record_to is not None:
            append_records(record_to, [row])

    for trial, start in enumerate(starts):
        try:
            states, sensors = expert(model, start)[:2]
        except RuntimeError as error:  # the expert could not attempt it
            refusals.append(f"trial {trial}: {error}")
            continue
        record(EXPERT_NAME, trial, states, sensors)
    for trial, start in enumerate(starts):
        states, sensors = run_sensor_episode(
            backend,
            floor.act,
            start,
            steps=protocol.steps,
            control_interval=protocol.control_interval,
        )
        record(FLOOR_NAME, trial, states, sensors)

    expert_rows = [r for r in records if r.policy == EXPERT_NAME]
    floor_rows = [r for r in records if r.policy == FLOOR_NAME]
    expert_successes = sum(r.success for r in expert_rows)
    floor_successes = sum(r.success for r in floor_rows)
    stages = funnel(expert_rows) if expert_rows else {}
    reasons: list[str] = []
    if refusals:
        reasons.append("the expert could not attempt " + "; ".join(refusals))
    if expert_successes < protocol.trials:
        names = [name for name, _ in protocol.milestones]
        counts = stages.get(EXPERT_NAME, [])
        where = (
            " — funnel "
            + " · ".join(
                f"{n} {c}/{protocol.trials}" for n, c in zip(names, counts, strict=True)
            )
            if names and counts
            else ""
        )
        reasons.append(
            f"the expert passed {expert_successes}/{protocol.trials} paired "
            f"trials{where}"
        )
    if floor_successes:
        reasons.append(
            f"the floor policy ({floor.name}) passed "
            f"{floor_successes}/{protocol.trials}: the referee rewards doing "
            "nothing, so this is not a test"
        )
    return Acceptance(
        task=task.stamp or task.name,
        accepted=not reasons,
        reasons=tuple(reasons),
        expert_successes=expert_successes,
        floor_successes=floor_successes,
        trials=protocol.trials,
        refusals=tuple(refusals),
        funnel=stages,
        records=tuple(records),
    )
