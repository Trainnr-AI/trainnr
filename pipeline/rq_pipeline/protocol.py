"""The vocabulary every layer shares: what an episode is and how it is judged.

Standard library only, at the `bundles` tier (docs/22 §3): a third-party
task package imports THIS to describe its task, never the evaluation
harness, the physics engine or the env. Moved here 2026-08-26 from
`evaluate/harness.py` and `evaluate/vision.py` (which re-export the
names) so that "tasks depend on evaluate" stopped being an edge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Milestone = tuple[str, Callable[[Any, Any, int], bool]]


RGB_CHANNELS = 3  # every camera here renders RGB; the exporters and the env agree


@dataclass(frozen=True)
class CameraSpec:
    """One camera: the observation key it feeds and the render geometry."""

    key: str  # observation.images.<key>
    camera_name: str  # the camera's name in the model
    width: int
    height: int


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
    from the engine — the ALOHA rig's reset state has both arms
    straight up on a path that jams the grippers together at over 1 kN,
    while the SO-101 tasks were tuned from theirs.

    `milestones` is an ORDERED chain of (name, predicate(states, sensors,
    step) -> bool) — Arena's progress tracking (docs/30 §7 row 39),
    computed offline over the episode we already keep. Never a verdict:
    `success` alone decides; milestones say how far a failure got
    ("moved 4/4, lifted 0/4"), which a row of zeros hides.
    """

    trials: int
    steps: int
    control_interval: int
    perturb: Callable[[int, Any], Any]
    success: Callable[[Any, Any], bool]
    home: str | None = None
    milestones: tuple[Milestone, ...] = ()
    # What the referee needs beyond the FULLPHYSICS row and the sensors —
    # e.g. ("particle_q",) for a deformable object. An engine declares what
    # it exposes (`Engine.observables`); the harness refuses a mismatch
    # before a trial is spent. Empty for every rigid task today.
    observables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("trials", self.trials),
            ("steps", self.steps),
            ("control_interval", self.control_interval),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        names = [name for name, _ in self.milestones]
        if len(set(names)) != len(names):
            raise ValueError(f"milestone names must be unique, got {names}")

    @property
    def control_ticks(self) -> int:
        """Control ticks per episode; the last may hold fewer physics
        steps when `steps` is not a multiple of `control_interval`."""
        return -(-self.steps // self.control_interval)


def events_for(
    protocol: EpisodeProtocol, states: Any, sensors: Any
) -> tuple[dict, ...]:
    """Walk the episode once: evaluate only the chain's current
    milestone at each physics step, advance at most one per step, and
    record the first step each fires (Arena's tracker rule, offline).
    Returns the fired milestones in order — an empty tuple means the
    policy never reached the first one."""
    events: list[dict] = []
    index = 0
    for step in range(len(states)):
        if index >= len(protocol.milestones):
            break
        name, predicate = protocol.milestones[index]
        if predicate(states, sensors, step):
            events.append({"index": index, "name": name, "step": step})
            index += 1
    return tuple(events)


def protocol_fields(protocol: EpisodeProtocol) -> dict[str, Any]:
    """The scalar half of a protocol — what a record can carry, including
    the milestone chain's names so a reader knows how long a complete
    funnel is."""
    fields: dict[str, Any] = {
        "trials": protocol.trials,
        "steps": protocol.steps,
        "control_interval": protocol.control_interval,
        "home": protocol.home,
        "milestones": [name for name, _ in protocol.milestones],
    }
    if protocol.observables:  # absent when empty: every existing hash stands
        fields["observables"] = list(protocol.observables)
    return fields
