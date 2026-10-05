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


class PlacementChecks:
    """The verdicts a start earns, by name — the keys of every
    `EpisodeRecord.placement` entry. All three are REQUIRED today: a
    start failing any one is refused before the trial is spent
    (docs/e2e-research/42 §3, A3: cheap checks, no silent fallback)."""

    IN_LIMITS = "in_limits"  # the body's centre inside the declared x/y bands
    ON_SUPPORT = "on_support"  # footprint inside the support's, bottom on its top
    NO_OVERLAP = "no_overlap"  # no penetration with anything but the support
    ALL = (IN_LIMITS, ON_SUPPORT, NO_OVERLAP)


@dataclass(frozen=True)
class Placement:
    """Where a free body must START: resting on `support` (a geom name),
    its centre inside the optional `x`/`y` bands, penetrating nothing
    else. Declared by the task beside `perturb`, so "where a part
    starts" is protocol content — hashed, recorded, and checked on the
    simulator's own geometry before an episode is spent. `tolerance_m`
    is how far the body's bottom may sit above or below the support's
    top and still count as resting (Arena's 5 mm; docs/42 §3 A3)."""

    body: str
    support: str
    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    tolerance_m: float = 0.005

    def __post_init__(self) -> None:
        for axis, band in (("x", self.x), ("y", self.y)):
            if band is not None and not band[0] < band[1]:
                raise ValueError(
                    f"{self.body}: {axis} band must be (low, high), got {band}"
                )
        if self.tolerance_m <= 0:
            raise ValueError(f"{self.body}: tolerance_m must be positive")


def placement_fields(placement: Placement) -> dict[str, Any]:
    return {
        "body": placement.body,
        "support": placement.support,
        "x": list(placement.x) if placement.x else None,
        "y": list(placement.y) if placement.y else None,
        "tolerance_m": placement.tolerance_m,
    }


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
    step) -> bool) — Arena's progress tracking (docs/e2e-research/39),
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
    # before a trial is spent. Empty for every rigid task today, and the
    # gate is all that exists: the referee call gains the extras with the
    # first engine that has any (harness.Engine).
    observables: tuple[str, ...] = ()
    # Where each free body must start (`Placement`); the engine checks
    # every trial's start against these before any episode is spent.
    placements: tuple[Placement, ...] = ()
    # How many control ticks of a policy's predicted chunk are executed
    # before it is asked again (`evaluate/scheduler.py`). None: the policy
    # replans on its own terms (a LeRobot checkpoint's `n_action_steps`),
    # and the record says so by its absence. A number here is hashed with
    # the trials: two runs differing only in replan rate are two protocols.
    executed_horizon: int | None = None

    def __post_init__(self) -> None:
        if self.executed_horizon is not None and self.executed_horizon <= 0:
            raise ValueError(
                f"executed_horizon must be positive, got {self.executed_horizon}"
            )
        for field_name, value in (
            ("trials", self.trials),
            ("steps", self.steps),
            ("control_interval", self.control_interval),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        bodies = [placement.body for placement in self.placements]
        if len(set(bodies)) != len(bodies):
            raise ValueError(f"a body is placed twice: {sorted(bodies)}")
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
    if protocol.placements:  # likewise
        fields["placements"] = [placement_fields(p) for p in protocol.placements]
    if protocol.executed_horizon is not None:  # likewise
        fields["executed_horizon"] = protocol.executed_horizon
    return fields
