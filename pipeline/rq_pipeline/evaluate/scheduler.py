"""The action scheduler: a chunk-predicting policy, executed on OUR
horizon (docs/e2e-research/43 §3, item 1).

A chunking policy answers one observation with a chunk of future
actions, `(horizon, nu)`. Which prefix of it is executed before the
policy is asked again is a fact about the evaluation — Arena's
`ActionChunkScheduler` picked Cosmos's 16 by eye; LeRobot hides its own
`n_action_steps` inside the checkpoint. Here it is `EpisodeProtocol.
executed_horizon`, hashed with the trials, and this module executes it:
`ActionScheduler` holds the chunk, hands out one action per control tick,
and re-asks the policy every `executed_horizon` ticks. Every fetch is
checked (docs/43 §3 item 5): a 2-D chunk, `nu` wide, at least
`executed_horizon` long — each turns a silent zero action into a refused
run. Standard library only; the chunk is whatever array the policy
returns, indexed row by row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

CHUNK_RANK = 2  # (horizon, nu)


@dataclass(frozen=True)
class ChunkPolicy:
    """A policy that predicts a chunk: `predict(observation) -> (horizon,
    nu)` over the env's raw observation, and `reset()` before every
    episode (whatever it caches must not leak between trials)."""

    name: str
    predict: Callable[[Mapping[str, Any]], Any]
    reset: Callable[[], None]


class ActionScheduler:
    """Execute `executed_horizon` actions of each predicted chunk, then
    ask again. `act(observation)` is the per-tick policy the env and
    train-watch drive; `reset()` drops the held chunk — call it where
    the policy's own `reset` is called, before every episode."""

    def __init__(
        self, policy: ChunkPolicy, *, executed_horizon: int, nu: int, latency: int = 0
    ) -> None:
        if executed_horizon <= 0:
            raise ValueError(
                f"executed_horizon must be positive, got {executed_horizon}"
            )
        if latency < 0:
            raise ValueError(f"latency must be >= 0 ticks, got {latency}")
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.policy = policy
        self.executed_horizon = executed_horizon
        self.nu = nu
        # The inference budget in ticks (docs/e2e-research/71): a chunk
        # asked at tick t takes over at t + latency, and until then the
        # chunk in flight keeps supplying its own later rows - the
        # committed / execution / discarded partition of a real robot,
        # emulated exactly. 0 is the synchronous loop (the default).
        self.latency = latency
        self._tick = 0
        self._current: tuple[Any, int] | None = None  # (chunk, origin tick)
        self._pending: tuple[Any, int, int] | None = None  # (chunk, origin, arrival)
        self._asked_at = 0
        self.fetches = (
            0  # how often the policy was asked: the replan count a record can carry
        )

    def reset(self) -> None:
        self._tick = 0
        self._current = None
        self._pending = None
        self._asked_at = 0
        self.fetches = 0
        self.policy.reset()

    def act(self, observation: Mapping[str, Any]) -> Any:
        tick = self._tick
        if self._pending is not None and self._pending[2] <= tick:
            self._current = self._pending[:2]
            self._pending = None
        first = self._current is None
        if first or tick - self._asked_at >= self.executed_horizon:
            chunk = self._checked(self.policy.predict(observation))
            self._asked_at = tick
            self.fetches += 1
            if first or self.latency == 0:
                # The first chunk takes over at once: nothing is in flight
                # to bridge the wait (SmoothRL holds the robot still).
                self._current = (chunk, tick)
            else:
                self._pending = (chunk, tick, tick + self.latency)
        assert self._current is not None
        chunk, origin = self._current
        action = chunk[tick - origin]
        self._tick += 1
        return action

    def _checked(self, chunk: Any) -> Any:
        shape = tuple(getattr(chunk, "shape", ()))
        if len(shape) != CHUNK_RANK:
            raise ValueError(
                f"{self.policy.name}: a chunk must be (horizon, nu), got shape {shape}"
            )
        horizon, width = shape
        if width != self.nu:
            raise ValueError(
                f"{self.policy.name}: chunk is {width} wide, the model has "
                f"{self.nu} actuators"
            )
        needed = self.executed_horizon + self.latency
        if horizon < needed:
            raise ValueError(
                f"{self.policy.name}: chunk has {horizon} steps, fewer than the "
                f"protocol's executed_horizon {self.executed_horizon} plus the "
                f"latency {self.latency}"
            )
        return chunk
