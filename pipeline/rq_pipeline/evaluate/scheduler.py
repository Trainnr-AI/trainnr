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

    def __init__(self, policy: ChunkPolicy, *, executed_horizon: int, nu: int) -> None:
        if executed_horizon <= 0:
            raise ValueError(
                f"executed_horizon must be positive, got {executed_horizon}"
            )
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.policy = policy
        self.executed_horizon = executed_horizon
        self.nu = nu
        self._chunk: Any = None
        self._cursor = 0
        self.fetches = (
            0  # how often the policy was asked: the replan count a record can carry
        )

    def reset(self) -> None:
        self._chunk = None
        self._cursor = 0
        self.fetches = 0
        self.policy.reset()

    def act(self, observation: Mapping[str, Any]) -> Any:
        if self._chunk is None or self._cursor >= self.executed_horizon:
            self._chunk = self._checked(self.policy.predict(observation))
            self._cursor = 0
            self.fetches += 1
        action = self._chunk[self._cursor]
        self._cursor += 1
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
        if horizon < self.executed_horizon:
            raise ValueError(
                f"{self.policy.name}: chunk has {horizon} steps, fewer than the "
                f"protocol's executed_horizon {self.executed_horizon}"
            )
        return chunk
