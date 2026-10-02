"""The replay SmoothRL's two processes share (Algorithm 1, lines 12-15
and 20), as files: the rollout side appends one shard per episode, the
learner samples across shards. Framework-free numpy; the transition is
the paper's tuple, named.

Symbols, once: `n` the budget, `s` the state, `a~[0,n)` the committed
rows the robot actually ran, `a[n,2n)` the execution rows it issued,
`a_target` the BC anchor over `[0,2n)` (the base's reference, or a
human's chunk), `r` the reward summed over the `2n` frames, and the
primed quantities the same at `t + 2n`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

SHARD_GLOB = "episode-*.npz"


@dataclass(frozen=True)
class Transition:
    """One decision of the timed loop, with its bootstrap ingredients."""

    state: Any  # (state_dim,)
    reference: Any  # (2n, nu) - the base's chunk the actor corrected
    committed: Any  # (n, nu) - a~[0,n), what ran while inferring
    execution: Any  # (n, nu) - a[n,2n), what this chunk issued
    target: Any  # (2n, nu) - the BC anchor
    reward: float  # summed over the 2n frames
    next_state: Any
    next_reference: Any
    next_committed: Any
    done: bool


FIELDS = tuple(f.name for f in fields(Transition))


class ReplayShards:
    """A directory of episodes, one `.npz` each, stacked per field."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, episode: Sequence[Transition]) -> Path:
        if not episode:
            raise ValueError("an episode shard needs at least one transition")
        index = len(list(self.shards()))
        path = self.root / f"episode-{index:05d}.npz"
        staging = path.with_suffix(".tmp.npz")
        # Any: numpy's stub types **kwargs against savez's own keywords.
        arrays: dict[str, Any] = {
            name: np.stack([np.asarray(getattr(t, name)) for t in episode])
            for name in FIELDS
        }
        np.savez(staging, **arrays)
        staging.replace(path)  # a reader never sees half a shard
        return path

    def shards(self) -> Iterator[Path]:
        return iter(sorted(self.root.glob(SHARD_GLOB)))

    def load(self) -> dict[str, np.ndarray]:
        """Every transition across shards, concatenated per field."""
        parts = [np.load(p) for p in self.shards()]
        if not parts:
            return {name: np.zeros((0,)) for name in FIELDS}
        return {name: np.concatenate([p[name] for p in parts]) for name in FIELDS}

    def sample(self, batch: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """`batch` transitions drawn uniformly across every shard."""
        table = self.load()
        count = len(table["reward"])
        if count == 0:
            raise ValueError(f"no transitions under {self.root}")
        picked = rng.integers(0, count, size=batch)
        return {name: table[name][picked] for name in FIELDS}
