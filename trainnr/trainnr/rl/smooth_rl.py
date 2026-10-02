"""SmoothRL's objective, transcribed (docs/e2e-research/71 §4a).

Astribot's SmoothRL (arXiv:2608.29768) ships no code. This module is
the paper's specification as code: the chunk partition its timed loop
induces, the chunk-skip backup its critic fits, the smoothness penalty
its actor pays, and every knob as a named field whose comment says
whether the paper states it (with the section) or we chose it. Pure
numpy, so the mechanism can be tested against hand-built cases before
any network exists; the torch actor and critic will live beside it and
use these functions, never their own copies.

Symbols, once: `n` is the inference budget in control frames, `H` the
chunk the policy emits, `gamma` the discount, `r` the reward summed over
the `2n` executed frames, `a[i:j]` rows `i..j-1` of a chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DERIVATIVE_ORDERS = 3  # velocity, acceleration, jerk - Eq. (5)
CHUNK_RANK = 2  # (T, nu)


@dataclass(frozen=True)
class ChunkRegions:
    """Frame ranges of one chunk under a budget `n` (paper §3.1, Fig. 2b):
    committed `[0, n)` - already issued by the previous chunk while this
    one was inferred; execution `[n, 2n)` - the only rows the robot runs
    from this chunk; discarded `[2n, H)` - superseded before they run."""

    budget: int
    horizon: int

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError(f"the budget n must be >= 1 frame, got {self.budget}")
        if self.horizon < 2 * self.budget:
            raise ValueError(
                f"a chunk of {self.horizon} frames cannot cover a budget of "
                f"{self.budget}: the objective needs H >= 2n (§3.3)"
            )

    @property
    def committed(self) -> slice:
        return slice(0, self.budget)

    @property
    def execution(self) -> slice:
        return slice(self.budget, 2 * self.budget)

    @property
    def discarded(self) -> slice:
        return slice(2 * self.budget, self.horizon)

    @property
    def span(self) -> int:
        """The frames the objective is defined over: `2n`, never `H`."""
        return 2 * self.budget


def chunk_skip_target(reward: Any, gamma: float, span: int, q_next: Any) -> Any:
    """The critic's target for one transition (§3.3, Eq. 3 restricted to
    the span): `r + gamma^(2n) * Q'`, where `r` already sums the span's
    frames and `Q'` is the pessimistic bootstrap at `s_{t+2n}`."""
    return np.asarray(reward, dtype=float) + gamma**span * np.asarray(
        q_next, dtype=float
    )


def smoothness_penalty(chunk: Any, weights: tuple[float, float, float]) -> float:
    """Eq. (5)'s third term for one chunk `(T, nu)`: `Σ_k w_k ‖Δ^k a‖²`
    over k = 1, 2, 3 (velocity, acceleration, jerk), the relaxation of
    the executable set `E` (§3.3)."""
    rows = np.asarray(chunk, dtype=float)
    if rows.ndim != CHUNK_RANK:
        raise ValueError(f"a chunk is (T, nu), got shape {rows.shape}")
    total = 0.0
    diff = rows
    for weight in weights[:DERIVATIVE_ORDERS]:
        diff = np.diff(diff, axis=0)
        total += float(weight) * float(np.sum(diff**2))
    return total


def gradient_mask(regions: ChunkRegions) -> Any:
    """Which rows of the objective's `[0, 2n)` span carry the value
    gradient (§3.3, "value gradient truncation"): the execution region
    only. Multiply the actor's chunk by it before the critic, keep the
    committed rows as a stop-gradient constant, and the gradient path
    is the paper's; all ones is the ablation the paper never ran."""
    mask = np.zeros(regions.span)
    mask[regions.execution] = 1.0
    return mask


@dataclass(frozen=True)
class SmoothRLKnobs:
    """Every number the instantiation needs. `stated` fields cite the
    paper; the others are ours (standard TD3 / REDQ practice) and say so,
    so a reader never mistakes a choice for a transcription."""

    budget_frames: int = 6  # stated: §4.1 - n = 6 at 30 Hz (200 ms)
    horizon: int = 32  # stated: §4.1 - H = 32
    hidden: int = 512  # stated: §4.3 - 3-layer MLPs, 512 units, LayerNorm (§3.4)
    layers: int = 3  # stated: §4.3
    correction_bound: float = 0.05  # stated: §4.3 - the residual's box
    update_to_data: int = 5  # stated: §4.3 - G
    actor_delay: int = 5  # stated: §4.3 - D
    batch_size: int = 256  # stated: §4.3
    seed_rollouts: int = 50  # stated: §4.4 - base-only rollouts seed the replay
    gamma: float = 0.99  # OURS: the paper states no discount
    tau: float = 0.005  # OURS: TD3's soft-update rate
    learning_rate: float = 3e-4  # OURS: not stated
    critics: int = 10  # OURS: REDQ's N; the paper says "N" only
    critic_subset: int = 2  # OURS: REDQ's M
    target_noise: float = 0.2  # OURS: TD3's default
    target_noise_clip: float = 0.5  # OURS: TD3's default
    bc_weight: float = 1.0  # OURS: w_bc unstated
    smooth_weight: float = 1.0  # OURS: w_smooth unstated
    derivative_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # OURS: w_k
    truncate: bool = True  # stated: §3.3; False is the ablation the paper lacks

    @property
    def regions(self) -> ChunkRegions:
        return ChunkRegions(self.budget_frames, self.horizon)

    @classmethod
    def ours(cls) -> tuple[str, ...]:
        """The fields the paper does not state, by name."""
        return (
            "gamma",
            "tau",
            "learning_rate",
            "critics",
            "critic_subset",
            "target_noise",
            "target_noise_clip",
            "bc_weight",
            "smooth_weight",
            "derivative_weights",
        )
