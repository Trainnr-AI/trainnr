"""SmoothRL's instantiation in torch (docs/e2e-research/71 §3.4, §4a):
a bounded residual actor on a frozen base's chunk, an ensemble of chunk
critics that see the committed region, and the actor loss of Eq. (5)
with the value gradient truncated to the execution region.

The base policy is abstract here - anything that yields a reference
chunk `(H, nu)` for a state - so the same head sits on an ACT student,
a diffusion policy or a VLA. Needs the `train` extra (torch); the pure
pieces it uses live in `rl/smooth_rl.py`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rq_pipeline.rl.smooth_rl import SmoothRLKnobs, gradient_mask


def mlp(inputs: int, outputs: int, hidden: int, layers: int) -> nn.Sequential:
    """3 x 512 with LayerNorm on every hidden layer (paper §3.4, §4.3)."""
    blocks: list[nn.Module] = []
    width = inputs
    for _ in range(layers):
        blocks += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        width = hidden
    blocks.append(nn.Linear(width, outputs))
    return nn.Sequential(*blocks)


class ResidualActor(nn.Module):
    """`[s, reference chunk[0:2n)] -> bounded correction (2n, nu)`, added
    to the reference (§3.4 "the actor"); the box is `correction_bound`."""

    def __init__(self, state_dim: int, nu: int, knobs: SmoothRLKnobs) -> None:
        super().__init__()
        self.regions = knobs.regions
        self.nu = nu
        self.bound = knobs.correction_bound
        span = self.regions.span
        self.net = mlp(state_dim + span * nu, span * nu, knobs.hidden, knobs.layers)

    def forward(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        span = self.regions.span
        flat = reference[:, :span].reshape(len(state), -1)
        correction = torch.tanh(self.net(torch.cat([state, flat], dim=1))) * self.bound
        return reference[:, :span] + correction.view(len(state), span, self.nu)


class ChunkCritic(nn.Module):
    """`Q(s, committed, execution)` (§3.3): the committed region is the
    in-flight action, part of the state of a concurrent decision."""

    def __init__(self, state_dim: int, nu: int, knobs: SmoothRLKnobs) -> None:
        super().__init__()
        span = knobs.regions.span
        self.net = mlp(state_dim + span * nu, 1, knobs.hidden, knobs.layers)

    def forward(self, state: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, chunk.reshape(len(state), -1)], dim=1))


def penalty(chunk: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    """Eq. (5)'s smoothness term, differentiable, per batch element."""
    total = torch.zeros(len(chunk), device=chunk.device)
    diff = chunk
    for weight in weights:
        diff = diff[:, 1:] - diff[:, :-1]
        total = total + weight * diff.pow(2).sum(dim=(1, 2))
    return total


def evaluated_chunk(
    chunk: torch.Tensor, committed: torch.Tensor, knobs: SmoothRLKnobs
) -> torch.Tensor:
    """What the critic is asked about: the committed rows the robot ran,
    as a constant, and the actor's execution rows (§3.3). With
    `knobs.truncate=False` the actor's whole span goes in - the
    ablation the paper never ran."""
    if not knobs.truncate:
        return chunk
    mask = torch.as_tensor(
        gradient_mask(knobs.regions), dtype=chunk.dtype, device=chunk.device
    )
    return committed * (1 - mask)[None, :, None] + chunk * mask[None, :, None]


def actor_loss(  # noqa: PLR0913 - Eq. (5)'s terms, each named
    actor: ResidualActor,
    critics: list[ChunkCritic],
    *,
    state: torch.Tensor,
    reference: torch.Tensor,
    committed: torch.Tensor,
    target: torch.Tensor,
    knobs: SmoothRLKnobs,
) -> torch.Tensor:
    """Eq. (5): `-Q(s, sg[committed], a_exec) + w_bc ||a - target||^2 +
    w_smooth * penalty`, the value gradient reaching the actor only
    through the execution rows (§3.3 "value gradient truncation").
    `knobs.truncate=False` lets it through the whole span - the
    ablation the paper never ran."""
    chunk = actor(state, reference)
    evaluated = evaluated_chunk(chunk, committed, knobs)
    values = (
        torch.stack([critic(state, evaluated) for critic in critics]).min(dim=0).values
    )
    bc = (chunk - target).pow(2).sum(dim=(1, 2))
    smooth = penalty(chunk, knobs.derivative_weights)
    return (
        -values.squeeze(-1) + knobs.bc_weight * bc + knobs.smooth_weight * smooth
    ).mean()


def critic_target(  # noqa: PLR0913 - the backup's ingredients, each named
    actor_target: ResidualActor,
    critic_targets: list[ChunkCritic],
    *,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    next_reference: torch.Tensor,
    next_committed: torch.Tensor,
    knobs: SmoothRLKnobs,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Algorithm 1 lines 22-24: the target actor's chunk with clipped
    noise, the committed region the next state actually carries, a
    pessimistic minimum over a random subset of `critic_subset` target
    critics, discounted by `gamma^(2n)` (§3.3)."""
    regions = actor_target.regions
    with torch.no_grad():
        chunk = actor_target(next_state, next_reference)
        noise = (
            torch.randn(chunk.shape, generator=generator, device=chunk.device)
            * knobs.target_noise
        ).clamp(-knobs.target_noise_clip, knobs.target_noise_clip)
        evaluated = evaluated_chunk(chunk + noise, next_committed, knobs)
        picked = torch.randperm(len(critic_targets), generator=generator)[
            : knobs.critic_subset
        ]
        q_next = (
            torch.stack([critic_targets[int(i)](next_state, evaluated) for i in picked])
            .min(dim=0)
            .values
        )
        return reward.view(-1, 1) + knobs.gamma**regions.span * q_next


__all__: list[Any] = [
    "ChunkCritic",
    "ResidualActor",
    "actor_loss",
    "critic_target",
    "evaluated_chunk",
    "mlp",
    "penalty",
]
