"""SmoothRL's torch half (docs/e2e-research/71 §4a): the truncation the
paper claims, checked on our own objective - the value's gradient
reaches the committed rows only when truncation is off - and the
backup's discount by the span."""

from __future__ import annotations

import unittest

from tests._extras import needs_train

STATE, NU, BUDGET, HORIZON = 4, 3, 2, 6


def _setup(truncate: bool):
    import torch  # noqa: PLC0415

    from trainnr.rl.residual import ChunkCritic, ResidualActor  # noqa: PLC0415
    from trainnr.rl.smooth_rl import SmoothRLKnobs  # noqa: PLC0415

    torch.manual_seed(0)
    knobs = SmoothRLKnobs(
        budget_frames=BUDGET, horizon=HORIZON, hidden=16, layers=2, truncate=truncate
    )
    actor = ResidualActor(STATE, NU, knobs)
    critics = [ChunkCritic(STATE, NU, knobs) for _ in range(2)]
    state = torch.randn(5, STATE)
    reference = torch.randn(5, HORIZON, NU)
    committed = torch.randn(5, knobs.regions.span, NU)
    return knobs, actor, critics, state, reference, committed


@needs_train
class TheTruncation(unittest.TestCase):
    def test_committed_rows_carry_value_gradient_only_without_truncation(self) -> None:
        import torch  # noqa: PLC0415

        from trainnr.rl.residual import evaluated_chunk  # noqa: PLC0415

        for truncate, expect_grad in ((True, False), (False, True)):
            knobs, actor, critics, state, reference, committed = _setup(truncate)
            chunk = actor(state, reference).detach().requires_grad_(True)
            value = (
                torch.stack(
                    [
                        c(state, evaluated_chunk(chunk, committed, knobs))
                        for c in critics
                    ]
                )
                .min(dim=0)
                .values.sum()
            )
            (grad,) = torch.autograd.grad(value, chunk)
            on_committed = grad[:, knobs.regions.committed].abs().sum().item()
            on_execution = grad[:, knobs.regions.execution].abs().sum().item()
            self.assertGreater(on_execution, 0.0)
            self.assertEqual(
                on_committed > 0.0, expect_grad, msg=f"truncate={truncate}"
            )

    def test_the_actor_loss_is_finite_and_trains(self) -> None:
        import torch  # noqa: PLC0415

        from trainnr.rl.residual import actor_loss  # noqa: PLC0415

        knobs, actor, critics, state, reference, committed = _setup(True)
        loss = actor_loss(
            actor,
            critics,
            state=state,
            reference=reference,
            committed=committed,
            target=reference[:, : knobs.regions.span],
            knobs=knobs,
        )
        loss.backward()
        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_the_backup_discounts_by_the_span(self) -> None:
        import dataclasses  # noqa: PLC0415

        import torch  # noqa: PLC0415

        from trainnr.rl.residual import (  # noqa: PLC0415
            critic_target,
            evaluated_chunk,
        )

        knobs, actor, critics, state, reference, committed = _setup(True)
        quiet = dataclasses.replace(knobs, target_noise=0.0)
        target = critic_target(
            actor,
            critics,
            reward=torch.ones(5),
            next_state=state,
            next_reference=reference,
            next_committed=committed,
            knobs=quiet,
            generator=torch.Generator().manual_seed(1),
        )
        with torch.no_grad():
            evaluated = evaluated_chunk(actor(state, reference), committed, knobs)
            q_min = (
                torch.stack([c(state, evaluated) for c in critics]).min(dim=0).values
            )
        self.assertTrue(
            torch.allclose(
                target, 1.0 + knobs.gamma**knobs.regions.span * q_min, atol=1e-6
            )
        )


if __name__ == "__main__":
    unittest.main()
