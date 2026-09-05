"""SmoothRL transcribed (docs/e2e-research/71 §4a): the partition, the
backup, the penalty and the gradient mask, against hand-built cases."""

from __future__ import annotations

import unittest

import numpy as np

from rq_pipeline.rl.smooth_rl import (
    ChunkRegions,
    SmoothRLKnobs,
    chunk_skip_target,
    gradient_mask,
    smoothness_penalty,
)


class ThePartition(unittest.TestCase):
    def test_the_papers_own_numbers(self) -> None:
        # §4.1: n = 6, H = 32 - committed and execution 12 frames, 20 discarded
        regions = ChunkRegions(6, 32)
        rows = np.arange(32)
        self.assertEqual(rows[regions.committed].tolist(), list(range(0, 6)))
        self.assertEqual(rows[regions.execution].tolist(), list(range(6, 12)))
        self.assertEqual(len(rows[regions.discarded]), 20)
        self.assertEqual(regions.span, 12)

    def test_a_chunk_shorter_than_two_budgets_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ChunkRegions(6, 11)
        with self.assertRaises(ValueError):
            ChunkRegions(0, 4)


class TheBackup(unittest.TestCase):
    def test_the_discount_is_raised_to_the_span(self) -> None:
        regions = ChunkRegions(2, 8)
        target = chunk_skip_target(reward=1.0, gamma=0.5, span=regions.span, q_next=8.0)
        self.assertAlmostEqual(
            float(target), 1.0 + 0.5**4 * 8.0
        )  # 1.5, not 1 + 0.5 * 8


class ThePenalty(unittest.TestCase):
    def test_orders_are_weighted_separately(self) -> None:
        ramp = np.arange(5, dtype=float)[:, None]  # velocity 1, higher orders 0
        self.assertAlmostEqual(smoothness_penalty(ramp, (1.0, 7.0, 9.0)), 4.0)
        quad = (np.arange(5, dtype=float) ** 2)[:, None]  # acceleration 2 everywhere
        self.assertAlmostEqual(smoothness_penalty(quad, (0.0, 1.0, 0.0)), 3 * 4.0)


class TheGradientMask(unittest.TestCase):
    def test_only_the_execution_rows_carry_the_gradient(self) -> None:
        self.assertEqual(gradient_mask(ChunkRegions(2, 6)).tolist(), [0, 0, 1, 1])


class TheKnobs(unittest.TestCase):
    def test_stated_and_ours_are_told_apart(self) -> None:
        knobs = SmoothRLKnobs()
        self.assertEqual(knobs.regions.span, 12)
        self.assertIn("gamma", SmoothRLKnobs.ours())
        self.assertNotIn("budget_frames", SmoothRLKnobs.ours())


if __name__ == "__main__":
    unittest.main()
