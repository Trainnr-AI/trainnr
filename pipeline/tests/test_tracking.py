"""The locomotion judgment, declared once for the evaluation and the gate."""

from __future__ import annotations

import unittest

from rq_pipeline.evaluate.tracking import (
    ERR_FLOOR_MPS,
    ERR_RATIO_BOUND,
    TrackingOutcome,
    criterion_text,
)


class TheRule(unittest.TestCase):
    def test_survived_and_tracked_make_a_success(self) -> None:
        ok = TrackingOutcome(steps=100, fell=False, mean_err=0.1, mean_cmd=0.5)
        self.assertTrue(ok.survived)
        self.assertTrue(ok.tracked)
        self.assertTrue(ok.success)
        self.assertLess(ok.err_ratio, ERR_RATIO_BOUND)

    def test_a_fall_fails_whatever_the_tracking(self) -> None:
        fell = TrackingOutcome(steps=10, fell=True, mean_err=0.0, mean_cmd=0.5)
        self.assertTrue(fell.tracked)
        self.assertFalse(fell.success)

    def test_the_denominator_floors_at_a_slow_command(self) -> None:
        slow = TrackingOutcome(steps=100, fell=False, mean_err=0.06, mean_cmd=0.01)
        self.assertAlmostEqual(slow.err_ratio, 0.06 / ERR_FLOOR_MPS)
        self.assertFalse(slow.tracked)

    def test_the_criterion_states_the_bound(self) -> None:
        self.assertEqual(criterion_text(), f"survived and err_ratio<{ERR_RATIO_BOUND}")
