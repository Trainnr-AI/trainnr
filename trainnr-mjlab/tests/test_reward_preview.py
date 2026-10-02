"""The preview's summary: per term, over a rollout nobody trained."""

from __future__ import annotations

import unittest

from trainnr_mjlab.reward_preview import CONTROLLERS, summarize


class TheSummary(unittest.TestCase):
    def test_per_term_statistics_and_the_total(self) -> None:
        rows = [{"pose": 0.5, "slip": -0.1}, {"pose": 0.7, "slip": -0.3}]
        out = summarize(rows, fell=1)
        self.assertEqual(out["steps"], 2)
        self.assertEqual(out["falls"], 1)
        self.assertAlmostEqual(out["terms"]["pose"]["mean"], 0.6)
        self.assertAlmostEqual(out["terms"]["slip"]["min"], -0.3)
        self.assertAlmostEqual(out["total_mean"], 0.4)

    def test_the_two_controllers(self) -> None:
        self.assertEqual(CONTROLLERS, ("untrained", "stand"))
