"""The exact power module: size under the null, the measured sizing
pins for the C1 table (docs/e2e-research/62 §3), symmetry, refusals."""

from __future__ import annotations

import unittest

from trainnr.stats.power import fisher_power, trials_needed

ALPHA = 0.05
POWER = 0.8


class TheSize(unittest.TestCase):
    def test_never_exceeds_alpha_under_the_null(self) -> None:
        for rate in (0.3, 0.6, 0.9):
            self.assertLessEqual(fisher_power(rate, rate, 40, alpha=ALPHA), ALPHA)

    def test_symmetric_in_the_two_sides(self) -> None:
        self.assertAlmostEqual(
            fisher_power(0.3, 0.8, 25, alpha=ALPHA),
            fisher_power(0.8, 0.3, 25, alpha=ALPHA),
            places=12,
        )


class TheSizing(unittest.TestCase):
    def test_the_c1_table_rows(self) -> None:
        # Measured 2026-08-31, pinned: the docs/62 §3 table is these calls.
        self.assertEqual(trials_needed(0.5, 0.9, alpha=ALPHA, power=POWER), 23)
        self.assertEqual(trials_needed(0.3, 0.7, alpha=ALPHA, power=POWER), 29)
        self.assertEqual(trials_needed(0.6, 0.9, alpha=ALPHA, power=POWER), 36)

    def test_a_smaller_gap_needs_more_trials(self) -> None:
        big = trials_needed(0.4, 0.9, alpha=ALPHA, power=POWER)
        small = trials_needed(0.6, 0.8, alpha=ALPHA, power=POWER, limit=400)
        self.assertLess(big, small)

    def test_an_effect_too_small_for_the_limit_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than 10 trials"):
            trials_needed(0.60, 0.62, alpha=ALPHA, power=POWER, limit=10)


class TheRefusals(unittest.TestCase):
    def test_bad_arguments_are_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            fisher_power(0.5, 0.9, 10, alpha=1.5)
        with self.assertRaisesRegex(ValueError, "rate_b"):
            fisher_power(0.5, 1.9, 10, alpha=ALPHA)
        with self.assertRaisesRegex(ValueError, "trials"):
            fisher_power(0.5, 0.9, 0, alpha=ALPHA)
        with self.assertRaisesRegex(ValueError, "power"):
            trials_needed(0.5, 0.9, alpha=ALPHA, power=1.0)


if __name__ == "__main__":
    unittest.main()
