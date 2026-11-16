"""A tracking policy's evaluation (docs/86 §5.2): each trial's command
lag is drawn from the seed in whole control ticks within the task's bound,
and the conditions name the evaluation's files."""

from __future__ import annotations

import unittest

from trainnr_mjlab.tracking_verdict import CONTROL_HZ, lag_ticks, suffix_of


class TheLag(unittest.TestCase):
    def test_no_lag_is_zero_ticks_everywhere(self) -> None:
        self.assertEqual(lag_ticks(0.0, 40, 1000).tolist(), [0] * 40)

    def test_the_lag_stays_within_the_bound_and_covers_it(self) -> None:
        ticks = lag_ticks(60.0, 400, 1000)
        most = round(60.0 * CONTROL_HZ / 1000)
        self.assertEqual(ticks.min(), 0)
        self.assertEqual(ticks.max(), most)

    def test_the_same_seed_draws_the_same_lags(self) -> None:
        self.assertEqual(
            lag_ticks(40.0, 40, 7).tolist(), lag_ticks(40.0, 40, 7).tolist()
        )
        self.assertNotEqual(
            lag_ticks(40.0, 40, 7).tolist(), lag_ticks(40.0, 40, 8).tolist()
        )


class TheFiles(unittest.TestCase):
    def test_the_conditions_name_the_files(self) -> None:
        self.assertEqual(
            suffix_of("cuda:0", pushes=True, lag_ms=20), "tracking-push-lag20-cuda"
        )
        self.assertEqual(
            suffix_of("cpu", pushes=False, lag_ms=0), "tracking-still-lag0-cpu"
        )


if __name__ == "__main__":
    unittest.main()
