"""Smoothness columns: the batch and the streaming meter agree, and the
numbers mean what they say."""

from __future__ import annotations

import unittest

import numpy as np

from rq_pipeline.evaluate.smoothness import Smoothness, SmoothnessMeter


class TheNumbers(unittest.TestCase):
    def test_a_constant_stream_is_perfectly_smooth(self) -> None:
        self.assertEqual(Smoothness.of(np.ones((8, 3))), Smoothness(0.0, 0.0, 0.0))

    def test_a_ramp_has_velocity_only_in_the_actions_units_per_second(self) -> None:
        ramp = (0.5 * np.arange(10))[:, None]  # one actuator, +0.5 per tick
        s = Smoothness.of(ramp, dt=0.02)
        self.assertAlmostEqual(s.velocity, 25.0)  # 0.5 per tick / 0.02 s
        self.assertAlmostEqual(s.acceleration, 0.0)
        self.assertAlmostEqual(s.jerk, 0.0)
        self.assertEqual(
            set(s.columns()), {"rms_velocity", "rms_acceleration", "rms_jerk"}
        )

    def test_a_short_stream_reports_zero_not_nan(self) -> None:
        self.assertEqual(Smoothness.of(np.zeros((1, 2))), Smoothness(0.0, 0.0, 0.0))

    def test_only_two_dimensional_streams(self) -> None:
        with self.assertRaises(ValueError):
            Smoothness.of(np.zeros(5))


class TheMeter(unittest.TestCase):
    def test_streaming_equals_the_batch_for_every_world(self) -> None:
        rng = np.random.default_rng(3)
        worlds, ticks, nu = 2, 40, 4
        stream = rng.normal(size=(ticks, worlds, nu))
        meter = SmoothnessMeter(worlds, dt=0.02)
        for tick in range(ticks):
            meter.observe(stream[tick])
        for w in range(worlds):
            batch = Smoothness.of(stream[:, w, :], dt=0.02)
            got = meter.result(w)
            for k in ("velocity", "acceleration", "jerk"):
                self.assertAlmostEqual(getattr(got, k), getattr(batch, k), places=9)


if __name__ == "__main__":
    unittest.main()
