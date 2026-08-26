"""The MJX-Warp engine's acceptance gauntlet (docs/e2e-research/49).

A second engine earns entry by contract, not by trust: same census
through the same gate, the same rollout call shape, and a MEASURED
divergence bound against the reference instrument on the same compiled
model. The probe measured float32-scale divergence (2.7e-5 m on an
impact scene); the bound here is 1e-3 — loose enough to never flake,
tight enough that a conversion-layer bug (wrong state layout, dropped
actuator, integrator mismatch) fails by orders of magnitude.
"""

import importlib.util
import unittest

MJX_PRESENT = importlib.util.find_spec("mujoco") is not None and (
    importlib.util.find_spec("mujoco.mjx") is not None
)


@unittest.skipUnless(MJX_PRESENT, "mjx extra not installed (uv sync --extra mjx)")
class MJXWarpGauntlet(unittest.TestCase):
    def _both(self):
        from test_mujoco_backend import PENDULUM  # noqa: PLC0415

        from rq_pipeline.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415

        cpu = MuJoCoBackend()
        cpu.load_mjcf_string(PENDULUM)
        warp = MJXWarpBackend()
        warp.load_model(cpu.model)  # one compile, two instruments
        return cpu, warp

    def test_census_parity_through_the_same_gate(self) -> None:
        cpu, warp = self._both()
        a, b = cpu.counts(), warp.counts()
        self.assertEqual(
            (a.actuators, a.sensors, a.geoms, a.cameras),
            (b.actuators, b.sensors, b.geoms, b.cameras),
        )

    def test_instrument_names_engine_versions_and_device(self) -> None:
        import mujoco  # noqa: PLC0415

        _, warp = self._both()
        stamp = warp.instrument
        self.assertIn("mjx-warp", stamp)
        self.assertIn(mujoco.__version__, stamp)
        self.assertIn("warp-", stamp)

    def test_divergence_from_the_reference_instrument_is_bounded(self) -> None:
        import numpy as np  # noqa: PLC0415

        cpu, warp = self._both()
        rng = np.random.default_rng(7)
        steps, nbatch = 50, 3
        initial = np.tile(cpu.default_initial_state(), (nbatch, 1))
        initial[:, 1] += rng.uniform(-0.3, 0.3, nbatch)  # vary the hinge
        controls = np.repeat(
            rng.uniform(-0.4, 0.4, (nbatch, 1, cpu.model.nu)), steps, axis=1
        )
        reference = cpu.rollout(initial, controls)
        candidate = warp.rollout(initial, controls)
        self.assertEqual(candidate.shape, reference.shape)
        gap = float(np.max(np.abs(candidate - reference)))
        # float32 vs float64 on a smooth scene: measured ~1e-5 scale.
        # A layout or conversion bug is orders of magnitude, not this.
        self.assertLess(gap, 1e-3, f"divergence {gap} exceeds the bound")

    def test_shape_refusals_match_the_reference(self) -> None:
        import numpy as np  # noqa: PLC0415

        _, warp = self._both()
        with self.assertRaises(ValueError):
            warp.rollout(np.zeros(3), np.zeros((1, 2, 1)))
        with self.assertRaises(ValueError):
            warp.rollout(np.zeros((2, 3)), np.zeros((1, 2, 1)))


if __name__ == "__main__":
    unittest.main()
