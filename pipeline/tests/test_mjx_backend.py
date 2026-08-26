"""The MJX-Warp engine's acceptance gauntlet (docs/e2e-research/49).

A second engine earns entry by contract, not by trust: same census
through the same gate, the same rollout call shape, and a MEASURED
divergence bound against the reference instrument on the same compiled
model. The probe measured float32-scale divergence (2.7e-5 m on an
impact scene); the bound here is 1e-3 — loose enough to never flake,
tight enough that a conversion-layer bug (wrong state layout, dropped
actuator, integrator mismatch) fails by orders of magnitude.

Measured 2026-08-27 on the RTX 3090 Ti, the real scene: the kitting
bundle, 2 worlds x 20 steps from the neutral pose, diverges from CPU
MuJoCo by 9.6e-4 at step 9 (7 of 20 steps within 1e-6). The pendulum
bound holds; a contact scene reaches it in ten steps — the GPU path is
a different instrument, statistically not bitwise comparable (docs/49),
and any certificate from it says so in its stamp.
"""

import importlib.util
import unittest

MJX_PRESENT = importlib.util.find_spec("mujoco") is not None and (
    importlib.util.find_spec("mujoco.mjx") is not None
)


@unittest.skipUnless(MJX_PRESENT, "mjx extra not installed (uv sync --extra mjx)")
class MJXWarpGauntlet(unittest.TestCase):
    def _both(self):
        from rq_pipeline.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import MuJoCoBackend  # noqa: PLC0415
        from tests.test_mujoco_backend import PENDULUM  # noqa: PLC0415

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
        self.assertTrue(stamp.startswith(f"mjx-warp-{mujoco.__version__}+warp-"))
        self.assertEqual(warp.name, "mjx-warp")  # the name follows the impl

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

    def test_a_busy_scene_is_refused_without_explicit_sizing(self) -> None:
        """The kitting bundle dumped core on the GPU with MJX's default
        capacities (2026-08-27); the backend refuses before Warp runs."""
        from rq_pipeline.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415
        from rq_pipeline.tasks.aloha2 import build_kitting  # noqa: PLC0415

        warp = MJXWarpBackend()
        warp.load_spec(build_kitting().spec)
        self.assertGreater(warp.counts().geoms, warp.SIZING_REQUIRED_ABOVE_GEOMS)
        with self.assertRaises(ValueError):
            warp.rollout(warp.default_initial_state()[None, :], [[[0.0] * 14]])

    def test_the_gpu_engine_is_registered(self) -> None:
        from rq_pipeline.physics.mjx_backend import MJXWarpBackend  # noqa: PLC0415
        from rq_pipeline.physics.registry import GPU_ENGINE, resolve  # noqa: PLC0415

        self.assertIs(resolve(GPU_ENGINE).build, MJXWarpBackend)

    def test_stepper_is_batched_and_agrees_with_rollout_and_the_reference(
        self,
    ) -> None:
        """The fifth door: a batched stepping loop. Its rows must be the
        rollout's rows (same kernels, one extra forward per step) and
        within the gauntlet's bound of the CPU stepper, world by world."""
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.evaluate.harness import Stepper  # noqa: PLC0415
        from rq_pipeline.physics.mujoco_backend import (  # noqa: PLC0415
            Stepper as CpuStepper,
        )

        cpu, warp = self._both()
        rng = np.random.default_rng(11)
        nbatch, ticks, substeps = 3, 5, 4
        initial = np.tile(cpu.default_initial_state(), (nbatch, 1))
        initial[:, 1] += rng.uniform(-0.3, 0.3, nbatch)
        controls = rng.uniform(-0.4, 0.4, (nbatch, ticks, cpu.model.nu))
        stepper = warp.stepper(initial, ticks * substeps)
        self.assertIsInstance(stepper, Stepper)
        self.assertEqual(stepper.sensordata.shape, (nbatch, cpu.model.nsensordata))
        for tick in range(ticks):
            stepper.advance(controls[:, tick], substeps)
        self.assertTrue(stepper.done)
        self.assertEqual(
            stepper.states.shape, (nbatch, ticks * substeps, initial.shape[1])
        )
        self.assertEqual(stepper.sensors.shape[:2], (nbatch, ticks * substeps))
        rollout = warp.rollout(initial, np.repeat(controls, substeps, axis=1))
        gap = float(np.max(np.abs(stepper.states - rollout)))
        self.assertLess(gap, 1e-4, f"stepper vs rollout {gap}")
        for world in range(nbatch):
            reference = CpuStepper(cpu.model, initial[world], ticks * substeps)
            for tick in range(ticks):
                reference.advance(controls[world, tick], substeps)
            gap = float(np.max(np.abs(stepper.states[world] - reference.states)))
            self.assertLess(gap, 1e-3, f"world {world} vs the reference {gap}")
        stepper.advance(controls[:, 0], substeps)  # past the budget: a no-op
        self.assertEqual(stepper.step, ticks * substeps)

    def test_shape_refusals_match_the_reference(self) -> None:
        import numpy as np  # noqa: PLC0415

        _, warp = self._both()
        with self.assertRaises(ValueError):
            warp.rollout(np.zeros(3), np.zeros((1, 2, 1)))
        with self.assertRaises(ValueError):
            warp.rollout(np.zeros((2, 3)), np.zeros((1, 2, 1)))


if __name__ == "__main__":
    unittest.main()
