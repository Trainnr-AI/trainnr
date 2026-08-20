"""The assembly menu: car, arm, mobile manipulator — composed and driven."""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None

CROUCH = [0.0, -1.9, 1.9, 1.3, 0.0, 0.3]
HOME = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class AssemblyMenu(unittest.TestCase):
    def test_every_combination_composes_with_the_expected_census(self) -> None:
        from rq_pipeline.tasks.components import compose  # noqa: PLC0415

        expected = {
            (True, False): (2, 2),  # car: two motors, two encoders
            (False, True): (6, 12),  # arm: six position actuators, 12 sensors
            (True, True): (8, 14),  # mobile manipulator: both, car first
        }
        for (car, arm), (nu, nsensor) in expected.items():
            model = compose(car=car, arm=arm).compile()
            self.assertEqual((model.nu, model.nsensor), (nu, nsensor), (car, arm))

    def test_empty_assembly_refused(self) -> None:
        from rq_pipeline.tasks.components import compose  # noqa: PLC0415

        with self.assertRaises(ValueError):
            compose(car=False, arm=False)


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class MobileManipulatorDrives(unittest.TestCase):
    def _settled(self, arm_target):
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.tasks.components import compose  # noqa: PLC0415

        model = compose(car=True, arm=True).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        initial = data.qpos[6:12].copy()
        # Gentle interpolated ramp: a step command at kp=50 whacks the
        # chassis hard enough to matter (measured during design).
        for step in range(1200):
            alpha = min(1.0, step / 800)
            if step % 10 == 0:
                blend = (1 - alpha) * initial + alpha * np.asarray(arm_target)
                data.ctrl[:] = [0.0, 0.0, *blend]
            mujoco.mj_step(model, data)
        return model, data

    def test_drives_forward_and_stays_flat(self) -> None:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        model, data = self._settled(CROUCH)
        start = data.qpos[:2].copy()
        for step in range(2500):
            if step % 10 == 0:
                data.ctrl[:2] = [0.8, 0.8]
            mujoco.mj_step(model, data)
        displacement = float(np.linalg.norm(data.qpos[:2] - start))
        self.assertGreater(displacement, 0.5)
        # Pitch stays negligible: the tipping failures of the first three
        # design probes must not return.
        w, x, y, z = data.qpos[3:7]
        self.assertLess(abs(2 * (w * y - z * x)), 0.05)

    def test_arm_fully_extended_does_not_tip_the_car(self) -> None:
        _model, data = self._settled(HOME)
        self.assertAlmostEqual(float(data.qpos[2]), 0.052, delta=0.005)
        w, x, y, z = data.qpos[3:7]
        self.assertLess(abs(2 * (w * y - z * x)), 0.05)


if __name__ == "__main__":
    unittest.main()
