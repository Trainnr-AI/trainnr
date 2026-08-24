"""The real rig's twin: measured geometry, exact servos, no tipping."""

import importlib.util
import unittest

MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(MUJOCO_PRESENT, "sim extra not installed (uv sync --extra sim)")
class YellowRigTwin(unittest.TestCase):
    def test_census_and_servo_fidelity(self) -> None:
        import mujoco  # noqa: PLC0415

        from rq_pipeline.tasks.yellow import compose_rig  # noqa: PLC0415

        model = compose_rig(car=True).compile()
        self.assertEqual(model.nu, 7)  # 2 wheels + 5 servos
        self.assertEqual(model.nsensor, 8)  # 2 encoders + 6 joint sensors
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)
        # Every servo reaches its command (the degrees-vs-radians bug
        # regression: with ranges misread as degrees, all joints pinned
        # at ±1.4° and achieved ~15% of command).
        for j in range(5):
            data.ctrl[:] = [0.0] * model.nu
            data.ctrl[2 + j] = 0.23
            for _ in range(600):
                mujoco.mj_step(model, data)
            self.assertAlmostEqual(
                float(data.sensordata[2 + j]), 0.23, delta=0.02, msg=f"servo {j}"
            )
            data.ctrl[:] = [0.0] * model.nu
            for _ in range(400):
                mujoco.mj_step(model, data)

    def test_rearward_reach_does_not_tip(self) -> None:
        import math  # noqa: PLC0415

        import mujoco  # noqa: PLC0415

        from rq_pipeline.tasks.yellow import compose_rig  # noqa: PLC0415

        model = compose_rig(car=True).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.ctrl[3] = 1.5  # waist to horizontal, worst lever
        for _ in range(1500):
            mujoco.mj_step(model, data)
        w, x, y, z = data.qpos[3:7]
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
        self.assertLess(abs(pitch), 2.0)


if __name__ == "__main__":
    unittest.main()
