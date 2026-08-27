"""The real rig's twin: measured geometry, exact servos, no tipping."""

import unittest

from tests._extras import needs_sim


@needs_sim
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


@needs_sim
class RearPick(unittest.TestCase):
    def _lift(self, dx, dy, close=True):
        import mujoco  # noqa: PLC0415

        from rq_pipeline.tasks.yellow import (  # noqa: PLC0415
            REAR_GRASP_POINT,
            REAR_PICK_SEQUENCE,
            compose_rig,
        )

        scene = compose_rig(car=True)
        cube = scene.worldbody.add_body(
            name="prop",
            pos=[REAR_GRASP_POINT[0] + dx, REAR_GRASP_POINT[1] + dy, 0.0125],
        )
        cube.add_freejoint()
        cube.add_geom(
            name="prop_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.0125, 0.0125, 0.0125],
            mass=0.015,
            friction=[2.0, 0.02, 0.001],
        )
        model = scene.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cq = model.jnt_qposadr[model.body("prop").jntadr[0]]
        for pose, seconds in REAR_PICK_SEQUENCE:
            ctrl = list(pose)
            if not close:
                ctrl[4] = 0.5
            for step in range(int(seconds * 500)):
                if step % 10 == 0:
                    data.ctrl[:] = [0.0, 0.0, *ctrl]
                mujoco.mj_step(model, data)
        return float(data.qpos[cq + 2])

    def test_picks_at_centre_and_basin_corner(self) -> None:
        self.assertGreater(self._lift(0.0, 0.0), 0.05)
        self.assertGreater(self._lift(0.008, 0.008), 0.05)

    def test_no_close_no_lift(self) -> None:
        self.assertLess(self._lift(0.0, 0.0, close=False), 0.05)


if __name__ == "__main__":
    unittest.main()
