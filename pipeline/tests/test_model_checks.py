"""The fail-loudly gate for silently-dead imports."""

import unittest

from rq_pipeline.robot.model_checks import DeadModelError, assert_model_alive


class AssertModelAlive(unittest.TestCase):
    def test_healthy_robot_passes(self) -> None:
        assert_model_alive(actuators=6, sensors=4, geoms=30, source="so101.xml")

    def test_zero_actuators_refused(self) -> None:
        # The USD-import failure shape: plausible file, inert robot.
        with self.assertRaises(DeadModelError) as caught:
            assert_model_alive(actuators=0, sensors=4, geoms=30, source="robot.usdz")
        self.assertIn("0 actuators", str(caught.exception))
        self.assertIn("robot.usdz", str(caught.exception))

    def test_scene_bundle_may_skip_sensors_but_not_geometry(self) -> None:
        assert_model_alive(
            actuators=1,
            sensors=0,
            geoms=200,
            source="scene.usdz",
            expect_sensors=False,
        )
        with self.assertRaises(DeadModelError):
            assert_model_alive(
                actuators=1,
                sensors=0,
                geoms=0,
                source="scene.usdz",
                expect_sensors=False,
            )


if __name__ == "__main__":
    unittest.main()
