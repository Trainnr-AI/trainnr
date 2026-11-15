"""A deployment of a robot on position servos says so (issue #107): the
servo's joints carry no PD gains (a simulated `<motor>`'s 1.0 and -0.0
mean nothing on the hardware), and `servos` names the actuator bundle and
the operating point its law assumes, which the robot's servos must match."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from trainnr.deploy.manifest import SERVO_LAW_BAM

from trainnr_mjlab import BamActuator, BamActuatorCfg
from trainnr_mjlab.walk_export import _servos

REPO = Path(__file__).resolve().parents[2]
ACTUATOR_BUNDLE = REPO / "robots" / "actuator-bundles" / "xl330.m6.bundle.json"


class ServoGroups(unittest.TestCase):
    def test_a_bam_group_names_its_bundle_joints_and_operating_point(self) -> None:
        cfg = BamActuatorCfg.from_bundle(
            ACTUATOR_BUNDLE, target_names_expr=(".*",), physics_dt=0.005, vin=7.0
        )
        servo = BamActuator.__new__(BamActuator)  # the live group, not compiled
        servo.cfg = cfg
        servo._target_names = ["robot/left_knee", "robot/right_knee"]
        pd = SimpleNamespace(cfg=None, target_names=["robot/neck"])
        (group,) = _servos(SimpleNamespace(actuators=[servo, pd]))
        self.assertEqual(group["law"], SERVO_LAW_BAM)
        self.assertEqual(group["actuator"], cfg.stamp)
        self.assertEqual(group["joints"], ["left_knee", "right_knee"])
        self.assertEqual(group["vin_v"], 7.0)
        self.assertEqual(group["kp"], cfg.law.kp)


if __name__ == "__main__":
    unittest.main()
