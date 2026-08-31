"""The servo-DR rule's one home: the both-terms gain scaling that a
second copy drifted from (review 2026-09-01), and the rig predicates."""

import unittest

from tests._extras import needs_sim


@needs_sim
class TheGainRule(unittest.TestCase):
    def _spec(self):
        import mujoco  # noqa: PLC0415

        return mujoco.MjSpec.from_string("""
        <mujoco>
          <worldbody>
            <body name="arm_link">
              <joint name="arm_j" type="hinge" damping="2.0"/>
              <geom size="0.05"/>
            </body>
            <body name="cube" pos="1 0 0">
              <freejoint name="cube_free"/>
              <geom size="0.05"/>
            </body>
          </worldbody>
          <actuator><position joint="arm_j" kp="10"/></actuator>
        </mujoco>
        """)

    def test_both_gain_terms_scale_together(self) -> None:
        # gainprm[0] alone moves the SETPOINT, not the stiffness — the
        # 2026-08-26 bug this rule exists to prevent.
        from rq_pipeline.physics.servo_dr import scale_servo_dynamics  # noqa: PLC0415

        spec = self._spec()
        gain0 = spec.actuators[0].gainprm[0]
        bias1 = spec.actuators[0].biasprm[1]
        scale_servo_dynamics(spec, damping_scale=1.0, gain_scale=2.0)
        self.assertAlmostEqual(spec.actuators[0].gainprm[0], gain0 * 2.0)
        self.assertAlmostEqual(spec.actuators[0].biasprm[1], bias1 * 2.0)

    def test_the_rigs_only_disagreement_is_which_joints(self) -> None:
        from rq_pipeline.physics.servo_dr import (  # noqa: PLC0415
            damped_joints,
            named_joints,
            scale_servo_dynamics,
        )

        spec = self._spec()
        scale_servo_dynamics(spec, damping_scale=3.0, gain_scale=1.0)
        self.assertAlmostEqual(spec.joints[0].damping[0], 6.0)  # damped joint
        # The name predicate skips what the damping one would take.
        spec2 = self._spec()
        scale_servo_dynamics(
            spec2,
            damping_scale=3.0,
            gain_scale=1.0,
            joints=named_joints(("nothing_matches",)),
        )
        self.assertAlmostEqual(spec2.joints[0].damping[0], 2.0)
        self.assertTrue(damped_joints(spec.joints[0]))


if __name__ == "__main__":
    unittest.main()
