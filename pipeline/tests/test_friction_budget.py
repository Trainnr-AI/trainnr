"""The M1-M6 torque budget, checked against hand-computed values —
independent of the implementation, straight off docs/e2e-research/53's
equations (BAM/ICRA 2025's theory), not against the code that computes
them.
"""

import math
import unittest


class TorqueBudget(unittest.TestCase):
    def test_m1_is_coulomb_viscous(self) -> None:
        from rq_pipeline.robot.friction_budget import (  # noqa: PLC0415
            FrictionParams,
            friction_torque_budget,
        )

        params = FrictionParams(friction_viscous=0.05, friction_base=0.1)
        # tau_fm = Kv|theta_dot| + Kc — tau_m/tau_e must be IGNORED entirely.
        got = friction_torque_budget(params, theta_dot=2.0, tau_m=99.0, tau_e=-99.0)
        self.assertAlmostEqual(got, 0.05 * 2.0 + 0.1)

    def test_m2_adds_stribeck_envelope(self) -> None:
        from rq_pipeline.robot.friction_budget import (  # noqa: PLC0415
            FrictionParams,
            friction_torque_budget,
        )

        params = FrictionParams(
            friction_viscous=0.05,
            friction_base=0.1,
            friction_stribeck=0.2,
            dtheta_stribeck=1.5,
            alpha=1.2,
        )
        theta_dot = 0.7
        envelope = math.exp(-(abs(theta_dot / 1.5) ** 1.2))
        expected = 0.05 * abs(theta_dot) + 0.1 + envelope * 0.2
        got = friction_torque_budget(params, theta_dot=theta_dot, tau_m=0.0, tau_e=0.0)
        self.assertAlmostEqual(got, expected)

    def test_m3_undirected_load_term(self) -> None:
        from rq_pipeline.robot.friction_budget import (  # noqa: PLC0415
            FrictionParams,
            friction_torque_budget,
        )

        params = FrictionParams(
            friction_viscous=0.05, friction_base=0.1, load_friction_base=0.3
        )
        # tau_fm = Kv|theta_dot| + Kc + Kl|tau_m - tau_e|
        expected = 0.05 * 1.0 + 0.1 + 0.3 * abs(4.0 - (-1.0))
        got = friction_torque_budget(params, theta_dot=1.0, tau_m=4.0, tau_e=-1.0)
        self.assertAlmostEqual(got, expected)

    def test_m5_directional_load_replaces_undirected(self) -> None:
        from rq_pipeline.robot.friction_budget import (  # noqa: PLC0415
            FrictionParams,
            friction_torque_budget,
        )

        params = FrictionParams(
            friction_viscous=0.0,
            friction_base=0.0,
            friction_stribeck=0.1,
            dtheta_stribeck=1.0,
            alpha=1.0,
            load_friction_motor=2.0,
            load_friction_external=0.5,
            load_friction_motor_stribeck=0.2,
            load_friction_external_stribeck=0.1,
        )
        theta_dot, tau_m, tau_e = 0.3, 3.0, -2.0
        envelope = math.exp(-(abs(theta_dot / 1.0) ** 1.0))
        base = abs(2.0 * tau_m - 0.5 * tau_e)
        stribeck = 0.1 + abs(0.2 * tau_m - 0.1 * tau_e)
        expected = base + envelope * stribeck
        got = friction_torque_budget(params, theta_dot, tau_m, tau_e)
        self.assertAlmostEqual(got, expected)

    def test_m6_quadratic_term_picks_the_smaller_torque_side(self) -> None:
        from rq_pipeline.robot.friction_budget import (  # noqa: PLC0415
            FrictionParams,
            friction_torque_budget,
        )

        params = FrictionParams(
            friction_viscous=0.0,
            friction_base=0.0,
            friction_stribeck=0.0,
            dtheta_stribeck=1.0,
            alpha=1.0,
            load_friction_motor=0.0,
            load_friction_external=0.0,
            load_friction_motor_stribeck=0.0,
            load_friction_external_stribeck=0.0,
            load_friction_motor_quad=7.0,
            load_friction_external_quad=3.0,
        )
        # BAM's reference (bam/model.py, read verbatim 2026-09-01): the
        # quadratic pays ONLY when the torques OPPOSE in sign. This test
        # once used same-sign torques and pinned the ungated variant —
        # the drift the kernel-parity review caught.
        # opposing, |tau_m| > |tau_e| -> Q = Keq * tau_e^2.
        got_motor_dominant = friction_torque_budget(
            params, theta_dot=0.0, tau_m=10.0, tau_e=-1.0
        )
        self.assertAlmostEqual(got_motor_dominant, 3.0 * 1.0**2)
        # opposing, |tau_e| > |tau_m| -> Q = Kmq * tau_m^2.
        got_external_dominant = friction_torque_budget(
            params, theta_dot=0.0, tau_m=-1.0, tau_e=10.0
        )
        self.assertAlmostEqual(got_external_dominant, 7.0 * 1.0**2)
        # SAME sign -> the gate zeroes the whole quadratic.
        self.assertAlmostEqual(
            friction_torque_budget(params, theta_dot=0.0, tau_m=10.0, tau_e=1.0), 0.0
        )
        # The tie pays nothing (neither strict inequality holds).
        self.assertAlmostEqual(
            friction_torque_budget(params, theta_dot=0.0, tau_m=5.0, tau_e=-5.0), 0.0
        )

    def test_absent_terms_are_zero_not_a_silent_guess(self) -> None:
        from rq_pipeline.robot.friction_budget import (  # noqa: PLC0415
            FrictionParams,
        )

        params = FrictionParams(friction_viscous=0.05, friction_base=0.1)
        self.assertIsNone(params.friction_stribeck)
        self.assertIsNone(params.load_friction_motor)
        self.assertIsNone(params.load_friction_motor_quad)

    def test_from_json_ignores_servo_model_fields(self) -> None:
        from rq_pipeline.robot.friction_budget import FrictionParams  # noqa: PLC0415

        raw = {
            "kt": 1.2,
            "R": 2.7,
            "armature": 0.02,
            "model": "m1",
            "actuator": "sts3215",
            "friction_viscous": 0.03,
            "friction_base": 0.05,
        }
        params = FrictionParams.from_json(raw)
        self.assertEqual(params.friction_viscous, 0.03)
        self.assertEqual(params.friction_base, 0.05)


if __name__ == "__main__":
    unittest.main()
