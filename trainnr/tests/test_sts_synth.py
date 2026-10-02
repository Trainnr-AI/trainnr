"""The synthetic STS3215 study's regression anchors.

The full 32-cell matrix lives in data/sts3215-synthetic-identifiability
.json and docs/26; these tests pin the two load-bearing cells so the
findings cannot silently rot: clean recovery must stay exact (the
convention-identity guarantee), and the recommended protocol
(position + load, velocity NOT fitted) must keep working at the bus
rate the TTL reality allows.
"""

import unittest

from tests._extras import needs_sim


@needs_sim
class SyntheticSts(unittest.TestCase):
    def test_clean_condition_recovers_truth_exactly(self) -> None:
        """All 12 clean cells recovered truth to machine precision; pin one.

        This is the convention-identity guarantee: truth and fit share
        one rollout and one grid. Three sampling-convention bugs each
        produced a biased fit with tight intervals before this held —
        the failure class Paper 0 named (R13), now fenced.
        """
        from trainnr.robot.sts_synth import (  # noqa: PLC0415
            PARAMETERS,
            Condition,
            run_condition,
        )

        result = run_condition(
            Condition(50, ("pos", "vel", "load"), False), seconds=3.0
        )
        truth = {spec.name: spec.nominal for spec in PARAMETERS}
        for parameter in result.parameters:
            self.assertTrue(parameter.pinned, parameter.name)
            self.assertAlmostEqual(
                parameter.estimate, truth[parameter.name], places=5, msg=parameter.name
            )

    def test_pos_plus_load_survives_the_servo_at_bus_rate(self) -> None:
        """The recommended protocol: position + load, 25 Hz, servo-real.

        The study's headline — the torque-side register anchors the fit
        through quantization and the dead zone even at the slowest
        plausible bus rate, while the derived-velocity register poisons
        it (worst cells -53%..-88%, confidently reported pinned).
        """
        from trainnr.robot.sts_synth import (  # noqa: PLC0415
            PARAMETERS,
            Condition,
            run_condition,
        )

        result = run_condition(Condition(25, ("pos", "load"), True), seconds=6.0)
        truth = {spec.name: spec.nominal for spec in PARAMETERS}
        for parameter in result.parameters:
            error = abs(parameter.estimate - truth[parameter.name])
            self.assertLess(
                error / truth[parameter.name], 0.10, f"{parameter.name}: {error:.4g}"
            )


if __name__ == "__main__":
    unittest.main()
