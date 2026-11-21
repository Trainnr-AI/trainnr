"""A fitted servo's firmware (#101): a family BAM measured comes from the
table; one trainnr's `servo-bench` fitted carries its own constants in the
bundle's context; one with neither is refused by name."""

from __future__ import annotations

import unittest

from trainnr_mjlab.firmware import FIRMWARE, firmware_of


class TheFirmware(unittest.TestCase):
    def test_the_table_then_the_bundles_own_then_a_refusal(self) -> None:
        self.assertIs(firmware_of({"params": {"actuator": "xl330"}}), FIRMWARE["xl330"])
        fitted = firmware_of(
            {
                "params": {"actuator": "scs0009"},
                "context": {"firmware": {"vin": 6.0, "kp": 32, "error_gain": 1.25}},
            }
        )
        self.assertEqual((fitted.vin, fitted.kp, fitted.error_gain), (6.0, 32.0, 1.25))
        self.assertEqual(fitted.max_pwm, 1.0)
        with self.assertRaisesRegex(KeyError, "no firmware constants"):
            firmware_of({"params": {"actuator": "mystery"}})


if __name__ == "__main__":
    unittest.main()
