"""Pin the twin ↔ firmware ↔ tools constant mirrors to one truth.

The repo's recurring bug shape is two implementations of one fact, and
review 2026-08-24 found the yellow rig's numbers living in four places
(yellow.py, servo.rs, main.rs, tools/sim-errand.py) connected only by
manual arithmetic — with one drift already shipped (mime holds). These
tests regex-read the OTHER sides from source and assert equality, the
same pin-both-ends idea test_wire uses with a committed recording.

If a test here fails, the fix is never to edit the test: retune ONE
side deliberately and regenerate the other, then both match again.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import ClassVar

from rq_pipeline.tasks.yellow import (
    AIR_PICK_SEQUENCE,
    AIR_TUCK,
    FIRMWARE_SLEW_US_PER_S,
    PULSE_CEILING_US,
    PULSE_FLOOR_US,
    REAR_PICK_SEQUENCE,
    SALUTE_HALF_S,
    pose_to_pulses_us,
)

REPO = Path(__file__).resolve().parents[2]
SERVO_RS = (REPO / "firmware" / "pico-odom" / "src" / "servo.rs").read_text()
MAIN_RS = (REPO / "firmware" / "pico-odom" / "src" / "main.rs").read_text()
SIM_ERRAND = (REPO / "tools" / "sim-errand.py").read_text()


def rust_const(source: str, name: str) -> float:
    match = re.search(rf"const {name}: \w+ = ([\d.]+)", source)
    assert match, f"const {name} not found"
    return float(match.group(1))


class FetchPickTable(unittest.TestCase):
    """servo.rs FETCH_PICK is generated from AIR_PICK_SEQUENCE — prove it."""

    def parsed_table(self):
        block = re.search(
            r"const FETCH_PICK: \[\(\[i32; 5\], u64\); (\d+)\] = \[(.*?)\];",
            SERVO_RS,
            re.DOTALL,
        )
        assert block, "FETCH_PICK not found"
        rows = re.findall(r"\(\[([-\d, ]+)\], (\d+)\)", block.group(2))
        return int(block.group(1)), [
            ([int(v) for v in offsets.split(",")], int(hold)) for offsets, hold in rows
        ]

    def test_row_count_matches(self):
        count, rows = self.parsed_table()
        self.assertEqual(count, len(rows))
        self.assertEqual(count, len(AIR_PICK_SEQUENCE))

    def test_every_offset_and_hold_derives_from_the_twin(self):
        _, rows = self.parsed_table()
        for (pose, hold_s), (offsets, hold_ms) in zip(
            AIR_PICK_SEQUENCE, rows, strict=True
        ):
            expected = [p - 1500 for p in pose_to_pulses_us(pose)]
            self.assertEqual(offsets, expected, f"pose {pose}")
            self.assertEqual(hold_ms, round(hold_s * 1000), f"pose {pose}")

    def test_clamp_band_matches_firmware(self):
        self.assertEqual(rust_const(SERVO_RS, "FETCH_MIN_US"), PULSE_FLOOR_US)
        self.assertEqual(rust_const(SERVO_RS, "FETCH_MAX_US"), PULSE_CEILING_US)

    def test_slew_rate_matches_firmware(self):
        step = rust_const(SERVO_RS, "FETCH_STEP_US")
        # 3 µs per 40 ms tick = 75 µs/s
        self.assertEqual(step * 25, FIRMWARE_SLEW_US_PER_S)

    def test_salute_cadence_matches_firmware(self):
        # servo.rs waves via Timer::after_millis(350) per half-wave
        self.assertIn("Timer::after_millis(350)", SERVO_RS)
        self.assertEqual(SALUTE_HALF_S, 0.35)

    def test_arm_sign_covers_five_channels(self):
        match = re.search(r"const ARM_SIGN: \[i32; 5\] = \[([-\d, ]+)\]", SERVO_RS)
        assert match
        self.assertEqual(len(match.group(1).split(",")), 5)


class SimErrandMirror(unittest.TestCase):
    """tools/sim-errand.py hand-mirrors fetch_forever — hold it to that."""

    PAIRS: ClassVar = [
        ("CREEP_M_PER_S", "CREEP_M_PER_S"),
        ("TURN_RAD_PER_S", "TURN_RAD_PER_S"),
        ("CENTRE_DEADBAND", "CENTRE_DEADBAND"),
        ("AREA_ARRIVED", "AREA_ARRIVED"),
        ("ARRIVE_GLANCES", "ARRIVE_GLANCES"),
        ("ARC_M_PER_S", "ARC_M_PER_S"),
        ("ARC_RAD_PER_S", "ARC_RAD_PER_S"),
        ("ARC_X_MAX", "ARC_X_MAX"),
    ]
    MS_PAIRS: ClassVar = [
        ("TURN_BURST_MS", "TURN_BURST_S"),
        ("GLANCE_MS", "GLANCE_S"),
        ("SPIN_MS", "SPIN_S"),
        ("BACK_MS", "BACK_S"),
        ("CARRY_MS", "CARRY_S"),
    ]

    def python_const(self, name: str) -> float:
        match = re.search(rf"^{name} = ([\d.]+)", SIM_ERRAND, re.MULTILINE)
        assert match, f"{name} not found in sim-errand.py"
        return float(match.group(1))

    def test_direct_constants_match(self):
        for rust_name, py_name in self.PAIRS:
            self.assertEqual(
                rust_const(MAIN_RS, rust_name),
                self.python_const(py_name),
                rust_name,
            )

    def test_millisecond_constants_match(self):
        for rust_name, py_name in self.MS_PAIRS:
            self.assertAlmostEqual(
                rust_const(MAIN_RS, rust_name) / 1000.0,
                self.python_const(py_name),
                msg=rust_name,
            )


class AirMimeInvariants(unittest.TestCase):
    def test_waist_and_yaw_stay_home_in_every_air_pose(self):
        # ch1 is an EMPTY channel on the metal and ch0 is hand-aligned;
        # an AIR pose commanding either is a bug by construction.
        for pose, _ in AIR_PICK_SEQUENCE:
            self.assertEqual(pose[0], 0.0, pose)
            self.assertEqual(pose[1], 0.0, pose)

    def test_air_pulses_inside_the_servo_band_unclamped(self):
        for pose, _ in AIR_PICK_SEQUENCE:
            for angle in pose:
                raw = round(1500 + angle / 0.00165)
                self.assertTrue(PULSE_FLOOR_US <= raw <= PULSE_CEILING_US, pose)

    def test_rear_translation_is_clamped_not_overdriven(self):
        # REAR_REACH's waist (1.50 rad → 2409 µs raw) sits past the
        # SG90's stop; the translation must cap it at the firmware band.
        for pose, _ in REAR_PICK_SEQUENCE:
            for pulse in pose_to_pulses_us(list(pose)):
                self.assertTrue(PULSE_FLOOR_US <= pulse <= PULSE_CEILING_US)

    def test_tuck_matches_between_air_and_salute_base(self):
        self.assertEqual(AIR_PICK_SEQUENCE[0][0], AIR_TUCK)


if __name__ == "__main__":
    unittest.main()
