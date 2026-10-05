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

import os
import re
import unittest
from pathlib import Path
from typing import ClassVar

from trainnr.tasks.yellow import (
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
# The firmware moved with the rig to a private archive on 2026-10-02. The
# firmware halves of these mirrors run when a checkout of it is named;
# without one they skip by name, so the Python-only halves still hold on
# every machine.
RIG = Path(os.environ.get("TRAINNR_RIG_DIR", REPO))
_PICO_ODOM = RIG / "firmware" / "pico-odom" / "src"
HAVE_FIRMWARE = (_PICO_ODOM / "servo.rs").exists()
needs_firmware = unittest.skipUnless(
    HAVE_FIRMWARE,
    "the rig firmware is not checked out here "
    "(TRAINNR_RIG_DIR=<a checkout of the rig's archive>)",
)
SERVO_RS = (
    (_PICO_ODOM / "servo.rs").read_text(encoding="utf-8") if HAVE_FIRMWARE else ""
)
MAIN_RS = (_PICO_ODOM / "main.rs").read_text(encoding="utf-8") if HAVE_FIRMWARE else ""
SIM_ERRAND = (REPO / "tools" / "sim-errand.py").read_text(encoding="utf-8")
BRIDGE = (REPO / "tools" / "udp-wire-bridge.py").read_text(encoding="utf-8")


def rust_fn(source: str, name: str) -> str:
    """The body of one fn — scoping matters: CREEP/TURN/DEADBAND exist
    in BOTH fetch_forever and chase_forever with different values, and
    a whole-file first-match pin held only because of declaration order
    (found in review 2026-08-26)."""
    match = re.search(rf"fn {name}[^{{]*\{{", source)
    assert match, f"fn {name} not found"
    depth, index = 1, match.end()
    while depth and index < len(source):
        depth += {"{": 1, "}": -1}.get(source[index], 0)
        index += 1
    return source[match.start() : index]


def rust_const(source: str, name: str) -> float:
    # Underscored literals (10_000) parse; [\d.] alone read 10_000 as 10.
    match = re.search(rf"const {name}: \w+ = ([\d_.]+)", source)
    assert match, f"const {name} not found"
    return float(match.group(1).replace("_", ""))


FETCH_FOREVER = rust_fn(MAIN_RS, "fetch_forever") if HAVE_FIRMWARE else ""
FETCH_ARM = rust_fn(SERVO_RS, "fetch_arm") if HAVE_FIRMWARE else ""


@needs_firmware
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
        # Scoped to fetch_arm: a whole-file assertIn was satisfiable by
        # any other 350 ms timer anywhere in the servo module.
        self.assertIn("Timer::after_millis(350)", FETCH_ARM)
        self.assertEqual(SALUTE_HALF_S, 0.35)

    def test_arm_sign_is_the_measured_vector(self):
        # The dance measured [+1, +1, -1, -1, +1] (2026-08-24); pinning
        # only the length pinned the shape and not the fact.
        match = re.search(r"const ARM_SIGN: \[i32; 5\] = \[([-\d, ]+)\]", SERVO_RS)
        assert match
        signs = [int(value) for value in match.group(1).split(",")]
        self.assertEqual(signs, [1, 1, -1, -1, 1])


@needs_firmware
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
                rust_const(FETCH_FOREVER, rust_name),
                self.python_const(py_name),
                rust_name,
            )

    def test_millisecond_constants_match(self):
        for rust_name, py_name in self.MS_PAIRS:
            self.assertAlmostEqual(
                rust_const(FETCH_FOREVER, rust_name) / 1000.0,
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


@needs_firmware
class WireClockAndPort(unittest.TestCase):
    """The wire's clock and the radio's port: mirrors found unpinned in
    the 2026-08-26 review. STATUS_HZ scales every timestamp the
    identification consumes — a REPORT_MS retune would silently rescale
    the flagship fits."""

    def test_status_hz_matches_report_ms(self):
        from trainnr.collect.frames import STATUS_HZ  # noqa: PLC0415

        report_ms = rust_const(MAIN_RS, "REPORT_MS")
        self.assertEqual(STATUS_HZ, 1000.0 / report_ms)

    def test_bridge_port_matches_firmware(self):
        rust_port = rust_const(MAIN_RS, "TELEMETRY_PORT")
        match = re.search(r"^TELEMETRY_PORT = (\d+)", BRIDGE, re.MULTILINE)
        assert match, "bridge port constant not found"
        self.assertEqual(float(match.group(1)), rust_port)

    def test_autonomy_clock_is_single_sourced(self):
        # One module-level definition each; the grace was once
        # hand-copied into two functions. sim-errand deliberately does
        # NOT mirror these (its prediction covers one errand, not the
        # rest between laps) — this pin guards the firmware side only.
        self.assertEqual(MAIN_RS.count("const AUTOSTART_GRACE_MS"), 1)
        self.assertEqual(MAIN_RS.count("const LAP_REST_SECS"), 1)
        self.assertEqual(rust_const(MAIN_RS, "AUTOSTART_GRACE_MS"), 10_000)
