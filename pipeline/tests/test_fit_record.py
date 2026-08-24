"""Fit records: measurements become bundle artifacts, repeat runs compared."""

import tempfile
import unittest
from pathlib import Path

from rq_pipeline.robot.fit_record import (
    cross_run_spread,
    load_fit_records,
    spread_summary,
    write_fit_record,
)
from rq_pipeline.robot.identify import IdentificationResult, IdentifiedParameter

ANCHOR = "armature=0.0004 anchored from reflected rotor inertia (bundle README)"
UNITS = {
    "left_gear": "N*m per duty percent (motor gear)",
    "left_damping": "N*m*s/rad (joint damping)",
}


def _result(gear: float, damping: float) -> IdentificationResult:
    return IdentificationResult(
        parameters=(
            IdentifiedParameter(
                name="left_gear",
                estimate=gear,
                half_width=1e-6,
                allowed_range=0.00098,
                pinned=True,
            ),
            IdentifiedParameter(
                name="left_damping",
                estimate=damping,
                half_width=2e-5,
                allowed_range=0.00995,
                pinned=True,
            ),
        ),
        confidence=0.95,
    )


class WriteAndReload(unittest.TestCase):
    def test_two_runs_roundtrip_and_spread_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            write_fit_record(
                bundle,
                _result(0.00023, 0.00090),
                robot="rig-drivetrain",
                recording="sweep-a@aaaaaaaaaaaa",
                anchor=ANCHOR,
                units=UNITS,
            )
            write_fit_record(
                bundle,
                _result(0.00024, 0.00094),
                robot="rig-drivetrain",
                recording="sweep-b@bbbbbbbbbbbb",
                anchor=ANCHOR,
                units=UNITS,
            )
            records = load_fit_records(bundle)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].parameters[0].name, "left_gear")
        spread = cross_run_spread(records)
        self.assertEqual(spread["left_gear"], (0.00023, 0.00024))
        # The rehearsal's second finding made this line the point: the
        # cross-run damping spread (4e-5) dwarfs the per-run half-width
        # (2e-5), so the summary must side with the spread.
        self.assertIn("trust the spread", spread_summary(records))

    def test_refit_of_same_recording_overwrites_not_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for damping in (0.0009, 0.0011):
                write_fit_record(
                    bundle,
                    _result(0.00023, damping),
                    robot="rig-drivetrain",
                    recording="sweep-a@aaaaaaaaaaaa",
                    anchor=ANCHOR,
                    units=UNITS,
                )
            records = load_fit_records(bundle)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].parameters[1].estimate, 0.0011)

    def test_empty_bundle_has_no_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_fit_records(Path(directory)), ())


class Refusals(unittest.TestCase):
    def test_unstamped_recording_refused(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ValueError),
        ):
            write_fit_record(
                Path(directory),
                _result(0.00023, 0.0009),
                robot="rig-drivetrain",
                recording="sweep-a",
                anchor=ANCHOR,
                units=UNITS,
            )

    def test_path_separator_in_recording_refused(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ValueError),
        ):
            write_fit_record(
                Path(directory),
                _result(0.00023, 0.0009),
                robot="rig-drivetrain",
                recording="../escape@aaaaaaaaaaaa",
                anchor=ANCHOR,
                units=UNITS,
            )

    def test_missing_anchor_refused(self) -> None:
        # The torque-scale finding as an artifact rule: no anchor
        # statement, no record.
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ValueError),
        ):
            write_fit_record(
                Path(directory),
                _result(0.00023, 0.0009),
                robot="rig-drivetrain",
                recording="sweep-a@aaaaaaaaaaaa",
                anchor="   ",
                units=UNITS,
            )


if __name__ == "__main__":
    unittest.main()
