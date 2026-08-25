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


class SpreadRecord(unittest.TestCase):
    def _bundle_with_two_runs(self, directory: str) -> Path:
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
        return bundle

    def test_spread_json_written_and_not_reloaded_as_a_fit(self) -> None:
        import json  # noqa: PLC0415

        from rq_pipeline.robot.fit_record import write_spread_record  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            bundle = self._bundle_with_two_runs(directory)
            path = write_spread_record(bundle)
            payload = json.loads(path.read_text())
            self.assertEqual(len(payload["summarizes"]), 2)
            self.assertIn("left_damping", payload["spread"])
            # SPREAD.json must never come back as a fit record.
            self.assertEqual(len(load_fit_records(bundle)), 2)

    def test_spread_refuses_a_single_run(self) -> None:
        from rq_pipeline.robot.fit_record import write_spread_record  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            write_fit_record(
                bundle,
                _result(0.00023, 0.0009),
                robot="rig-drivetrain",
                recording="sweep-a@aaaaaaaaaaaa",
                anchor=ANCHOR,
                units=UNITS,
            )
            with self.assertRaises(ValueError):
                write_spread_record(bundle)

    def test_unbounded_half_width_roundtrips_and_verdict_survives(self) -> None:
        """The feature that broke strict JSON once, and the divergence
        the review caught: an unbounded interval must not make the
        summary print "runs agree" for the unpinned parameter."""
        from rq_pipeline.robot.fit_record import spread_summary  # noqa: PLC0415

        unbounded = IdentificationResult(
            parameters=(
                IdentifiedParameter(
                    name="scale_ref",
                    estimate=0.001,
                    half_width=float("inf"),
                    allowed_range=2e-6,
                    pinned=False,
                ),
            ),
            confidence=0.95,
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for stamp_name, estimate in (
                ("run-a@aaaaaaaaaaaa", 0.001),
                ("run-b@bbbbbbbbbbbb", 0.002),
            ):
                record = IdentificationResult(
                    parameters=(
                        IdentifiedParameter(
                            name="scale_ref",
                            estimate=estimate,
                            half_width=float("inf"),
                            allowed_range=2e-6,
                            pinned=False,
                        ),
                    ),
                    confidence=0.95,
                )
                write_fit_record(
                    bundle,
                    record,
                    robot="r",
                    recording=stamp_name,
                    anchor=ANCHOR,
                    units={"scale_ref": "anchored"},
                )
            records = load_fit_records(bundle)
            self.assertEqual(records[0].parameters[0].half_width, float("inf"))
            summary = spread_summary(records)
            self.assertIn("unbounded", summary)
            self.assertNotIn("EXCEEDS", summary)
        del unbounded

    def test_legacy_record_without_new_fields_loads_with_none(self) -> None:
        import json  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            fits = Path(directory) / "fits"
            fits.mkdir()
            legacy = {
                "robot": "old",
                "recording": "sweep-x@cccccccccccc",
                "anchor": ANCHOR,
                "confidence": 0.95,
                "parameters": [
                    {
                        "name": "gain",
                        "estimate": 1.0,
                        "half_width": 0.1,
                        "allowed_range": 2.0,
                        "pinned": True,
                    }
                ],
                "created_utc": "2026-08-23T00:00:00+00:00",
            }
            (fits / "sweep-x@cccccccccccc.json").write_text(json.dumps(legacy))
            records = load_fit_records(Path(directory))
            self.assertIsNone(records[0].profile)
            self.assertIsNone(records[0].units)

    def test_unknown_field_names_the_file(self) -> None:
        import json  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            fits = Path(directory) / "fits"
            fits.mkdir()
            (fits / "weird@dddddddddddd.json").write_text(
                json.dumps({"robot": "x", "parameters": [], "from_the_future": 1})
            )
            with self.assertRaises(ValueError) as caught:
                load_fit_records(Path(directory))
            self.assertIn("weird@dddddddddddd", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
