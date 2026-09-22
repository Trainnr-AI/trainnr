"""The locomotion certificate's judgment: the declared criterion,
spelled a second time so the tool cannot drift from what the
certificate claims (G4, docs/e2e-research/63 §2.4)."""

from __future__ import annotations

import unittest

from rq_mjlab.walk_verdict import ERR_FLOOR, ERR_RATIO_BOUND, EpisodeOutcome


class TheCriterion(unittest.TestCase):
    def test_success_needs_both_milestones(self) -> None:
        walked = EpisodeOutcome(steps=1000, fell=False, mean_err=0.05, mean_cmd=0.3)
        self.assertTrue(walked.survived and walked.tracked and walked.success)
        fell = EpisodeOutcome(steps=412, fell=True, mean_err=0.05, mean_cmd=0.3)
        self.assertFalse(fell.success)
        drifted = EpisodeOutcome(steps=1000, fell=False, mean_err=0.2, mean_cmd=0.3)
        self.assertFalse(drifted.tracked or drifted.success)

    def test_the_ratio_is_the_standing_still_gap_longhand(self) -> None:
        # err_ratio = mean_err / max(mean_cmd, ERR_FLOOR); tracked means
        # the policy closes at least half the gap standing still leaves.
        outcome = EpisodeOutcome(steps=1000, fell=False, mean_err=0.12, mean_cmd=0.3)
        self.assertAlmostEqual(outcome.err_ratio, 0.12 / 0.3)
        self.assertTrue(outcome.err_ratio < ERR_RATIO_BOUND)

    def test_a_gentle_command_floors_the_denominator(self) -> None:
        # A near-zero command must not make any wobble an automatic fail
        # (or a trivial pass): the denominator floors at ERR_FLOOR m/s.
        idle = EpisodeOutcome(steps=1000, fell=False, mean_err=0.04, mean_cmd=0.01)
        self.assertAlmostEqual(idle.err_ratio, 0.04 / ERR_FLOOR)
        self.assertTrue(idle.tracked)
        wobbly = EpisodeOutcome(steps=1000, fell=False, mean_err=0.06, mean_cmd=0.01)
        self.assertFalse(wobbly.tracked)


if __name__ == "__main__":
    unittest.main()


class TheBlankedCamera(unittest.TestCase):
    """The campaign 3 control (docs/07 2026-09-03): the student judged
    with its image zeroed, the row and the file saying so."""

    def test_blanked_keeps_shape_and_dtype_and_zeroes_every_pixel(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_mjlab.walk_verdict import blanked  # noqa: PLC0415

        frames = np.full((3, 4, 5, 3), 200, dtype=np.uint8)
        blank = blanked(frames)
        self.assertEqual(blank.shape, frames.shape)
        self.assertEqual(blank.dtype, frames.dtype)
        self.assertEqual(int(blank.max()), 0)

    def test_a_blanked_run_never_lands_on_the_sighted_file(self) -> None:
        from rq_mjlab.walk_verdict import verdict_suffix  # noqa: PLC0415

        sighted = verdict_suffix("cuda", student=True, blank_camera=False)
        blank = verdict_suffix("cuda", student=True, blank_camera=True)
        self.assertEqual(sighted, "student-cuda")  # campaign 2/3's headline file
        self.assertEqual(blank, "student-blank-cuda")
        self.assertEqual(
            verdict_suffix("cuda", student=True, blank_camera=False, blank_state=True),
            "student-blank-state-cuda",
        )
        self.assertEqual(
            verdict_suffix("cuda", student=True, blank_camera=True, blank_state=True),
            "student-blank-both-cuda",
        )
        self.assertEqual(
            verdict_suffix("cpu", student=False, blank_camera=False), "cpu"
        )

    def test_the_flag_exists_and_defaults_off(self) -> None:
        import sys  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from rq_mjlab.walk_verdict import parse_args  # noqa: PLC0415

        with mock.patch.object(sys, "argv", ["walk_verdict", "model.pt"]):
            self.assertFalse(parse_args().blank_camera)
        with mock.patch.object(
            sys, "argv", ["walk_verdict", "model.pt", "--blank-camera"]
        ):
            self.assertTrue(parse_args().blank_camera)


class TheRotation(unittest.TestCase):
    """A certificate is never silently replaced: the same checkpoint
    judged three times under three protocols (here, three instruments)
    leaves three files (E0, docs/78, 2026-09-22)."""

    def test_three_instruments_three_files(self) -> None:
        import json  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from rq_mjlab.walk_verdict import (  # noqa: PLC0415
            EpisodeOutcome,
            write_certificate,
        )

        outcome = EpisodeOutcome(steps=1000, fell=False, mean_err=0.05, mean_cmd=0.8)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for version in ("3.11.0", "3.13.0", "3.14.0"):
                instrument = f"mjlab-1.6.0+mujoco-{version}+warp-1.17.0+cuda"
                write_certificate(
                    out,
                    "cuda",
                    [outcome, outcome],
                    source="go2-walk@test",
                    policy_name="model_1499",
                    instrument=instrument,
                    protocol={"seed": 1000, "instrument": instrument},
                    identity={},
                    trials=2,
                )
            files = sorted(p.name for p in out.glob("walk-verdict-*.json"))
            self.assertEqual(len(files), 3)
            instruments = {
                json.loads((out / f).read_text())["instrument"] for f in files
            }
            self.assertEqual(len(instruments), 3)
            self.assertIn("walk-verdict-cuda.json", files)
