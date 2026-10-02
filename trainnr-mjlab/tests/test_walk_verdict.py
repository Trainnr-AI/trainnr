"""The locomotion certificate's judgment: the declared criterion,
spelled a second time so the tool cannot drift from what the
certificate claims (G4, docs/e2e-research/63 §2.4)."""

from __future__ import annotations

import unittest

from trainnr_mjlab.walk_verdict import ERR_FLOOR, ERR_RATIO_BOUND, EpisodeOutcome


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


class TheBlankedCamera(unittest.TestCase):
    """The campaign 3 control (docs/07 2026-09-03): the student judged
    with its image zeroed, the row and the file saying so."""

    def test_blanked_keeps_shape_and_dtype_and_zeroes_every_pixel(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr_mjlab.walk_verdict import blanked  # noqa: PLC0415

        frames = np.full((3, 4, 5, 3), 200, dtype=np.uint8)
        blank = blanked(frames)
        self.assertEqual(blank.shape, frames.shape)
        self.assertEqual(blank.dtype, frames.dtype)
        self.assertEqual(int(blank.max()), 0)

    def test_a_blanked_run_never_lands_on_the_sighted_file(self) -> None:
        from trainnr_mjlab.walk_verdict import verdict_suffix  # noqa: PLC0415

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

        from trainnr_mjlab.walk_verdict import parse_args  # noqa: PLC0415

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

        from trainnr_mjlab.walk_verdict import (  # noqa: PLC0415
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


class TheSceneGate(unittest.TestCase):
    """A scene-trained checkpoint is judged on its scene, a plane one on
    the plane; the identity says which, and the other is refused."""

    def test_the_scene_must_match_the_identity(self) -> None:
        from trainnr_mjlab.walk_view import require_same_identity  # noqa: PLC0415

        require_same_identity({}, {"robot": "go2@1"})  # the plane, both
        require_same_identity({"scene": "fake@1"}, {"scene": "fake@1"})
        with self.assertRaisesRegex(SystemExit, "trained on fake@1.*the plane"):
            require_same_identity({"scene": "fake@1"}, {"robot": "go2@1"})
        with self.assertRaisesRegex(SystemExit, "trained on the plane.*fake@1"):
            require_same_identity({}, {"scene": "fake@1"})
        # the robot and the actuator too; a key the run never wrote is not held
        require_same_identity({"robot": "go2@1"}, {"robot": "go2@1", "actuator": "a@1"})
        with self.assertRaisesRegex(SystemExit, "mismatch on robot"):
            require_same_identity({"robot": "go2@1"}, {"robot": "go1@1"})

    def test_the_actor_sees_a_camera_only_if_it_trained_with_one(self) -> None:
        from trainnr_mjlab.walk_view import trained_with_cameras  # noqa: PLC0415

        self.assertFalse(trained_with_cameras({}))
        self.assertFalse(trained_with_cameras({"cameras": "none"}))
        self.assertTrue(trained_with_cameras({"cameras": "head 64x64 rgb"}))


class TheCrossWorld(unittest.TestCase):
    """--judge-in-fit judges a policy in another robot world: its own fit
    by default, another fit, or the declared constants; the file and the
    protocol say it is a cross-evaluation, never the policy's certificate."""

    def test_the_world_a_policy_is_judged_in(self) -> None:
        from trainnr_mjlab.walk_verdict import (  # noqa: PLC0415
            JUDGE_DECLARED,
            judged_fit,
        )

        fitted = {"fit": "fit@0ad6202797c5"}
        self.assertEqual(judged_fit(None, fitted), "fit@0ad6202797c5")
        self.assertIsNone(judged_fit(None, {}))
        self.assertIsNone(judged_fit(JUDGE_DECLARED, fitted))
        self.assertEqual(judged_fit("fit@abc", {}), "fit@abc")

    def test_the_record_names_both_worlds(self) -> None:
        from trainnr_mjlab.walk_verdict import cross_identity  # noqa: PLC0415
        from trainnr_mjlab.walks import Identity  # noqa: PLC0415

        # declared-trained, judged in fit b: the judged basis never sits
        # under the trained key
        built = {"robot": "go2@1", "fit": "fit@b", "fit_basis": "b's robot"}
        judged = {"fit": "fit@b", "fit_basis": "b's robot"}
        record = cross_identity(built, {}, judged)
        self.assertNotIn(Identity.FIT, record)
        self.assertNotIn(Identity.FIT_BASIS, record)
        self.assertEqual(record[Identity.JUDGED_IN_FIT], "fit@b")
        self.assertEqual(record[Identity.JUDGED_FIT_BASIS], "b's robot")
        # fit-trained, judged on the declared constants
        trained = {"fit": "fit@b", "fit_basis": "b's robot"}
        record = cross_identity({"robot": "go2@1"}, trained, {})
        self.assertEqual(record[Identity.FIT], "fit@b")
        self.assertEqual(record[Identity.FIT_BASIS], "b's robot")
        self.assertEqual(record[Identity.JUDGED_IN_FIT], "declared")
        self.assertNotIn(Identity.JUDGED_FIT_BASIS, record)

    def test_a_cross_still_holds_the_robot_and_actuator(self) -> None:
        """The cross gate lets only the fit move (walk_verdict's override)."""
        from trainnr_mjlab.walk_view import require_same_identity  # noqa: PLC0415

        trained = {"robot": "go2@1", "actuator": "a@1", "fit": "fit@a"}
        built = {"robot": "go2@1", "actuator": "a@1", "fit": "fit@b"}
        require_same_identity({**trained, "fit": built["fit"]}, built)
        with self.assertRaisesRegex(SystemExit, "mismatch on robot"):
            require_same_identity(
                {**trained, "fit": built["fit"]}, {**built, "robot": "go1@1"}
            )

    def test_a_cross_into_its_own_world_is_refused(self) -> None:
        from trainnr_mjlab.walk_verdict import require_another_world  # noqa: PLC0415

        with self.assertRaisesRegex(SystemExit, "own certificate"):
            require_another_world({"fit": "fit@b"}, {"fit": "fit@b"})
        with self.assertRaisesRegex(SystemExit, "declared"):
            require_another_world({}, {})
        require_another_world({"fit": "fit@b"}, {})  # fit -> declared: a cross
        require_another_world({}, {"fit": "fit@b"})  # declared -> fit: a cross

    def test_the_running_job_names_its_world(self) -> None:
        import sys  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from trainnr_mjlab.walk_verdict import (  # noqa: PLC0415
            parse_args,
            verdict_job_name,
        )

        argv = ["walk_verdict", "runs/go2-c2/model_1499.pt"]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(verdict_job_name(parse_args()), "go2-c2")
        cross = [*argv, "--judge-in-fit", "fit@ab", "--judge-at-scale", "0.8"]
        with mock.patch.object(sys, "argv", [*cross, "--judge-param", "kp"]):
            self.assertEqual(
                verdict_job_name(parse_args()), "go2-c2 in fit@ab at kp x0.8"
            )

    def test_the_cross_file_names_its_world(self) -> None:
        from trainnr_mjlab.walk_verdict import cross_world_word  # noqa: PLC0415

        self.assertEqual(cross_world_word("fit@0ad6202797c5"), "fit-0ad6202797c5")
        self.assertEqual(cross_world_word("declared"), "declared")

    def test_the_flag_exists_and_defaults_off(self) -> None:
        import sys  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from trainnr_mjlab.walk_verdict import parse_args  # noqa: PLC0415

        with mock.patch.object(sys, "argv", ["walk_verdict", "model.pt"]):
            self.assertIsNone(parse_args().judge_in_fit)
        with mock.patch.object(
            sys, "argv", ["walk_verdict", "model.pt", "--judge-in-fit", "declared"]
        ):
            self.assertEqual(parse_args().judge_in_fit, "declared")


if __name__ == "__main__":
    unittest.main()
