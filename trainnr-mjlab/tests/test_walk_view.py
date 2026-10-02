"""The walk view's own pieces, with fakes: the Studio's Commands tab on
mjlab's joystick handles, and the project's newest checkpoint."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from trainnr.project.locate import POLICIES_FOLDER, RUNS_FOLDER

from trainnr_mjlab.walk_view import Joystick, latest_checkpoint

TWIST = "twist"  # the mailbox word under test; the stream names the real one


class TheJoystick(unittest.TestCase):
    def _term(self) -> SimpleNamespace:
        # mjlab's velocity term as `compute` sees it: the three handles.
        return SimpleNamespace(
            _joystick_enabled=None, _joystick_sliders=None, _joystick_get_env_idx=None
        )

    def test_a_twist_lands_in_the_terms_handles_for_its_world(self) -> None:
        term = self._term()
        stick = Joystick(term, axes=3, command=TWIST)
        self.assertFalse(term._joystick_enabled.value)
        stick.on_command(TWIST, 2 * 3 + 1, 0.75)  # world 2, axis 1 (vy)
        self.assertTrue(term._joystick_enabled.value)
        self.assertEqual(term._joystick_get_env_idx(), 2)
        self.assertEqual([s.value for s in term._joystick_sliders], [0.0, 0.75, 0.0])
        stick.on_command(TWIST, 2 * 3 + 0, -1.0)
        self.assertEqual(term._joystick_sliders[0].value, -1.0)

    def test_a_release_hands_the_command_back_and_other_words_are_ignored(self) -> None:
        term = self._term()
        stick = Joystick(term, axes=3, command=TWIST)
        stick.on_command(TWIST, 0, 0.5)
        stick.on_command("pause", 0, 9.0)  # not this tab's word
        self.assertEqual(term._joystick_sliders[0].value, 0.5)
        stick.on_command(TWIST, -1, 0.0)
        self.assertFalse(term._joystick_enabled.value)

    def test_a_term_without_the_handles_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(SystemExit, "_joystick_enabled"):
            Joystick(SimpleNamespace(), axes=3, command=TWIST)


class TheNewestCheckpoint(unittest.TestCase):
    def test_a_policy_artifact_wins_over_a_run_and_newest_by_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / RUNS_FOLDER / "a" / "model_10.pt"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"0")
            new = root / RUNS_FOLDER / "b" / "model_2.pt"
            new.parent.mkdir(parents=True)
            new.write_bytes(b"0")
            later = time.time() + 5
            import os  # noqa: PLC0415

            os.utime(new, (later, later))
            self.assertEqual(latest_checkpoint(root), new)
            judged = root / POLICIES_FOLDER / "p" / "model_2.pt"
            judged.parent.mkdir(parents=True)
            judged.write_bytes(b"0")
            self.assertEqual(
                latest_checkpoint(root), judged, "a judged checkpoint first"
            )

    def test_a_scene_trained_policy_is_skipped_for_the_plain_walk(self) -> None:
        """go2-scene-c1 (235-wide, a staged scene) was newest and crashed the
        plain walk scene; its identity is read through the run its policy
        names, and the newest plane policy is played instead."""
        import json  # noqa: PLC0415
        import os  # noqa: PLC0415

        from trainnr.project.kinds import (  # noqa: PLC0415
            IDENTITY_FILE,
            POLICY_FILE,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run, scene in (("plane-run", None), ("scene-run", "garden@abc")):
                folder = root / RUNS_FOLDER / run
                folder.mkdir(parents=True)
                identity = {"scene": scene} if scene else {}
                (folder / IDENTITY_FILE).write_text(json.dumps(identity))
            plane = root / POLICIES_FOLDER / "plane-model_1" / "model_1.pt"
            scene = root / POLICIES_FOLDER / "scene-model_1" / "model_1.pt"
            for checkpoint, run in ((plane, "plane-run"), (scene, "scene-run")):
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"0")
                (checkpoint.parent / POLICY_FILE).write_text(
                    json.dumps({"run": f"{run}@0123456789ab"})
                )
            later = time.time() + 5
            os.utime(scene, (later, later))  # the scene policy is newest
            self.assertEqual(latest_checkpoint(root), plane)
            from trainnr_mjlab.walk_view import trained_identity  # noqa: PLC0415

            self.assertEqual(
                trained_identity(scene),
                {"scene": "garden@abc"},
                "a policy's identity is its run's, the project found from its path",
            )

    def test_only_scene_policies_are_refused_by_name(self) -> None:
        import json  # noqa: PLC0415

        from trainnr.project.kinds import IDENTITY_FILE  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / RUNS_FOLDER / "scene-run"
            run.mkdir(parents=True)
            (run / IDENTITY_FILE).write_text(json.dumps({"scene": "garden@abc"}))
            (run / "model_1.pt").write_bytes(b"0")
            with self.assertRaisesRegex(SystemExit, "staged scene"):
                latest_checkpoint(root)

    def test_an_empty_project_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "train a walk first"):
                latest_checkpoint(Path(tmp))


if __name__ == "__main__":
    unittest.main()
