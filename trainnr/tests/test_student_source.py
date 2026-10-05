"""A student checkpoint runs no code it names, and comes from the project or
a Hugging Face repo id: LeRobot imports the processor step classes a
checkpoint's JSON names, so a crafted one ran code when it loaded
(security review, 2026-10-05)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trainnr.envs.lerobot_policy import check_processors
from trainnr.mcp_actions import student_source
from trainnr.project import create_project


def _processors(folder: Path, steps: list[dict]) -> None:
    (folder / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))


class OnlyLeRobotsOwnSteps(unittest.TestCase):
    def test_registered_and_lerobot_classes_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _processors(
                Path(tmp),
                [
                    {"registry_name": "normalizer_processor", "config": {}},
                    {"class": "lerobot.processor.normalize_processor.Normalizer"},
                ],
            )
            check_processors(Path(tmp))

    def test_any_other_class_is_refused_by_name(self) -> None:
        for target in ("os.system", "evil.Step", "lerobotx.Step", ""):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                _processors(Path(tmp), [{"class": target}])
                with self.assertRaisesRegex(ValueError, "not LeRobot's own"):
                    check_processors(Path(tmp))


class WhereAStudentComesFrom(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project = create_project(Path(self._tmp.name) / "p", "p")
        patch = mock.patch.dict(os.environ, {"TRAINNR_PROJECT": str(self.project.root)})
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_a_folder_in_the_project_and_a_repo_id_are_taken(self) -> None:
        folder = self.project.root / "runs" / "student"
        folder.mkdir(parents=True)
        self.assertEqual(student_source("runs/student"), str(folder.resolve()))
        self.assertEqual(student_source("org/vision-student"), "org/vision-student")

    def test_a_folder_elsewhere_is_refused(self) -> None:
        outside = Path(self._tmp.name) / "elsewhere"
        outside.mkdir()
        for value in (str(outside), "../elsewhere", "~/x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                student_source(value)

    def test_a_name_that_is_neither_is_refused(self) -> None:
        # one slash, no such folder, and not a repo id's shape
        with self.assertRaises(ValueError):
            student_source("org/na me")

    def test_a_bare_name_is_a_run_folder_of_the_project(self) -> None:
        self.assertEqual(
            student_source("student"),
            str((self.project.root / "runs" / "student").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
