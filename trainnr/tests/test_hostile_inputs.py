"""Inputs a shared project or a prompt could carry, refused by name
(security review, 2026-10-04): a model folder linking outside itself, a
name that escapes on Windows, a manifest naming a program, an endless
capture window."""

import os
import tempfile
import unittest
from pathlib import Path

from trainnr.project.locate import plain_name
from trainnr.robot.onboarding import copy_model_folder


class TheModelFolder(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_a_link_out_of_the_folder_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / "secret.txt"
            secret.write_text("private")
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "robot.xml").write_text("<mujoco/>")
            (model / "notes.txt").symlink_to(secret)
            with self.assertRaisesRegex(ValueError, "links outside"):
                copy_model_folder(model, Path(tmp) / "copy")
            self.assertFalse((Path(tmp) / "copy").exists())

    def test_a_link_inside_the_folder_is_copied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            (model / "assets").mkdir(parents=True)
            (model / "assets" / "a.stl").write_text("solid")
            if os.name != "nt":
                (model / "b.stl").symlink_to(model / "assets" / "a.stl")
            copy_model_folder(model, Path(tmp) / "copy")
            self.assertTrue((Path(tmp) / "copy" / "assets" / "a.stl").is_file())


class Names(unittest.TestCase):
    def test_names_that_escape_on_any_os_are_refused(self) -> None:
        for bad in ("D:evil", "a\\b", "con", "NUL.txt", "a\nb", "../x", ".hidden"):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                plain_name(bad)
        self.assertEqual(plain_name("go2-walk_2"), "go2-walk_2")
        self.assertEqual(plain_name("console"), "console")


class TheUnitreeManifest(unittest.TestCase):
    def test_a_manifest_naming_a_program_path_is_refused(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from trainnr.deploy.unitree_stage import facts_of  # noqa: PLC0415

        for controller in ("/usr/bin/evil", "../../x", "ok"):
            manifest = SimpleNamespace(
                root=Path("/d"),
                unitree=SimpleNamespace(controller=controller, robot="go2"),
            )
            with self.subTest(controller=controller):
                if controller == "ok":
                    self.assertEqual(facts_of(manifest).controller, "ok")
                else:
                    with self.assertRaises(ValueError):
                        facts_of(manifest)


if __name__ == "__main__":
    unittest.main()
