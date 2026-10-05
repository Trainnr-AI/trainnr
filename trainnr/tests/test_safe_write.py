"""A project from someone else can carry links where trainnr writes; no
write goes through one (the security review of 2026-10-05 reproduced an
overwrite through `.index/project.json.tmp`, a deletion through a linked
`.index/commands` and an append through a linked `studio.log`)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from trainnr import safe_write
from trainnr.project import create_project
from trainnr.project.control import _prune
from trainnr.project.files import write_json


class NoWriteThroughALink(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.project = create_project(self.tmp / "shared", "shared")
        self.victim = self.tmp / "victim"
        self.victim.mkdir()
        self.secret = self.victim / "secret.txt"
        self.secret.write_text("keep me")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_linked_target_is_refused_and_left_alone(self) -> None:
        planted = self.project.root / ".index" / "project.json"
        planted.parent.mkdir(exist_ok=True)
        if planted.exists():
            planted.unlink()
        planted.symlink_to(self.secret)
        with self.assertRaises(safe_write.LinkRefusedError):
            write_json(planted, {"x": 1})
        self.assertEqual(self.secret.read_text(), "keep me")

    def test_a_linked_folder_inside_the_project_is_refused(self) -> None:
        runs = self.project.root / "runs"
        if runs.is_dir():
            runs.rmdir()
        runs.symlink_to(self.victim, target_is_directory=True)
        with self.assertRaises(safe_write.LinkRefusedError):
            write_json(self.project.root / "runs" / "a" / "training.json", {})
        self.assertEqual(sorted(p.name for p in self.victim.iterdir()), ["secret.txt"])

    def test_a_planted_staging_name_is_never_followed(self) -> None:
        target = self.project.root / ".index" / "x.json"
        target.parent.mkdir(exist_ok=True)
        (target.parent / "x.json.tmp").symlink_to(self.secret)  # the old staging name
        write_json(target, {"ok": True})
        self.assertEqual(json.loads(target.read_text()), {"ok": True})
        self.assertEqual(self.secret.read_text(), "keep me")

    def test_pruning_a_linked_commands_folder_deletes_nothing(self) -> None:
        for i in range(5):
            (self.victim / f"{i}.json").write_text("{}")
        commands = self.project.root / ".index" / "commands"
        commands.parent.mkdir(exist_ok=True)
        commands.symlink_to(self.victim, target_is_directory=True)
        with self.assertRaises(safe_write.LinkRefusedError):
            _prune(commands, keep=1, root=self.project.root)
        self.assertEqual(len(list(self.victim.glob("*.json"))), 5)

    def test_a_linked_log_is_not_appended_to(self) -> None:
        log = self.project.root / ".index" / "studio.log"
        log.parent.mkdir(exist_ok=True)
        log.symlink_to(self.secret)
        with self.assertRaises(safe_write.LinkRefusedError):
            safe_write.open_append(log)
        self.assertEqual(self.secret.read_text(), "keep me")

    def test_a_project_that_is_itself_a_link_still_works(self) -> None:
        """A user's projects home is often a link (and macOS's /tmp is)."""
        alias = self.tmp / "alias"
        alias.symlink_to(self.project.root, target_is_directory=True)
        out = write_json(alias / ".index" / "y.json", {"y": 1})
        self.assertEqual(json.loads(out.read_text()), {"y": 1})

    def test_a_write_leaves_no_staging_file_and_keeps_the_umask(self) -> None:
        target = self.project.root / "notes" / "a.txt"
        safe_write.write_text(target, "hello")
        self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["a.txt"])
        umask = os.umask(0)
        os.umask(umask)
        self.assertEqual(target.stat().st_mode & 0o777, 0o666 & ~umask)


if __name__ == "__main__":
    unittest.main()
