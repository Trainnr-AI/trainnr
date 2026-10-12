"""Which agents are connected, as files (`trainnr.mcp_link`): each server
keeps its own while it serves and removes only its own on the way out.
The reading half is the Studio's, in Rust, and is tested there
(`model.rs::read_agent_link`)."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from trainnr.mcp_link import (
    HEARTBEAT_EVERY_S,
    LINK_DIR,
    SCHEMA,
    SWEEP_AFTER_S,
    AgentLink,
    link_dir,
    link_path,
    sweep,
)
from trainnr.paths import HOME_ENV


class WhereTheFilesLive(unittest.TestCase):
    def test_one_file_per_process_under_the_user_home(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {HOME_ENV: tmp}),
        ):
            self.assertEqual(link_dir(), Path(tmp) / LINK_DIR)
            self.assertEqual(link_path(), Path(tmp) / LINK_DIR / f"{os.getpid()}.json")
            self.assertEqual(link_path(4242), Path(tmp) / LINK_DIR / "4242.json")

    def test_they_are_not_inside_a_project(self) -> None:
        """An agent is connected to the machine, not to a project: the
        Welcome page shows the answer before any project exists."""
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {HOME_ENV: tmp}),
        ):
            self.assertNotIn("projects", link_path().parts)


class OpenAndClose(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)

    def path(self, name: str) -> Path:
        return self.home / LINK_DIR / f"{name}.json"

    def read(self, name: str) -> dict:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def test_open_writes_this_process_and_close_removes_it(self) -> None:
        link = AgentLink(self.path("a")).open()
        try:
            raw = self.read("a")
            self.assertEqual(raw["schema"], SCHEMA)
            self.assertEqual(raw["pid"], os.getpid())
            self.assertEqual(raw["calls"], 0)
            self.assertIsNone(raw["last_tool"])
        finally:
            link.close()
        self.assertFalse(self.path("a").exists())

    def test_a_tool_call_is_counted_and_named(self) -> None:
        """`touch` moves the counters in memory; the heartbeat writes
        them within the second."""
        with AgentLink(self.path("a")) as link:
            link.touch("describe_project")
            link.touch("list_robots")
            want = 2
            deadline = time.time() + HEARTBEAT_EVERY_S * 10
            while self.read("a")["calls"] != want and time.time() < deadline:
                time.sleep(HEARTBEAT_EVERY_S / 4)
            raw = self.read("a")
        self.assertEqual(raw["calls"], want)
        self.assertEqual(raw["last_tool"], "list_robots")

    def test_the_heartbeat_moves_while_the_server_idles(self) -> None:
        """An agent that has called nothing for an hour is still
        connected, so the heartbeat cannot wait for a tool call."""
        with AgentLink(self.path("a")):
            first = self.read("a")["heartbeat"]
            deadline = time.time() + HEARTBEAT_EVERY_S * 10
            later = first
            while later <= first and time.time() < deadline:
                time.sleep(HEARTBEAT_EVERY_S / 4)
                later = self.read("a")["heartbeat"]
            self.assertGreater(later, first, "the heartbeat never moved")

    def test_two_sessions_keep_their_own_files(self) -> None:
        """Two Claude Code windows, or Claude Code and Cursor, are
        ordinary. One file for the machine had them overwriting each
        other every heartbeat (review, 2026-10-11)."""
        with AgentLink(self.path("a")) as first:
            with AgentLink(self.path("b")) as second:
                first.touch("list_robots")
                second.touch("describe_project")
                self.assertTrue(self.path("a").exists())
                self.assertTrue(self.path("b").exists())
            # The second left; the first is untouched and still serving.
            self.assertFalse(self.path("b").exists(), "only its own goes")
            self.assertTrue(self.path("a").exists(), "the other survives")
        self.assertFalse(self.path("a").exists())

    def test_an_unwritable_home_never_takes_the_server_down(self) -> None:
        """Serving matters more than recording that it is serving."""
        link = AgentLink(self.path("a"))
        with mock.patch(
            "trainnr.mcp_link.write_text", side_effect=OSError("read-only")
        ):
            link.open()  # must not raise
        link.close()


class Sweep(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / LINK_DIR
        self.folder.mkdir(parents=True)

    def write(self, name: str, age: float) -> Path:
        path = self.folder / f"{name}.json"
        path.write_text(
            json.dumps({"schema": SCHEMA, "pid": 1, "heartbeat": time.time() - age}),
            encoding="utf-8",
        )
        return path

    def test_debris_goes_and_a_live_file_stays(self) -> None:
        """A session killed before it could tidy up leaves its file for
        ever; the next server to start removes it, by age alone."""
        old = self.write("111", SWEEP_AFTER_S + 60)
        fresh = self.write("222", 0.0)
        self.assertEqual(sweep(self.folder), 1)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists(), "a serving session is not debris")

    def test_a_file_it_cannot_read_is_left_alone(self) -> None:
        junk = self.folder / "333.json"
        junk.write_text("{not json", encoding="utf-8")
        other = self.folder / "notes.txt"
        other.write_text("hello", encoding="utf-8")
        self.assertEqual(sweep(self.folder), 0)
        self.assertTrue(junk.exists())
        self.assertTrue(other.exists())

    @unittest.skipIf(os.name == "nt", "symbolic links need privileges on Windows")
    def test_a_linked_folder_is_never_swept(self) -> None:
        """Debris is removed through `safe_write`, never through a link:
        a linked `agents` folder would point the sweep somewhere else."""
        self.write("111", SWEEP_AFTER_S + 60)
        linked = Path(self.tmp.name) / "elsewhere"
        linked.symlink_to(self.folder, target_is_directory=True)
        self.assertEqual(sweep(linked), 0)
        self.assertTrue((self.folder / "111.json").exists())

    def test_a_missing_folder_is_not_an_error(self) -> None:
        self.assertEqual(sweep(Path(self.tmp.name) / "nope"), 0)

    def test_open_sweeps_before_it_writes(self) -> None:
        old = self.write("111", SWEEP_AFTER_S + 60)
        link = AgentLink(self.folder / "222.json").open()
        try:
            self.assertFalse(old.exists())
            self.assertTrue((self.folder / "222.json").exists())
        finally:
            link.close()


if __name__ == "__main__":
    unittest.main()
