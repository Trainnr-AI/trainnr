"""Whether an agent is connected, as a file (`trainnr.mcp_link`): the
server keeps it while it serves and removes it on the way out. The
reading half is the Studio's, in Rust, and is tested there
(`model.rs::read_agent_link`)."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from trainnr.mcp_link import HEARTBEAT_EVERY_S, SCHEMA, AgentLink, link_path
from trainnr.paths import HOME_ENV


class LinkHome(unittest.TestCase):
    def test_the_file_sits_under_the_user_home(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {HOME_ENV: tmp}),
        ):
            self.assertEqual(link_path(), Path(tmp) / "agent.json")

    def test_it_is_not_inside_a_project(self) -> None:
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
        self.path = Path(self.tmp.name) / "agent.json"

    def test_open_writes_this_process_and_close_removes_it(self) -> None:
        link = AgentLink(self.path).open()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema"], SCHEMA)
            self.assertEqual(raw["pid"], os.getpid())
            self.assertEqual(raw["calls"], 0)
            self.assertIsNone(raw["last_tool"])
        finally:
            link.close()
        self.assertFalse(self.path.exists())

    def test_a_tool_call_is_counted_and_named(self) -> None:
        with AgentLink(self.path) as link:
            link.touch("describe_project")
            link.touch("list_robots")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["calls"], 2)
        self.assertEqual(raw["last_tool"], "list_robots")

    def test_the_heartbeat_moves_while_the_server_idles(self) -> None:
        """An agent that has called nothing for an hour is still connected,
        so the heartbeat cannot wait for a tool call."""
        with AgentLink(self.path) as link:
            first = json.loads(self.path.read_text(encoding="utf-8"))["heartbeat"]
            deadline = time.time() + HEARTBEAT_EVERY_S * 10
            later = first
            while later <= first and time.time() < deadline:
                time.sleep(HEARTBEAT_EVERY_S / 4)
                later = json.loads(self.path.read_text(encoding="utf-8"))["heartbeat"]
            self.assertGreater(later, first, "the heartbeat never moved")
            self.assertTrue(link.path.exists())

    def test_an_unwritable_home_never_takes_the_server_down(self) -> None:
        """Serving matters more than recording that it is serving."""
        link = AgentLink(Path(self.tmp.name) / "no" / "such" / "agent.json")
        with mock.patch(
            "trainnr.mcp_link.write_text", side_effect=OSError("read-only")
        ):
            link.touch("list_robots")  # must not raise
        link.close()


if __name__ == "__main__":
    unittest.main()
