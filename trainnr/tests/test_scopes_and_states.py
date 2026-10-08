"""What a list says and what a stage counts (the second fresh-install
audit on macOS, 2026-10-08): list_robots said nothing of where a robot
came from, and list_projects counted nine stages where the Studio
counted eight."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trainnr.project import create_project, locate
from trainnr.project.index import index_project, write_index


class FromAFreshHome(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        env = mock.patch.dict(os.environ, {"TRAINNR_HOME": str(self.tmp / "home")})
        env.start()
        self.addCleanup(env.stop)
        for name in ("TRAINNR_PROJECT", "TRAINNR_PROJECTS"):
            os.environ.pop(name, None)
        for patcher in (
            mock.patch.object(locate, "checkout_projects", return_value=None),
            mock.patch.object(locate, "_session_root", None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_each_robot_says_whether_it_is_the_projects_or_the_librarys(self) -> None:
        from trainnr.mcp_server import create_project_dir, list_robots  # noqa: PLC0415

        made = create_project_dir("demo", "demo")
        own = Path(made["root"]) / "robots" / "mine"
        own.mkdir(parents=True)
        (own / "robot.xml").write_text("<mujoco/>")
        scopes = {r["name"]: r["scope"] for r in list_robots()}
        self.assertEqual(scopes["mine"], "project")
        self.assertEqual(scopes["microduck"], "library")

    def test_a_walk_project_counts_the_stages_its_loop_passes(self) -> None:
        from trainnr.mcp_server import list_project_dirs  # noqa: PLC0415

        project = create_project(
            locate.projects_home() / "walker", "walker", loop="reinforcement"
        )
        write_index(project, index_project(project), previews=False)
        raw = json.loads((project.root / ".index" / "project.json").read_text())
        needed = sum(1 for s in raw["states"] if s.get("needed", True))
        self.assertLess(needed, len(raw["states"]))  # a walk has no dataset stage
        listed = {p["name"]: p for p in list_project_dirs()}
        self.assertEqual(listed["walker"]["stages"], needed)


if __name__ == "__main__":
    unittest.main()
