"""A robot onboarded into a project is found before the library, and the
bundle record names the model file that was compiled."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import rq_pipeline.mcp_server as server
from rq_pipeline.bundles.bundle import BUNDLE_FILE, model_file_of, read_bundle_record
from rq_pipeline.bundles.locate import (
    add_search_root,
    bundle_dirs,
    bundle_file,
    find_bundle,
    robots_dir,
)
from rq_pipeline.mcp_actions import Actions
from rq_pipeline.mcp_jobs import JobManager
from rq_pipeline.project import PROJECT_ENV, create_project
from tests._extras import needs_sim

TWO_FILES = """<mujoco model="two"><worldbody><body name="b"><joint name="j"/>
<geom size="0.1"/></body></worldbody></mujoco>"""


class SearchOrder(unittest.TestCase):
    def test_a_project_root_is_searched_before_the_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_robots = Path(tmp) / "robots"
            (project_robots / "aloha2-nominal").mkdir(parents=True)
            add_search_root(project_robots)
            self.assertEqual(
                find_bundle("aloha2-nominal"), project_robots / "aloha2-nominal"
            )
            self.assertEqual(
                bundle_file("aloha2-nominal", "x.xml"),
                project_robots / "aloha2-nominal" / "x.xml",
            )
            self.assertEqual(
                bundle_dirs()["aloha2-nominal"], project_robots / "aloha2-nominal"
            )
            # A name nowhere resolves into the library, so the error names it.
            self.assertTrue(
                str(bundle_file("ghost", "m.xml")).startswith(str(robots_dir()))
            )
            self.assertIsNone(find_bundle("ghost"))


@needs_sim
class Onboarding(unittest.TestCase):
    def test_onboarding_records_the_model_file_it_compiled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            source.mkdir()
            (source / "two.xml").write_text(TWO_FILES)
            (source / "two_mjx.xml").write_text(
                TWO_FILES + "\n<!-- the larger twin -->\n"
            )
            into = Path(tmp) / "robots"
            actions = Actions(JobManager(Path(tmp) / "runs"), env_file=None)
            out = actions.onboard_robot(str(source / "two.xml"), "two", into=str(into))
            self.assertTrue(out["stamp"].startswith("two@"))
            record = read_bundle_record(into / "two")
            self.assertEqual(record["model_file"], "two.xml")
            self.assertEqual(record["census"]["joints"], 1)
            self.assertEqual(model_file_of(into / "two"), into / "two" / "two.xml")
            self.assertTrue((into / "two" / BUNDLE_FILE).is_file())
            with self.assertRaises(FileExistsError):
                actions.onboard_robot(str(source / "two.xml"), "two", into=str(into))
            with self.assertRaises(ValueError):
                actions.onboard_robot(str(source / "two.xml"), "a/b", into=str(into))

    def test_the_door_lands_in_the_open_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            source.mkdir()
            (source / "two.xml").write_text(TWO_FILES)
            project = create_project(Path(tmp) / "p", "p")
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                out = server.onboard_robot(str(source / "two.xml"), "two")
                self.assertEqual(out["status"], "done")
                self.assertTrue((project.root / "robots" / "two" / "two.xml").is_file())
                index = json.loads(
                    (project.root / ".index" / "project.json").read_text()
                )
                self.assertEqual([a["kind"] for a in index["artifacts"]], ["robot"])
                self.assertEqual(
                    server.onboard_robot("/nowhere/x.xml", "x")["status"], "refused"
                )
                self.assertIn("two", server.bundle_names())
            finally:
                os.environ.pop(PROJECT_ENV, None)
