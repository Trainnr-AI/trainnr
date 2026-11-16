"""Project templates (`trainnr.templates`, docs/86 §6): every step proves
a loop state and names only tools that exist; a template sets the
project's loop; the index walks its steps."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_mcp, needs_sim
from trainnr.project.index import STATES, index_project
from trainnr.project.locate import LOOPS, Project, create_project
from trainnr.templates import TEMPLATES, resolve

NO_DATASET = "data generated"


class EveryTemplate(unittest.TestCase):
    def test_steps_prove_loop_states_in_the_loop_it_names(self) -> None:
        names = [name for name, _ in STATES]
        for template in TEMPLATES.values():
            with self.subTest(template=template.name):
                self.assertIn(template.loop, LOOPS)
                self.assertTrue(template.steps)
                for step in template.steps:
                    self.assertIn(step.stage, names)
                stages = [s.stage for s in template.steps]
                self.assertEqual(len(stages), len(set(stages)), "one step per stage")
                if template.loop == "reinforcement":
                    self.assertNotIn(NO_DATASET, stages)

    def test_an_unknown_template_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(KeyError, "humanoid-motion-from-video"):
            resolve("no-such-template")


@needs_mcp
@needs_sim
class TheNamedTools(unittest.TestCase):
    def test_every_named_tool_is_served(self) -> None:
        from trainnr.mcp_server import build_server, tool_api  # noqa: PLC0415

        served = {tool["name"] for tool in tool_api(build_server())}
        for template in TEMPLATES.values():
            for step in template.steps:
                if step.tool is not None:
                    self.assertIn(step.tool, served, f"{template.name}: {step.title}")


class AProjectFromATemplate(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_takes_the_template_s_loop_and_walks_its_steps(self) -> None:
        template = TEMPLATES["humanoid-motion-from-video"]
        project = create_project(self.home / "g1", "g1", template=template.name)
        manifest = project.manifest()
        self.assertEqual(manifest.template, template.name)
        self.assertEqual(manifest.loop, "reinforcement")
        plan = index_project(project).template
        assert plan is not None
        self.assertEqual(plan["title"], template.title)
        titles = [s["title"] for s in plan["steps"]]
        self.assertEqual(titles, [s.title for s in template.steps])
        self.assertFalse(any(s["done"] for s in plan["steps"]))

    def test_a_step_is_done_by_its_own_artifact_not_by_its_stage(self) -> None:
        """A plain recording proves Telemetry but not "Turn a video into
        motion"; a tracking run proves its step once finished, and reads
        running while it trains (2026-11-16)."""
        import json  # noqa: PLC0415

        from trainnr.templates import TRACKING_TASK  # noqa: PLC0415

        project = create_project(
            self.home / "g1", "g1", template="humanoid-motion-from-video"
        )

        def recording(name: str, collection: str) -> None:
            folder = project.recordings / name
            folder.mkdir(parents=True)
            (folder / "recording.json").write_text(
                json.dumps(
                    {
                        "schema": "trainnr-recording/1",
                        "source": name,
                        "adapter": "x",
                        "duration_s": 1.0,
                        "channels": [],
                        "collection": collection,
                    }
                )
            )

        def run(name: str, status: str, iterations: int) -> None:
            folder = project.root / "runs" / name
            folder.mkdir(parents=True)
            (folder / "identity.json").write_text(json.dumps({"task": TRACKING_TASK}))
            (folder / "training.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "iterations": 2000,
                        "iterations_logged": iterations,
                    }
                )
            )

        def steps() -> dict[str, dict]:
            plan = index_project(Project(project.root)).template
            assert plan is not None
            return {s["title"]: s for s in plan["steps"]}

        recording("telemetry", "teleop")
        self.assertFalse(steps()["Turn a video into motion"]["done"])
        recording("dance", "video")
        video = steps()["Turn a video into motion"]
        self.assertTrue(video["done"])
        self.assertTrue(video["proved_by"].startswith("dance@"))

        run("long", "running", 113)
        training = steps()["Train it to track the motion"]
        self.assertFalse(training["done"])
        self.assertEqual(training["running"], "113 of 2000 iterations")
        run("stopped", "unrecorded", 281)
        self.assertFalse(steps()["Train it to track the motion"]["done"])
        run("check", "done", 30)
        training = steps()["Train it to track the motion"]
        self.assertTrue(training["done"])
        self.assertIn(", 30 iterations", training["proved_by"])

    def test_refuses_a_loop_the_template_does_not_have(self) -> None:
        with self.assertRaisesRegex(ValueError, "reinforcement loop"):
            create_project(
                self.home / "g1",
                "g1",
                loop="imitation",
                template="humanoid-motion-from-video",
            )
        self.assertFalse((self.home / "g1").exists())

    def test_a_blank_project_has_no_plan(self) -> None:
        project = create_project(self.home / "blank", "blank")
        self.assertIsNone(index_project(project).template)

    def test_a_template_this_version_does_not_know_keeps_its_name(self) -> None:
        project = create_project(self.home / "later", "later")
        raw = project.manifest_path.read_text(encoding="utf-8")
        project.manifest_path.write_text(
            raw.replace('"template": ""', '"template": "from-a-later-version"'),
            encoding="utf-8",
        )
        plan = index_project(Project(project.root)).template
        later = "from-a-later-version"
        self.assertEqual(plan, {"name": later, "title": later, "steps": []})


if __name__ == "__main__":
    unittest.main()
