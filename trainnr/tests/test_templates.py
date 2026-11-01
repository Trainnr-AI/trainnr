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
