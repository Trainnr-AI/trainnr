"""The Studio's forms over the tools (`trainnr.forms`): each one runs a
tool that exists, asks for every argument that tool requires, and lands
on a page the Studio has (the desktop app plan, docs/85, 2026-10-11)."""

from __future__ import annotations

import io
import json
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests._extras import needs_mcp, needs_sim
from trainnr import cli, forms

STUDIO_SRC = Path(__file__).resolve().parents[2] / "crates" / "trainnr-studio" / "src"
PAGES_RS = (STUDIO_SRC / "pages.rs").read_text(encoding="utf-8")


def studio_pages() -> set[str]:
    """The Studio's page titles, as `Section::parse` reads them."""
    body = PAGES_RS[PAGES_RS.index("pub fn title(self)") :]
    body = body[: body.index("\n    }\n")]
    return {title.lower() for title in re.findall(r'=> "([^"]+)",', body)}


@needs_mcp
@needs_sim
class EveryForm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from trainnr.mcp_server import build_server, tool_api  # noqa: PLC0415

        cls.api = {tool["name"]: tool for tool in tool_api(build_server())}

    def test_runs_a_tool_that_exists_with_every_argument_it_needs(self) -> None:
        for form in forms.FORMS:
            with self.subTest(tool=form.tool):
                schema = self.api[form.tool]["inputSchema"]
                self.assertLessEqual(set(form.fields), set(schema["properties"]))
                self.assertLessEqual(set(schema.get("required", [])), set(form.fields))
                self.assertLessEqual(set(form.hints), set(form.fields))

    def test_lands_on_a_page_the_studio_has(self) -> None:
        pages = studio_pages()
        self.assertIn("robots", pages)
        for form in forms.FORMS:
            with self.subTest(tool=form.tool):
                self.assertIn(form.page, pages)

    def test_describe_fills_each_field_from_the_schema(self) -> None:
        described = forms.describe(list(self.api.values()))
        self.assertEqual(described["schema"], forms.SCHEMA)
        self.assertEqual(len(described["forms"]), len(forms.FORMS))
        robot = next(f for f in described["forms"] if f["tool"] == "onboard_robot")
        model = robot["fields"][0]
        self.assertEqual((model["name"], model["type"]), ("model_path", "string"))
        self.assertTrue(model["required"])
        self.assertIn("urdf", model["files"])
        drift = next(f for f in described["forms"] if f["tool"] == "check_drift")
        self.assertEqual([f["kind"] for f in drift["fields"]], ["robot", "recording"])

    def test_registry_choices_are_filled_when_described(self) -> None:
        described = forms.describe(list(self.api.values()))["forms"]
        choices = {
            (form["tool"], f["name"]): f["choices"]
            for form in described
            for f in form["fields"]
        }
        self.assertIn("iit-go2-chirp", choices[("ingest_public_log", "log")])
        self.assertIn("mujoco", choices[("gate_deployment", "runtime")])
        self.assertIn("trainnr/go2-walk", choices[("create_task", "family")])

    def test_a_registry_that_fails_leaves_a_text_field(self) -> None:
        def broken() -> tuple[str, ...]:
            raise RuntimeError("no registry")

        self.assertEqual(forms._called(broken), [])

    def test_a_missing_tool_leaves_no_dead_button(self) -> None:
        api = [t for n, t in self.api.items() if n != "train_walk"]
        tools = [f["tool"] for f in forms.describe(api)["forms"]]
        self.assertNotIn("train_walk", tools)

    def test_the_command_prints_them(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["forms"]), 0)
        self.assertEqual(json.loads(out.getvalue())["schema"], forms.SCHEMA)


if __name__ == "__main__":
    unittest.main()
