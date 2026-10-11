"""`trainnr call` runs one tool by name, as an agent would: the same
registry, argument checks and refusals. The Studio's buttons run it, so a
person in the app and an agent over MCP share one tool surface (the
desktop app plan, docs/85, 2026-10-11)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from tests._extras import needs_mcp, needs_sim
from trainnr import cli
from trainnr.project import locate


@needs_mcp
@needs_sim
class CallingAToolByName(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        env = mock.patch.dict(os.environ, {"TRAINNR_HOME": str(self.home)})
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

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["call", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_a_tool_runs_and_prints_its_result_as_json(self) -> None:
        code, out, _ = self.run_cli(
            "create_project", "--args", '{"path": "mine", "name": "mine"}'
        )
        self.assertEqual(code, 0)
        made = json.loads(out)
        self.assertEqual(made["status"], "done")
        # Places, not spellings: macOS's /var is a link to /private/var.
        self.assertEqual(
            Path(made["root"]).resolve(), (self.home / "projects" / "mine").resolve()
        )

    def test_a_name_is_never_a_home_folder(self) -> None:
        """`~root` typed as a project name expanded to /root (review,
        2026-10-11); a name is a folder in the projects home, refused when
        it is not one."""
        for name in ("~root", "~no-such-user-here"):
            arguments = json.dumps({"name": name})
            code, out, _ = self.run_cli("create_project", "--args", arguments)
            with self.subTest(name=name):
                self.assertEqual(code, cli.REFUSED_EXIT)
                self.assertIn("plain word", json.loads(out)["reason"])
        self.assertFalse((Path.home() / "project.json").exists())

    def test_a_refusal_is_a_result_with_its_own_exit_code(self) -> None:
        code, out, _ = self.run_cli("use_project", "--args", '{"project": "none"}')
        self.assertEqual(code, cli.REFUSED_EXIT)
        self.assertEqual(json.loads(out)["status"], "refused")

    def test_an_unknown_tool_or_bad_arguments_fail_on_one_line(self) -> None:
        code, out, err = self.run_cli("no_such_tool")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("no tool 'no_such_tool'", err)
        code, _, err = self.run_cli("create_project", "--args", '{"bogus": 1}')
        self.assertEqual(code, 1)
        self.assertIn("bogus", err)
        code, _, err = self.run_cli("create_project", "--args", "[1]")
        self.assertEqual(code, 1)
        self.assertIn("must be a JSON object", err)

    def test_list_names_every_tool_the_agent_sees(self) -> None:
        from trainnr.mcp_server import build_server, tool_api  # noqa: PLC0415

        code, out, _ = self.run_cli("--list")
        self.assertEqual(code, 0)
        names = out.split()
        self.assertEqual(names, [t["name"] for t in tool_api(build_server())])
        self.assertIn("create_project", names)


if __name__ == "__main__":
    unittest.main()
