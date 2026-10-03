"""Every tool's failure reaches the agent with its reason. The MCP SDK turns
any exception it did not expect into the bare "Error executing tool <name>"
and drops the text, so on a fresh clone the natural first call,
`describe_project`, read as a crash (review, 2026-10-03). Each tool is
called here with no project selected and with arguments that name nothing,
in a scratch home, with job launching refused so nothing real runs."""

import asyncio
import os
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests._extras import needs_mcp, needs_sim

BARE = re.compile(r"^Error executing tool \w+:?\s*$")
BANNED = re.compile(
    r"\bdoors?\b|\bpress(ed|es|ing)?\b|referee|\bthe box\b|GPU box|operator|"
    r"flagship|docs/\d\d|SPREAD|Neverwhere|\bthe rig\b|honesty",
    re.IGNORECASE,
)


def _dummy(schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if kind is None and "anyOf" in schema:
        kind = next(
            (s.get("type") for s in schema["anyOf"] if s.get("type") != "null"), None
        )
    return {
        "string": "nope",
        "integer": 1,
        "number": 1.0,
        "boolean": False,
        "array": [],
        "object": {},
    }.get(kind, "nope")


def _text(result: Any) -> str:
    return " ".join(
        getattr(part, "text", "") for part in getattr(result, "content", []) or []
    ).strip()


@needs_mcp
@needs_sim
class EveryToolSaysWhy(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp())
        env = {"TRAINNR_HOME": str(self.home), "TRAINNR_PROJECTS": str(self.home / "p")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("TRAINNR_PROJECT", None)
        refuse = mock.patch(
            "trainnr.mcp_jobs.JobManager.start",
            side_effect=RuntimeError("job launching is disabled in this test"),
        )
        refuse.start()
        self.addCleanup(refuse.stop)
        # a checkout's projects/default would be opened as the fallback
        none = mock.patch("trainnr.project.locate.checkout_projects", return_value=None)
        none.start()
        self.addCleanup(none.stop)
        from trainnr import mcp_server  # noqa: PLC0415

        self.server = mcp_server.build_server()

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """What the agent reads: the reply's text, or the error message the
        protocol layer sends for a failed call (raised as ToolError here)."""
        from mcp.server.mcpserver.exceptions import ToolError  # noqa: PLC0415

        try:
            result = asyncio.run(self.server.call_tool(name, arguments))
        except ToolError as exc:
            return str(exc)
        text = _text(result)
        if getattr(result, "is_error", False) or getattr(result, "isError", False):
            return text
        return text or "(ok)"  # an empty list is a reply, not a failure

    def test_no_tool_answers_with_a_bare_error(self) -> None:
        tools = self.server._tool_manager.list_tools()
        self.assertGreater(len(tools), 70)
        bare = {}
        for tool in tools:
            if tool.name in {"launch_studio", "start_capture"}:
                continue  # a window or a socket; their refusals are tested elsewhere
            props = tool.parameters.get("properties", {})
            args = {
                k: _dummy(v)
                for k, v in props.items()
                if k in tool.parameters.get("required", [])
            }
            text = self.call(tool.name, args)
            if not text or BARE.match(text):
                bare[tool.name] = text
        self.assertEqual(bare, {}, "tools whose reply carries no reason")

    def test_no_project_names_the_way_out(self) -> None:
        text = self.call("describe_project", {})
        self.assertIn("create_project", text)
        self.assertIn("use_project", text)

    def test_unknown_names_and_unsafe_ids_are_refused_with_their_reason(self) -> None:
        self.assertIn("nope", self.call("describe_bundle", {"name": "nope"}))
        self.assertIn("not a job id", self.call("cancel_job", {"job_id": "../x"}))
        self.assertIn("no project", self.call("use_project", {"project": "nope"}))

    def test_descriptions_are_plain_and_annotated(self) -> None:
        for tool in self.server._tool_manager.list_tools():
            with self.subTest(tool=tool.name):
                self.assertIsNone(
                    BANNED.search(tool.description or ""), tool.description
                )
                self.assertLessEqual(
                    len((tool.description or "").split("\n\n")[0]), 260
                )
                self.assertIsNotNone(tool.annotations)
        names = {t.name: t for t in self.server._tool_manager.list_tools()}
        self.assertTrue(names["describe_project"].annotations.read_only_hint)
        self.assertTrue(names["cancel_job"].annotations.destructive_hint)
        self.assertFalse(names["train_walk"].annotations.read_only_hint)


if __name__ == "__main__":
    unittest.main()
