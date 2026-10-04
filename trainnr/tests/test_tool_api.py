"""The MCP tool API as a contract: its snapshot, its naming rules, its
size, and that the trimmed schemas still accept what they did.

Every tool's name, description and input schema is read by the agent at
the start of every session, so a rename is a breaking change, a vague
word costs a wrong call, and every token is paid for on every session.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests._extras import needs_mcp, needs_sim

PACKAGE = Path(__file__).resolve().parent.parent
REPO = PACKAGE.parent
SNAPSHOT = PACKAGE / "tests" / "api" / "tools.json"

# The verbs a tool name may start with: list_ for a collection, describe_
# for one item, and the actions the loop takes.
VERBS = frozenset(
    {
        "assay", "attribute", "cancel", "capture", "check", "compare", "control",
        "create", "describe", "evaluate", "export", "gate", "generate", "identify",
        "import", "ingest", "launch", "list", "multiply", "onboard", "open", "play",
        "preflight", "preview", "quit", "read", "run", "screenshot", "set", "show",
        "stage", "start", "stop", "train", "use",
    }
)  # fmt: skip
# Argument names that said what the code called a thing, not what the
# agent knows it as: an existing artifact is named by its kind.
BANNED_ARGUMENTS = frozenset(
    {"slug", "task_id", "run", "certificate", "demos_dir", "overlay"}
)
# Words from inside the project that mean nothing to an agent reading a
# description; "actuator bundle" is the published term and stays.
BANNED_WORDS = re.compile(
    r"\bdoors?\b|\bpress(ed|es|ing)?\b|\breferees?\b|(?<!actuator )\bbundles?\b|"
    r"\bthe box\b|\boperators?\b",
    re.IGNORECASE,
)
# The trimmed surface measured 6,566 by this proxy (6,175 cl100k tokens)
# on 2026-10-04, down from 9,192 (9,161); growth past 5% is a decision.
TOKEN_PROXY_AT_TRIM = 6566
CHARS_PER_TOKEN = 3.6
TOKEN_BUDGET = round(TOKEN_PROXY_AT_TRIM * 1.05)


def _api() -> list[dict[str, Any]]:
    from trainnr.mcp_server import build_server, tool_api  # noqa: PLC0415

    return tool_api(build_server(plugins=False))


@needs_mcp
@needs_sim
class TheSnapshot(unittest.TestCase):
    def test_the_api_matches_its_snapshot(self) -> None:
        recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        self.assertTrue(
            _api() == recorded,
            "the MCP tool API changed: run python3 tools/api-snapshot.py --write "
            "and review the diff",
        )

    def test_the_script_checks_it(self) -> None:
        done = subprocess.run(
            [sys.executable, str(REPO / "tools" / "api-snapshot.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)


@needs_mcp
@needs_sim
class TheNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = _api()

    def test_every_name_starts_with_a_known_verb(self) -> None:
        for tool in self.api:
            with self.subTest(tool=tool["name"]):
                self.assertRegex(tool["name"], r"^[a-z]+(_[a-z0-9]+)*$")
                self.assertIn(tool["name"].split("_", 1)[0], VERBS)

    def test_a_list_takes_no_required_argument(self) -> None:
        for tool in self.api:
            if tool["name"].startswith("list_"):
                with self.subTest(tool=tool["name"]):
                    self.assertEqual(tool["inputSchema"].get("required", []), [])

    def test_no_argument_has_a_banned_name(self) -> None:
        for tool in self.api:
            names = set(tool["inputSchema"].get("properties", {}))
            with self.subTest(tool=tool["name"]):
                self.assertEqual(names & BANNED_ARGUMENTS, set())

    def test_no_description_uses_an_inside_word(self) -> None:
        for tool in self.api:
            text = tool["description"] + json.dumps(tool["inputSchema"])
            with self.subTest(tool=tool["name"]):
                self.assertIsNone(BANNED_WORDS.search(text), text)


@needs_mcp
@needs_sim
class TheBudget(unittest.TestCase):
    def test_the_surface_stays_within_its_token_budget(self) -> None:
        text = "".join(
            tool["name"] + tool["description"] + json.dumps(tool["inputSchema"])
            for tool in _api()
        )
        tokens = round(len(text) / CHARS_PER_TOKEN)
        print(f"\nMCP tool surface: ~{tokens} tokens (budget {TOKEN_BUDGET})")
        self.assertLessEqual(tokens, TOKEN_BUDGET)


class TheTrim(unittest.TestCase):
    def test_titles_go_and_an_optional_value_is_its_type(self) -> None:
        from trainnr.mcp_server import trim_schema  # noqa: PLC0415

        schema = {
            "title": "fArguments",
            "type": "object",
            "properties": {
                "title": {"title": "Title", "type": "string"},
                "robot": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Robot",
                },
                "must": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "c": {"$ref": "#/$defs/C", "default": None},
            },
            "required": ["title", "must"],
            "$defs": {"C": {"title": "C", "type": "object", "properties": {}}},
        }
        self.assertEqual(
            trim_schema(schema),
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "robot": {"type": "string", "default": None},
                    # required and nullable: null is a real answer, kept
                    "must": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "c": {"type": "object", "properties": {}, "default": None},
                },
                "required": ["title", "must"],
            },
        )


@needs_mcp
@needs_sim
class TheCallsStillValidate(unittest.TestCase):
    """The advertised schema is trimmed; what a call may pass is not."""

    def setUp(self) -> None:
        from trainnr.mcp_server import build_server  # noqa: PLC0415

        self.tools = build_server(plugins=False)._tool_manager

    def validate(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.tools.get_tool(name)
        return tool.fn_metadata.arg_model.model_validate(arguments)

    def test_omitted_and_null_optionals_are_accepted(self) -> None:
        for arguments in ({"checkpoint": "a"}, {"checkpoint": "a", "robot": None}):
            with self.subTest(arguments=arguments):
                self.assertIsNone(self.validate("evaluate_walk", arguments).robot)
        held = self.validate("evaluate_walk", {"checkpoint": "a", "conditions": None})
        self.assertIsNone(held.conditions)

    def test_conditions_take_their_four_fields_and_refuse_others(self) -> None:
        from pydantic import ValidationError  # noqa: PLC0415

        held = self.validate(
            "evaluate_walk",
            {"checkpoint": "a", "conditions": {"in_fit": "declared", "delay": 2}},
        )
        self.assertEqual(held.conditions, {"in_fit": "declared", "delay": 2})
        with self.assertRaises(ValidationError):
            self.validate(
                "evaluate_walk", {"checkpoint": "a", "conditions": {"dly": 2}}
            )

    def test_show_in_studio_takes_exactly_one(self) -> None:
        from trainnr.mcp_server import show_in_studio  # noqa: PLC0415

        for arguments in ({}, {"artifact": "a", "stream": "b"}):
            with self.subTest(arguments=arguments):
                self.assertEqual(show_in_studio(**arguments)["status"], "refused")


class _ExitedProcess:
    pid = os.getpid()

    def wait(self) -> int:
        return 0


def _spawn(argv: Any, cwd: Path, log_path: Path) -> _ExitedProcess:
    Path(log_path).write_text("stubbed\n")
    return _ExitedProcess()


@needs_mcp
@needs_sim
class TheJobTable(unittest.TestCase):
    """A job started after create_project is found by the job readers. The
    server's job manager was built at start, rooted before any project
    existed, so describe_job answered "no job …; known: []" for a job the
    starter had written into the project chosen through the projects
    home's `.current` (an agent run, 2026-10-04)."""

    def setUp(self) -> None:
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        env = {"TRAINNR_HOME": str(self.home), "TRAINNR_PROJECTS": str(self.home / "p")}
        for patcher in (
            mock.patch.dict(os.environ, env),
            mock.patch("trainnr.project.locate.checkout_projects", return_value=None),
            mock.patch("trainnr.mcp_jobs._spawn", _spawn),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("TRAINNR_PROJECT", None)
        from trainnr.mcp_server import build_server  # noqa: PLC0415

        self.server = build_server(plugins=False)  # before any project, as at start

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        result = asyncio.run(self.server.call_tool(name, arguments))
        self.assertFalse(result.is_error, result)
        content = result.structured_content or {}
        return content.get("result", content)

    def test_the_readers_find_what_the_starter_wrote(self) -> None:
        made = self.call("create_project", {"path": "demo", "name": "demo"})
        self.assertNotEqual(made.get("status"), "refused", made)
        # a starter that builds its own job manager per call, as most do
        handle = self.call(
            "ingest_public_log", {"log": "iit-go2-chirp", "accept_unlicensed": True}
        )
        job_id = handle["job_id"]
        self.assertIn(str(self.home), handle["log"])
        status = self.call("describe_job", {"job_id": job_id})
        for _ in range(100):  # the watcher records the stub's exit
            if status["state"] != "running":
                break
            time.sleep(0.02)
            status = self.call("describe_job", {"job_id": job_id})
        self.assertEqual(status["job_id"], job_id)
        listed = [job["job_id"] for job in self.call("list_jobs", {})]
        self.assertIn(job_id, listed)


if __name__ == "__main__":
    unittest.main()
