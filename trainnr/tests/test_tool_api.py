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
# agent knows it as: an existing artifact is named by its kind. Every
# argument the 2026-10-04 rename retired (CHANGELOG), at any depth.
BANNED_ARGUMENTS = frozenset(
    {
        "slug", "task_id", "run", "certificate", "demos_dir", "overlay",
        "seeds_dir", "recording_name", "record", "judge_in_fit",
        "judge_at_scale", "judge_param",
    }
)  # fmt: skip
# Words from inside the project that mean nothing to an agent reading a
# description; "actuator bundle" is the published term and stays.
BANNED_WORDS = re.compile(
    r"\bdoors?\b|\bpress(ed|es|ing)?\b|\breferees?\b|(?<!actuator )\bbundles?\b|"
    r"\bthe box\b|\boperators?\b",
    re.IGNORECASE,
)
# What a session reads at start: the server's instructions and every
# tool's name, description, schema and annotations. The trim took the
# tools alone from 9,192 by this proxy to 6,566 (6,175 cl100k tokens);
# refusing unknown arguments added 189; with the instructions and the
# annotations counted the surface measured 7,875 (2026-10-04). The public
# review the same day added describe_experiment, truer descriptions and a
# one-line description on the eleven arguments agents misread (recipe,
# fit, task, checkpoint, evaluation, unevaluated, student, device,
# command, follow, the USD options): 8,449. Growth past 5% is a decision.
TOKEN_PROXY_AT_TRIM = 8449
CHARS_PER_TOKEN = 3.6
TOKEN_BUDGET = round(TOKEN_PROXY_AT_TRIM * 1.05)


def _api(plugins: bool = False) -> list[dict[str, Any]]:
    from trainnr.mcp_server import build_server, tool_api  # noqa: PLC0415

    return tool_api(build_server(plugins=plugins))


def _argument_names(schema: Any) -> set[str]:
    """Every property name in `schema`, nested objects and arrays included."""
    if isinstance(schema, list):
        return set().union(*(_argument_names(item) for item in schema))
    if not isinstance(schema, dict):
        return set()
    names = set(schema.get("properties", {}))
    for value in schema.values():
        names |= _argument_names(value)
    return names


@needs_mcp
@needs_sim
class TheStatedCount(unittest.TestCase):
    """The README states how many tools the server has; a count that
    drifted from the server (it said 77 after a merge made it 74,
    2026-10-04) tells every reader something false."""

    def test_the_readme_states_the_servers_tool_count(self) -> None:
        count = len(_api())
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        stated = {
            int(n)
            for n in re.findall(r"MCP-(\d+)%20tools|(\d+) tools", readme)
            for n in n
            if n
        }
        self.assertTrue(stated, "the README states no tool count")
        self.assertEqual(
            stated,
            {count},
            f"README states {sorted(stated)} tools; the server has {count}",
        )


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
    """The rules hold for every tool the server lists, an installed
    plugin's included; only the snapshot is of the built-ins alone."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = _api(plugins=True)

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
            names = _argument_names(tool["inputSchema"])
            with self.subTest(tool=tool["name"]):
                self.assertEqual(names & BANNED_ARGUMENTS, set())

    def test_the_lint_reads_nested_arguments(self) -> None:
        nested = {"properties": {"conditions": {"properties": {"judge_param": {}}}}}
        self.assertEqual(_argument_names(nested), {"conditions", "judge_param"})

    def test_a_list_or_describe_is_read_only(self) -> None:
        for tool in self.api:
            if tool["name"].startswith(("list_", "describe_")):
                with self.subTest(tool=tool["name"]):
                    self.assertTrue(tool["annotations"].get("readOnlyHint"))
                    self.assertFalse(tool["annotations"].get("destructiveHint"))

    def test_every_tool_refuses_an_unknown_argument(self) -> None:
        for tool in self.api:
            with self.subTest(tool=tool["name"]):
                self.assertIs(tool["inputSchema"].get("additionalProperties"), False)

    def test_no_description_uses_an_inside_word(self) -> None:
        for tool in self.api:
            text = tool["description"] + json.dumps(tool["inputSchema"])
            with self.subTest(tool=tool["name"]):
                self.assertIsNone(BANNED_WORDS.search(text), text)


@needs_mcp
@needs_sim
class TheBudget(unittest.TestCase):
    def test_the_surface_stays_within_its_token_budget(self) -> None:
        from trainnr.mcp_server import build_server  # noqa: PLC0415

        text = (build_server(plugins=False).instructions or "") + "".join(
            tool["name"]
            + tool["description"]
            + json.dumps(tool["inputSchema"])
            + json.dumps(tool["annotations"])
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
                    # omitted means null: no default that fails its type
                    "robot": {"type": "string"},
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

    def test_an_old_argument_name_is_refused_not_ignored(self) -> None:
        """An ignored argument ran the call with its defaults: an old
        `overlay` made a task with no settings, an old `judge_in_fit`
        judged a policy in its own world (review, 2026-10-04)."""
        from pydantic import ValidationError  # noqa: PLC0415

        for name, arguments in (
            ("create_task", {"family": "f", "name": "n", "overlay": {"x": 1}}),
            ("evaluate_walk", {"checkpoint": "a", "judge_in_fit": "fit@abc"}),
            ("export_deployment", {"experiment": "e", "certificate": "X"}),
            ("list_findings", {"prefixx": "a"}),
        ):
            with (
                self.subTest(tool=name),
                self.assertRaisesRegex(ValidationError, "extra_forbidden"),
            ):
                self.validate(name, arguments)

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


class TheCitedEvaluation(unittest.TestCase):
    """export_deployment cites an evaluation by its index stamp; anything
    else is refused with this checkpoint's, never written into the
    manifest for the gate to find nothing (an agent run, 2026-10-04)."""

    def test_an_unknown_evaluation_is_refused_with_the_known_ones(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from trainnr.mcp_server import _unknown_evaluation  # noqa: PLC0415

        policy = SimpleNamespace(stamp="p@1")
        artifacts = [
            SimpleNamespace(kind="certificate", stamp="e@1", cites={"policy": "p@1"}),
            SimpleNamespace(kind="certificate", stamp="e@2", cites={"policy": "q@1"}),
            SimpleNamespace(kind="run", stamp="r@1", cites={}),
        ]
        self.assertEqual(_unknown_evaluation(artifacts, None, policy), "")
        self.assertEqual(_unknown_evaluation(artifacts, "e@1", policy), "")
        reason = _unknown_evaluation(artifacts, "runs/r/verdict", policy)
        self.assertIn("['e@1']", reason)
        self.assertNotIn("e@2", reason)


@needs_mcp
@needs_sim
class OneNamePerEvaluation(unittest.TestCase):
    """list_evaluations names a project evaluation by its stamp, and that
    one value is what describe_evaluation and export_deployment take. The
    same argument once meant a run folder in one tool and a stamp in the
    other; an agent passed the folder and the gate judged nothing
    (2026-10-04)."""

    def setUp(self) -> None:
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        root = Path(home.name)
        env = {"TRAINNR_HOME": str(root), "TRAINNR_PROJECTS": str(root / "p")}
        for patcher in (
            mock.patch.dict(os.environ, env),
            mock.patch("trainnr.project.locate.checkout_projects", return_value=None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("TRAINNR_PROJECT", None)
        from trainnr.mcp_server import create_project_dir  # noqa: PLC0415

        create_project_dir("demo", "demo")
        folder = root / "p" / "demo" / "certificates" / "walk-model_9-n2"
        folder.mkdir(parents=True)
        (folder / "certificate.json").write_text(
            json.dumps(
                {"policy": "walk-model_9@aa", "run": "walk@bb", "successes": 1,
                 "trials": 2, "ci95": [0.01, 0.99]}
            )
        )  # fmt: skip
        (folder / "records-cpu.jsonl").write_text(
            '{"trial": 0, "success": true, "steps": 5}\n'
            '{"trial": 1, "success": false, "steps": 3}\n'
        )

    def test_the_listed_name_is_the_one_describe_takes(self) -> None:
        from trainnr.mcp_server import (  # noqa: PLC0415
            describe_evaluation,
            list_evaluations,
        )

        (row,) = [r for r in list_evaluations() if r["evaluation"].startswith("walk-")]
        self.assertRegex(row["evaluation"], r"^walk-model_9-n2@[0-9a-f]{12}$")
        self.assertEqual(row["policy"], "walk-model_9@aa")
        self.assertEqual(row["success"], "1 / 2")
        detail = describe_evaluation(row["evaluation"])
        self.assertEqual(detail["successes"], 1)
        self.assertEqual([e["success"] for e in detail["episodes"]], [True, False])

    def test_a_folder_path_is_refused_naming_the_stamp(self) -> None:
        from trainnr.mcp_server import describe_evaluation  # noqa: PLC0415

        with self.assertRaisesRegex(KeyError, r"walk-model_9-n2@"):
            describe_evaluation("runs/walk/verdict")


class TheDefaultCitation(unittest.TestCase):
    """Export cites the newest evaluation in the policy's own world; a
    newer run under a delay, a scaled gain or another fit is a stress
    result, never the deployment's certificate (tool review, 2026-10-04)."""

    def test_a_newer_perturbed_run_is_not_the_default(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from trainnr.mcp_server import newest_evaluation_of  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                ("own", {"judged_at": "law DR: fit fit@a"}, "2026-10-01"),
                ("delay", {"judged_at": "law DR: fit fit@a", "delay": 2}, "2026-10-02"),
                (
                    "cross",
                    {"judged_at": "CROSS-evaluation: trained in a"},
                    "2026-10-03",
                ),
                ("knob", {"judged_at": "kp at fit x 0.8"}, "2026-10-04"),
            ]
            artifacts = []
            for name, protocol, updated in rows:
                (root / name).mkdir()
                (root / name / "certificate.json").write_text(
                    json.dumps({"protocol": protocol})
                )
                artifacts.append(
                    SimpleNamespace(
                        kind="certificate",
                        stamp=f"{name}@1",
                        path=name,
                        cites={"policy": "p@1"},
                        updated=updated,
                    )
                )
            self.assertEqual(
                newest_evaluation_of(artifacts, "p@1", root).stamp, "own@1"
            )
            self.assertEqual(newest_evaluation_of(artifacts[1:], "p@1", root), None)


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

    def test_an_unnamed_training_run_lands_in_the_project(self) -> None:
        """Unnamed, a g3 run was written under the checkout's runs/, outside
        the project, and every step after it lost the experiment."""
        made = self.call("create_project", {"path": "demo", "name": "demo"})
        self.assertNotEqual(made.get("status"), "refused", made)
        handle = self.call(
            "train_walk", {"recipe": "full", "iterations": 1, "robot": "go2"}
        )
        self.assertNotEqual(handle.get("status"), "refused", handle)
        argv = self.call("describe_job", {"job_id": handle["job_id"]})["argv"]
        log_dir = Path(argv[argv.index("--log-dir") + 1])
        self.assertEqual(log_dir.parent, self.home / "p" / "demo" / "runs")
        self.assertTrue(log_dir.name.startswith("go2-walk-"), log_dir)
        self.assertEqual(handle["experiment"], log_dir.name)
        self.assertIn("evaluate_walk", handle["next"])

    def test_evaluate_reads_the_walk_from_the_experiment(self) -> None:
        """evaluate_walk asked for a `task` argument it does not take; the
        experiment's identity names its robot, as export reads it."""
        self.call("create_project", {"path": "demo", "name": "demo"})
        run = self.home / "p" / "demo" / "runs" / "c0"
        run.mkdir(parents=True)
        (run / "model_9.pt").write_bytes(b"")
        (run / "identity.json").write_text(json.dumps({"robot": "go2@abc"}))
        handle = self.call("evaluate_walk", {"checkpoint": "c0", "trials": 2})
        self.assertNotEqual(handle.get("status"), "refused", handle)
        argv = self.call("describe_job", {"job_id": handle["job_id"]})["argv"]
        self.assertEqual(argv[argv.index("--robot") + 1], "go2")
        self.assertTrue(
            argv[argv.index("trainnr_mjlab.walk_verdict") + 1].endswith("model_9.pt")
        )

    def test_a_declared_task_is_described_by_its_name(self) -> None:
        """create_task then describe_task refused the task just made: the
        tool read only the registry (tool review, 2026-10-04)."""
        self.call("create_project", {"path": "demo", "name": "demo"})
        made = self.call("create_task", {"family": "trainnr/kitting", "name": "mine"})
        self.assertEqual(made["status"], "done", made)
        detail = self.call("describe_task", {"task": "mine"})
        self.assertTrue(detail["declared"])
        self.assertEqual(detail["family"], "trainnr/kitting")
        self.assertEqual(detail["next"], "check_task('mine')")

    def test_training_never_overwrites_and_says_what_it_saves(self) -> None:
        """A named run on an existing experiment wrote over its checkpoints
        and identity; a smoke saved nothing and said nothing (tool review,
        2026-10-04)."""
        self.call("create_project", {"path": "demo", "name": "demo"})
        (self.home / "p" / "demo" / "runs" / "c0").mkdir(parents=True)
        for arguments, said in (
            ({"recipe": "full", "robot": "go2", "name": "c0"}, "already exists"),
            ({"recipe": "smoke", "robot": "go2", "name": "c1"}, "saves nothing"),
            ({"recipe": "g3", "robot": "go2"}, "recipe is one of"),
            ({"recipe": "full", "robot": "go2", "iterations": 0}, "at least 1"),
        ):
            with self.subTest(arguments=arguments):
                reply = self.call("train_walk", arguments)
                self.assertEqual(reply.get("status"), "refused", reply)
                self.assertIn(said, reply["reason"])


if __name__ == "__main__":
    unittest.main()
