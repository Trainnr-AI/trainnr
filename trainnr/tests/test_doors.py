"""The server-level doors and the registries they read, against a fake
spawner and a temporary project: the walk resolved from the project's
declared task, the gate runtime refused by name, the shared lookups, the
acceptance smoke's command line — the contracts the review of 2026-09-12
found untested.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import trainnr.mcp_server as server
from trainnr.bundles.locate import add_search_root, search_roots
from trainnr.mcp_actions import unitree_reference
from trainnr.mcp_jobs import JobHandle, JobManager
from trainnr.project import create_project
from trainnr.project.locate import PROJECT_ENV, Project
from trainnr.project.task_ref import (
    DECLARED,
    TaskReference,
    read_task_reference,
    task_references,
    write_task_reference,
)
from trainnr.tasks.experts import expert_for, experts
from trainnr.tasks.registry import (
    _REGISTRY,
    register,
    register_expert,
    resolve,
    tasks,
)
from trainnr.tasks.walks import walk_robot, walk_robots

TOOLS = Path(__file__).resolve().parents[2] / "tools"


class _Spawned:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: Any, cwd: Any, log_path: Any) -> SimpleNamespace:
        self.calls.append(list(argv))
        Path(log_path).write_text("")
        return SimpleNamespace(pid=os.getpid(), wait=lambda: 0)


@contextmanager
def project_with(*walks: str) -> Iterator[tuple[Project, _Spawned]]:
    """A temporary current project declaring the named walk families
    (`microduck-walk`, …) as tasks, with the doors' jobs faked."""
    with tempfile.TemporaryDirectory() as tmp:
        project = create_project(Path(tmp) / "p", "p")
        for family in walks:
            write_task_reference(
                project,
                f"trainnr/{family}",
                f"{family}@{'0' * 12}",
                kind=DECLARED,
                spec={"dr_span": 0.05},
                name=family.replace("-walk", "-flat"),
            )
        spawned = _Spawned()
        original = server._jobs_root
        os.environ[PROJECT_ENV] = str(project.root)
        # The doors build their own JobManager over the project root;
        # the fake spawner rides in through the manager's constructor.
        real_manager = JobManager.__init__

        def fake_init(self: JobManager, runs_root: Path, **_: Any) -> None:
            real_manager(self, runs_root, spawner=spawned)

        JobManager.__init__ = fake_init  # type: ignore[method-assign]
        try:
            yield project, spawned
        finally:
            JobManager.__init__ = real_manager  # type: ignore[method-assign]
            server._jobs_root = original
            os.environ.pop(PROJECT_ENV, None)


class TheRegistryMarksTheWalks(unittest.TestCase):
    def test_walk_families_carry_their_robot(self) -> None:
        self.assertEqual(walk_robots(), ("go1", "go2", "microduck"))
        self.assertEqual(walk_robot("go2-walk"), "go2")
        self.assertEqual(walk_robot("trainnr/microduck-walk"), "microduck")
        self.assertIsNone(walk_robot("kitting"))
        self.assertTrue(resolve("go2-walk").walk)
        self.assertFalse(resolve("kitting").walk)

    def test_the_kitting_expert_is_registered_on_its_family(self) -> None:
        self.assertIn("trainnr/kitting", experts())
        self.assertEqual(expert_for("kitting").__name__, "scripted_kitting_episode")
        with self.assertRaisesRegex(ValueError, "no scripted expert reviews"):
            expert_for("go2-walk")

    def test_an_expert_needs_its_family_first_and_only_once(self) -> None:
        with self.assertRaisesRegex(KeyError, "register its builder first"):
            register_expert("nobody")(lambda *a, **k: None)

        @register("review-me", rig="test", namespace="test")
        def build() -> None:
            return None

        @register_expert("review-me", namespace="test")
        def expert(*a: Any, **k: Any) -> None:
            return None

        try:
            self.assertIs(tasks()["test/review-me"].expert, expert)
            with self.assertRaisesRegex(ValueError, "already has an expert"):
                register_expert("review-me", namespace="test")(lambda *a, **k: 1)
        finally:
            _REGISTRY.pop("test/review-me")  # the registry is process-wide


class TaskReferences(unittest.TestCase):
    def test_a_reference_reads_back_as_it_was_written(self) -> None:
        with project_with("go2-walk") as (project, _):
            ref = read_task_reference(project, "go2-flat")
            self.assertIsInstance(ref, TaskReference)
            self.assertEqual(ref.task_id, "trainnr/go2-walk")
            self.assertEqual(ref.dr_span, 0.05)
            self.assertEqual(ref.kind, DECLARED)
            self.assertEqual([r.name for r in task_references(project)], ["go2-flat"])
            with self.assertRaisesRegex(FileNotFoundError, "no task 'ghost'"):
                read_task_reference(project, "ghost")


class TheWalkDoors(unittest.TestCase):
    def test_train_walk_takes_the_projects_one_declared_walk(self) -> None:
        with project_with("go2-walk") as (project, spawned):
            out = server.train_walk("smoke", iterations=2)
            self.assertIn("job_id", out)
            [argv] = spawned.calls
            self.assertEqual(argv[argv.index("--robot") + 1], "go2")
            self.assertEqual(argv[argv.index("--dr-span") + 1], "0.05")
            self.assertEqual(
                argv[argv.index("--task-stamp") + 1], "go2-walk@" + "0" * 12
            )
            trainer = argv[argv.index("-m") :]  # after uv's own --project
            self.assertEqual(trainer[trainer.index("--project") + 1], str(project.root))

    def test_train_walk_refuses_when_the_walk_is_ambiguous_or_absent(self) -> None:
        with project_with("go2-walk", "microduck-walk") as (_, spawned):
            out = server.train_walk("smoke")
            self.assertEqual(out["status"], "refused")
            self.assertIn("go2-flat", out["reason"])
            self.assertIn("microduck-flat", out["reason"])
            self.assertEqual(spawned.calls, [])
        with project_with() as (_, spawned):
            out = server.train_walk("smoke")
            self.assertEqual(out["status"], "refused")
            self.assertIn("declares no walk", out["reason"])
            out = server.train_walk("smoke", robot="spot")
            self.assertEqual(out["status"], "refused")
            self.assertIn("one of go1, go2, microduck", out["reason"])
            out = server.train_walk("smoke", task="ghost")
            self.assertEqual(out["status"], "refused")
            self.assertIn("ghost", out["reason"])

    def test_evaluate_walk_names_the_declared_walks_robot(self) -> None:
        with project_with("microduck-walk") as (project, spawned):
            # A checkpoint that is not there is refused before any job
            # spawns, naming the ones the run holds (2026-09-28).
            (project.root / "runs" / "x").mkdir(parents=True)
            (project.root / "runs" / "x" / "model_0.pt").write_bytes(b"x")
            out = server.evaluate_walk("runs/x/model_1.pt", trials=2)
            self.assertEqual(out["status"], "refused")
            self.assertIn("no checkpoint 'model_1.pt'", out["reason"])
            self.assertIn("model_0.pt", out["reason"])
            self.assertEqual(spawned.calls, [])
            (project.root / "runs" / "x" / "model_1.pt").write_bytes(b"x")
            server.evaluate_walk("runs/x/model_1.pt", trials=2)
            [argv] = spawned.calls
            self.assertEqual(argv[argv.index("-m") + 1], "trainnr_mjlab.walk_verdict")
            self.assertEqual(argv[argv.index("--robot") + 1], "microduck")

    def test_preview_rewards_reads_the_declared_walk(self) -> None:
        with project_with("go2-walk") as (project, spawned):
            server.preview_rewards("go2-flat", controller="stand")
            [argv] = spawned.calls
            self.assertEqual(argv[argv.index("--robot") + 1], "go2")
            self.assertEqual(
                argv[argv.index("--out") + 1],
                str(project.root / "tasks" / "go2-flat" / "preview-stand.json"),
            )
            out = server.preview_rewards("ghost")
            self.assertEqual(out["status"], "refused")


class TheGateDoor(unittest.TestCase):
    def test_an_unknown_runtime_is_refused_by_name(self) -> None:
        with project_with() as (_, spawned):
            out = server.gate_deployment("d", runtime="gazebo")
            self.assertEqual(out["status"], "refused")
            self.assertIn("gazebo", out["reason"])
            self.assertIn("mujoco, dds", out["reason"])
            self.assertEqual(spawned.calls, [])

    def test_a_runtime_for_another_platform_is_refused_by_name(self) -> None:
        with project_with() as (project, spawned):
            folder = project.folder("deploy") / "d"
            folder.mkdir(parents=True)
            (folder / "deploy.json").write_text("{}")
            with mock.patch("sys.platform", "darwin"):
                out = server.gate_deployment("d", trials=2, runtime="dds")
            self.assertEqual(out["status"], "refused")
            self.assertIn("linux only", out["reason"])
            self.assertIn("darwin", out["reason"])
            self.assertEqual(spawned.calls, [])

    def test_the_dds_runtime_rides_with_the_reference(self) -> None:
        with project_with() as (project, spawned):
            folder = project.folder("deploy") / "d"
            folder.mkdir(parents=True)
            (folder / "deploy.json").write_text("{}")
            with mock.patch("sys.platform", "linux"):
                out = server.gate_deployment("d", trials=2, runtime="dds")
            self.assertIn("job_id", out)
            [argv] = spawned.calls
            self.assertEqual(argv[argv.index("--runtime") + 1], "dds")
            self.assertEqual(
                argv[argv.index("--reference") + 1], str(unitree_reference())
            )

    def test_the_runtimes_are_listed_with_their_platforms(self) -> None:
        listed = {r["name"]: r for r in server.list_gate_runtimes()}
        self.assertEqual(listed["mujoco"]["platforms"], [])
        self.assertEqual(listed["dds"]["platforms"], ["linux"])


class TheSharedLookups(unittest.TestCase):
    def test_a_checkpoints_policy_and_its_newest_evaluation(self) -> None:
        art = SimpleNamespace
        run = art(kind="run", stamp="go2-c1@r1", cites={}, updated="")
        policy = art(kind="policy", stamp="go2-c1-model_7@p1", cites={}, updated="")
        other = art(
            kind="policy", stamp="x-model_7@p2", cites={"run": "go2-c1@r1"}, updated=""
        )
        old = art(
            kind="certificate",
            stamp="e-old@1",
            cites={"policy": "go2-c1-model_7@p1"},
            updated="2026-01-01",
        )
        new = art(
            kind="certificate",
            stamp="e-new@2",
            cites={"policy": "go2-c1-model_7@p1"},
            updated="2026-02-01",
        )
        artifacts = [run, other, policy, old, new]
        found = server.policy_of_checkpoint(
            artifacts, "go2-c1", "go2-c1@r1", "model_7.pt"
        )
        self.assertIs(found, other)  # the first mark that matches, by index order
        self.assertIs(
            server.policy_of_checkpoint(
                [run, policy], "go2-c1", "go2-c1@r1", "model_7.pt"
            ),
            policy,
        )
        self.assertIsNone(
            server.policy_of_checkpoint([run], "go2-c1", "go2-c1@r1", "m.pt")
        )
        self.assertIs(server.newest_evaluation_of(artifacts, "go2-c1-model_7@p1"), new)
        self.assertIsNone(server.newest_evaluation_of(artifacts, "nobody@0"))

    def test_the_walk_of_a_run_comes_from_its_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertIsNone(server._walk_of_run([], run_dir))
            (run_dir / "identity.json").write_text(json.dumps({"robot": "go2@abc"}))
            self.assertEqual(server._walk_of_run([], run_dir), "go2")
            (run_dir / "identity.json").write_text(json.dumps({"robot": "spot@abc"}))
            self.assertIsNone(server._walk_of_run([], run_dir))
            (run_dir / "identity.json").write_text(
                json.dumps({"robot": "spot@abc", "task": "trainnr/microduck-walk"})
            )
            self.assertEqual(server._walk_of_run([], run_dir), "microduck")


def _accept_task_module() -> Any:
    """`tools/accept-task.py` as a module (a script; the tests reach its
    functions by loading the file)."""
    import importlib.util  # noqa: PLC0415

    if str(TOOLS) not in sys.path:  # the tools import their `_lab` neighbour
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "accept_task", TOOLS / "accept-task.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheStudioDoorsResolveBareNames(unittest.TestCase):
    """`screenshot_studio(artifact="go2")` was refused as `no artifact "go2"`
    right after `onboard_robot` had answered with the version, while the
    fit and drift doors had resolved bare names since 2026-09-28 (the
    stranger test, 2026-10-03)."""

    def test_screenshot_open_show_and_compare_take_a_bare_name(self) -> None:
        with project_with("go2-walk") as (project, _):
            stamp = "go2-flat@" + "0" * 12  # the declared task, by its name
            sent: list[dict[str, Any]] = []

            def fake_screenshot(proj: Any, **kwargs: Any) -> dict[str, Any]:
                sent.append({"door": "screenshot", **kwargs})
                return {"status": "done", "path": "x.png"}

            def fake_command(proj: Any, verb: str, **kwargs: Any) -> dict[str, Any]:
                sent.append({"door": verb, **kwargs})
                return {"status": "done"}

            with (
                mock.patch("trainnr.project.control.screenshot", fake_screenshot),
                mock.patch("trainnr.project.control.command", fake_command),
                mock.patch(
                    "trainnr.project.control.wait_presented",
                    lambda *a, **k: {"presented": True},
                ),
            ):
                self.assertEqual(
                    server.screenshot_studio(
                        section="environments", artifact="go2-flat"
                    )["status"],
                    "done",
                )
                self.assertEqual(
                    server.open_in_studio(artifact="go2-flat")["status"], "done"
                )
                self.assertEqual(server.show_in_studio("go2-flat")["status"], "done")
                self.assertEqual(
                    server.compare_in_studio("go2-flat", stamp)["status"], "done"
                )
                # a full version passes through untouched
                server.open_in_studio(artifact=stamp)
                # a name nobody carries is refused by the same sentence as before
                refused = server.screenshot_studio(artifact="spot")
                self.assertEqual(refused["status"], "refused")
                self.assertIn("no artifact 'spot'", refused["reason"])
                self.assertEqual(server.show_in_studio("spot")["status"], "refused")
                self.assertEqual(
                    server.compare_in_studio("spot", stamp)["status"], "refused"
                )
            self.assertEqual(
                [d["door"] for d in sent],
                ["screenshot", "open", "show", "compare", "open"],
            )
            self.assertEqual(sent[0]["artifact"], stamp)
            self.assertEqual(sent[1]["artifact"], stamp)
            self.assertEqual(sent[2]["artifact"], stamp)
            self.assertEqual((sent[3]["a"], sent[3]["b"]), (stamp, stamp))
            self.assertEqual(sent[4]["artifact"], stamp)
            self.assertEqual(project.root, project.root)  # the fixture's project


class TheAcceptanceSmoke(unittest.TestCase):
    def test_a_robot_in_the_wrong_shape_is_refused_before_the_smoke(self) -> None:
        """A Menagerie Go2 (unnamed collision geoms, no foot sites) sent the
        smoke into mjlab's regex traceback and the verdict read "exited 1;
        see acceptance.log" (stranger test 2026-10-03). The census comes
        first and the verdict says what is missing and where the right
        model is; nothing is spawned."""
        module = _accept_task_module()
        calls: list[list[str]] = []

        def fake_run(
            argv: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[Any]:
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0)

        with project_with("go2-walk") as (project, _):
            ref = read_task_reference(project, "go2-flat")
            accepted = module.review_walk(
                project,
                ref.folder,
                ref,
                "go2",
                run=fake_run,
                prepare=lambda argv, cwd: None,
                shape_missing=lambda robot: ["geom FR_foot_collision", "site imu"],
            )
            self.assertFalse(accepted)
            self.assertEqual(calls, [])
            record = json.loads((ref.folder / "acceptance.json").read_text())
            self.assertFalse(record["accepted"])
            [reason] = record["reasons"]
            self.assertIn("missing geom FR_foot_collision, site imu", reason)
            self.assertIn("unitree_rl_mjlab", reason)
            self.assertIn("Menagerie", reason)
            self.assertIsNone(record["log"])
            self.assertEqual(record["instrument"], "unrecorded")

    def test_review_walk_spawns_the_train_doors_line_and_reads_the_mark(self) -> None:
        module = _accept_task_module()
        calls: list[list[str]] = []

        def fake_run(
            argv: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[Any]:
            calls.append(list(argv))
            kwargs["stdout"].write(
                "identity: {'actuator': 'go2-pd@abc'}\n[smoke] done\n"
            )
            return subprocess.CompletedProcess(argv, 0)

        with project_with("go2-walk") as (project, _):
            ref = read_task_reference(project, "go2-flat")
            prepared = []
            accepted = module.review_walk(
                project,
                ref.folder,
                ref,
                "go2",
                run=fake_run,
                prepare=lambda argv, cwd: prepared.append(list(argv)),
                shape_missing=lambda robot: [],
            )
            self.assertTrue(accepted)
            [argv] = calls
            self.assertEqual(prepared, [argv])  # made ready before it ran
            self.assertEqual(argv[argv.index("-m") + 1], "trainnr_mjlab.walk_train")
            self.assertEqual(argv[argv.index("--robot") + 1], "go2")
            self.assertEqual(argv[argv.index("--dr-span") + 1], "0.05")
            self.assertIn("--no-recorder", argv)
            trainer = argv[argv.index("-m") :]
            self.assertEqual(trainer[trainer.index("--project") + 1], str(project.root))
            record = json.loads((ref.folder / "acceptance.json").read_text())
            self.assertTrue(record["accepted"])
            self.assertEqual(record["instrument"], "go2-pd@abc")
            self.assertEqual(record["task"], "go2-walk@" + "0" * 12)


class TheSearchRoots(unittest.TestCase):
    def test_a_replacing_root_forgets_the_previous_project(self) -> None:
        before = search_roots()
        add_search_root(Path("/p/a"))
        add_search_root(Path("/p/b"), replace=True)
        self.assertEqual(search_roots()[:1], [Path("/p/b")])
        self.assertNotIn(Path("/p/a"), search_roots())
        # restore what the suite had
        for root in reversed(before[:-1]):
            add_search_root(root)


def _handle_shape(handle: JobHandle) -> None:
    del handle


if __name__ == "__main__":
    unittest.main()
