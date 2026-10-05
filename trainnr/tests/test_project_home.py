"""Where projects live and which one is current. Projects are the user's
data: they live in the projects home (`$TRAINNR_PROJECTS`, else
`$TRAINNR_HOME/projects`, else ~/trainnr/projects), never in a checkout or
a plugin's install folder, which an update replaces (review, 2026-10-03).
The current project is `$TRAINNR_PROJECT`, else the one last created or
chosen, else a checkout's `projects/default`, else none, refused by name."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from trainnr.mcp_jobs import JobManager
from trainnr.project import locate


class _Home(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        env = mock.patch.dict(os.environ, {"TRAINNR_HOME": str(self.tmp / "home")})
        env.start()
        self.addCleanup(env.stop)
        for name in ("TRAINNR_PROJECT", "TRAINNR_PROJECTS"):
            os.environ.pop(name, None)
        none = mock.patch.object(locate, "checkout_projects", return_value=None)
        none.start()
        self.addCleanup(none.stop)
        session = mock.patch.object(locate, "_session_root", None)
        session.start()
        self.addCleanup(session.stop)


class OneProcessOneProject(_Home):
    """Two agent sessions shared the projects home's `.current`: one
    session's use_project moved the other's work into its own project
    (review, 2026-10-04). A process keeps the project it chose; `.current`
    only seeds a new one."""

    def test_another_sessions_choice_does_not_move_this_one(self) -> None:
        mine = locate.create_project(locate.projects_home() / "mine", "mine")
        theirs = locate.create_project(locate.projects_home() / "theirs", "theirs")
        locate.use_project("mine")
        # another process writes its own choice into the shared file
        (locate.projects_home() / locate.CURRENT_FILE).write_text(str(theirs.root))
        self.assertEqual(locate.current_project().root, mine.root.absolute())
        # a new process (no choice yet) starts from the shared file
        with mock.patch.object(locate, "_session_root", None):
            self.assertEqual(locate.current_project().root, theirs.root)


class TheProjectsHome(_Home):
    def test_the_home_is_the_user_data_root_unless_named(self) -> None:
        self.assertEqual(locate.projects_home(), self.tmp / "home" / "projects")
        with mock.patch.dict(
            os.environ, {"TRAINNR_PROJECTS": str(self.tmp / "elsewhere")}
        ):
            self.assertEqual(locate.projects_home(), self.tmp / "elsewhere")

    def test_with_nothing_chosen_the_refusal_names_both_ways_out(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            locate.current_project()
        self.assertIn("create_project", str(caught.exception))
        self.assertIn("use_project", str(caught.exception))

    def test_a_created_and_remembered_project_is_current(self) -> None:
        root = locate.projects_home() / "demo"
        locate.create_project(root, "demo")
        locate.remember_project(root)
        self.assertEqual(locate.current_project().root, root.absolute())

    def test_use_project_takes_a_bare_name_and_refuses_a_missing_one(self) -> None:
        for name in ("a", "b"):
            locate.create_project(locate.projects_home() / name, name)
        self.assertEqual(locate.use_project("b").root, locate.projects_home() / "b")
        self.assertEqual(locate.current_project().root.name, "b")
        with self.assertRaises(FileNotFoundError):
            locate.use_project("nope")

    def test_the_variable_wins_over_the_remembered_project(self) -> None:
        for name in ("a", "b"):
            locate.create_project(locate.projects_home() / name, name)
        locate.use_project("a")
        with mock.patch.dict(
            os.environ, {"TRAINNR_PROJECT": str(locate.projects_home() / "b")}
        ):
            self.assertEqual(locate.current_project().root.name, "b")

    def test_a_checkouts_default_is_the_last_fallback(self) -> None:
        checkout_projects = self.tmp / "checkout" / "projects"
        locate.create_project(checkout_projects / "default", "default")
        with mock.patch.object(
            locate, "checkout_projects", return_value=checkout_projects
        ):
            self.assertEqual(locate.current_project().root.name, "default")
            listed = [p.root.name for p in locate.list_projects()]
        self.assertIn("default", listed)


class TheMcpProjectTools(_Home):
    def test_create_project_lands_in_the_home_and_becomes_current(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        made = mcp_server.create_project_dir("demo", "demo")
        self.assertEqual(made["root"], str(locate.projects_home() / "demo"))
        self.assertEqual(locate.current_project().root.name, "demo")
        listed = mcp_server.list_project_dirs()
        self.assertEqual([(p["name"], p["current"]) for p in listed], [("demo", True)])

    def test_use_project_refuses_by_name(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        self.assertEqual(mcp_server.use_project("nope")["status"], "refused")


class CancellingAJob(unittest.TestCase):
    """A cancel signals only the job's own process: a finished job, a pid
    the OS gave to another process, or a record from before the start
    time was kept, is refused rather than signalled."""

    def setUp(self) -> None:
        self.jobs = JobManager(Path(tempfile.mkdtemp()))

    def test_a_job_id_that_is_a_path_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a job id"):
            self.jobs.cancel("../escape")

    def test_a_running_job_is_cancelled_and_a_finished_one_is_not(self) -> None:
        try:
            import psutil  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("psutil (the mcp extra) is not installed")
        handle = self.jobs.start(
            "sleep", [sys.executable, "-c", "import time; time.sleep(30)"], Path.cwd()
        )
        time.sleep(0.2)
        answer = self.jobs.cancel(handle["job_id"])
        self.assertIn("cancelled", answer)
        self.jobs.join(5)
        again = self.jobs.cancel(handle["job_id"])
        self.assertEqual(again.get("status"), "refused")

    def test_a_reused_pid_is_not_signalled(self) -> None:
        handle = self.jobs.start("quick", [sys.executable, "-c", "pass"], Path.cwd())
        self.jobs.join(5)
        (self.jobs.jobs_dir / f"{handle['job_id']}.exit").unlink(missing_ok=True)
        record_path = self.jobs.jobs_dir / f"{handle['job_id']}.json"
        record = record_path.read_text()
        # the pid now names process 1, alive and started at boot: a reused pid
        import json  # noqa: PLC0415

        raw = json.loads(record)
        raw["pid"] = 1
        record_path.write_text(json.dumps(raw))
        answer = self.jobs.cancel(handle["job_id"])
        self.assertEqual(answer.get("status"), "refused")


class APluginThatFails(unittest.TestCase):
    def test_is_skipped_and_named_and_the_server_keeps_its_tools(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        class Broken:
            name = "broken"
            value = "nowhere:register"

            def load(self) -> object:
                raise ImportError("no module named nowhere")

        with (
            mock.patch("importlib.metadata.entry_points", return_value=[Broken()]),
            mock.patch("sys.stderr") as err,
        ):
            names = mcp_server.register_plugin_tools(object())
        self.assertEqual(names, [])
        written = "".join(c.args[0] for c in err.write.call_args_list)
        self.assertIn("broken", written)
        self.assertIn("skipped", written)


class TheCliWithoutTheMcpExtra(unittest.TestCase):
    def test_names_the_command_that_installs_it(self) -> None:
        code = (
            "import builtins, sys\n"
            "real = builtins.__import__\n"
            "def fake(name, *a, **k):\n"
            "    if name.startswith('mcp') or name == 'trainnr.mcp_server':\n"
            "        raise ImportError('No module named mcp')\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = fake\n"
            "from trainnr.cli import main\n"
            "sys.exit(main(['mcp']))\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--extra mcp", run.stderr)


if __name__ == "__main__":
    unittest.main()


class TheCheckoutSample(unittest.TestCase):
    """The checkout's empty sample is listed only when there is no other
    project; beside the user's own it read as a broken one (2026-10-04)."""

    def test_the_sample_steps_aside_for_real_projects(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        checkout = tmp / "checkout"
        locate.create_project(checkout / locate.SAMPLE_PROJECT, "sample")
        with (
            mock.patch.dict(os.environ, {"TRAINNR_HOME": str(tmp / "home")}),
            mock.patch.object(locate, "checkout_projects", return_value=checkout),
        ):
            os.environ.pop("TRAINNR_PROJECTS", None)
            self.assertEqual([p.root.name for p in locate.list_projects()], ["sample"])
            locate.create_project(locate.projects_home() / "go2", "go2")
            self.assertEqual([p.root.name for p in locate.list_projects()], ["go2"])
