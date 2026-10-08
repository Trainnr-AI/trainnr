"""The bugs the second fresh-install audit on macOS found (2026-10-08):
each test reproduces the case the audit reported."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trainnr import mcp_jobs
from trainnr.paths import CHECKOUT_ENV
from trainnr.project import locate
from trainnr.tasks.walks import DEFAULT_WALK, WalkTask


class AWalkNamesItsRobot(unittest.TestCase):
    """create_task("go1-walk") and create_task("go2-walk") at their
    defaults returned one hash, 0e7e123a7de7: the stamp hashed the
    spec's knobs and left the robot out."""

    @staticmethod
    def _walk(robot: str) -> WalkTask:
        return WalkTask(
            name=f"{robot}-walk",
            robot=robot,
            task_spec=DEFAULT_WALK,
            bundle_dir=None,
            spec=None,
        )

    def test_two_robots_at_the_same_knobs_have_two_hashes(self) -> None:
        go1, go2 = self._walk("go1").stamp, self._walk("go2").stamp
        self.assertNotEqual(go1.split("@", 1)[1], go2.split("@", 1)[1])

    def test_the_same_walk_twice_has_one_stamp(self) -> None:
        self.assertEqual(self._walk("go2").stamp, self._walk("go2").stamp)


class APluginInstallListsNoProjectsOfItsOwn(unittest.TestCase):
    """On a fresh install list_projects showed `sample` from the
    plugin's cache folder, a folder each update replaces."""

    def test_a_checkout_under_claudes_plugins_folder_is_not_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "claude"
            install = config / "plugins" / "cache" / "trainnr" / "trainnr" / "0.1.0"
            (install / "projects" / "sample").mkdir(parents=True)
            env = {"CLAUDE_CONFIG_DIR": str(config), CHECKOUT_ENV: str(install)}
            with mock.patch.dict(os.environ, env):
                self.assertIsNone(locate.checkout_projects())

    def test_a_developers_checkout_still_lists_its_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "src" / "trainnr"
            (clone / "projects" / "sample").mkdir(parents=True)
            env = {
                "CLAUDE_CONFIG_DIR": str(Path(tmp) / "claude"),
                CHECKOUT_ENV: str(clone),
            }
            with mock.patch.dict(os.environ, env):
                found = locate.checkout_projects()
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.resolve(), (clone / "projects").resolve())


class AJobDoesNotInheritTheServersVenv(unittest.TestCase):
    """Every job log began with uv's warning that VIRTUAL_ENV (the
    server's venv) did not match the job's project environment."""

    def test_the_spawned_environment_has_no_virtual_env(self) -> None:
        seen: dict[str, str] = {}

        def popen(*_args: object, **kwargs: object) -> mock.Mock:
            seen.update(kwargs["env"])  # type: ignore[arg-type]
            return mock.Mock(pid=os.getpid())

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"VIRTUAL_ENV": "/server/.venv"}),
            mock.patch.object(mcp_jobs.subprocess, "Popen", popen),
        ):
            mcp_jobs._spawn(["true"], Path(tmp), Path(tmp) / "job-1.log")
        self.assertNotIn("VIRTUAL_ENV", seen)
        self.assertEqual(seen[mcp_jobs.JOB_ID_ENV], "job-1")


if __name__ == "__main__":
    unittest.main()
