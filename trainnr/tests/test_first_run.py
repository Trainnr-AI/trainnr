"""A first run says what to do next (the second fresh-install audit on
macOS, 2026-10-08): describe_studio before any project, the next step
after create_project, and where a missing quickstart model comes from."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._fixtures import scratch_dir
from trainnr.project import locate
from trainnr.robot.onboarding import UNITREE_RL_MJLAB_URL, onboard


class AFreshHome(unittest.TestCase):
    """No project anywhere: an empty projects home, no checkout projects,
    no session project."""

    def setUp(self) -> None:
        self.tmp = scratch_dir(self)
        env = mock.patch.dict(os.environ, {"TRAINNR_HOME": str(self.tmp / "home")})
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

    def test_describe_studio_says_what_to_do_before_any_project(self) -> None:
        from trainnr.mcp_server import describe_studio  # noqa: PLC0415

        answer = describe_studio()
        self.assertFalse(answer["alive"])
        self.assertIsNone(answer["project"])
        self.assertIn("create_project", answer["next"])
        self.assertIn("launch_studio", answer["next"])
        self.assertIn("open_sample", answer["next"])
        self.assertTrue(answer["welcome"])
        self.assertEqual(answer["projects_home"], str(locate.projects_home()))

    def test_create_project_names_the_studio_as_the_next_step(self) -> None:
        from trainnr.mcp_server import create_project_dir  # noqa: PLC0415

        made = create_project_dir("demo", "demo")
        self.assertIn("launch_studio", made["next"])


class AMissingQuickstartModel(unittest.TestCase):
    """onboard_robot refused a missing go2.xml with only its path."""

    def test_a_path_inside_unitrees_repository_says_how_to_clone_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "unitree_rl_mjlab"
            model = (
                repo / "src" / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
            )
            with self.assertRaises(FileNotFoundError) as caught:
                onboard(model, "go2", Path(tmp) / "bundle")
            message = str(caught.exception)
            self.assertIn(f"git clone --depth 1 {UNITREE_RL_MJLAB_URL} {repo}", message)

    def test_any_other_missing_path_is_named_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "elsewhere" / "robot.xml"
            with self.assertRaises(FileNotFoundError) as caught:
                onboard(model, "robot", Path(tmp) / "bundle")
            self.assertEqual(str(caught.exception), f"no model file at {model}")


HOOK = Path(__file__).resolve().parents[2] / "tools" / "plugin-prefetch.sh"


@unittest.skipUnless(shutil.which("sh"), "the session hook is a POSIX sh script")
class TheFirstSession(unittest.TestCase):
    """After the install and the restart nothing said that trainnr was
    there or what to say; the hook now tells the model once."""

    def _run(self, data: Path, path: str) -> str:
        env = {
            **os.environ,
            "PATH": path,
            "TRAINNR_NO_PREFETCH": "1",
            "CLAUDE_PLUGIN_ROOT": str(HOOK.parents[1]),
            "CLAUDE_PLUGIN_DATA": str(data),
        }
        sh = shutil.which("sh")
        assert sh is not None
        done = subprocess.run(
            [sh, str(HOOK)], env=env, capture_output=True, text=True, check=True
        )
        return done.stdout

    @staticmethod
    def _path(with_uv: Path | None) -> str:
        """The test's PATH without any uv on it, and `with_uv` first."""
        dirs = [
            d
            for d in os.environ.get("PATH", "").split(os.pathsep)
            if d and not (Path(d) / "uv").exists()
        ]
        if with_uv is not None:
            dirs.insert(0, str(with_uv))
        return os.pathsep.join(dirs)

    @staticmethod
    def _fake_uv(into: Path) -> Path:
        into.mkdir()
        uv = into / "uv"
        uv.write_text("#!/bin/sh\nexit 0\n")
        uv.chmod(0o755)
        return into

    def test_the_first_session_hears_what_to_say_and_the_next_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._path(self._fake_uv(Path(tmp) / "bin"))
            first = self._run(Path(tmp) / "data", path)
            self.assertIn("create a project called go2", first)
            self.assertIn("launch_studio", first)
            self.assertEqual(self._run(Path(tmp) / "data", path), "")

    def test_without_uv_every_session_hears_how_to_install_it(self) -> None:
        # The server is `uv run …`; without uv Claude Code said only
        # "Connection closed" (native Windows, 2026-11-18).
        with tempfile.TemporaryDirectory() as tmp:
            path = self._path(None)
            self._run(Path(tmp) / "data", path)  # the first session's line
            again = self._run(Path(tmp) / "data", path)
            self.assertIn("uv, which is not on PATH", again)
            self.assertIn("install.ps1", again)  # Windows
            self.assertIn("install.sh", again)  # macOS and Linux


if __name__ == "__main__":
    unittest.main()
