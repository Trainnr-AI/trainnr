"""The engine the desktop app sets up when no checkout is there (docs/85,
2026-10-11): the runtime archive a release carries, the uv the engine
names to every job, and the names both halves must agree on."""

from __future__ import annotations

import importlib.util
import os
import re
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tomllib

from trainnr import paths

REPO = Path(__file__).resolve().parents[2]
ENGINE_RS = (REPO / "crates" / "trainnr-studio" / "src" / "engine.rs").read_text(
    encoding="utf-8"
)
SPAWN_RS = (REPO / "crates" / "trainnr-studio" / "src" / "spawn.rs").read_text(
    encoding="utf-8"
)


def _runtime_archive():
    spec = importlib.util.spec_from_file_location(
        "runtime_archive", REPO / "tools" / "runtime-archive.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constant(source: str, pattern: str) -> str:
    found = re.findall(pattern, source, flags=re.MULTILINE)
    if len(found) != 1:
        raise AssertionError(f"{pattern!r} matched {len(found)} times")
    return found[0]


class TheRuntimeArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _runtime_archive()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        cls.archive = cls.tool.build("9.9.9", cls.out / "a")
        cls.again = cls.tool.build("9.9.9", cls.out / "b")
        with tarfile.open(cls.archive) as tar:
            cls.names = tar.getnames()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_it_holds_what_a_job_runs_and_no_tests(self) -> None:
        top = "trainnr-runtime-v9.9.9/"
        self.assertTrue(all(n.startswith(top) for n in self.names))
        for needed in (
            "trainnr/pyproject.toml",
            "trainnr/uv.lock",
            "trainnr/trainnr/mcp_server.py",
            "trainnr-mjlab/pyproject.toml",
            "tools/accept-task.py",
            "tools/studio-present.py",
            "robots/actuators",
            "LICENSE",
            "NOTICE",
        ):
            with self.subTest(needed=needed):
                self.assertTrue(any(n.startswith(top + needed) for n in self.names))
        self.assertFalse([n for n in self.names if "/tests/" in n])

    def test_the_same_commit_gives_the_same_bytes_and_its_digest(self) -> None:
        self.assertEqual(self.archive.read_bytes(), self.again.read_bytes())
        listed = Path(f"{self.archive}.sha256").read_text().split()
        self.assertEqual(listed[1], self.archive.name)
        import hashlib  # noqa: PLC0415

        self.assertEqual(
            listed[0], hashlib.sha256(self.archive.read_bytes()).hexdigest()
        )

    def test_it_stays_lean(self) -> None:
        """About 9 MB on 2026-10-11; a doubling is a decision, not a drift."""
        self.assertLess(self.archive.stat().st_size, 20_000_000)


class TheEnginesUv(unittest.TestCase):
    def test_jobs_run_the_uv_the_engine_names(self) -> None:
        from trainnr.mcp_actions import _uv  # noqa: PLC0415
        from trainnr.mcp_jobs import uv_prepare_argv  # noqa: PLC0415

        with mock.patch.dict(os.environ, {paths.UV_ENV: "/engine/bin/uv"}):
            argv = _uv(Path("/x"), "sim")
            self.assertEqual(argv[0], "/engine/bin/uv")
            prepared = uv_prepare_argv([*argv, "tool.py"])
            self.assertEqual(prepared[0], "/engine/bin/uv")
        with mock.patch.dict(os.environ, {paths.UV_ENV: ""}):
            self.assertEqual(paths.uv_program(), "uv")


class BothHalvesAgree(unittest.TestCase):
    def test_the_names_the_studio_and_python_share(self) -> None:
        self.assertEqual(
            constant(ENGINE_RS, r'^pub const UV_ENV: &str = "([^"]+)";'), paths.UV_ENV
        )
        self.assertEqual(
            constant(SPAWN_RS, r'^pub const REPO_MARKER: &str = "([^"]+)";'),
            paths.CHECKOUT_MARKER.as_posix(),
        )
        self.assertEqual(
            constant(ENGINE_RS, r'^const RUNTIME_NAME: &str = "([^"]+)";'),
            _runtime_archive().NAME,
        )

    def test_the_core_extras_exist(self) -> None:
        core = re.findall(
            r'"(\w+)"',
            constant(ENGINE_RS, r"^pub const CORE_EXTRAS: &\[&str\] = &\[([^\]]+)\];"),
        )
        pyproject = tomllib.loads((REPO / "trainnr" / "pyproject.toml").read_text())
        self.assertLessEqual(
            set(core), set(pyproject["project"]["optional-dependencies"])
        )
        self.assertIn(
            "mcp", core, "the buttons run trainnr call, which needs the server"
        )


if __name__ == "__main__":
    unittest.main()
