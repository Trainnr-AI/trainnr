"""A job's exit code lands whoever launched it: the runner wrapping
the tool writes the `.exit` file itself, so a door called from a
script that returned still leaves the code the Studio reads."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from rq_pipeline.mcp_jobs import JobManager, exit_path_for, runner_argv

WAIT_S = 20.0


class TheRunner(unittest.TestCase):
    def test_the_exit_code_is_recorded_by_the_runner_itself(self) -> None:
        root = Path(tempfile.mkdtemp())
        jobs = JobManager(root)
        handle = jobs.start(
            "fake", [sys.executable, "-c", "print('hi'); raise SystemExit(3)"], root
        )
        exit_file = exit_path_for(Path(handle["log"]))
        deadline = time.time() + WAIT_S
        while not exit_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        jobs.join(WAIT_S)
        self.assertEqual(exit_file.read_text(), "3")
        self.assertIn("hi", Path(handle["log"]).read_text())
        self.assertEqual(jobs.status(handle["job_id"])["state"], "failed (exit 3)")

    def test_the_runner_command_line(self) -> None:
        argv = runner_argv(["uv", "run", "x"], Path("/j/a.exit"))
        self.assertEqual(
            argv[1:],
            [
                "-m",
                "rq_pipeline.mcp_jobs",
                "--exit-file",
                "/j/a.exit",
                "--",
                "uv",
                "run",
                "x",
            ],
        )


class TheEnvironmentIsReadyBeforeTheJob(unittest.TestCase):
    """A door launches `uv run --no-sync`; the sync runs once, under a
    per-project lock, before the job (four launches raced on 2026-09-25)."""

    def test_the_prepare_line_is_the_same_uv_run_with_the_sync_on(self) -> None:
        from rq_pipeline.mcp_jobs import uv_prepare_argv  # noqa: PLC0415

        launch = [
            "uv",
            "run",
            "--no-sync",
            "--project",
            "/p",
            "--extra",
            "sim",
            "python",
            "-m",
            "x",
            "--k",
        ]
        self.assertEqual(
            uv_prepare_argv(launch),
            ["uv", "run", "--project", "/p", "--extra", "sim", "python", "-c", "pass"],
        )
        self.assertIsNone(uv_prepare_argv(["uv", "run", "--project", "/p", "python"]))
        self.assertIsNone(uv_prepare_argv(["/box/wsl-run.sh", "cargo", "run"]))

    def test_one_lock_per_project(self) -> None:
        from rq_pipeline.mcp_jobs import uv_lock_path, uv_project  # noqa: PLC0415

        here = Path("/p")
        a = uv_lock_path(
            ["uv", "run", "--no-sync", "--project", "/p/a", "python"], here
        )
        self.assertEqual(
            a,
            uv_lock_path(["uv", "run", "--no-sync", "--project=/p/a", "python"], here),
        )
        self.assertEqual(
            a, uv_lock_path(["uv", "run", "--project", "a", "python"], here)
        )
        self.assertNotEqual(
            a,
            uv_lock_path(
                ["uv", "run", "--no-sync", "--project", "/p/b", "python"], here
            ),
        )
        # no --project: the directory the command runs in
        self.assertEqual(uv_project(["uv", "run", "python"], here), here.resolve())

    def test_the_lock_admits_one_holder_at_a_time(self) -> None:
        import threading  # noqa: PLC0415
        import time  # noqa: PLC0415
        from tempfile import TemporaryDirectory  # noqa: PLC0415

        from rq_pipeline.mcp_jobs import exclusive  # noqa: PLC0415

        inside, most = [0], [0]
        guard = threading.Lock()

        def hold(path: Path) -> None:
            with exclusive(path):
                with guard:
                    inside[0] += 1
                    most[0] = max(most[0], inside[0])
                time.sleep(0.05)
                with guard:
                    inside[0] -= 1

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.lock"
            threads = [threading.Thread(target=hold, args=(path,)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(most[0], 1)

    def test_a_non_uv_launch_is_not_prepared(self) -> None:
        from rq_pipeline.mcp_jobs import prepare_uv  # noqa: PLC0415

        prepare_uv(["cargo", "run", "--release"], Path())  # returns, runs nothing


@unittest.skipIf(os.name == "nt", "the stand-in uv is a POSIX shell script")
class ThePrepareRuns(unittest.TestCase):
    """prepare_uv against a stand-in `uv` on PATH: its output reaches this
    process's own (the job's log), a failure and a missing uv are refused
    by name, and the runner records a refused prepare as the job's exit."""

    ARGV = ("uv", "run", "--no-sync", "--project", ".", "python", "-m", "x")

    def stand_in(self, tmp: Path, code: int) -> dict[str, str]:
        uv = tmp / "uv"
        uv.write_text(f'#!/bin/sh\necho stand-in uv "$@"\nexit {code}\n')
        uv.chmod(0o755)
        return {**os.environ, "PATH": f"{tmp}{os.pathsep}{os.environ['PATH']}"}

    def test_a_good_sync_passes_and_a_bad_one_is_refused(self) -> None:
        from unittest import mock  # noqa: PLC0415

        from rq_pipeline.mcp_jobs import (  # noqa: PLC0415
            EnvironmentNotReadyError,
            prepare_uv,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, self.stand_in(Path(tmp), 0)):
                prepare_uv(self.ARGV, Path(tmp))
            with (
                mock.patch.dict(os.environ, self.stand_in(Path(tmp), 7)),
                self.assertRaisesRegex(EnvironmentNotReadyError, "exit 7"),
            ):
                prepare_uv(self.ARGV, Path(tmp))
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                (Path(tmp) / "uv").unlink()
                with self.assertRaisesRegex(EnvironmentNotReadyError, "not on PATH"):
                    prepare_uv(self.ARGV, Path(tmp))

    def test_the_runner_records_a_refused_prepare(self) -> None:
        from unittest import mock  # noqa: PLC0415

        from rq_pipeline.mcp_jobs import PREPARE_FAILED, main  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            exit_file = Path(tmp) / "job.exit"
            with (
                mock.patch.dict(os.environ, self.stand_in(Path(tmp), 1)),
                mock.patch("os.getcwd", return_value=tmp),
            ):
                code = main(["--exit-file", str(exit_file), "--", *self.ARGV])
            self.assertEqual(code, PREPARE_FAILED)
            self.assertEqual(exit_file.read_text().strip(), str(PREPARE_FAILED))


class TheLockGivesUp(unittest.TestCase):
    def test_a_held_lock_ends_in_a_refusal_by_name(self) -> None:
        import threading  # noqa: PLC0415

        from rq_pipeline.mcp_jobs import (  # noqa: PLC0415
            EnvironmentNotReadyError,
            exclusive,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.lock"
            held, release = threading.Event(), threading.Event()

            def holder() -> None:
                with exclusive(path):
                    held.set()
                    release.wait(5)

            thread = threading.Thread(target=holder)
            thread.start()
            held.wait(5)
            try:
                with (
                    self.assertRaisesRegex(EnvironmentNotReadyError, "still held"),
                    exclusive(path, wait_s=0.3),
                ):
                    pass
            finally:
                release.set()
                thread.join()
