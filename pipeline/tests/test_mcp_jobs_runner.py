"""A job's exit code lands whoever launched it: the runner wrapping
the tool writes the `.exit` file itself, so a door called from a
script that returned still leaves the code the Studio reads."""

from __future__ import annotations

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
