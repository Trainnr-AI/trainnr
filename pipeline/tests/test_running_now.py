"""The job table holds every running piece of work, whoever started it
(2026-09-25: a gate run from a terminal showed "idle" in the Studio)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from rq_pipeline import mcp_jobs
from rq_pipeline.mcp_jobs import (
    JOB_ID_ENV,
    JOBS_DIR_ENV,
    SOURCE_AGENT,
    SOURCE_DOOR,
    SOURCE_ENV,
    SOURCE_TOOL,
    STATE_DIED,
    STATE_RUNNING,
    JobManager,
    JobRecord,
    read_status,
    track,
)


def _table(root: Path) -> Path:
    return root / mcp_jobs.JOBS_DIR_NAME


class TheTracker(unittest.TestCase):
    def test_a_tool_run_enters_the_table_says_its_stage_and_its_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = _table(Path(tmp))
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(SOURCE_ENV, None)
                os.environ.pop(JOB_ID_ENV, None)
                with track(
                    "gate", jobs_dir=table, name="go2-c2", viewport="deploy:go2-c2"
                ) as t:
                    t.stage("trials")
                    t.progress(7, 20, "trials", "trial 7 of 20")
                    record = JobRecord.read(table / f"{t.job_id}.json")
                    status = read_status(table / f"{t.job_id}{mcp_jobs.STATUS_SUFFIX}")
                    self.assertEqual(record.source, SOURCE_TOOL)
                    self.assertEqual(record.pid, os.getpid())
                    self.assertEqual(record.viewport, "deploy:go2-c2")
                    self.assertFalse((table / f"{t.job_id}.exit").exists())
                    assert status is not None
                    self.assertEqual((status.done, status.total), (7, 20))
                    self.assertEqual(status.stage, "trial 7 of 20")
            self.assertEqual((table / f"{t.job_id}.exit").read_text(), "0")
            self.assertIn("trials", (table / f"{t.job_id}.log").read_text())

    def test_an_agent_says_so_and_an_unknown_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = _table(Path(tmp))
            with mock.patch.dict(os.environ, {SOURCE_ENV: SOURCE_AGENT}):
                with track("attribution", jobs_dir=table) as t:
                    pass
                self.assertEqual(
                    JobRecord.read(table / f"{t.job_id}.json").source, SOURCE_AGENT
                )
            with (
                mock.patch.dict(os.environ, {SOURCE_ENV: "robot"}),
                self.assertRaisesRegex(ValueError, "robot"),
                track("x", jobs_dir=table),
            ):
                pass

    def test_a_failure_and_an_interrupt_record_their_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = _table(Path(tmp))
            for error, code in (
                (RuntimeError("no"), "1"),
                (KeyboardInterrupt(), "130"),
            ):
                with self.assertRaises(type(error)), track("x", jobs_dir=table) as t:
                    raise error
                self.assertEqual((table / f"{t.job_id}.exit").read_text(), code)

    def test_progress_is_written_atomically_and_at_most_so_often(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = _table(Path(tmp))
            with track("x", jobs_dir=table) as t:
                path = table / f"{t.job_id}{mcp_jobs.STATUS_SUFFIX}"
                t.progress(1, 1000, "it")
                first = path.stat().st_mtime_ns
                for i in range(2, 50):
                    t.progress(i, 1000, "it")  # inside STATUS_EVERY_S: not written
                self.assertEqual(json.loads(path.read_text())["done"], 1)
                self.assertEqual(path.stat().st_mtime_ns, first)
                t.progress(1000, 1000, "it")  # the last step always lands
                self.assertEqual(json.loads(path.read_text())["done"], 1000)
                self.assertFalse(list(table.glob(".*.tmp")))


class TheTable(unittest.TestCase):
    def test_a_dead_run_is_said_died_not_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = _table(Path(tmp))
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            JobRecord(
                id="gate-dead0000",
                tool="gate",
                argv=["gate"],
                cwd=tmp,
                log=str(table / "gate-dead0000.log"),
                pid=child.pid,
                started=time.time(),
                source=SOURCE_TOOL,
            ).write(table / "gate-dead0000.json")
            status = JobManager(Path(tmp)).status("gate-dead0000")
            self.assertEqual(status["state"], STATE_DIED)

    def test_a_door_job_and_a_tool_run_are_listed_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = _table(root)
            manager = JobManager(root)
            handle = manager.start(
                "sleep", [sys.executable, "-c", "import time; time.sleep(30)"], root
            )
            try:
                with track("gate", jobs_dir=table, name="go2-c2") as t:
                    t.progress(3, 4, "trials")
                    listed = {row["job_id"]: row for row in manager.list()}
                    self.assertIn(handle["job_id"], listed)
                    self.assertEqual(listed[handle["job_id"]]["source"], SOURCE_DOOR)
                    self.assertEqual(listed[t.job_id]["state"], STATE_RUNNING)
                    self.assertEqual(listed[t.job_id]["done"], 3)
            finally:
                manager.cancel(handle["job_id"])
                manager.join(timeout=10)

    def test_a_door_s_child_adopts_the_door_s_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = _table(Path(tmp))
            JobRecord(
                id="gate-door0000",
                tool="gate",
                argv=["gate"],
                cwd=tmp,
                log=str(table / "gate-door0000.log"),
                pid=os.getpid(),
                started=time.time(),
            ).write(table / "gate-door0000.json")
            env = {JOB_ID_ENV: "gate-door0000", JOBS_DIR_ENV: str(table)}
            with (
                mock.patch.dict(os.environ, env),
                track(
                    "gate", jobs_dir=table, name="go2-c2", viewport="deploy:go2-c2"
                ) as t,
            ):
                t.progress(2, 4, "trials")
            self.assertEqual(t.job_id, "gate-door0000")
            self.assertEqual(len(list(table.glob("*.json"))), 1)
            record = JobRecord.read(table / "gate-door0000.json")
            self.assertEqual(
                (record.name, record.viewport), ("go2-c2", "deploy:go2-c2")
            )
            # the runner, not the child, records a door job's exit
            self.assertFalse((table / "gate-door0000.exit").exists())


if __name__ == "__main__":
    unittest.main()
