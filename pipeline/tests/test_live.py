"""A run training inside a project shows in the Studio as it goes: its
console log becomes its training record, marked running until the
trainer's done line lands."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.project.live import (
    TRAIN_LOG,
    TRAINING_FILE,
    refresh_project,
    refresh_training,
    run_folders,
    stale,
)
from rq_pipeline.project.locate import Project

LOG = """[g3] 4096 envs on cuda:0, 8000 iterations
                           Learning iteration 0/8000
                            Mean reward: -0.89
                         Iteration time: 0.95s
                           Learning iteration 1/8000
                            Mean reward: 12.5
                         Iteration time: 1.51s
"""


class TheLiveRecord(unittest.TestCase):
    def _project(self, log: str) -> tuple[Project, Path]:
        tmp = Path(tempfile.mkdtemp())
        (tmp / "project.json").write_text(
            json.dumps(
                {"schema": "trainnr-project/1", "name": "t", "created": "2026-09-10"}
            )
        )
        run = tmp / "runs" / "go2-c1"
        run.mkdir(parents=True)
        (run / TRAIN_LOG).write_text(log)
        return Project(tmp), run

    def test_a_growing_log_becomes_a_running_record_and_refreshes_once(self) -> None:
        project, run = self._project(LOG)
        self.assertEqual(run_folders(project), [run])
        self.assertTrue(stale(run))
        self.assertEqual(refresh_project(project), [run])
        record = json.loads((run / TRAINING_FILE).read_text())
        self.assertEqual(record["status"], "running")
        self.assertEqual(record["iterations"], 8000)
        self.assertEqual(record["iterations_logged"], 2)
        self.assertEqual(record["final"]["reward"], 12.5)
        # Nothing grew: no second refresh.
        self.assertFalse(stale(run))
        self.assertEqual(refresh_project(project), [])

    def test_the_done_line_closes_the_record(self) -> None:
        project, run = self._project(LOG + "[g3] done - checkpoints in runs/go2-c1\n")
        record = refresh_training(run)
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "done")

    def test_a_log_with_no_iteration_writes_nothing(self) -> None:
        project, run = self._project("[g3] identity: {}\n")
        self.assertIsNone(refresh_training(run))
        self.assertFalse((run / TRAINING_FILE).exists())
