"""A run training inside a project shows in the Studio as it goes: its
console log becomes its training record, marked running until the
trainer's done line lands."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.project.kinds import stamp_run
from rq_pipeline.project.live import (
    TRAIN_LOG,
    TRAINING_FILE,
    VERDICT_DIR,
    refresh_project,
    refresh_training,
    refresh_verdicts,
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
        _, run = self._project(LOG + "[g3] done - checkpoints in runs/go2-c1\n")
        record = refresh_training(run)
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "done")

    def test_a_log_with_no_iteration_writes_nothing(self) -> None:
        _, run = self._project("[g3] identity: {}\n")
        self.assertIsNone(refresh_training(run))
        self.assertFalse((run / TRAINING_FILE).exists())


class TheLiveVerdict(unittest.TestCase):
    """A checkpoint certified inside the project shows as an evaluation
    citing a policy, the next tick, and only once."""

    def _run(self) -> tuple[Project, Path]:
        tmp = Path(tempfile.mkdtemp())
        (tmp / "project.json").write_text(
            json.dumps(
                {"schema": "trainnr-project/1", "name": "t", "created": "2026-09-10"}
            )
        )
        run = tmp / "runs" / "go2-c1"
        (run / VERDICT_DIR).mkdir(parents=True)
        (run / "identity.json").write_text(
            json.dumps({"robot": "go2@5003bf617b5f", "actuator": "pd@3b68", "seed": 42})
        )
        (run / "model_1400.pt").write_bytes(b"weights")
        verdict = {
            "source": "go2-walk@83d8a180bfda",
            "policy": "model_1400",
            "successes": 40,
            "trials": 40,
            "ci95": [0.9119, 1.0],
        }
        (run / VERDICT_DIR / "walk-verdict-cuda.json").write_text(json.dumps(verdict))
        (run / VERDICT_DIR / "records-cuda.jsonl").write_text("{}\n")
        return Project(tmp), run

    def test_a_verdict_becomes_a_policy_and_an_evaluation_once(self) -> None:
        project, _ = self._run()
        written = refresh_verdicts(project)
        out = project.root / "certificates" / "go2-c1-model_1400-cuda"
        self.assertEqual(written, [out])
        certificate = json.loads((out / "certificate.json").read_text())
        self.assertEqual(certificate["successes"], 40)
        self.assertEqual(certificate["robot"], "go2@5003bf617b5f")
        self.assertEqual(certificate["task"], "go2-walk@83d8a180bfda")
        self.assertTrue(certificate["policy"].startswith("go2-c1-model_1400@"))
        self.assertTrue(certificate["run"].startswith("go2-c1@"))
        self.assertTrue((out / "records-cuda.jsonl").is_file())
        policy = project.root / "policies" / "go2-c1-model_1400"
        self.assertTrue((policy / "model_1400.pt").is_file())
        manifest = json.loads((policy / "policy.json").read_text())
        self.assertEqual(manifest["iterations"], 1401)
        self.assertEqual(manifest["run"], certificate["run"])
        # The next tick finds nothing new.
        self.assertEqual(refresh_verdicts(project), [])

    def test_a_verdict_without_its_checkpoint_waits(self) -> None:
        project, run = self._run()
        (run / "model_1400.pt").unlink()
        self.assertEqual(refresh_verdicts(project), [])
        self.assertFalse((project.root / "certificates").exists())

    def test_a_landing_checkpoint_does_not_move_the_run_version(self) -> None:
        project, run = self._run()
        before = stamp_run(run)
        (run / "model_1450.pt").write_bytes(b"more weights")
        self.assertEqual(stamp_run(run), before)
        refresh_verdicts(project)
        policy = json.loads(
            (
                project.root / "policies" / "go2-c1-model_1400" / "policy.json"
            ).read_text()
        )
        self.assertEqual(policy["run"], before)
