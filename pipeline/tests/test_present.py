"""The presenters: every kind the index can hold streams into a
recording as itself, and a kind with nothing to show says so by name."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.project import Kind, create_project, index_project
from rq_pipeline.project.present import _PRESENTERS, present

try:
    import rerun as rr
except ImportError:  # pragma: no cover - the viz extra is absent
    rr = None


def _memory_stream() -> object:
    assert rr is not None
    stream = rr.RecordingStream("trainnr-test")
    stream.memory_recording()
    return stream


class Presenters(unittest.TestCase):
    def setUp(self) -> None:
        if rr is None:  # pragma: no cover - the viz extra is absent
            self.skipTest("rerun not installed")

    def _project(self, tmp: Path):
        project = create_project(tmp / "p", "p")
        run = project.root / "runs" / "a"
        run.mkdir(parents=True)
        (run / "identity.json").write_text(
            json.dumps({"robot": "microduck@14b51b63c52e", "seed": 1})
        )
        (run / "training.json").write_text(
            json.dumps(
                {
                    "schema": "trainnr-training/1",
                    "trainer": "rsl_rl",
                    "columns": ["iteration", "reward", "episode_length"],
                    "curve": [[0, -1.0, 20.0], [10, 5.0, None], [20, 9.0, 300.0]],
                    "final": {"reward": 9.0},
                }
            )
        )
        (run / "model_1.pt").write_bytes(b"\x00")
        policy = project.root / "policies" / "a"
        policy.mkdir(parents=True)
        (policy / "model_1.pt").write_bytes(b"\x00")
        (policy / "policy.json").write_text(
            json.dumps({"schema": "trainnr-policy/1", "iterations": 20})
        )
        findings = project.root / "findings"
        findings.mkdir(parents=True, exist_ok=True)
        (findings / "f.json").write_text(
            json.dumps(
                {
                    "id": "f-2026-09-09",
                    "date": "2026-09-09",
                    "claim": "Two arms differ. The rest is detail.",
                    "outcome": {
                        "arms": {
                            "point": {"successes": 3, "trials": 4},
                            "wide": {"successes": 1, "trials": 4},
                        }
                    },
                    "caveats": ["small n"],
                }
            )
        )
        return project

    def test_every_index_kind_has_a_presenter_or_is_named(self) -> None:
        missing = {k.value for k in Kind} - {k.value for k in _PRESENTERS}
        # Fits ride in bundles: a robot's drawer shows them.
        self.assertEqual(missing, {"fit"})

    def test_rl_run_policy_and_finding_present_as_themselves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(Path(tmp))
            index = index_project(project)
            stream = _memory_stream()
            by_kind = {a.kind: a for a in index.artifacts}
            run = _PRESENTERS[Kind.RUN](project, by_kind["run"], stream)
            self.assertIn("run/reward", run["paths"])
            self.assertIn("run/episode_length", run["paths"])
            self.assertEqual(run["view"], "time series")
            policy = _PRESENTERS[Kind.POLICY](project, by_kind["policy"], stream)
            self.assertIn("policy/facts", policy["paths"])
            self.assertEqual(policy["view"], "evaluations")  # no robot in this project
            finding = _PRESENTERS[Kind.FINDING](project, by_kind["finding"], stream)
            self.assertEqual(finding["paths"], ["finding/reading", "finding/outcome"])
            self.assertEqual(finding["view"], "outcome + reading")

    def test_an_unknown_stamp_and_an_unshowable_kind_are_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = self._project(Path(tmp))
            with self.assertRaises(KeyError):
                present(project, "ghost@000000000000")
