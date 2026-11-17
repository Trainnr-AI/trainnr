"""Video to a G1 motion, as the tools refuse it (docs/86 §5.3): nothing
starts without GEM-X and the retargeter installed, without the footage,
or over an existing recording; tracking takes only a reference, and a
tracking task is accepted only on one."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests._extras import needs_mcp
from trainnr.project import create_project
from trainnr.project.locate import PROJECT_ENV
from trainnr.project.task_ref import declare_task, read_task_reference
from trainnr.tasks.tracking import TrackingSpec
from trainnr.video import nvidia

TOOLS = Path(__file__).resolve().parents[2] / "tools"


class TheInstalls(unittest.TestCase):
    def test_a_missing_install_names_the_file_and_the_setup_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {nvidia.GEMX_ENV: tmp, nvidia.SOMA_ENV: tmp}
            with mock.patch.dict(os.environ, env):
                why = nvidia.missing()
        assert why is not None
        self.assertIn("GEM-X is not installed", why)
        self.assertIn(nvidia.SETUP, why)

    def test_both_installs_complete_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for root, needs in (("g", nvidia.GEMX_NEEDS), ("s", nvidia.SOMA_NEEDS)):
                for need in needs:
                    (Path(tmp) / root / need).parent.mkdir(parents=True, exist_ok=True)
                    (Path(tmp) / root / need).touch()
            env = {nvidia.GEMX_ENV: f"{tmp}/g", nvidia.SOMA_ENV: f"{tmp}/s"}
            with mock.patch.dict(os.environ, env):
                self.assertIsNone(nvidia.missing())


@needs_mcp
class TheTools(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = create_project(Path(self._tmp.name) / "p", "p")
        env = mock.patch.dict(os.environ, {PROJECT_ENV: str(self.project.root)})
        env.start()
        self.addCleanup(env.stop)

    def test_unknown_footage_is_refused_by_name(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        answer = mcp_server.retarget_video("no-such-clip")
        self.assertEqual(answer["status"], "refused")
        self.assertIn("no-such-clip", answer["reason"])

    def test_tracking_takes_a_reference_not_any_recording(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        folder = self.project.recordings / "plain"
        folder.mkdir(parents=True)
        (folder / "recording.json").write_text(
            json.dumps(
                {
                    "schema": "trainnr-recording/1",
                    "source": "x",
                    "adapter": "wire",
                    "duration_s": 0.0,
                    "channels": [],
                }
            )
        )
        answer = mcp_server.train_tracking("plain")
        self.assertEqual(answer["status"], "refused")
        self.assertIn("retarget_video", answer["reason"])
        self.assertEqual(
            mcp_server.train_tracking("plain", iterations=0)["status"], "refused"
        )


def _recording(project: Any, name: str, *, video: bool) -> Path:
    folder = project.recordings / name
    (folder / "raw").mkdir(parents=True)
    record = {
        "schema": "trainnr-recording/1",
        "source": "x",
        "adapter": "wire",
        "duration_s": 0.0,
        "channels": [],
    }
    if video:
        record["collection"] = "video"
        (folder / "raw" / "motion.npz").touch()
    (folder / "recording.json").write_text(json.dumps(record))
    return folder


def _accept_task_module() -> Any:
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if str(TOOLS) not in sys.path:  # the tools import their `_lab` neighbour
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "check_task", TOOLS / "accept-task.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheTrackingTask(unittest.TestCase):
    """The tracking family (docs/86 §5.2): declared over a reference, and
    accepted only when the reference is one and a few iterations run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = create_project(Path(self._tmp.name) / "p", "p")

    def test_the_spec_refuses_no_reference_and_a_lag_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs a reference"):
            declare_task(self.project, "g1-tracking", "t", {})
        with self.assertRaisesRegex(ValueError, "lag_ms"):
            TrackingSpec(reference="r", lag_ms=500)

    def test_a_recording_not_made_from_video_is_rejected_before_the_smoke(
        self,
    ) -> None:
        _recording(self.project, "plain", video=False)
        declare_task(self.project, "g1-tracking", "t", {"reference": "plain"})
        ref = read_task_reference(self.project, "t")
        calls: list[Any] = []
        accepted = _accept_task_module().review_tracking(
            self.project,
            ref.folder,
            ref,
            run=lambda argv, **kw: calls.append(argv),
            prepare=lambda argv, cwd: None,
        )
        self.assertFalse(accepted)
        self.assertEqual(calls, [])
        record = json.loads((ref.folder / "acceptance.json").read_text())
        self.assertIn("not made from video", " ".join(record["reasons"]))
        self.assertIn("raw/motion.npz", " ".join(record["reasons"]))

    def test_a_reference_runs_the_trainers_line_and_cites_the_task(self) -> None:
        _recording(self.project, "dance", video=True)
        declare_task(self.project, "g1-tracking", "t", {"reference": "dance"})
        ref = read_task_reference(self.project, "t")
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            calls.append(list(argv))
            kwargs["stdout"].write("[tracking] done in 9 s\n")
            return subprocess.CompletedProcess(argv, 0)

        accepted = _accept_task_module().review_tracking(
            self.project, ref.folder, ref, run=fake_run, prepare=lambda a, c: None
        )
        self.assertTrue(accepted)
        [argv] = calls
        trainer = argv[argv.index("-m") :]
        self.assertEqual(trainer[1], "trainnr_mjlab.tracking_train")
        self.assertEqual(trainer[trainer.index("--task-stamp") + 1], ref.stamp)
        self.assertTrue(
            trainer[trainer.index("--reference-stamp") + 1].startswith("dance@")
        )
        self.assertIn("--no-recorder", trainer)
        # the smoke's run folder is not left in the project as an experiment
        log_dir = Path(trainer[trainer.index("--log-dir") + 1])
        self.assertFalse(log_dir.is_relative_to(self.project.root))


@needs_mcp
class TheTrackingEvaluation(unittest.TestCase):
    """`evaluate_walk` on a tracking run judges it under its tracking task:
    the task's pushes, lag and trials, on the run's reference."""

    def setUp(self) -> None:
        from trainnr.templates import TRACKING_TASK  # noqa: PLC0415

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = create_project(Path(self._tmp.name) / "p", "p")
        env = mock.patch.dict(os.environ, {PROJECT_ENV: str(self.project.root)})
        env.start()
        self.addCleanup(env.stop)
        _recording(self.project, "dance", video=True)
        self.run_dir = self.project.root / "runs" / "dance-run"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "model_9.pt").touch()
        self.identity = {"task": TRACKING_TASK, "reference": "dance"}
        self._write_identity()

    def _write_identity(self) -> None:
        (self.run_dir / "identity.json").write_text(json.dumps(self.identity))

    def _evaluate(self, **kwargs: Any) -> tuple[dict[str, Any], list[dict]]:
        from trainnr import mcp_server  # noqa: PLC0415

        calls: list[dict] = []

        def fake(_self: Any, checkpoint: str, **kw: Any) -> dict[str, Any]:
            calls.append({"checkpoint": checkpoint, **kw})
            return {"job_id": "evaluate-tracking-x", "status": "started"}

        with mock.patch("trainnr.mcp_actions.Actions.evaluate_tracking", fake):
            answer = mcp_server.evaluate_walk("dance-run/model_9", **kwargs)
        return answer, calls

    def test_the_task_over_the_reference_sets_the_conditions(self) -> None:
        declare_task(
            self.project,
            "g1-tracking",
            "dance-tracking",
            {"reference": "dance", "lag_ms": 40, "pushes": False, "trials": 12},
        )
        answer, [call] = self._evaluate()
        self.assertEqual(answer["job_id"], "evaluate-tracking-x")
        self.assertEqual((call["lag_ms"], call["pushes"]), (40, False))
        self.assertEqual(call["trials"], 12)  # the task's, the call left 40
        self.assertTrue(call["task_stamp"].startswith("g1-tracking@"))
        self.assertTrue(call["reference_stamp"].startswith("dance@"))
        _, [named] = self._evaluate(trials=8)
        self.assertEqual(named["trials"], 8)

    def test_walk_arguments_and_two_candidate_tasks_are_refused(self) -> None:
        answer, calls = self._evaluate(robot="go2")
        self.assertEqual(answer["status"], "refused")
        self.assertIn("robot", answer["reason"])
        for name in ("a", "b"):
            declare_task(self.project, "g1-tracking", name, {"reference": "dance"})
        answer, calls = self._evaluate()
        self.assertEqual(answer["status"], "refused")
        self.assertIn("a, b", answer["reason"])
        self.assertEqual(calls, [])
        # a run that cites its task is judged under it, whatever else exists
        self.identity["tracking_task"] = read_task_reference(self.project, "b").stamp
        self._write_identity()
        _, [call] = self._evaluate()
        self.assertEqual(call["task_stamp"], self.identity["tracking_task"])


if __name__ == "__main__":
    unittest.main()
