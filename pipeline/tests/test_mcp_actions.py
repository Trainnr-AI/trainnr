"""S2's doors, against a fake spawner — the command lines ARE the
contract (docs/64 §6: "each with a test that fakes the heavy call"),
plus the job manager's lifecycle and the onboarding refusals.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rq_pipeline.mcp_actions import (
    PIPELINE_DIR,
    RQ_MJLAB_DIR,
    STUDIO_DIR,
    TOOLS_DIR,
    TRAIN_PYTHON,
    Actions,
)
from rq_pipeline.mcp_jobs import JobManager


class _FakeProcess:
    def __init__(self, pid: int = 4242, code: int = 0) -> None:
        self.pid = pid
        self._code = code

    def wait(self) -> int:
        return self._code


class _FakeSpawner:
    """Records every spawn; touches the log so status can tail it."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, argv, cwd, log_path) -> _FakeProcess:
        self.calls.append((list(argv), Path(cwd)))
        Path(log_path).write_text("line one\nline two\n")
        return _FakeProcess()


def harness(tmp: str) -> tuple[Actions, _FakeSpawner]:
    spawner = _FakeSpawner()
    return Actions(JobManager(Path(tmp), spawner=spawner)), spawner


UV_PIPELINE = ["uv", "run", "--project", str(PIPELINE_DIR)]
UV_MJLAB = ["uv", "run", "--project", str(RQ_MJLAB_DIR), "python"]


class TheDoors(unittest.TestCase):
    def test_generate_demos_spawns_the_kitting_press_verbatim(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, spawner = harness(tmp)
            handle = actions.generate_demos(episodes=3, seed=7, out="runs/x")
            [(argv, cwd)] = spawner.calls
            self.assertEqual(
                argv,
                [
                    *UV_PIPELINE,
                    "--extra",
                    "sim",
                    "python",
                    str(TOOLS_DIR / "kitting-demos.py"),
                    "3",
                    "runs/x",
                    "--seed",
                    "7",
                ],
            )
            self.assertEqual(cwd, PIPELINE_DIR)
            self.assertTrue(str(handle["job_id"]).startswith("generate-demos-"))

    def test_zero_episodes_refuses_before_spawning(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, spawner = harness(tmp)
            with self.assertRaises(ValueError):
                actions.generate_demos(episodes=0)
            self.assertEqual(spawner.calls, [])

    def test_the_chain_runs_through_the_train_venv(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, spawner = harness(tmp)
            actions.run_chain(name="demo", episodes=2, steps=300, from_stage="train")
            [(argv, _cwd)] = spawner.calls
            self.assertEqual(argv[0], str(TRAIN_PYTHON))
            self.assertEqual(argv[1], str(TOOLS_DIR / "e2e-smoke.py"))
            self.assertIn("--from", argv)
            self.assertEqual(argv[argv.index("--name") + 1], "demo")

    def test_the_walk_trains_and_certifies_in_the_rq_mjlab_venv(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, spawner = harness(tmp)
            actions.train_walk(agent="smoke", iterations=5)
            actions.certify_walk("runs/x/model_100.pt", trials=8, device="cpu")
            (train_argv, train_cwd), (cert_argv, _c) = spawner.calls
            self.assertEqual(
                train_argv,
                [
                    *UV_MJLAB,
                    "-m",
                    "rq_mjlab.walk_train",
                    "--agent",
                    "smoke",
                    "--iterations",
                    "5",
                ],
            )
            self.assertEqual(train_cwd, RQ_MJLAB_DIR)
            self.assertEqual(
                cert_argv[cert_argv.index("-m") + 1], "rq_mjlab.walk_verdict"
            )
            self.assertIn("--device", cert_argv)

    def test_the_studio_launches_release_in_its_crate(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, spawner = harness(tmp)
            actions.open_studio()
            [(argv, cwd)] = spawner.calls
            self.assertEqual(argv, ["cargo", "run", "--release"])
            self.assertEqual(cwd, STUDIO_DIR)


class TheJobLifecycle(unittest.TestCase):
    def test_status_reports_done_with_the_log_tail(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, _spawner = harness(tmp)
            handle = actions.open_studio()
            status = actions.job_status(str(handle["job_id"]))
            # The fake process exits 0 instantly; the watcher thread
            # records it (poll once — daemon thread scheduling).
            for _ in range(50):
                status = actions.job_status(str(handle["job_id"]))
                if status["state"] != "running":
                    break
            self.assertEqual(status["state"], "done")
            self.assertEqual(status["log_tail"], ["line one", "line two"])

    def test_an_unknown_job_is_refused_naming_the_known(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, _spawner = harness(tmp)
            actions.open_studio()
            with self.assertRaises(KeyError) as ctx:
                actions.job_status("nope-123")
            self.assertIn("studio-", str(ctx.exception))

    def test_list_is_newest_first(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, _spawner = harness(tmp)
            first = actions.open_studio()
            second = actions.train_walk()
            listed = actions.list_jobs()
            self.assertEqual(
                [job["job_id"] for job in listed],
                [second["job_id"], first["job_id"]],
            )


class Onboarding(unittest.TestCase):
    def test_a_missing_mjcf_is_refused_by_path(self) -> None:
        with TemporaryDirectory() as tmp:
            actions, _spawner = harness(tmp)
            with self.assertRaises(FileNotFoundError):
                actions.onboard_robot(f"{tmp}/ghost.xml", "ghost")

    def test_an_existing_bundle_is_never_overwritten(self) -> None:
        actions, _spawner = harness("/tmp")
        with self.assertRaises(FileExistsError) as ctx:
            actions.onboard_robot(__file__, "microduck")  # exists in robots/
        self.assertIn("never overwrites", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
