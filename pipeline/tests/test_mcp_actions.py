"""S2's doors, against a fake spawner — the command lines ARE the
contract (docs/64 §6: "each with a test that fakes the heavy call"),
plus the job manager's lifecycle and the onboarding refusals.
"""

from __future__ import annotations

import os
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from rq_pipeline.mcp_actions import (
    PIPELINE_DIR,
    RQ_MJLAB_DIR,
    STUDIO_DIR,
    TOOLS_DIR,
    TRAIN_PYTHON,
    WSL_RUN,
    Actions,
)
from rq_pipeline.mcp_jobs import JobManager


class _FakeProcess:
    def __init__(self, pid: int = os.getpid(), code: int = 0) -> None:
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


@contextmanager
def harness(env_file: Path | None = None) -> Iterator[tuple[Actions, _FakeSpawner]]:
    """A fake-spawned Actions over a temp runs root. env_file=None: the
    pins below are the PLATFORM-FREE command lines; the environment's
    own wrapping has its own test. The watchers are JOINED before the
    tempdir goes — a fake process exits instantly, and its watcher was
    still writing the exit file while rmtree ran (2026-09-02)."""
    with TemporaryDirectory() as tmp:
        spawner = _FakeSpawner()
        jobs = JobManager(Path(tmp), spawner=spawner)
        try:
            yield Actions(jobs, env_file=env_file), spawner
        finally:
            jobs.join()


UV_PIPELINE = ["uv", "run", "--project", str(PIPELINE_DIR)]
UV_MJLAB = ["uv", "run", "--project", str(RQ_MJLAB_DIR), "python"]


class TheDoors(unittest.TestCase):
    def test_generate_demos_spawns_the_kitting_press_verbatim(self) -> None:
        with harness() as (actions, spawner):
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

    def test_accept_task_reviews_a_declared_task_in_its_project(self) -> None:
        with harness() as (actions, spawner):
            handle = actions.accept_task("tray-far", "/p/aloha")
            [(argv, cwd)] = spawner.calls
            self.assertEqual(
                argv,
                [
                    *UV_PIPELINE,
                    "--extra",
                    "sim",
                    "python",
                    str(TOOLS_DIR / "accept-task.py"),
                    "--project",
                    "/p/aloha",
                    "--name",
                    "tray-far",
                ],
            )
            self.assertEqual(cwd, PIPELINE_DIR)
            self.assertTrue(str(handle["job_id"]).startswith("accept-task-"))
            with self.assertRaises(ValueError):
                actions.accept_task("a/b", "/p")

    def test_press_planned_spawns_the_planner_press_on_the_task(self) -> None:
        with harness() as (actions, spawner):
            handle = actions.press_planned("block_stack", episodes=4, seed=9)
            [(argv, cwd)] = spawner.calls
            self.assertEqual(
                argv,
                [
                    *UV_PIPELINE,
                    "--extra",
                    "sim",
                    "--extra",
                    "viz",
                    "python",
                    str(TOOLS_DIR / "planner-demos.py"),
                    "block_stack",
                    "--episodes",
                    "4",
                    "--seed",
                    "9",
                    "--dr-span",
                    "0.0",
                ],
            )
            self.assertEqual(cwd, PIPELINE_DIR)
            self.assertTrue(str(handle["job_id"]).startswith("press-planned-"))

    def test_press_planned_shards_only_when_asked(self) -> None:
        with harness() as (actions, spawner):
            actions.press_planned("lift", shards=4)
            [(argv, _)] = spawner.calls
            self.assertEqual(argv[-2:], ["--shards", "4"])

    def test_zero_episodes_refuses_before_spawning(self) -> None:
        with harness() as (actions, spawner):
            with self.assertRaises(ValueError):
                actions.generate_demos(episodes=0)
            self.assertEqual(spawner.calls, [])

    def test_the_chain_runs_through_the_train_venv(self) -> None:
        with harness() as (actions, spawner):
            actions.run_chain(name="demo", episodes=2, steps=300, from_stage="train")
            [(argv, _cwd)] = spawner.calls
            self.assertEqual(argv[0], str(TRAIN_PYTHON))
            self.assertEqual(argv[1], str(TOOLS_DIR / "e2e-smoke.py"))
            self.assertIn("--from", argv)
            self.assertEqual(argv[argv.index("--name") + 1], "demo")

    def test_the_walk_trains_and_certifies_in_the_rq_mjlab_venv(self) -> None:
        with harness() as (actions, spawner):
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
        with harness() as (actions, spawner):
            actions.open_studio()
            [(argv, cwd)] = spawner.calls
            self.assertEqual(argv, ["cargo", "run", "--release"])
            self.assertEqual(cwd, STUDIO_DIR)


class TheStudentCertificate(unittest.TestCase):
    def test_a_student_rides_the_same_door_with_its_horizon(self) -> None:
        with harness() as (actions, spawner):
            actions.certify_walk(
                "runs/x/model_1.pt",
                trials=8,
                student="runs/s/pretrained_model",
                horizon=10,
            )
            [(argv, _)] = spawner.calls
            self.assertEqual(
                argv[-4:], ["--student", "runs/s/pretrained_model", "--horizon", "10"]
            )


class ThePressWalkDoor(unittest.TestCase):
    def test_press_walk_rolls_the_newest_checkpoint_by_default(self) -> None:
        with harness() as (actions, spawner):
            actions.press_walk(episodes=4, worlds=3, seed=9, out="runs/w")
            [(argv, cwd)] = spawner.calls
            self.assertEqual(
                argv,
                [
                    *UV_MJLAB,
                    "-m",
                    "rq_mjlab.walk_press",
                    "--latest",
                    "--out",
                    "runs/w",
                    "--episodes",
                    "4",
                    "--worlds",
                    "3",
                    "--seed",
                    "9",
                ],
            )
            self.assertEqual(cwd, RQ_MJLAB_DIR)


class TheJobLifecycle(unittest.TestCase):
    def test_status_reports_done_with_the_log_tail(self) -> None:
        with harness() as (actions, _spawner):
            handle = actions.open_studio()
            # The fake process exits 0 instantly; join the watcher rather
            # than poll (a poll saw "ended (unrecorded)" first whenever
            # the fake pid was dead on the box, 2026-09-02).
            actions.jobs.join()
            status = actions.job_status(str(handle["job_id"]))
            self.assertEqual(status["state"], "done")
            self.assertEqual(status["log_tail"], ["line one", "line two"])

    def test_an_unknown_job_is_refused_naming_the_known(self) -> None:
        with harness() as (actions, _spawner):
            actions.open_studio()
            with self.assertRaises(KeyError) as ctx:
                actions.job_status("nope-123")
            self.assertIn("studio-", str(ctx.exception))

    def test_list_is_newest_first(self) -> None:
        with harness() as (actions, _spawner):
            first = actions.open_studio()
            second = actions.train_walk()
            listed = actions.list_jobs()
            self.assertEqual(
                [job["job_id"] for job in listed],
                [second["job_id"], first["job_id"]],
            )


class Onboarding(unittest.TestCase):
    def test_a_missing_mjcf_is_refused_by_path(self) -> None:
        with harness() as (actions, _spawner), self.assertRaises(FileNotFoundError):
            ghost = actions.jobs.jobs_dir.parent / "ghost.xml"
            actions.onboard_robot(str(ghost), "ghost")

    def test_an_existing_bundle_is_never_overwritten(self) -> None:
        with harness() as (actions, _spawner):
            with self.assertRaises(FileExistsError) as ctx:
                actions.onboard_robot(__file__, "microduck")  # exists in robots/
            self.assertIn("never overwrites", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TheLaunchEnvironment(unittest.TestCase):
    """On a box with a launch env file (Linux: pipeline/wsl.env), every
    door carries it — a warp child without CUDA's library path falls to
    the CPU silently (the first GPU-box run of the doors, 2026-09-02)."""

    def test_uv_doors_pass_the_env_file_to_uv(self) -> None:
        with harness(env_file=Path("/box/wsl.env")) as (actions, spawner):
            actions.certify_walk("runs/x/model_1.pt", trials=2)
            [(argv, _)] = spawner.calls
            self.assertEqual(argv[:6], [*UV_MJLAB[:4], "--env-file", "/box/wsl.env"])

    def test_the_train_venv_chain_runs_under_wsl_run(self) -> None:
        with harness(env_file=Path("/box/wsl.env")) as (actions, spawner):
            actions.run_chain(name="t")
            [(argv, _)] = spawner.calls
            self.assertEqual(argv[:2], [str(WSL_RUN), str(TRAIN_PYTHON)])

    def test_the_studio_drops_the_wayland_display_on_that_box(self) -> None:
        with harness(env_file=Path("/box/wsl.env")) as (actions, spawner):
            actions.open_studio()
            [(argv, _)] = spawner.calls
            self.assertEqual(
                argv, ["env", "-u", "WAYLAND_DISPLAY", "cargo", "run", "--release"]
            )

    def test_without_an_env_file_nothing_is_wrapped(self) -> None:
        with harness() as (actions, spawner):
            actions.open_studio()
            [(argv, _)] = spawner.calls
            self.assertEqual(argv, ["cargo", "run", "--release"])
