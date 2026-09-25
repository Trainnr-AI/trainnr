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
    UNITREE_REFERENCE_DEFAULT,
    UNITREE_REFERENCE_ENV,
    WSL_RUN,
    Actions,
    unitree_reference,
    walk_train_argv,
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
        self.prepared: list[list[str]] = []

    def prepare(self, argv, cwd) -> None:
        self.prepared.append(list(argv))

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
        jobs = JobManager(Path(tmp), spawner=spawner, preparer=spawner.prepare)
        try:
            yield Actions(jobs, env_file=env_file), spawner
        finally:
            jobs.join()


UV_PIPELINE = ["uv", "run", "--no-sync", "--project", str(PIPELINE_DIR)]
UV_MJLAB = ["uv", "run", "--no-sync", "--project", str(RQ_MJLAB_DIR), "python"]


class TheDoors(unittest.TestCase):
    def test_generate_kitting_demos_spawns_the_kitting_tool_verbatim(self) -> None:
        with harness() as (actions, spawner):
            handle = actions.generate_kitting_demos(episodes=3, seed=7, out="runs/x")
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
            self.assertTrue(str(handle["job_id"]).startswith("generate-kitting-demos-"))

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

    def test_generate_planned_demos_spawns_the_planner_on_the_task(self) -> None:
        with harness() as (actions, spawner):
            handle = actions.generate_planned_demos("block_stack", episodes=4, seed=9)
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
            self.assertTrue(str(handle["job_id"]).startswith("generate-planned-demos-"))

    def test_generate_planned_demos_shards_only_when_asked(self) -> None:
        with harness() as (actions, spawner):
            actions.generate_planned_demos("lift", shards=4)
            [(argv, _)] = spawner.calls
            self.assertEqual(argv[-2:], ["--shards", "4"])

    def test_zero_episodes_refuses_before_spawning(self) -> None:
        with harness() as (actions, spawner):
            with self.assertRaises(ValueError):
                actions.generate_kitting_demos(episodes=0)
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
            actions.train_walk(agent="smoke", iterations=5, robot="microduck")
            actions.evaluate_walk(
                "runs/x/model_100.pt", trials=8, device="cpu", robot="microduck"
            )
            (train_argv, train_cwd), (cert_argv, _c) = spawner.calls
            self.assertEqual(
                train_argv,
                [
                    *UV_MJLAB,
                    "-m",
                    "rq_mjlab.walk_train",
                    "--agent",
                    "smoke",
                    "--robot",
                    "microduck",
                    "--iterations",
                    "5",
                ],
            )
            self.assertEqual(train_cwd, RQ_MJLAB_DIR)
            self.assertEqual(
                cert_argv[cert_argv.index("-m") + 1], "rq_mjlab.walk_verdict"
            )
            self.assertIn("--device", cert_argv)
            self.assertEqual(cert_argv[cert_argv.index("--robot") + 1], "microduck")

    def test_the_go2_walk_trains_in_its_project(self) -> None:
        with harness() as (actions, spawner):
            actions.train_walk(
                "g3", robot="go2", project="/p/go2", log_dir="/p/go2/runs/first", seed=7
            )
            [(argv, _cwd)] = spawner.calls
            tail = argv[argv.index("--robot") :]
            self.assertEqual(
                tail,
                [
                    "--robot",
                    "go2",
                    "--project",
                    "/p/go2",
                    "--seed",
                    "7",
                    "--log-dir",
                    "/p/go2/runs/first",
                ],
            )
            with self.assertRaisesRegex(ValueError, "one of go1, go2, microduck"):
                actions.train_walk("smoke", robot="spot")
            with self.assertRaisesRegex(ValueError, "name the robot"):
                actions.train_walk("smoke", robot=None)

    def test_the_walk_argv_is_one_line_for_the_door_and_the_smoke(self) -> None:
        argv = walk_train_argv(
            agent="smoke",
            robot="go2",
            project="/p",
            envs=2,
            iterations=2,
            dr_span=0.1,
            task_stamp="go2-walk@abc",
            recorder=False,
            env_file=Path("/box/wsl.env"),
        )
        self.assertEqual(
            argv,
            [
                *UV_MJLAB[:5],
                "--env-file",
                "/box/wsl.env",
                "python",
                "-m",
                "rq_mjlab.walk_train",
                "--agent",
                "smoke",
                "--robot",
                "go2",
                "--project",
                "/p",
                "--envs",
                "2",
                "--iterations",
                "2",
                "--dr-span",
                "0.1",
                "--task-stamp",
                "go2-walk@abc",
                "--no-recorder",
            ],
        )
        with self.assertRaisesRegex(ValueError, "one of"):
            walk_train_argv(agent="smoke", robot="spot")

    def test_a_scene_rides_on_the_evaluation_argv(self) -> None:
        with harness() as (actions, spawner):
            actions.evaluate_walk(
                "runs/c/model_9.pt", robot="go2", scene="/p/scenes/hurdle"
            )
            [(argv, _)] = spawner.calls
            self.assertEqual(argv[-2:], ["--scene", "/p/scenes/hurdle"])

    def test_a_scene_rides_on_the_walk_argv(self) -> None:
        argv = walk_train_argv(
            agent="smoke", robot="go2", scene="/p/scenes/hurdle", env_file=None
        )
        self.assertEqual(argv[-2:], ["--scene", "/p/scenes/hurdle"])
        self.assertNotIn("--scene", walk_train_argv(agent="smoke", robot="go2"))
        blind = walk_train_argv(
            agent="smoke", robot="go2", scene="/p/scenes/hurdle", cameras=False
        )
        self.assertEqual(blind[-1], "--no-cameras")
        # the camera switch is the scene's; a plane never carries it
        self.assertNotIn(
            "--no-cameras", walk_train_argv(agent="smoke", robot="go2", cameras=False)
        )

    def test_the_gate_names_its_runtime_and_the_reference(self) -> None:
        with harness() as (actions, spawner):
            actions.gate_deployment("d", project="/p", trials=4, runtime="dds")
            [(argv, _)] = spawner.calls
            self.assertEqual(argv[argv.index("--runtime") + 1], "dds")
            self.assertEqual(
                argv[argv.index("--reference") + 1], str(unitree_reference())
            )
            with self.assertRaisesRegex(ValueError, "unknown gate runtime"):
                actions.gate_deployment("d", project="/p", runtime="gazebo")

    def test_the_reference_comes_from_the_environment(self) -> None:
        os.environ[UNITREE_REFERENCE_ENV] = "/ref/checkout"
        try:
            self.assertEqual(unitree_reference(), Path("/ref/checkout"))
        finally:
            os.environ.pop(UNITREE_REFERENCE_ENV, None)
        self.assertEqual(unitree_reference(), UNITREE_REFERENCE_DEFAULT.expanduser())

    def test_the_studio_launches_release_in_its_crate(self) -> None:
        with harness() as (actions, spawner):
            actions.open_studio()
            [(argv, cwd)] = spawner.calls
            self.assertEqual(argv, ["cargo", "run", "--release"])
            self.assertEqual(cwd, STUDIO_DIR)


class TheStudentEvaluation(unittest.TestCase):
    def test_a_student_rides_the_same_door_with_its_horizon(self) -> None:
        with harness() as (actions, spawner):
            actions.evaluate_walk(
                "runs/x/model_1.pt",
                trials=8,
                student="runs/s/pretrained_model",
                horizon=10,
                robot="microduck",
            )
            [(argv, _)] = spawner.calls
            self.assertEqual(
                argv[-4:], ["--student", "runs/s/pretrained_model", "--horizon", "10"]
            )


class TheCrossEvaluation(unittest.TestCase):
    def test_a_cliff_rung_in_another_world_rides_the_same_door(self) -> None:
        with harness() as (actions, spawner):
            actions.evaluate_walk(
                "runs/x/model_1.pt",
                robot="go2",
                judge_in_fit="fit@ab",
                judge_at_scale=0.8,
                judge_param="kp",
                delay=2,
            )
            [(argv, _)] = spawner.calls
            self.assertEqual(
                argv[-8:],
                [
                    "--judge-in-fit",
                    "fit@ab",
                    "--judge-at-scale",
                    "0.8",
                    "--judge-param",
                    "kp",
                    "--delay",
                    "2",
                ],
            )

    def test_the_policy_own_world_adds_nothing(self) -> None:
        with harness() as (actions, spawner):
            actions.evaluate_walk("runs/x/model_1.pt", robot="go2")
            [(argv, _)] = spawner.calls
            for flag in ("--judge-in-fit", "--judge-at-scale", "--delay"):
                self.assertNotIn(flag, argv)


class TheWalkDemosDoor(unittest.TestCase):
    def test_generate_walk_demos_on_a_scene_names_the_robot_and_the_scene(self) -> None:
        with harness() as (actions, spawner):
            actions.generate_walk_demos(
                checkpoint="runs/c/model_9.pt",
                episodes=2,
                worlds=4,
                out="runs/w",
                robot="go2",
                scene="/p/scenes/hurdle",
                project="/p",
                frame_size=(160, 120),
            )
            [(argv, _cwd)] = spawner.calls
            press = argv[argv.index("rq_mjlab.walk_press") :]  # after uv's own flags
            self.assertEqual(press[1], "runs/c/model_9.pt")
            self.assertEqual(press[press.index("--robot") + 1], "go2")
            self.assertEqual(press[press.index("--scene") + 1], "/p/scenes/hurdle")
            self.assertEqual(press[press.index("--project") + 1], "/p")
            self.assertEqual(
                press[press.index("--width") + 1 : press.index("--width") + 4],
                ["160", "--height", "120"],
            )
            with self.assertRaisesRegex(ValueError, "one of"):
                actions.generate_walk_demos(robot="spot")

    def test_generate_walk_demos_rolls_the_newest_checkpoint_by_default(self) -> None:
        with harness() as (actions, spawner):
            actions.generate_walk_demos(episodes=4, worlds=3, seed=9, out="runs/w")
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
            second = actions.train_walk(robot="microduck")
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
            actions.evaluate_walk("runs/x/model_1.pt", trials=2, robot="microduck")
            [(argv, _)] = spawner.calls
            self.assertEqual(argv[:7], [*UV_MJLAB[:5], "--env-file", "/box/wsl.env"])

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


class ThePreviewDoor(unittest.TestCase):
    def test_preview_rewards_spawns_the_preview_in_the_walk_venv(self) -> None:
        with harness() as (actions, spawner):
            actions.preview_rewards(
                robot="go2",
                controller="stand",
                seconds=3.0,
                project="/p",
                out="/p/tasks/t/preview-stand.json",
            )
            [(argv, _cwd)] = spawner.calls
            self.assertIn("rq_mjlab.reward_preview", argv)
            # uv's own --project comes first; the tool's flags follow the module.
            argv = argv[argv.index("rq_mjlab.reward_preview") :]
            for flag, value in (
                ("--robot", "go2"),
                ("--controller", "stand"),
                ("--seconds", "3.0"),
                ("--project", "/p"),
            ):
                self.assertEqual(argv[argv.index(flag) + 1], value)
            with self.assertRaises(ValueError):
                actions.preview_rewards(robot="spot")


class ThePlayDoor(unittest.TestCase):
    def test_play_walk_opens_the_checkpoint_in_the_walk_venv(self) -> None:
        with harness() as (actions, spawner):
            actions.play_walk(
                "/p/runs/r/model_7999.pt", robot="go2", envs=4, project="/p"
            )
            [(argv, _cwd)] = spawner.calls
            argv = argv[argv.index("rq_mjlab.walk_play") :]
            self.assertEqual(argv[1], "/p/runs/r/model_7999.pt")
            self.assertEqual(argv[argv.index("--robot") + 1], "go2")
            self.assertEqual(argv[argv.index("--envs") + 1], "4")
            self.assertEqual(argv[argv.index("--project") + 1], "/p")
