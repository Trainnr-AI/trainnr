"""The MCP surface's query functions, against the real repo — bundles,
actuators, tasks, engines, runs. The server framing itself needs the
`mcp` extra and gets one guarded construction test; everything else here
runs on any venv, because the functions deliberately import no MCP.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._extras import needs_mcp, needs_numpy, needs_sim
from trainnr import mcp_server
from trainnr.mcp_server import (
    bundle_names,
    create_project_dir,
    describe_actuator,
    describe_dataset,
    describe_evaluation,
    describe_friction,
    describe_project,
    describe_robot,
    describe_task,
    list_actuators,
    list_engines,
    list_evaluations,
    list_experiments,
    list_robots,
    list_tasks,
)
from trainnr.project import PROJECT_ENV


class Bundles(unittest.TestCase):
    def test_the_census_names_the_known_bundles_and_not_the_library(self) -> None:
        names = bundle_names()
        self.assertIn("rig-drivetrain", names)
        self.assertIn("so101-nominal", names)
        self.assertNotIn("actuators", names)

    def test_every_bundle_carries_a_stamp_and_files(self) -> None:
        for bundle in list_robots():
            with self.subTest(bundle=bundle["name"]):
                self.assertIn("@", bundle["stamp"])
                self.assertTrue(bundle["stamp"].startswith(bundle["name"] + "@"))
                self.assertTrue(bundle["files"])

    def test_the_drivetrain_detail_has_profile_fits_and_spread(self) -> None:
        detail = describe_robot("rig-drivetrain")
        self.assertEqual(detail["profile"]["name"], "rig-drivetrain")
        self.assertIn("SPREAD.json", detail["fits"])
        # The flagship verdict rides along verbatim — the whole point of
        # this window is that an agent reads the same record the repo
        # trusts, not a summary of it.
        self.assertTrue(any(name.startswith("sweep-") for name in detail["fits"]))

    def test_an_unknown_bundle_is_refused_naming_what_exists(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            describe_robot("no-such-robot")
        self.assertIn("rig-drivetrain", str(ctx.exception))

    def test_the_library_directory_is_refused_as_a_bundle(self) -> None:
        with self.assertRaises(KeyError):
            describe_robot("actuators")


class Actuators(unittest.TestCase):
    def test_all_eight_servos_with_provenance(self) -> None:
        described = list_actuators()
        self.assertEqual(len(described), 9)  # eight vendored fits + our XL330 refit
        for servo in described:
            if servo["source"] == "bam-refit":
                continue  # one tier, its own provenance (docs/e2e-research/72)
            with self.subTest(actuator=servo["actuator"]):
                self.assertEqual(servo["source"], "bam")
                self.assertEqual(servo["tiers"], ["m1", "m2", "m3", "m4", "m5", "m6"])

    def test_one_servo_in_full_carries_the_published_kt(self) -> None:
        detail = describe_actuator("feetech_sts3215_7_4V", "m1")
        self.assertAlmostEqual(detail["servo"]["kt"], 1.1776311631627974)
        self.assertEqual(detail["provenance"]["source"], "bam")


class ActuatorBundles(unittest.TestCase):
    def test_the_committed_store_lists_with_stamps_and_advisories(self) -> None:
        described = [b for a in list_actuators() for b in a["bundles"]]
        self.assertEqual(len(described), 49)  # 8 motors x m1..m6 + the XL330 refit
        for entry in described:
            with self.subTest(file=entry["file"]):
                self.assertIn("@", entry["stamp"])
                if "refit" in entry["file"]:
                    # The refit carries its interval: no uncertainty advisory.
                    self.assertFalse(
                        any("uncertainty" in a for a in entry["advisories"])
                    )
                    continue
                # Vendored point estimates: the honesty advisories ride
                # on every listing, not just the deep view.
                self.assertTrue(any("uncertainty" in a for a in entry["advisories"]))

    def test_one_bundle_in_full_carries_bams_params_verbatim(self) -> None:
        detail = describe_actuator("feetech_sts3215_7_4V", "m6")
        self.assertEqual(detail["bundle"]["params"]["actuator"], "sts3215")
        self.assertIn("alpha", detail["bundle"]["checks"]["near_search_bound"])

    def test_an_unknown_bundle_is_refused_naming_the_store(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            describe_actuator("no-such-servo", "m6")
        self.assertIn("feetech_sts3215_7_4V", str(ctx.exception))


class Datasheets(unittest.TestCase):
    def test_a_batch_summary_folds_with_its_keep_rate_bound(self) -> None:
        from trainnr.project.locate import create_project  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as root:
            made = create_project(Path(root), "p")
            tmp = str(made.runs / "batch")
            Path(tmp).mkdir(parents=True)
            episode = Path(tmp) / "episode_0000"
            episode.mkdir()
            (episode / "manifest.json").write_text(
                json.dumps(
                    {
                        "seed": 1,
                        "attempt": 4,
                        "task": "t@1",
                        "expert": "e@1",
                        "instrument": "i",
                        "dynamics": {"kt": 1.3},
                        "dynamics_basis": "identified-interval",
                        "draws": {},
                        "retries": [],
                        "control_hz": 50,
                        "frame_every_control_ticks": 1,
                    }
                )
            )
            with mock.patch.dict(os.environ, {PROJECT_ENV: str(made.root)}):
                sheet = describe_dataset(tmp)
            self.assertEqual(sheet["episodes"], 1)
            self.assertEqual(sheet["keep_rate_bound"], 0.25)
            self.assertEqual(sheet["bases"], ("identified-interval",))


class Registries(unittest.TestCase):
    def test_tasks_include_both_rigs(self) -> None:
        rigs = {entry["rig"] for entry in list_tasks()}
        self.assertTrue({"aloha2", "so101"} <= rigs)
        self.assertIn("go2", rigs)  # the walk families (docs/77)

    def test_engines_include_both_backends(self) -> None:
        names = {entry["name"] for entry in list_engines()}
        self.assertIn("mujoco", names)
        self.assertIn("mjx-warp", names)

    @needs_sim
    def test_built_tasks_report_spec_and_stamp_honestly(self) -> None:
        # Task.stamp is str | None BY DESIGN: only a spec-carrying task
        # has content to hash, and a code-only task saying "no stamp" is
        # the honest answer, not a gap. Both kinds must exist and both
        # must round-trip through the window unchanged.
        from trainnr.tasks.walks import walk_robot  # noqa: PLC0415

        details = []
        for t in list_tasks():
            try:
                details.append(describe_task(t["task"]))
            except FileNotFoundError as why:
                # A walk family builds on the project's robot; with no
                # project open it refuses by name, which is the answer.
                self.assertIsNotNone(walk_robot(t["task"]))
                self.assertIn("onboard_robot", str(why))
        with_spec = [d for d in details if "spec" in d]
        self.assertTrue(with_spec, "no spec-carrying task in the registry?")
        for detail in details:
            with self.subTest(task=detail["task"]):
                if "spec" in detail:
                    self.assertIn("@", detail["stamp"])
                else:
                    self.assertIsNone(detail["stamp"])


# The records these tests fold are `tools/e2e-smoke.py --name smoke`'s own
# output (`runs/smoke-eval`, gitignored): on a clone that never ran the chain
# they are absent, and the tests say so instead of failing (2026-09-27).
SMOKE_EVAL = Path(__file__).resolve().parents[1] / "runs" / "smoke-eval"


@unittest.skipUnless(
    SMOKE_EVAL.is_dir(),
    f"no {SMOKE_EVAL.relative_to(SMOKE_EVAL.parents[2])}: "
    "run `tools/e2e-smoke.py --name smoke` first",
)
class Evals(unittest.TestCase):
    def test_the_real_smoke_records_fold_with_funnel_and_successes(self) -> None:
        # Against the committed smoke runs — the same records the panel
        # and the certificate read.
        runs = {entry["evaluation"] for entry in list_evaluations()}
        self.assertIn("smoke-eval", runs)
        detail = describe_evaluation("smoke-eval")
        data = detail["files"]["episodes.jsonl"]
        self.assertEqual(data["records"], 2)
        self.assertLessEqual(data["successes"], data["records"])
        for counts in data["funnel"].values():
            self.assertEqual(len(counts), len(data["milestones"]))

    def test_a_run_without_records_is_refused_naming_those_that_have_them(
        self,
    ) -> None:
        with self.assertRaises(KeyError) as ctx:
            describe_evaluation("no-such-run")
        self.assertIn("smoke-eval", str(ctx.exception))

    def test_an_empty_runs_root_lists_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_evaluations(Path(tmp)), [])


class FrictionCurves(unittest.TestCase):
    @needs_numpy
    def test_the_curve_is_plottable_and_load_adds_friction(self) -> None:
        curve = describe_friction("feetech_sts3215_7_4V", "m6", points=11)
        self.assertEqual(len(curve["velocity"]), 11)
        self.assertEqual(len(curve["unloaded"]), 11)
        self.assertEqual(len(curve["loaded"]), 11)
        self.assertEqual(curve["velocity"][0], 0.0)
        # M6 carries load-dependent terms: under external torque the
        # budget must exceed the unloaded budget at every velocity.
        for unloaded, loaded in zip(curve["unloaded"], curve["loaded"], strict=True):
            self.assertGreater(loaded, unloaded)


class Runs(unittest.TestCase):
    def test_the_open_projects_runs_are_listed_with_their_manifests(self) -> None:
        """A walk's experiments live in the project's `runs/` with
        `training.json` and `identity.json`, not in the legacy root: the
        README's "query and learn" row answered [] for them until
        2026-10-03 (the stranger test)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy"
            (legacy / "t9").mkdir(parents=True)
            (legacy / "t9" / "run.json").write_text(json.dumps({"name": "t9"}))
            project = create_project_dir("p", str(root / "p"))
            run = Path(project["root"]) / "runs" / "g3-150"
            run.mkdir(parents=True)
            (run / "training.json").write_text(json.dumps({"iterations": 150}))
            (run / "identity.json").write_text(json.dumps({"robot": "go2@abc"}))
            (run / "verdict").mkdir()
            (run / "verdict" / "records-cuda.jsonl").write_text("{}\n{}\n")
            (Path(project["root"]) / "runs" / "empty").mkdir()
            with (
                mock.patch.dict(os.environ, {PROJECT_ENV: project["root"]}),
                mock.patch.object(mcp_server, "_runs_root", return_value=legacy),
            ):
                runs = list_experiments()
                records = list_evaluations()
            # an explicit root is that root alone, as before
            self.assertEqual(
                list_experiments(legacy),
                [{"experiment": "t9", "manifest": {"name": "t9"}}],
            )
            # the project's first, one short row each; curves stay out
            self.assertEqual([r["experiment"] for r in runs], ["g3-150", "t9"])
            project_run = runs[0]
            self.assertEqual(project_run["where"], str(run))
            self.assertEqual(project_run["iterations"], 150)
            self.assertEqual(project_run["robot"], "go2@abc")
            self.assertNotIn("training", project_run)
            with mock.patch.dict(os.environ, {PROJECT_ENV: project["root"]}):
                full = mcp_server.describe_experiment("g3-150")
                with self.assertRaisesRegex(KeyError, "it has"):
                    mcp_server.describe_experiment("nope")
            self.assertEqual(full["training"], {"iterations": 150})
            # a long curve comes back sampled, its last row kept
            curve = [[i, float(i)] for i in range(500)]
            (run / "training.json").write_text(
                json.dumps({"iterations": 500, "curve": curve})
            )
            with mock.patch.dict(os.environ, {PROJECT_ENV: project["root"]}):
                sampled = mcp_server.describe_experiment("g3-150")["training"]
            self.assertEqual(len(sampled["curve"]), mcp_server.CURVE_ROWS)
            self.assertEqual(sampled["curve"][-1], [499, 499.0])
            self.assertEqual(sampled["curve_rows_recorded"], 500)
            self.assertEqual(full["identity"], {"robot": "go2@abc"})
            # A project's evaluations are listed from its index by stamp
            # (test_tool_api.OneNamePerEvaluation); a verdict with no
            # checkpoint beside it never becomes one, and the legacy root
            # here holds no episode records (2026-10-04).
            self.assertEqual(records, [])

    def test_without_a_project_the_runs_are_the_legacy_root_alone(self) -> None:
        nowhere = Path(tempfile.gettempdir()) / "trainnr-no-such-runs"
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(mcp_server, "_runs_root", return_value=nowhere),
        ):
            os.environ.pop(PROJECT_ENV, None)
            self.assertEqual(list_experiments(), [])
            self.assertEqual(list_evaluations(), [])

    def test_manifests_are_read_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t9-watch").mkdir()
            manifest = {"name": "t9", "steps": 300}
            (root / "t9-watch" / "run.json").write_text(json.dumps(manifest))
            runs = list_experiments(root)
            self.assertEqual(runs, [{"experiment": "t9-watch", "manifest": manifest}])

    def test_a_missing_runs_directory_is_empty_not_an_error(self) -> None:
        self.assertEqual(list_experiments(Path("/nonexistent/runs")), [])


class TheCaptureDoors(unittest.TestCase):
    """The live-capture doors (built 2026-09-09, registered 2026-09-24:
    the module existed and no agent could call it)."""

    def _free_port(self) -> int:
        import socket  # noqa: PLC0415

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_start_status_stop_round_trip_and_the_refusals(self) -> None:
        import os  # noqa: PLC0415
        import socket  # noqa: PLC0415
        import time  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from trainnr.mcp_server import (  # noqa: PLC0415
            describe_capture,
            start_capture,
            stop_capture,
        )
        from trainnr.robots.capture import FAILED, IDLE, LISTENING  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            made = create_project_dir("p", str(Path(tmp) / "p"), "test")
            with mock.patch.dict(os.environ, {PROJECT_ENV: made["root"]}):
                self.assertEqual(describe_capture()["state"], IDLE)
                self.assertEqual(stop_capture()["status"], "refused")
                port = self._free_port()
                started = start_capture("session-1", port=port, window_s=30.0)
                self.assertEqual(started["status"], "done", started)
                self.assertEqual(started["capture"]["state"], LISTENING)
                self.assertEqual(describe_capture()["state"], LISTENING)
                # a second listener for the same project is refused by name
                again = start_capture("session-2", port=self._free_port())
                self.assertEqual(again["status"], "refused")
                self.assertIn("session-1", again["reason"])
                sent = 12
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    for i in range(sent):
                        s.sendto(
                            f"n={i} t={i * 20} v=0.0".encode(), ("127.0.0.1", port)
                        )
                        time.sleep(0.002)
                deadline = time.time() + 5.0
                while describe_capture()["datagrams"] < sent and time.time() < deadline:
                    time.sleep(0.05)
                stopped = stop_capture()
                self.assertEqual(stopped["status"], "done", stopped)
                # a dozen unparseable lines: the ingest fails by name, the
                # state says so, and nothing is left behind
                self.assertIn(stopped["capture"]["state"], (FAILED, "ingested"))
                self.assertEqual(
                    describe_capture()["state"], stopped["capture"]["state"]
                )
                self.assertEqual(stop_capture()["status"], "refused")
                self.assertFalse(
                    list((Path(made["root"]) / "recordings").glob(".capture-*"))
                )

    def test_without_a_project_every_capture_door_refuses(self) -> None:
        import os  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from trainnr.mcp_server import (  # noqa: PLC0415
            describe_capture,
            start_capture,
            stop_capture,
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {PROJECT_ENV: str(Path(tmp) / "nowhere")}),
        ):
            for door in (describe_capture, stop_capture):
                self.assertEqual(door()["status"], "refused")
            self.assertEqual(start_capture("x")["status"], "refused")


class TheProjectDoors(unittest.TestCase):
    def test_create_then_describe_names_every_state_missing(self) -> None:
        import os  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            made = create_project_dir("p", str(Path(tmp) / "p"), "test")
            self.assertEqual(made["name"], "p")
            with mock.patch.dict(os.environ, {PROJECT_ENV: made["root"]}):
                index = describe_project()
            self.assertEqual(index["project"], "p")
            self.assertEqual(index["artifacts"], [])
            self.assertFalse(any(s["present"] for s in index["states"]))
            self.assertIn("record", index["next_move"])
            self.assertTrue((Path(made["root"]) / ".index" / "project.json").is_file())

    def test_describe_returns_the_written_index_with_its_previews(self) -> None:
        """The agent must see the same index the Studio reads — including
        the preview paths write_index records (a stale in-memory copy once
        said None while the file on disk had the picture, 2026-09-09)."""
        import os  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from trainnr.project import current_project  # noqa: PLC0415
        from trainnr.project.ingest import ingest  # noqa: PLC0415

        repo = Path(__file__).resolve().parents[2]
        wire = repo / "recordings" / "chase-arm-2026-08-17.wire"
        with tempfile.TemporaryDirectory() as tmp:
            made = create_project_dir("p", str(Path(tmp) / "p"), "test")
            with mock.patch.dict(os.environ, {PROJECT_ENV: made["root"]}):
                ingest(current_project(), wire, name="chase")
                index = describe_project()
            rec = next(a for a in index["artifacts"] if a["kind"] == "recording")
            self.assertIsNotNone(
                rec["preview"], "preview path missing from the returned index"
            )
            self.assertTrue((Path(made["root"]) / rec["preview"]).is_file())

    def test_the_committed_sample_describes(self) -> None:
        import os  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        sample = Path(__file__).resolve().parents[2] / "projects" / "sample"
        with mock.patch.dict(os.environ, {PROJECT_ENV: str(sample)}):
            index = describe_project(refresh=False)
        self.assertEqual(index["project"], "sample")


class ServerFraming(unittest.TestCase):
    @needs_mcp
    def test_the_server_builds_with_every_tool_registered(self) -> None:
        from trainnr.mcp_server import build_server  # noqa: PLC0415

        server = build_server()
        self.assertEqual(server._lowlevel_server.name, "trainnr")

    @needs_mcp
    def test_every_door_is_registered(self) -> None:
        """A door is a module-level function that can answer with a
        refusal; one written and not registered (play_walk, 2026-09-23)
        answers nobody."""
        import inspect  # noqa: PLC0415

        from trainnr import mcp_server  # noqa: PLC0415

        doors = {
            name
            for name, f in vars(mcp_server).items()
            if inspect.isfunction(f)
            and f.__module__ == mcp_server.__name__
            and not name.startswith("_")
            and "Refusal" in str(f.__annotations__.get("return", ""))
        }
        registered = {  # by the function, not the tool's name (create_project)
            t.fn.__name__ for t in mcp_server.build_server()._tool_manager.list_tools()
        }
        self.assertEqual(doors - registered, set())


class StudioDoors(unittest.TestCase):
    def test_every_studio_door_answers_without_a_studio(self) -> None:
        """No window, no waiting: describe reports it, every act door refuses
        by name and points at launch_studio, the event log is empty."""
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from trainnr.mcp_server import (  # noqa: PLC0415
            compare_in_studio,
            control_simulator,
            describe_studio,
            open_in_studio,
            read_studio_events,
            run_simulation,
            screenshot_studio,
            set_simulator_input,
            set_simulator_view,
            set_studio_panels,
            set_studio_time,
            show_in_studio,
        )
        from trainnr.project import PROJECT_ENV, create_project  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            create_project(Path(tmp) / "p", "p")
            os.environ[PROJECT_ENV] = str(Path(tmp) / "p")
            try:
                self.assertFalse(describe_studio()["alive"])
                for answer in (
                    open_in_studio(section="robots"),
                    show_in_studio("a@000000000000"),
                    compare_in_studio("a@000000000000", "b@000000000000"),
                    set_studio_time(play=True),
                    set_studio_panels(blueprint="expand"),
                    screenshot_studio(),
                    run_simulation("kitting"),
                    control_simulator(play=False),
                    set_simulator_input(0.5, actuator="left/waist"),
                    set_simulator_view("contactforce", True),
                ):
                    self.assertEqual(answer["status"], "refused")
                    self.assertIn("launch_studio", answer["reason"])
                self.assertIn("no page", open_in_studio(section="dance")["reason"])
                # The walk's twist axes: named from a fixed set, valued unless
                # handing back; never more than one input per call.
                self.assertIn("vx", set_simulator_input(1.0, command="vq")["reason"])
                self.assertIn("value", set_simulator_input(command="vx")["reason"])
                self.assertIn(
                    "exactly one",
                    set_simulator_input(1.0, actuator="a", command="vx")["reason"],
                )
                self.assertIn(
                    "launch_studio", set_simulator_input(command="own")["reason"]
                )
                self.assertIn("on", set_simulator_view(group=1)["reason"])
                self.assertIn(
                    "launch_studio",
                    open_in_studio(section="evaluations", view="matrix")["reason"],
                )
                self.assertIn(
                    "launch_studio", open_in_studio(search="narrow")["reason"]
                )
                self.assertIn(
                    "not one of", set_studio_panels(selection="hide")["reason"]
                )
                self.assertEqual(read_studio_events(), [])
            finally:
                os.environ.pop(PROJECT_ENV, None)


class TaskDoors(unittest.TestCase):
    """A4: an environment declared by conversation; refusals by name
    before any scene is built."""

    def test_the_doors_refuse_by_name_without_building(self) -> None:
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from trainnr.mcp_server import (  # noqa: PLC0415
            check_task,
            create_task,
            list_task_families,
        )
        from trainnr.project import PROJECT_ENV, create_project  # noqa: PLC0415

        families = list_task_families()
        self.assertIn("trainnr/kitting", families)
        self.assertEqual(families["trainnr/kitting"]["fields"]["trials"]["default"], 4)
        with tempfile.TemporaryDirectory() as tmp:
            create_project(Path(tmp) / "p", "p")
            os.environ[PROJECT_ENV] = str(Path(tmp) / "p")
            try:
                bad = create_task("kitting", "x", {"tray_centre": [0, 0]})
                self.assertEqual(bad["status"], "refused")
                self.assertIn("tray_center", bad["reason"])
                fixed = create_task("trainnr/reach", "x", {})
                self.assertEqual(fixed["status"], "refused")
                self.assertIn("trainnr/kitting", fixed["reason"])
                self.assertEqual(create_task("acme/pour", "x", {})["status"], "refused")
                ghost = check_task("ghost")
                self.assertEqual(ghost["status"], "refused")
                self.assertIn("ghost", ghost["reason"])
            finally:
                os.environ.pop(PROJECT_ENV, None)


if __name__ == "__main__":
    unittest.main()


class TheFocusDoor(unittest.TestCase):
    def test_the_live_name_is_matched_on_its_letters(self) -> None:
        """The SDK folds a stamp's `@` into `-` and appends the recording
        id; a name that is not streaming is seen, not acked (2026-09-27)."""
        from trainnr.mcp_server import recording_matches  # noqa: PLC0415

        stamp = "go2-scene-c1-model_1000-cuda-seed1000-n40-pdf8c16@a95789ca057b"
        self.assertTrue(
            recording_matches(
                stamp,
                "go2-scene-c1-model_1000-cuda-seed1000-n40-pdf8c16-a95789ca057bf66d",
            )
        )
        self.assertTrue(
            recording_matches("trainnr-sim-deploy:go2-c2", "trainnr-sim-deploy-go2-c2")
        )
        self.assertFalse(
            recording_matches("no-such-recording", "trainnr-sim-deploy-go2-c2")
        )
        self.assertFalse(recording_matches("x", None))


class ArtifactsByName(unittest.TestCase):
    def test_a_bare_name_resolves_when_it_is_the_only_one(self) -> None:
        """`identify_system("go2", …)` right after `onboard_robot` answered
        `go2@e5aa641994fc` was refused as "no artifact 'go2'" (2026-09-28)."""
        from types import SimpleNamespace as A  # noqa: PLC0415

        from trainnr.mcp_server import pick_artifact  # noqa: PLC0415

        go2 = A(stamp="go2@e5aa641994fc", kind="robot")
        chirp = A(stamp="go2@f7f50914a129", kind="recording")
        arts = [go2, chirp, A(stamp="go2-c1@1111", kind="run")]
        self.assertIs(pick_artifact(arts, "go2@e5aa641994fc", "robot", "p"), go2)
        self.assertIs(pick_artifact(arts, "go2", "robot", "p"), go2)
        self.assertIs(pick_artifact(arts, "go2", "recording", "p"), chirp)
        with self.assertRaisesRegex(KeyError, "2 artifacts named 'go2'.*e5aa641994fc"):
            pick_artifact(arts, "go2", None, "p")
        with self.assertRaisesRegex(KeyError, "no artifact 'go1' in p"):
            pick_artifact(arts, "go1", "robot", "p")
        with self.assertRaisesRegex(ValueError, "is a recording, not a robot"):
            pick_artifact(arts, "go2@f7f50914a129", "robot", "p")
        # a bare name never matches a different kind
        with self.assertRaisesRegex(KeyError, "no artifact 'go2-c1'"):
            pick_artifact(arts, "go2-c1", "robot", "p")


class TheCheckpointCheck(unittest.TestCase):
    def test_a_missing_checkpoint_is_refused_naming_the_ones_there(self) -> None:
        from trainnr.mcp_server import checkpoint_missing  # noqa: PLC0415
        from trainnr.project.locate import RUNS_FOLDER  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / RUNS_FOLDER / "c0"
            run.mkdir(parents=True)
            for n in (0, 50, 100, 149):
                (run / f"model_{n}.pt").write_bytes(b"x")
            self.assertIsNone(checkpoint_missing(str(run / "model_149.pt"), Path(tmp)))
            # a checkpoint runs code on load: none from outside the project
            self.assertIn("no project", checkpoint_missing("c0/model_149.pt", None))
            with tempfile.TemporaryDirectory() as other:
                outside = Path(other) / "model_1.pt"
                outside.write_bytes(b"x")
                self.assertIn(
                    "outside the current project",
                    checkpoint_missing(str(outside), Path(tmp)),
                )
            self.assertIsNone(checkpoint_missing("c0/model_149.pt", Path(tmp)))
            why = checkpoint_missing("c0/model_150.pt", Path(tmp))
            self.assertIn("no checkpoint 'model_150.pt'", why)
            self.assertTrue(
                why.endswith("model_0.pt, model_50.pt, model_100.pt, model_149.pt")
            )
            self.assertIn(
                "is not a run folder", checkpoint_missing("nope/model_1.pt", Path(tmp))
            )


class PublicLogLicences(unittest.TestCase):
    """Review 2026-10-03: the ingest tool refuses data whose publisher
    states no licence, naming the state and the flag, before any job."""

    def test_unlicensed_log_is_refused_until_accepted(self) -> None:
        from trainnr.mcp_server import ingest_public_log  # noqa: PLC0415
        from trainnr.robots import public_logs  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            made = create_project_dir("p", str(Path(tmp) / "p"), "test")
            with (
                mock.patch.dict(os.environ, {PROJECT_ENV: made["root"]}),
                mock.patch.object(public_logs, "locate", return_value=None),
            ):
                refused = ingest_public_log("go2-leg-odometry")
                self.assertEqual(refused["status"], "refused")
                self.assertIn("unlabelled", refused["reason"])
                self.assertIn("accept_unlicensed", refused["reason"])
                started = mock.MagicMock(return_value={"job_id": "j"})
                with mock.patch(
                    "trainnr.mcp_actions.Actions.ingest_public_log", started
                ):
                    handle = ingest_public_log(
                        "go2-leg-odometry", accept_unlicensed=True
                    )
                self.assertEqual(handle["job_id"], "j")
                self.assertEqual(handle["status"], "started")
                self.assertIn("describe_job", handle["next"])  # a job, said so
                self.assertTrue(started.call_args.kwargs["accept_unlicensed"])
