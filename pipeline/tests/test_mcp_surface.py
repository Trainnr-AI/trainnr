"""The MCP surface's query functions, against the real repo — bundles,
actuators, tasks, engines, runs. The server framing itself needs the
`mcp` extra and gets one guarded construction test; everything else here
runs on any venv, because the functions deliberately import no MCP.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.mcp_server import (
    bundle_names,
    describe_actuator,
    describe_actuator_bundle,
    describe_actuator_bundles,
    describe_actuators,
    describe_bundle,
    describe_bundles,
    describe_datasheet,
    describe_engines,
    describe_eval,
    describe_runs,
    describe_task,
    describe_tasks,
    friction_curve,
    list_eval_records,
)
from tests._extras import needs_mcp, needs_numpy, needs_sim


class Bundles(unittest.TestCase):
    def test_the_census_names_the_known_bundles_and_not_the_library(self) -> None:
        names = bundle_names()
        self.assertIn("rig-drivetrain", names)
        self.assertIn("so101-nominal", names)
        self.assertNotIn("actuators", names)

    def test_every_bundle_carries_a_stamp_and_files(self) -> None:
        for bundle in describe_bundles():
            with self.subTest(bundle=bundle["name"]):
                self.assertIn("@", bundle["stamp"])
                self.assertTrue(bundle["stamp"].startswith(bundle["name"] + "@"))
                self.assertTrue(bundle["files"])

    def test_the_drivetrain_detail_has_profile_fits_and_spread(self) -> None:
        detail = describe_bundle("rig-drivetrain")
        self.assertEqual(detail["profile"]["name"], "rig-drivetrain")
        self.assertIn("SPREAD.json", detail["fits"])
        # The flagship verdict rides along verbatim — the whole point of
        # this window is that an agent reads the same record the repo
        # trusts, not a summary of it.
        self.assertTrue(any(name.startswith("sweep-") for name in detail["fits"]))

    def test_an_unknown_bundle_is_refused_naming_what_exists(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            describe_bundle("no-such-robot")
        self.assertIn("rig-drivetrain", str(ctx.exception))

    def test_the_library_directory_is_refused_as_a_bundle(self) -> None:
        with self.assertRaises(KeyError):
            describe_bundle("actuators")


class Actuators(unittest.TestCase):
    def test_all_eight_servos_with_provenance(self) -> None:
        described = describe_actuators()
        self.assertEqual(len(described), 9)  # eight vendored fits + our XL330 refit
        for servo in described:
            if servo["source"] == "bam-refit":
                continue  # one tier, its own provenance (docs/e2e-research/72)
            with self.subTest(slug=servo["slug"]):
                self.assertEqual(servo["source"], "bam")
                self.assertEqual(servo["tiers"], ["m1", "m2", "m3", "m4", "m5", "m6"])

    def test_one_servo_in_full_carries_the_published_kt(self) -> None:
        detail = describe_actuator("feetech_sts3215_7_4V", "m1")
        self.assertAlmostEqual(detail["servo"]["kt"], 1.1776311631627974)
        self.assertEqual(detail["provenance"]["source"], "bam")


class ActuatorBundles(unittest.TestCase):
    def test_the_committed_store_lists_with_stamps_and_advisories(self) -> None:
        described = describe_actuator_bundles()
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
        detail = describe_actuator_bundle("feetech_sts3215_7_4V", "m6")
        self.assertEqual(detail["bundle"]["params"]["actuator"], "sts3215")
        self.assertIn("alpha", detail["bundle"]["checks"]["near_search_bound"])

    def test_an_unknown_bundle_is_refused_naming_the_store(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            describe_actuator_bundle("no-such-servo", "m6")
        self.assertIn("feetech_sts3215_7_4V.m6.bundle.json", str(ctx.exception))


class Datasheets(unittest.TestCase):
    def test_a_batch_summary_folds_with_its_keep_rate_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
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
            sheet = describe_datasheet(tmp)
            self.assertEqual(sheet["episodes"], 1)
            self.assertEqual(sheet["keep_rate_bound"], 0.25)
            self.assertEqual(sheet["bases"], ("identified-interval",))


class Registries(unittest.TestCase):
    def test_tasks_include_both_rigs(self) -> None:
        rigs = {entry["rig"] for entry in describe_tasks()}
        self.assertEqual(rigs, {"aloha2", "so101"})

    def test_engines_include_both_backends(self) -> None:
        names = {entry["name"] for entry in describe_engines()}
        self.assertIn("mujoco", names)
        self.assertIn("mjx-warp", names)

    @needs_sim
    def test_built_tasks_report_spec_and_stamp_honestly(self) -> None:
        # Task.stamp is str | None BY DESIGN: only a spec-carrying task
        # has content to hash, and a code-only task saying "no stamp" is
        # the honest answer, not a gap. Both kinds must exist and both
        # must round-trip through the window unchanged.
        details = [describe_task(t["task_id"]) for t in describe_tasks()]
        with_spec = [d for d in details if "spec" in d]
        self.assertTrue(with_spec, "no spec-carrying task in the registry?")
        for detail in details:
            with self.subTest(task=detail["task_id"]):
                if "spec" in detail:
                    self.assertIn("@", detail["stamp"])
                else:
                    self.assertIsNone(detail["stamp"])


class Evals(unittest.TestCase):
    def test_the_real_smoke_records_fold_with_funnel_and_successes(self) -> None:
        # Against the committed smoke runs — the same records the panel
        # and the certificate read.
        runs = {entry["run"] for entry in list_eval_records()}
        self.assertIn("smoke-eval", runs)
        detail = describe_eval("smoke-eval")
        data = detail["files"]["episodes.jsonl"]
        self.assertEqual(data["records"], 2)
        self.assertLessEqual(data["successes"], data["records"])
        for counts in data["funnel"].values():
            self.assertEqual(len(counts), len(data["milestones"]))

    def test_a_run_without_records_is_refused_naming_those_that_have_them(
        self,
    ) -> None:
        with self.assertRaises(KeyError) as ctx:
            describe_eval("no-such-run")
        self.assertIn("smoke-eval", str(ctx.exception))

    def test_an_empty_runs_root_lists_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_eval_records(Path(tmp)), [])


class FrictionCurves(unittest.TestCase):
    @needs_numpy
    def test_the_curve_is_plottable_and_load_adds_friction(self) -> None:
        curve = friction_curve("feetech_sts3215_7_4V", "m6", points=11)
        self.assertEqual(len(curve["velocity"]), 11)
        self.assertEqual(len(curve["unloaded"]), 11)
        self.assertEqual(len(curve["loaded"]), 11)
        self.assertEqual(curve["velocity"][0], 0.0)
        # M6 carries load-dependent terms: under external torque the
        # budget must exceed the unloaded budget at every velocity.
        for unloaded, loaded in zip(curve["unloaded"], curve["loaded"], strict=True):
            self.assertGreater(loaded, unloaded)


class Runs(unittest.TestCase):
    def test_manifests_are_read_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t9-watch").mkdir()
            manifest = {"name": "t9", "steps": 300}
            (root / "t9-watch" / "run.json").write_text(json.dumps(manifest))
            runs = describe_runs(root)
            self.assertEqual(runs, [{"run": "t9-watch", "manifest": manifest}])

    def test_a_missing_runs_directory_is_empty_not_an_error(self) -> None:
        self.assertEqual(describe_runs(Path("/nonexistent/runs")), [])


class ServerFraming(unittest.TestCase):
    @needs_mcp
    def test_the_server_builds_with_every_tool_registered(self) -> None:
        from rq_pipeline.mcp_server import build_server  # noqa: PLC0415

        server = build_server()
        self.assertEqual(server._lowlevel_server.name, "robotiq")


if __name__ == "__main__":
    unittest.main()
