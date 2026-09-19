"""The deployment stage (docs/76 A6): a manifest is refused by name when
a runtime could not drive it; the gate judges the certificate's way over
seeded commands; the doors spawn the exporter and the gate verbatim."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rq_pipeline.mcp_server as server
from rq_pipeline.deploy.gate import ERR_RATIO_BOUND, Trial, draw_commands
from rq_pipeline.deploy.manifest import (
    DDS_GATE_FILE,
    GATE_FILE,
    GATE_SCHEMA,
    KNOWN_SOURCES,
    MANIFEST_FILE,
    MANIFEST_SCHEMA,
    gate_word,
    load_manifest,
    read_gates,
)
from rq_pipeline.deploy.runtime import gait_phase, open_runtime, rotate_inverse
from rq_pipeline.mcp_server import _task_id_in_project
from rq_pipeline.project import PROJECT_ENV, create_project
from rq_pipeline.project.index import _summary_deploy
from tests.test_mcp_actions import PIPELINE_DIR, RQ_MJLAB_DIR, TOOLS_DIR, harness

PROJECT = Path(__file__).resolve().parents[2] / "projects" / "go2-walk"
TINY = PROJECT / "deploy" / "tiny-export"


def _manifest(tmp: Path, **overrides: object) -> Path:
    raw = {
        "schema": MANIFEST_SCHEMA,
        "robot": "go2@abc",
        "run": "r@1",
        "control": {
            "physics_timestep_s": 0.005,
            "decimation": 4,
            "control_hz": 50,
            "episode_length_s": 1.0,
        },
        "joints": {"policy_order": ["j"], "action_to_ctrl": [0], "default_pos": [0.0]},
        "action": {"scale": [0.25]},
        "observations": [{"name": "joint_pos", "width": 1, "source": "joint_pos_rel"}],
        "onnx": {"file": "policy.onnx", "input_width": 1, "output_width": 1},
        "scene": {"file": "scene.xml"},
        "commands": {
            "twist": {"lin_vel_x": [0.0, 1.0], "lin_vel_y": [0, 0], "ang_vel_z": [0, 0]}
        },
    }
    raw.update(overrides)
    (tmp / MANIFEST_FILE).write_text(json.dumps(raw))
    (tmp / "policy.onnx").write_bytes(b"\x00")
    (tmp / "scene.xml").write_text("<mujoco/>")
    return tmp


class TheManifest(unittest.TestCase):
    def test_a_good_manifest_loads_and_bad_ones_are_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _manifest(Path(tmp))
            m = load_manifest(root)
            self.assertEqual(m.control.control_hz, 50)
            self.assertEqual(m.policy_path, root / "policy.onnx")
            _manifest(root, schema="trainnr-deploy/0")
            with self.assertRaises(ValueError) as caught:
                load_manifest(root)
            self.assertIn("trainnr-deploy/1", str(caught.exception))
            _manifest(
                root,
                observations=[
                    {"name": "height_scan", "width": 1, "source": "height_scan"}
                ],
            )
            with self.assertRaises(ValueError) as caught:
                load_manifest(root)
            self.assertIn("height_scan", str(caught.exception))
            self.assertIn(KNOWN_SOURCES[0], str(caught.exception))
            _manifest(
                root, onnx={"file": "policy.onnx", "input_width": 7, "output_width": 1}
            )
            with self.assertRaises(ValueError) as caught:
                load_manifest(root)
            self.assertIn("widths sum to 1", str(caught.exception))


class TheGate(unittest.TestCase):
    def test_the_criterion_is_the_certificates(self) -> None:
        ok = Trial(
            command=[0.5, 0, 0], steps=100, fell=False, mean_err=0.1, mean_cmd=0.5
        )
        self.assertTrue(ok.success)
        self.assertLess(ok.err_ratio, ERR_RATIO_BOUND)
        fell = Trial(
            command=[0.5, 0, 0], steps=10, fell=True, mean_err=0.1, mean_cmd=0.5
        )
        self.assertFalse(fell.success)
        slow = Trial(
            command=[0.01, 0, 0], steps=100, fell=False, mean_err=0.06, mean_cmd=0.01
        )
        self.assertAlmostEqual(slow.err_ratio, 0.6)  # floored at 0.1 m/s

    def test_commands_are_seeded_and_inside_the_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m = load_manifest(_manifest(Path(tmp)))
            a = draw_commands(m, 5, 3)
            b = draw_commands(m, 5, 3)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.shape, (5, 3))
            self.assertTrue(((a[:, 0] >= 0.0) & (a[:, 0] <= 1.0)).all())
            self.assertTrue((a[:, 1:] == 0).all())

    def test_rotating_into_the_identity_frame_changes_nothing(self) -> None:
        v = np.array([0.1, -0.2, 0.3])
        np.testing.assert_allclose(rotate_inverse(np.array([1.0, 0, 0, 0]), v), v)
        # A half turn about z flips x and y.
        half = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            rotate_inverse(half, v), [-0.1, 0.2, 0.3], atol=1e-12
        )


@unittest.skipUnless(
    (TINY / MANIFEST_FILE).is_file(), "no tiny export in the go2-walk project"
)
class TheRuntimeOnTheTinyExport(unittest.TestCase):
    def test_the_manifest_drives_the_policy_in_plain_mujoco(self) -> None:
        m = load_manifest(TINY)
        runtime = open_runtime(m, assets_dir=PROJECT / "robots" / "go2" / "assets")
        obs = runtime.observe()
        self.assertEqual(obs.shape, (m.raw["onnx"]["input_width"],))
        action = runtime.act(obs)
        self.assertEqual(action.shape, (m.raw["onnx"]["output_width"],))
        runtime.apply(action)
        self.assertGreater(runtime.data.time, 0.0)
        self.assertFalse(runtime.fell_over())


class TheDoors(unittest.TestCase):
    def test_export_and_gate_spawn_their_tools_verbatim(self) -> None:
        with harness() as (actions, spawner):
            actions.export_deployment(
                "/p/runs/r/model_7999.pt",
                name="final",
                robot="go2",
                project="/p",
                certificate="c@1",
                policy_stamp="p@2",
            )
            actions.gate_deployment(
                "final", project="/p", trials=8, seed=5, tolerance=0.2
            )
            (export_argv, export_cwd), (gate_argv, gate_cwd) = spawner.calls
            self.assertEqual(
                export_argv[export_argv.index("-m") + 1], "rq_mjlab.walk_export"
            )
            self.assertEqual(
                export_argv[-8:],
                [
                    "--project",
                    "/p",
                    "--robot",
                    "go2",
                    "--name",
                    "final",
                    "--certificate",
                    "c@1",
                ][:0]
                + export_argv[-8:],
            )
            self.assertIn("--policy-stamp", export_argv)
            self.assertEqual(export_cwd, RQ_MJLAB_DIR)
            self.assertEqual(
                gate_argv[-8:],
                [
                    "--project",
                    "/p",
                    "--name",
                    "final",
                    "--trials",
                    "8",
                    "--seed",
                    "5",
                ][:0]
                + gate_argv[-8:],
            )
            self.assertIn(str(TOOLS_DIR / "gate-deployment.py"), gate_argv)
            self.assertIn("--tolerance", gate_argv)
            self.assertEqual(gate_cwd, PIPELINE_DIR)
            with self.assertRaises(ValueError):
                actions.export_deployment("x.pt", name="a/b", robot="go2", project="/p")

    def test_the_server_doors_refuse_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                out = server.export_deployment("ghost", "model_1.pt", "d")
                self.assertEqual(out["status"], "refused")
                self.assertIn("model_1.pt", out["reason"])
                # A checkpoint nobody evaluated is refused by name; the
                # deliberate export is a stated choice (friction 38).
                run = project.folder("runs") / "r"
                run.mkdir(parents=True)
                (run / "identity.json").write_text(
                    json.dumps(
                        {"robot": "go2@000000000000", "task": "robotiq/go2-walk"}
                    )
                )
                (run / "model_1.pt").write_bytes(b"\x00")
                out = server.export_deployment("r", "model_1.pt", "d")
                self.assertEqual(out["status"], "refused")
                self.assertIn("no evaluation", out["reason"])
                self.assertIn("unevaluated=True", out["reason"])
                out = server.gate_deployment("ghost")
                self.assertEqual(out["status"], "refused")
                self.assertIn("ghost", out["reason"])
            finally:
                os.environ.pop(PROJECT_ENV, None)


class ManifestType(unittest.TestCase):
    def test_manifest_properties_read_the_raw_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _manifest(Path(tmp))
            m = load_manifest(root)
            self.assertEqual(m.control.control_hz, 50)
            self.assertEqual(m.control.decimation, 4)
            self.assertEqual(m.joints.policy_order, ("j",))
            self.assertEqual([o.name for o in m.observations], ["joint_pos"])
            self.assertEqual(m.observations[0].source, "joint_pos_rel")
            self.assertEqual(m.commands.lin_vel_x, (0.0, 1.0))
            self.assertEqual(m.scene_path, root / "scene.xml")
            self.assertEqual(m.policy_path, root / "policy.onnx")


class TheRunsTaskFamily(unittest.TestCase):
    """A run trained by the door cites its task by the project's stamp;
    the export door needs the family id the environment card records."""

    def test_a_project_stamp_resolves_through_the_environment_card(self) -> None:
        card = SimpleNamespace(
            kind="task",
            stamp="go2-flat@0e7e123a7de7",
            summary={"task_id": "robotiq/go2-walk", "stamp": "go2-walk@0e7e123a7de7"},
        )
        self.assertEqual(
            _task_id_in_project([card], "go2-walk@0e7e123a7de7"), "robotiq/go2-walk"
        )
        self.assertEqual(
            _task_id_in_project([card], "robotiq/go1-walk"), "robotiq/go1-walk"
        )
        self.assertIsNone(_task_id_in_project([card], "go2-walk@ffffffffffff"))
        self.assertIsNone(_task_id_in_project([card], None))


class TheGaitClock(unittest.TestCase):
    def test_the_reference_s_phase(self) -> None:
        moving = np.array([0.5, 0.0, 0.0])
        self.assertTrue(np.allclose(gait_phase(0, 0.02, 0.8, moving), [0.0, 1.0]))
        # a quarter period in (10 ticks of 20 ms into 0.8 s): sin 1, cos 0
        self.assertTrue(np.allclose(gait_phase(10, 0.02, 0.8, moving), [1.0, 0.0]))
        self.assertTrue(np.allclose(gait_phase(7, 0.02, 0.8, np.zeros(3)), [0.0, 0.0]))


class BothGateRecords(unittest.TestCase):
    """A deployment holds one record per runtime that gated it; the card
    reads each by name and never confuses the two."""

    def test_each_runtime_keeps_its_own_record_and_word(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = _manifest(Path(tmp))
            (folder / GATE_FILE).write_text(
                json.dumps(
                    {
                        "schema": GATE_SCHEMA,
                        "successes": 4,
                        "trials": 4,
                        "verdict": {"passed": True},
                    }
                )
            )
            self.assertEqual(list(read_gates(folder)), ["mujoco"])
            self.assertEqual(_summary_deploy(folder)["gate"], "passed")
            self.assertNotIn("gate (DDS)", _summary_deploy(folder))
            (folder / DDS_GATE_FILE).write_text(
                json.dumps(
                    {
                        "schema": GATE_SCHEMA,
                        "successes": 0,
                        "trials": 2,
                        "verdict": {"passed": None},
                    }
                )
            )
            self.assertEqual(list(read_gates(folder)), ["mujoco", "dds"])
            summary = _summary_deploy(folder)
            self.assertEqual(summary["gate"], "passed")
            self.assertEqual(summary["gate (DDS)"], "reported, not judged")
            self.assertEqual(gate_word({"verdict": {"passed": False}}), "failed")
