"""The legged-joints method behind the identification seam: registered,
refuses by name, fits a synthetic Go2 log into a record that carries the
basis and the metrics, and lights the loop's Sys ID state with that
basis through the door."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tests._extras import needs_sim
from tests._fixtures import go2_model_or_skip

EXPLAINED_FLOOR = 0.95  # the synthetic chirp explains 99 % per joint (2026-09-24)


def _go2_bundle(into: Path) -> Path:
    xml = go2_model_or_skip()
    bundle = into / "go2"
    bundle.mkdir(parents=True)
    shutil.copy2(xml, bundle / xml.name)
    assets = xml.parent / "assets"
    if assets.is_dir():
        shutil.copytree(assets, bundle / "assets")
    return bundle


class Registry(unittest.TestCase):
    def test_the_method_is_registered_beside_the_drivetrain(self) -> None:
        from rq_pipeline.robot.methods import methods, resolve  # noqa: PLC0415

        self.assertIn("legged-joints", methods())
        self.assertEqual(resolve("legged-joints").name, "legged-joints")
        self.assertEqual(resolve("legged-joints").build().anchored, ())


@needs_sim
class Refusals(unittest.TestCase):
    def test_refuses_a_bundle_without_hinges_and_a_recording_without_torque(
        self,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.robot.legged_fit import LeggedJoints  # noqa: PLC0415
        from rq_pipeline.robots.recording import Channel, Recording  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "flat"
            flat.mkdir()
            (flat / "model.xml").write_text(
                '<mujoco><worldbody><body><joint name="j" axis="0 1 0"/>'
                '<geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>'
            )
            rec = root / "rec"
            times = np.arange(0, 1.0, 0.01)
            Recording(
                source="x",
                adapter="test",
                channels={
                    "joint.position": Channel(
                        "joint.position", times, np.zeros((100, 1)), "rad", ("j",)
                    ),
                    "joint.velocity": Channel(
                        "joint.velocity", times, np.zeros((100, 1)), "rad/s", ("j",)
                    ),
                },
            ).write(rec)
            method = LeggedJoints()
            self.assertIn("hinge joints, fewer than", method.accepts(flat, rec) or "")
            bundle = _go2_bundle(root)
            self.assertIn("not hinges of the model", method.accepts(bundle, rec) or "")


@needs_sim
class SyntheticFit(unittest.TestCase):
    def test_a_synthetic_log_fits_into_a_record_with_basis_and_metrics(self) -> None:
        from rq_pipeline.robot.legged_fit import LeggedJoints  # noqa: PLC0415
        from rq_pipeline.robot.methods import detect  # noqa: PLC0415
        from rq_pipeline.robot.quadruped_synth import simulate  # noqa: PLC0415
        from rq_pipeline.robots.recording import BASIS_SIMULATION  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _go2_bundle(root)
            recording, truth = simulate(bundle / "go2.xml")
            rec = root / "synthetic"
            recording.write(rec)
            self.assertEqual(detect(bundle, rec).name, "legged-joints")
            result, path = LeggedJoints().fit(bundle, rec, write=True)
            self.assertIsNotNone(path)
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(raw["basis"], BASIS_SIMULATION)
            self.assertEqual(raw["provenance"]["torque_source"], "measured effort")
            self.assertEqual(set(raw["metrics"]), set(truth))
            self.assertTrue(
                all(m["explained"] > EXPLAINED_FLOOR for m in raw["metrics"].values())
            )
            self.assertIn("TORQUE BALANCE", raw["anchor"])
            self.assertEqual(len(result.parameters), 36)
            self.assertEqual(raw["units"]["FL_calf_joint.armature"], "kg*m^2")


@needs_sim
class Door(unittest.TestCase):
    def test_the_door_lights_sys_id_with_the_recording_basis(self) -> None:
        from rq_pipeline.mcp_server import identify_system  # noqa: PLC0415
        from rq_pipeline.project import (  # noqa: PLC0415
            PROJECT_ENV,
            create_project,
            index_project,
        )
        from rq_pipeline.robot.quadruped_synth import simulate  # noqa: PLC0415
        from rq_pipeline.robots.recording import BASIS_SIMULATION  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            bundle = _go2_bundle(project.folder("robots"))
            recording, _truth = simulate(bundle / "go2.xml")
            # The recording enters through ingest as a directory artifact.
            staged = Path(tmp) / "staged"
            recording.write(staged)
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                index = index_project(project)
                robot = next(a.stamp for a in index.artifacts if a.kind == "robot")
                # A written recording directory is already the artifact shape;
                # copy it into the project's recordings folder under a name.
                target = project.folder("recordings") / "synthetic-chirp"
                shutil.copytree(staged, target)
                index = index_project(project)
                recording_stamp = next(
                    a.stamp for a in index.artifacts if a.kind == "recording"
                )
                before = next(s for s in index.states if s.name == "system identified")
                self.assertFalse(before.present)
                self.assertIsNone(before.basis)
                answer = identify_system(robot, recording_stamp)
                self.assertEqual(answer.get("status"), "done", answer)
                self.assertEqual(answer["method"], "legged-joints")
                self.assertTrue(answer["state"]["system identified"])
                self.assertEqual(answer["state"]["basis"], BASIS_SIMULATION)
                self.assertNotEqual(answer["robot"], robot)
            finally:
                os.environ.pop(PROJECT_ENV, None)
