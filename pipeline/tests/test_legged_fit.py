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


class TheFilterRuns(unittest.TestCase):
    """The balance's low-pass runs on what is fitted, or refuses: an
    hour-long 500 Hz log thinned by the sample budget alone fell to 17 Hz,
    under the 15 Hz band, and the filter silently did nothing while the
    record said it ran (review 2026-09-24)."""

    def test_the_stride_keeps_the_band_where_the_budget_would_not(self) -> None:
        from rq_pipeline.robot.legged_fit import Decimation  # noqa: PLC0415

        hour_at_500 = 500 * 3600
        by_budget = -(-hour_at_500 // Decimation().max_samples)
        stride = Decimation().stride(hour_at_500, 500.0, 15.0)
        self.assertEqual(by_budget, 30)  # 16.7 Hz: under the band
        self.assertEqual(stride, 8)  # 62.5 Hz: Nyquist twice the band
        self.assertEqual(Decimation().stride(hour_at_500, 500.0, 0.0), 30)

    def test_a_band_the_rate_cannot_hold_is_refused(self) -> None:
        import numpy as np  # noqa: PLC0415

        from rq_pipeline.robot.torque_balance import Bandwidth, lowpass  # noqa: PLC0415

        values = np.zeros((100, 2))
        with self.assertRaisesRegex(ValueError, "15 Hz band needs samples faster"):
            lowpass(values, 20.0, Bandwidth(cutoff_hz=15.0))
        self.assertIs(lowpass(values, 20.0, Bandwidth(cutoff_hz=0.0)), values)


@needs_sim
class VendorNamesAndIdleStarts(unittest.TestCase):
    """A recording named the way Unitree's bus names motors (`FR_hip`, not
    the model's `FR_hip_joint`) reaches the fit through the declared
    rename table, and a log whose motors idle at the start is judged on
    its whole effort channel (review 2026-09-24: every live capture was
    refused as 'not hinges of the model', and a two-sample look refused
    an idle start)."""

    def _vendor_named(self, root: Path) -> tuple[Path, Path]:
        import dataclasses  # noqa: PLC0415

        from rq_pipeline.robot.quadruped_synth import simulate  # noqa: PLC0415
        from rq_pipeline.robots.joint_orders import MODEL_JOINT_SUFFIX  # noqa: PLC0415

        bundle = _go2_bundle(root)
        recording, _truth = simulate(bundle / "go2.xml")
        channels = {}
        for name, channel in recording.channels.items():
            components = tuple(
                c.removesuffix(MODEL_JOINT_SUFFIX) for c in channel.components
            )
            channels[name] = dataclasses.replace(channel, components=components)
        effort = channels["joint.effort"]
        values = effort.values.copy()
        values[:2] = 0.0  # the motors idle at the start
        channels["joint.effort"] = dataclasses.replace(effort, values=values)
        for name in ("joint.command", "joint.kp", "joint.kd"):
            channels.pop(name, None)
        rec = root / "vendor"
        dataclasses.replace(recording, channels=channels).write(rec)
        return bundle, rec

    def test_accepts_and_fits_with_the_mapping_recorded(self) -> None:
        from rq_pipeline.robot.legged_fit import EFFORT, LeggedJoints  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            bundle, rec = self._vendor_named(Path(tmp))
            method = LeggedJoints()
            self.assertIsNone(method.accepts(bundle, rec))
            fit = method.fit_balance(bundle, rec)
        self.assertEqual(fit.torque_source, EFFORT)
        self.assertTrue(all(j.endswith("_joint") for j in fit.samples.joints))
        self.assertIn("unitree-motors-to-model", fit.base_handling)
        self.assertEqual(len(fit.balance.result.parameters), 36)


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
