"""The second gate's pieces without their stack: the pad's mapping,
the DDS runtime over a fake bus and pad, the staging layout."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rq_pipeline.deploy.dds_runtime import DdsRuntime
from rq_pipeline.deploy.gamepad import AXIS_MAX, axis_value, sticks_for_command
from rq_pipeline.deploy.manifest import Manifest
from rq_pipeline.deploy.unitree_stage import stage, write_sim_config

GO2_DEPLOY = Path(
    "/home/prakhar-pc/robotiq/projects/go2-walk/deploy/go2-c2-smoke-deploy"
)


def _still(_seconds: float) -> None:
    return None


class ThePad(unittest.TestCase):
    def test_axis_values_are_their_16_bit_reading(self) -> None:
        self.assertEqual(axis_value(1.0), AXIS_MAX)
        self.assertEqual(axis_value(-0.5), -round(AXIS_MAX / 2))
        self.assertEqual(axis_value(0.5, inverted=True), -round(AXIS_MAX / 2))
        self.assertEqual(axis_value(3.0), AXIS_MAX)

    def test_sticks_for_a_command_follow_their_controller(self) -> None:
        sticks = sticks_for_command(np.array([0.7, 0.2, -0.3]))
        self.assertAlmostEqual(sticks["ly"], 0.7)
        self.assertAlmostEqual(sticks["lx"], -0.2)
        self.assertAlmostEqual(sticks["rx"], 0.3)
        self.assertEqual(sticks["ry"], 0.0)


class _Pad:
    def __init__(self) -> None:
        self.log: list = []

    def sticks(self, **positions: float) -> None:
        self.log.append(("sticks", positions))

    def chord(self, trigger: str, button: str, hold_s: float = 0.0) -> None:
        self.log.append(("chord", trigger, button))

    def close(self) -> None:
        self.log.append(("close",))


class _Bus:
    def __init__(self, quat: list[float], velocity: list[float]) -> None:
        self.quat, self.velocity = np.asarray(quat, float), np.asarray(velocity, float)

    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray]:
        return self.quat, self.velocity


def _manifest() -> Manifest:
    raw = {"control": {"control_hz": 50}, "termination": {"fell_over_deg": 70.0}}
    return Manifest(root=Path("/nowhere"), raw=raw)


class TheRuntime(unittest.TestCase):
    def test_reset_walks_their_state_machine_through_the_pad(self) -> None:
        pad = _Pad()
        rt = DdsRuntime(
            _manifest(), bus=_Bus([1, 0, 0, 0], [0, 0, 0]), pad=pad, sleep=_still
        )
        rt.reset()
        chords = [e for e in pad.log if e[0] == "chord"]
        self.assertEqual(chords[:2], [("chord", "LT", "up")] * 2)
        self.assertEqual(chords[2:], [("chord", "RT", "A")] * 2)
        self.assertEqual(rt.ticks, 0)

    def test_apply_moves_the_sticks_to_the_command(self) -> None:
        pad = _Pad()
        rt = DdsRuntime(
            _manifest(), bus=_Bus([1, 0, 0, 0], [0, 0, 0]), pad=pad, sleep=_still
        )
        rt.command = np.array([0.5, 0.0, 0.1], dtype=np.float32)
        rt.apply(np.zeros(0))
        self.assertAlmostEqual(pad.log[-1][1]["ly"], 0.5)
        self.assertAlmostEqual(pad.log[-1][1]["rx"], -0.1, places=6)
        self.assertEqual(rt.ticks, 1)

    def test_velocity_is_rotated_into_the_body_and_a_fall_is_seen(self) -> None:
        # yawed 90 degrees about z: a world +x velocity reads as body -y
        half = np.sqrt(0.5)
        rt = DdsRuntime(
            _manifest(),
            bus=_Bus([half, 0, 0, half], [1.0, 0, 0]),
            pad=_Pad(),
            sleep=_still,
        )
        rt.observe()
        self.assertTrue(np.allclose(rt.base_velocity_b(), [0.0, -1.0, 0.0], atol=1e-9))
        self.assertFalse(rt.fell_over())
        on_its_back = DdsRuntime(
            _manifest(), bus=_Bus([0, 1, 0, 0], [0, 0, 0]), pad=_Pad(), sleep=_still
        )
        on_its_back.observe()
        self.assertTrue(on_its_back.fell_over())

    def test_the_pad_bounds_the_command_and_the_record_has_its_own_file(self) -> None:
        self.assertEqual(DdsRuntime.command_limit, 1.0)
        self.assertEqual(DdsRuntime.record_file, "gate-dds.json")


class TheStage(unittest.TestCase):
    @unittest.skipUnless(GO2_DEPLOY.is_dir(), "the smoke deployment lives on the box")
    def test_the_layout_their_controller_expects(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        reference = tmp / "ref"
        (reference / "deploy" / "robots" / "go2" / "build").mkdir(parents=True)
        (reference / "deploy" / "robots" / "go2" / "build" / "go2_ctrl").write_bytes(
            b"#!/bin/sh\n"
        )
        (reference / "deploy" / "robots" / "go2" / "config").mkdir()
        (reference / "deploy" / "robots" / "go2" / "config" / "config.yaml").write_text(
            "FSM: {}\n"
        )
        (reference / "simulate").mkdir()
        deployment = tmp / "dep"
        shutil.copytree(
            GO2_DEPLOY,
            deployment,
            ignore=shutil.ignore_patterns("unitree", "gate.json"),
        )
        manifest = Manifest(
            root=deployment, raw=json.loads((deployment / "deploy.json").read_text())
        )
        proj = stage(manifest, reference)
        self.assertTrue((proj / "build" / "go2_ctrl").is_file())
        self.assertTrue((proj / "config" / "config.yaml").is_file())
        version = proj / "config" / "policy" / "velocity" / "v0"
        self.assertTrue((version / "params" / "deploy.yaml").is_file())
        self.assertTrue((version / "exported" / "policy.onnx").is_file())
        config = write_sim_config(reference, device="/dev/input/js3")
        text = config.read_text()
        self.assertIn('robot: "go2"', text)
        self.assertIn("/dev/input/js3", text)
        self.assertIn('interface: "lo"', text)
