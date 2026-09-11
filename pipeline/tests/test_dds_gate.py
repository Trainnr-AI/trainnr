"""The second gate's pieces without their stack: the pad's mapping,
the DDS runtime over a fake bus and pad, the staging layout against a
fake reference tree, the platform seam."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.deploy.dds_runtime import (
    NETWORK,
    STATE_TIMEOUT_MS,
    DdsRuntime,
    SdkBus,
)
from rq_pipeline.deploy.gamepad import (
    AXIS_MAX,
    VirtualPad,
    axis_value,
    sticks_for_command,
)
from rq_pipeline.deploy.manifest import MANIFEST_SCHEMA, Manifest
from rq_pipeline.deploy.runtimes import runtime_spec
from rq_pipeline.deploy.unitree_stage import (
    REFERENCE_CACHE,
    REFERENCE_ENV,
    SIM_BINARY,
    STAGE_DIR,
    reference_dir,
    stage,
    stage_simulator,
)
from rq_pipeline.deploy.unitree_yaml import UNITREE_DEPLOY_FILE

UNITREE = {
    "robot": "go2",
    "controller": "go2_ctrl",
    "scene": "src/assets/robots/unitree_go2/xmls/scene_go2.xml",
    "source": "a test",
}


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
        self.log: list[tuple[Any, ...]] = []

    def sticks(self, **positions: float) -> None:
        self.log.append(("sticks", positions))

    def chord(self, trigger: str, button: str) -> None:
        self.log.append(("chord", trigger, button))

    def close(self) -> None:
        self.log.append(("close",))


class _Bus:
    def __init__(self, quat: list[float], velocity: list[float]) -> None:
        self.quat, self.velocity = np.asarray(quat, float), np.asarray(velocity, float)

    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray] | None:
        return self.quat, self.velocity


class _Silent:
    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray] | None:
        return None


def _manifest(root: Path = Path("/nowhere"), **overrides: object) -> Manifest:
    raw: dict[str, Any] = {
        "control": {
            "physics_timestep_s": 0.005,
            "decimation": 4,
            "control_hz": 50,
            "episode_length_s": 1.0,
        },
        "termination": {"fell_over_deg": 70.0},
        "unitree": UNITREE,
    }
    raw.update(overrides)
    return Manifest(root=root, raw=raw)


def _runtime(bus: Any, pad: Any = None, **kw: Any) -> DdsRuntime:
    return DdsRuntime(_manifest(), bus=bus, pad=pad or _Pad(), sleep=_still, **kw)


class TheRuntime(unittest.TestCase):
    def test_reset_walks_their_state_machine_through_the_pad(self) -> None:
        pad = _Pad()
        rt = _runtime(_Bus([1, 0, 0, 0], [0, 0, 0]), pad)
        rt.reset()
        chords = [e for e in pad.log if e[0] == "chord"]
        self.assertEqual(chords[:2], [("chord", "LT", "up")] * 2)
        self.assertEqual(chords[2:], [("chord", "RT", "A")] * 2)
        self.assertEqual(rt.ticks, 0)

    def test_apply_moves_the_sticks_to_the_command(self) -> None:
        pad = _Pad()
        rt = _runtime(_Bus([1, 0, 0, 0], [0, 0, 0]), pad)
        rt.command = np.array([0.5, 0.0, 0.1], dtype=np.float32)
        rt.apply(np.zeros(0))
        self.assertAlmostEqual(pad.log[-1][1]["ly"], 0.5)
        self.assertAlmostEqual(pad.log[-1][1]["rx"], -0.1, places=6)
        self.assertEqual(rt.ticks, 1)

    def test_velocity_is_rotated_into_the_body_and_a_fall_is_seen(self) -> None:
        # yawed 90 degrees about z: a world +x velocity reads as body -y
        half = np.sqrt(0.5)
        rt = _runtime(_Bus([half, 0, 0, half], [1.0, 0, 0]))
        rt.observe()
        self.assertTrue(np.allclose(rt.base_velocity_b(), [0.0, -1.0, 0.0], atol=1e-9))
        self.assertFalse(rt.fell_over())
        on_its_back = _runtime(_Bus([0, 1, 0, 0], [0, 0, 0]))
        on_its_back.observe()
        self.assertTrue(on_its_back.fell_over())

    def test_observe_paces_at_the_manifests_rate_on_the_injected_clock(self) -> None:
        slept: list[float] = []
        rt = DdsRuntime(
            _manifest(),
            bus=_Bus([1, 0, 0, 0], [0, 0, 0]),
            pad=_Pad(),
            sleep=slept.append,
            clock=lambda: 100.0,
        )
        rt.observe()
        rt.observe()
        self.assertEqual(len(slept), 2)
        self.assertAlmostEqual(slept[0], 0.02)
        self.assertAlmostEqual(slept[1], 0.04)

    def test_a_silent_bus_is_a_timeout_that_names_the_topic(self) -> None:
        rt = _runtime(_Silent())
        with self.assertRaises(TimeoutError) as caught:
            rt.observe()
        self.assertIn("sportmodestate", str(caught.exception))
        self.assertIn(str(STATE_TIMEOUT_MS), str(caught.exception))

    def test_the_pad_bounds_the_command_and_the_registry_names_the_record(self) -> None:
        self.assertEqual(DdsRuntime.command_limit, 1.0)
        self.assertEqual(runtime_spec("dds").record_file, "gate-dds.json")

    def test_the_instrument_names_the_manifests_controller(self) -> None:
        rt = _runtime(_Bus([1, 0, 0, 0], [0, 0, 0]))
        self.assertIn("go2_ctrl", rt.instrument)
        bare = DdsRuntime(
            _manifest(unitree=None), bus=_Silent(), pad=_Pad(), sleep=_still
        )
        self.assertIn("unrecorded", bare.instrument)


@unittest.skipIf(sys.platform == "linux", "the seam refuses only off Linux")
class ThePlatformSeam(unittest.TestCase):
    def test_the_linux_only_pieces_refuse_by_name_here(self) -> None:
        for make in (VirtualPad, SdkBus):
            with self.assertRaises(RuntimeError) as caught:
                make()
            self.assertIn("linux only", str(caught.exception))
            self.assertIn(sys.platform, str(caught.exception))


def _reference(tmp: Path) -> Path:
    """A fake checkout: their two binaries built, their controller config."""
    reference = tmp / "ref"
    ctrl = reference / "deploy" / "robots" / "go2"
    (ctrl / "build").mkdir(parents=True)
    (ctrl / "build" / "go2_ctrl").write_bytes(b"#!/bin/sh\n")
    (ctrl / "config").mkdir()
    (ctrl / "config" / "config.yaml").write_text("FSM: {}\n")
    (reference / "simulate" / "build").mkdir(parents=True)
    (reference / "simulate" / "build" / SIM_BINARY).write_bytes(b"#!/bin/sh\n")
    (reference / "simulate" / "config.yaml").write_text('robot: "g1"\n')
    return reference


def _deployment(tmp: Path, **overrides: object) -> Manifest:
    root = tmp / "dep"
    root.mkdir()
    raw: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "control": {
            "physics_timestep_s": 0.005,
            "decimation": 4,
            "control_hz": 50,
            "episode_length_s": 20.0,
        },
        "joints": {
            "policy_order": [f"j{i}" for i in range(12)],
            "action_to_ctrl": list(range(12)),
            "sdk_order_map": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
            "stiffness": [20.0] * 12,
            "damping": [1.0] * 12,
            "default_pos": [0.0] * 12,
        },
        "action": {"scale": [0.25] * 12, "offset": [0.0] * 12, "clip": None},
        "commands": {
            "twist": {
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-0.5, 0.5],
                "ang_vel_z": [-0.7, 0.7],
            }
        },
        "observations": [{"name": "joint_pos", "width": 12, "source": "joint_pos_rel"}],
        "onnx": {"file": "policy.onnx", "input_width": 12, "output_width": 12},
        "scene": {"file": "scene.xml"},
        "unitree": UNITREE,
    }
    raw.update(overrides)
    (root / "deploy.json").write_text(json.dumps(raw))
    (root / "policy.onnx").write_bytes(b"\x00")
    (root / "scene.xml").write_text("<mujoco/>")
    return Manifest(root=root, raw=raw)


class TheStage(unittest.TestCase):
    def test_the_layout_their_controller_expects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = _reference(Path(tmp))
            manifest = _deployment(Path(tmp))
            proj = stage(manifest, reference)
            self.assertEqual(proj, manifest.root / STAGE_DIR)
            self.assertTrue((proj / "build" / "go2_ctrl").is_file())
            self.assertFalse((proj / "build" / "go2_ctrl").is_symlink())
            self.assertTrue((proj / "config" / "config.yaml").is_file())
            version = proj / "config" / "policy" / "velocity" / "v0"
            self.assertTrue((version / "params" / UNITREE_DEPLOY_FILE).is_file())
            self.assertTrue((version / "exported" / "policy.onnx").is_file())

    def test_their_simulator_runs_from_our_folder_and_their_tree_is_untouched(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = _reference(Path(tmp))
            manifest = _deployment(Path(tmp))
            sim = stage_simulator(manifest, reference, device=Path("/dev/input/js3"))
            self.assertTrue((sim / "build" / SIM_BINARY).is_file())
            text = (sim / "config.yaml").read_text()
            self.assertIn('robot: "go2"', text)
            self.assertIn(str(reference / UNITREE["scene"]), text)
            self.assertIn("/dev/input/js3", text)
            self.assertIn(f'interface: "{NETWORK}"', text)
            self.assertEqual(
                (reference / "simulate" / "config.yaml").read_text(), 'robot: "g1"\n'
            )

    def test_a_deployment_without_their_stack_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = _reference(Path(tmp))
            manifest = _deployment(Path(tmp), unitree=None)
            with self.assertRaises(ValueError) as caught:
                stage(manifest, reference)
            self.assertIn(str(manifest.root), str(caught.exception))

    def test_a_missing_binary_names_the_build_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = _reference(Path(tmp))
            (reference / "deploy" / "robots" / "go2" / "build" / "go2_ctrl").unlink()
            with self.assertRaises(FileNotFoundError) as caught:
                stage(_deployment(Path(tmp)), reference)
            self.assertIn("go2_ctrl", str(caught.exception))

    def test_the_reference_comes_from_the_argument_the_env_or_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(reference_dir(Path(tmp)), Path(tmp).resolve())
            saved = os.environ.get(REFERENCE_ENV)
            os.environ[REFERENCE_ENV] = tmp
            try:
                self.assertEqual(reference_dir(), Path(tmp).resolve())
            finally:
                if saved is None:
                    os.environ.pop(REFERENCE_ENV, None)
                else:
                    os.environ[REFERENCE_ENV] = saved
            if saved is None:
                self.assertEqual(
                    reference_dir(), REFERENCE_CACHE.expanduser().resolve()
                )
