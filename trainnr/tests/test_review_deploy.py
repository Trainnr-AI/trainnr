"""Pins for the deploy-side review of 2026-09-24: each test names the bug
that existed, and fails on the code before its fix."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from trainnr.bundles.basis import BASIS_OWN, BASIS_PUBLIC, BASIS_UNKNOWN
from trainnr.deploy import gate as gate_mod
from trainnr.deploy import preflight as pf
from trainnr.deploy.dds_runtime import (
    MS_PER_S,
    PASSIVE_CHORD,
    STATE_TIMEOUT_MS,
    DdsRuntime,
    SdkBus,
)
from trainnr.deploy.manifest import GATE_SCHEMA, Manifest
from trainnr.project import PROJECT_ENV
from trainnr.project.index import (
    HIDDEN_SUMMARY_KEYS,
    Artifact,
    _states,
)
from trainnr.project.kinds import Kind

RANGES = SimpleNamespace(
    commands=SimpleNamespace(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
    )
)


def _gate_record(folder: Path, *, draw: str | None, seed: int = 1000) -> None:
    protocol: dict[str, Any] = {"trials": 20, "seed": seed}
    if draw is not None:
        protocol[gate_mod.DRAW_KEY] = draw
    (folder / "gate.json").write_text(
        json.dumps(
            {
                "schema": GATE_SCHEMA,
                "successes": 20,
                "trials": 20,
                "protocol": protocol,
                "verdict": {"passed": True},
            }
        )
    )


class TheGateIsPairedByTrialIndex(unittest.TestCase):
    """Bug: each axis was drawn as one vector the length of the trial
    count, so trial i of 4 was not trial i of 20 (pre-flight and
    attribution ran other commands than the gate they cited)."""

    def test_the_first_k_trials_are_the_same_at_any_count(self) -> None:
        few = gate_mod.draw_commands(RANGES, 4, 1000)
        many = gate_mod.draw_commands(RANGES, 20, 1000)
        np.testing.assert_array_equal(few, many[:4])

    def test_a_trials_axes_are_drawn_apart(self) -> None:
        """Found re-running the gate on 2026-09-24: a generator per axis
        drew one number three times, so forward, sideways and turn moved
        together (0.043, 0.043, 0.021)."""
        drawn = gate_mod.draw_commands(RANGES, 20, 1000)
        lo = np.array([-1.0, -0.5, -1.0])
        span = np.array([2.0, 1.0, 2.0])
        unit = (drawn - lo) / span  # each axis's own uniform
        self.assertFalse(np.allclose(unit[:, 0], unit[:, 1]))
        self.assertFalse(np.allclose(unit[:, 0], unit[:, 2]))

    def test_a_record_without_the_field_is_the_old_draw_and_refused(self) -> None:
        self.assertEqual(gate_mod.draw_of({"protocol": {}}), gate_mod.DRAW_BY_COUNT)
        with self.assertRaisesRegex(ValueError, "re-run the gate"):
            gate_mod.require_same_draw({"protocol": {}}, "the mujoco gate of x")
        gate_mod.require_same_draw(
            {"protocol": {gate_mod.DRAW_KEY: gate_mod.DRAW_NOW}}, "x"
        )

    def test_the_pre_flight_takes_the_gates_seed_and_refuses_another(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            seed, words = pf.gate_twists(folder, None)
            self.assertEqual(seed, gate_mod.DEFAULT_SEED)
            self.assertIn("no mujoco gate record", words)
            _gate_record(folder, draw=gate_mod.DRAW_NOW, seed=7)
            seed, words = pf.gate_twists(folder, None)
            self.assertEqual(seed, 7)
            self.assertIn("first", words)
            with self.assertRaisesRegex(ValueError, "seed 7"):
                pf.gate_twists(folder, 8)
            _gate_record(folder, draw=None)
            with self.assertRaisesRegex(ValueError, "by-count"):
                pf.gate_twists(folder, None)


class AttributionRefusesWithoutAProject(unittest.TestCase):
    """Bug 1: the door raised FileNotFoundError with no project open."""

    def test_the_door_refuses_by_name(self) -> None:
        from trainnr.mcp_server import attribute_deployment  # noqa: PLC0415

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {PROJECT_ENV: str(Path(tmp) / "nowhere")}),
        ):
            answer = attribute_deployment("x")
        self.assertEqual(answer["status"], "refused")
        self.assertIn("no project", answer["reason"])


def _artifact(kind: Kind, stamp: str, **summary: Any) -> Artifact:
    return Artifact(kind=kind.value, stamp=stamp, path=stamp, summary=summary)


class TheStageBasis(unittest.TestCase):
    """Bug 3: every present stage with no basis read "own robot" (a
    policy trained on a public-log fit said so on hover). Bug 4: the Sys
    ID basis ignored fit artifacts. Bug 5: `fit_bases` showed on every
    robot card."""

    def test_only_recordings_and_fits_have_a_basis(self) -> None:
        states = {
            s.name: s
            for s in _states(
                [
                    _artifact(Kind.RECORDING, "log@1", basis=BASIS_PUBLIC),
                    _artifact(Kind.ROBOT, "go2@1", files=["go2.xml"], fit_bases=[]),
                    _artifact(Kind.RUN, "run@1"),
                    _artifact(Kind.DEPLOY, "dep@1"),
                ]
            )
        }
        self.assertEqual(states["telemetry recorded"].basis, BASIS_PUBLIC)
        for name in ("asset onboarded", "policy trained", "deployment exported"):
            self.assertTrue(states[name].present, name)
            self.assertIsNone(states[name].basis, name)

    def test_a_silent_recording_is_the_operators_own(self) -> None:
        states = {s.name: s for s in _states([_artifact(Kind.RECORDING, "own@1")])}
        self.assertEqual(states["telemetry recorded"].basis, BASIS_OWN)

    def test_a_fit_artifacts_own_basis_counts(self) -> None:
        states = {
            s.name: s
            for s in _states(
                [
                    _artifact(Kind.FIT, "fit@1", basis=BASIS_OWN),
                    _artifact(
                        Kind.ROBOT, "go2@1", files=["fits"], fit_bases=[BASIS_PUBLIC]
                    ),
                ]
            )
        }
        self.assertEqual(states["system identified"].basis, BASIS_OWN)
        states = {s.name: s for s in _states([_artifact(Kind.FIT, "fit@1")])}
        self.assertEqual(states["system identified"].basis, BASIS_UNKNOWN)

    def test_the_fit_bases_are_hidden_from_the_card(self) -> None:
        self.assertIn("fit_bases", HIDDEN_SUMMARY_KEYS)


class _Pad:
    def __init__(self) -> None:
        self.log: list[tuple[str, ...]] = []

    def sticks(self, **positions: float) -> None:
        self.log.append(("sticks",))

    def chord(self, trigger: str, button: str) -> None:
        self.log.append(("chord", trigger, button))

    def close(self) -> None:
        self.log.append(("close",))


class _Bus:
    def __init__(self, *, fail_health: bool = False) -> None:
        self.fail_health = fail_health

    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0, 0, 0, 0]), np.zeros(3)

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array([0.0, 0.0, 0.3]), np.array([1.0, 0, 0, 0]), np.zeros(20)

    def health(self) -> dict[str, Any]:
        if self.fail_health:
            raise RuntimeError("no health yet")
        return {
            "quaternion": np.array([1.0, 0, 0, 0]),
            "gyroscope": np.zeros(3),
            "motor_speed": np.zeros(12),
            "motor_temperature": np.zeros(12),
            "battery_percent": 0.0,
        }


def _dds(bus: _Bus, pad: _Pad) -> DdsRuntime:
    manifest = Manifest(
        raw={
            "control": {
                "physics_timestep_s": 0.005,
                "decimation": 4,
                "control_hz": 50,
                "episode_length_s": 1.0,
            },
            "termination": {"fell_over_deg": 70.0},
        },
        root=Path("/nowhere"),
    )
    rt = DdsRuntime(manifest, bus=bus, pad=pad, sleep=lambda _s: None)
    rt._clock = lambda: 0.0  # no pacing under test
    return rt


class TheDdsStop(unittest.TestCase):
    def test_the_record_says_how_many_ticks_it_walked(self) -> None:
        """Bug 6: a STOP file ended the walk at tick 1 and the record still
        read "walking ... for 2 s"."""
        with tempfile.TemporaryDirectory() as tmp:
            stop = pf.request_stop(Path(tmp))
            out = pf.measure_dds_stop(
                _dds(_Bus(), _Pad()),
                np.array([0.5, 0, 0]),
                stand_in=True,
                stop_file=stop,
            )
        self.assertEqual(out["ticks_walked"], 0)
        self.assertIn("for 0 of", out["while"])

    def test_a_stale_stop_file_is_refused_by_name(self) -> None:
        """Bug 6: nothing cleared the STOP file, so the next run stopped
        at its first tick."""
        with tempfile.TemporaryDirectory() as tmp:
            pf.refuse_stale_stop(Path(tmp))  # nothing there: no refusal
            pf.request_stop(Path(tmp), "hand on the button")
            with self.assertRaisesRegex(ValueError, "hand on the button"):
                pf.refuse_stale_stop(Path(tmp))

    def test_passive_is_sent_when_the_walk_fails(self) -> None:
        """Bug 7: a health that could not be read after the handover left
        their controller in velocity mode."""
        pad = _Pad()
        with self.assertRaises(RuntimeError):
            pf.measure_dds_stop(
                _dds(_Bus(fail_health=True), pad), np.array([0.5, 0, 0]), stand_in=True
            )
        self.assertEqual(pad.log[-1], ("chord", *PASSIVE_CHORD))


class TheLink(unittest.TestCase):
    """Bug 8: `link_lost` measured the time since our own read (always ~0)
    and could never trip, yet was listed as covered; and the SDK's `Read`
    takes seconds, so a 1000 ms budget waited 1000 s."""

    def test_the_link_is_enforced_by_the_read_timeout(self) -> None:
        link = pf.WATCHDOGS["link_lost"]
        self.assertIsNone(link.reading)
        self.assertIsNotNone(link.enforced_by)
        health = pf.Health(
            tilt_rad=0.0, max_joint_velocity=0.0, max_angular_velocity=0.0
        )
        self.assertIsNone(health.value("link_lost"))
        self.assertEqual(pf.tripped(health), [])

    def test_the_sdk_read_is_given_seconds(self) -> None:
        waits: list[float] = []

        class Reader:
            def __init__(self, sample: Any) -> None:
                self.sample = sample

            def Read(self, timeout: float) -> Any:  # noqa: N802 - the SDK's name
                waits.append(timeout)
                return self.sample

        motor = SimpleNamespace(q=0.0, dq=0.0, temperature=30)
        low = SimpleNamespace(
            imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]),
            motor_state=[motor] * 20,
            bms_state=SimpleNamespace(soc=90),
        )
        state = SimpleNamespace(position=[0, 0, 0.3], velocity=[0, 0, 0])
        bus = object.__new__(SdkBus)
        bus.network, bus.motors, bus._pose = "lo", 12, None
        bus._state, bus._low = Reader(state), Reader(low)
        bus.latest(STATE_TIMEOUT_MS)
        self.assertLessEqual(max(waits), STATE_TIMEOUT_MS / MS_PER_S)


class ThePreflightCard(unittest.TestCase):
    def test_only_the_operators_own_robot_goes_unqualified(self) -> None:
        verdict = {"verdict": "passed 7/7"}
        self.assertEqual(pf.card_line({**verdict, "basis": BASIS_OWN}), "passed 7/7")
        self.assertIn("in simulation", pf.card_line({**verdict, "basis": "simulation"}))
        self.assertIn(
            "stand-in",
            pf.card_line({**verdict, "basis": "simulation", "stand_in": True}),
        )
        self.assertIn("on hearsay", pf.card_line({**verdict, "basis": "hearsay"}))
        fell = {
            **verdict,
            "basis": "simulation",
            "ramp_in": {pf.WITH_RAMP: {"stood": False}},
        }
        self.assertIn("did not stand", pf.card_line(fell))


class TheDeploymentsIdentity(unittest.TestCase):
    """A STOP file, a pre-flight record or an attribution beside the
    manifest moved the deployment's stamp (the review of 2026-09-24):
    they are records ABOUT the deployment, left out of its identity."""

    def test_run_records_leave_the_stamp_alone(self) -> None:
        from trainnr.bundles.hashing import bundle_hash  # noqa: PLC0415
        from trainnr.deploy.manifest import MANIFEST_FILE  # noqa: PLC0415
        from trainnr.project.kinds import (  # noqa: PLC0415
            artifact_hash,
            stamp_kind,
        )

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "dep"
            folder.mkdir()
            (folder / MANIFEST_FILE).write_text("{}", encoding="utf-8")
            (folder / "policy.onnx").write_bytes(b"weights")
            before = stamp_kind(Kind.DEPLOY, folder)
            # the same bytes as bundle_hash when nothing is left out
            self.assertEqual(artifact_hash(Kind.DEPLOY, folder), bundle_hash(folder))
            pf.request_stop(folder)
            (folder / pf.PREFLIGHT_FILE).write_text("{}", encoding="utf-8")
            (folder / "gate.json").write_text("{}", encoding="utf-8")
            (folder / "attribution.json.tmp").write_text("{", encoding="utf-8")
            self.assertEqual(stamp_kind(Kind.DEPLOY, folder), before)
            (folder / "policy.onnx").write_bytes(b"other weights")
            self.assertNotEqual(stamp_kind(Kind.DEPLOY, folder), before)


if __name__ == "__main__":
    unittest.main()
