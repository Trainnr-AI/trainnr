"""Pre-flight (docs/77 §10): the SDK's watchdog numbers, the checks and
their refusals by name, the ramp-in and the soft stop, the DDS runtime's
stand / handover / stop, the doors, the card and the drawer.

The tests that drive a real exported policy read the go2-c2 deployment
(`projects/` is not tracked): `TRAINNR_PREFLIGHT_DEPLOYMENT` names it,
else the checkout's own; skipped by name when neither exists."""

from __future__ import annotations

import copy
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np

import rq_pipeline.mcp_server as server
from rq_pipeline.deploy import preflight as pf
from rq_pipeline.deploy.dds_runtime import (
    HANDOVER_CHORD,
    PASSIVE_CHORD,
    STAND_CHORD,
    DdsRuntime,
)
from rq_pipeline.deploy.manifest import Manifest
from rq_pipeline.project import PROJECT_ENV, create_project
from rq_pipeline.project.details import preflight_sections
from tests._extras import needs_sim
from tests.test_mcp_actions import PIPELINE_DIR, TOOLS_DIR, harness

ENV = "TRAINNR_PREFLIGHT_DEPLOYMENT"
_DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "projects"
    / "go2-walk"
    / "deploy"
    / "go2-c2-deploy-cited"
)
DEPLOYMENT = Path(os.environ.get(ENV, str(_DEFAULT)))
ASSETS = DEPLOYMENT.parents[1] / "robots" / "go2" / "assets"
REAL = DEPLOYMENT.is_dir() and ASSETS.is_dir()
needs_deployment = unittest.skipUnless(
    REAL, f"no go2-c2 deployment at {DEPLOYMENT} (set {ENV})"
)

AT_REST = pf.Health(tilt_rad=0.0, max_joint_velocity=0.0, max_angular_velocity=0.0)


class TheWatchdogs(unittest.TestCase):
    def test_the_numbers_are_the_sdks(self) -> None:
        """unitree_sdk2 h2/common/terminations.hpp, read 2026-09-24."""
        limits = {w.name: w.limit for w in pf.WATCHDOGS.values()}
        self.assertEqual(
            limits,
            {
                "tilt": 1.0,
                "joint_velocity": 10.0,
                "angular_velocity": 6.0,
                "winding_temperature": 120.0,
                "casing_temperature": 85.0,
                "battery": 20.0,
                "link_lost": 1000.0,
            },
        )
        for w in pf.WATCHDOGS.values():
            self.assertTrue(w.sdk_function)

    def test_what_trips_and_battery_trips_below(self) -> None:
        self.assertEqual(pf.tripped(AT_REST), [])
        tilted = pf.Health(1.2, 0.0, 0.0, battery_percent=15.0)
        self.assertEqual(pf.tripped(tilted), ["tilt", "battery"])
        self.assertEqual(pf.tripped(pf.Health(0.0, 0.0, 0.0, battery_percent=90)), [])

    def test_a_stand_in_s_silent_zeros_are_unreported_a_robot_s_are_read(self) -> None:
        reading = {
            "quaternion": [1.0, 0, 0, 0],
            "gyroscope": [0.0, 0.0, 0.1],
            "motor_speed": [0.0] * 12,
            "motor_temperature": [0.0] * 12,
            "battery_percent": 0.0,
            "read_at": time.monotonic(),
        }
        stand_in = pf.health_of_lowstate(reading, stand_in=True)
        self.assertIsNone(stand_in.battery_percent)
        self.assertIsNone(stand_in.max_casing_temperature)
        self.assertEqual(pf.tripped(stand_in), [])
        robot = pf.health_of_lowstate(reading, stand_in=False)
        self.assertEqual(robot.battery_percent, 0.0)
        self.assertIn("battery", pf.tripped(robot))


class TheArithmetic(unittest.TestCase):
    def test_blend_is_clipped(self) -> None:
        a, b = np.zeros(2), np.ones(2)
        np.testing.assert_allclose(pf.blend(a, b, 0.25), [0.25, 0.25])
        np.testing.assert_allclose(pf.blend(a, b, 7.0), b)
        np.testing.assert_allclose(pf.blend(a, b, -1.0), a)

    def test_the_verdict_names_the_first_refusal(self) -> None:
        ok = pf.Check("a", True, "1", "2")
        na = pf.Check("b", None, "not run", "")
        bad = pf.Check("c", False, "47 vs 45", "equal")
        self.assertEqual(pf.verdict_word([ok, na]), "passed 1/1")
        self.assertEqual(pf.verdict_word([ok, bad]), "refused: c (47 vs 45)")

    def test_the_card_says_on_what_state(self) -> None:
        base = {"verdict": "passed 7/7", "basis": "simulation"}
        self.assertEqual(pf.card_line(base), "passed 7/7 in simulation")
        self.assertEqual(
            pf.card_line({**base, "stand_in": True}),
            "passed 7/7 on a simulation stand-in",
        )

    def test_transitions_come_from_the_manifest_else_the_defaults(self) -> None:
        m = Manifest(raw={"control": {"ramp_in_s": 2.5}}, root=Path("/x"))
        t = pf.transitions_of(m)
        self.assertEqual(t.ramp_in_s, 2.5)
        self.assertEqual(t.soft_stop_s, pf.TRANSITIONS.soft_stop_s)


@needs_sim
@needs_deployment
class OnTheGo2(unittest.TestCase):
    """The real exported go2-c2 policy in plain MuJoCo."""

    def _run(self, manifest: Manifest | None = None, **kw: Any) -> dict[str, Any]:
        return pf.preflight(
            DEPLOYMENT, assets_dir=ASSETS, manifest=manifest, measure=False, **kw
        )

    def _manifest(self) -> Manifest:
        m = pf.load_manifest(DEPLOYMENT)
        return Manifest(raw=copy.deepcopy(m.raw), root=m.root)

    def _check(self, record: dict[str, Any], name: str) -> dict[str, Any]:
        return next(c for c in record["checks"] if c["name"] == name)

    def test_the_exported_policy_passes_every_check(self) -> None:
        record = self._run()
        self.assertTrue(record["passed"], record["verdict"])
        self.assertEqual(record["verdict"], "passed 7/7")

    def test_a_wrong_observation_size_is_refused_and_nothing_drives(self) -> None:
        """unitree_rl_mjlab #25: the robot sent a vector the policy did not
        train on; found here before a motor moves."""
        m = self._manifest()
        m.raw["observations"] = m.raw["observations"][:-1]  # a term left out
        record = self._run(m)
        widths = self._check(record, "policy widths")
        self.assertFalse(widths["passed"])
        self.assertIn("policy takes 47", widths["measured"])
        self.assertTrue(record["verdict"].startswith("refused: policy widths"))
        rollout = self._check(record, "targets within the joints' ranges")
        self.assertIsNone(rollout["passed"])
        self.assertIn("refused first", rollout["detail"])

    def test_an_action_scale_copied_wrong_puts_targets_past_the_ranges(self) -> None:
        m = self._manifest()
        m.raw["action"]["scale"] = [s * 8 for s in m.raw["action"]["scale"]]
        record = self._run(m)
        targets = self._check(record, "targets within the joints' ranges")
        self.assertFalse(targets["passed"], targets["measured"])
        self.assertFalse(record["passed"])

    def test_gains_that_differ_from_training_are_refused(self) -> None:
        m = self._manifest()
        m.raw["joints"]["stiffness"] = [2 * k for k in m.raw["joints"]["stiffness"]]
        gains = self._check(self._run(m), "gains")
        self.assertFalse(gains["passed"])
        self.assertIn("trained", gains["detail"])

    def test_a_tilted_robot_is_refused_before_handover(self) -> None:
        tilted = pf.Health(
            tilt_rad=1.2, max_joint_velocity=0.0, max_angular_velocity=0.0
        )
        record = self._run(health=tilted, state_from="a test")
        state = self._check(record, "robot state before handover")
        self.assertFalse(state["passed"])
        self.assertIn("tilt", state["detail"])

    def test_the_ramp_takes_the_slam_out_of_the_first_tick(self) -> None:
        from rq_pipeline.deploy.runtime import open_runtime  # noqa: PLC0415

        runtime = open_runtime(pf.load_manifest(DEPLOYMENT), assets_dir=ASSETS)
        ramp = pf.measure_ramp(runtime, pf.TRANSITIONS)
        with_, without = ramp["with ramp"], ramp["without ramp"]
        self.assertLess(
            with_["first_tick_target_step_rad"],
            0.1 * without["first_tick_target_step_rad"],
        )
        self.assertLess(with_["max_force_step_nm"], without["max_force_step_nm"])
        self.assertTrue(with_["stood"])

    def test_a_soft_stop_never_zeroes_and_a_zeroed_one_drops_the_body(self) -> None:
        from rq_pipeline.deploy.runtime import open_runtime  # noqa: PLC0415

        runtime = open_runtime(pf.load_manifest(DEPLOYMENT), assets_dir=ASSETS)
        stop = pf.measure_stop(runtime, pf.TRANSITIONS)
        soft, zeroed = stop["soft"], stop["zeroed"]
        self.assertAlmostEqual(soft["seconds_to_damping"], pf.TRANSITIONS.soft_stop_s)
        self.assertLess(soft["max_body_fall_mps"], zeroed["max_body_fall_mps"])
        # zeroed motors spin a joint past the SDK's own watchdog
        self.assertGreater(
            zeroed["max_joint_speed_rad_s"], pf.WATCHDOGS["joint_velocity"].limit
        )
        self.assertLess(
            soft["max_joint_speed_rad_s"], pf.WATCHDOGS["joint_velocity"].limit
        )

    def test_the_operators_stop_file_stops_a_guarded_run_softly(self) -> None:
        from rq_pipeline.deploy.runtime import open_runtime  # noqa: PLC0415

        runtime = open_runtime(pf.load_manifest(DEPLOYMENT), assets_dir=ASSETS)
        with tempfile.TemporaryDirectory() as tmp:
            stop_file = Path(tmp) / pf.STOP_FILE
            guarded = pf.Guarded(runtime, pf.TRANSITIONS, stop_file=stop_file)
            guarded.handover()
            for _ in range(5):
                guarded.apply(runtime.act(runtime.observe()))
            self.assertIsNone(guarded.stopped_at)
            pf.request_stop(Path(tmp))
            guarded.apply(runtime.act(runtime.observe()))
            self.assertEqual(guarded.stop_reason, pf.OPERATOR_STOP)
            kp_now = guarded.log[-1]["kp"]
            self.assertGreater(kp_now, 0.0)  # blending, not zeroed
            guarded.restore()


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
    def __init__(self, speed: float = 0.0) -> None:
        self.speed = speed
        self.reads = 0

    def latest(self, timeout_ms: int) -> tuple[np.ndarray, np.ndarray]:
        self.reads += 1
        return np.array([1.0, 0, 0, 0]), np.array([0.0, 0.0, -0.2])

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array([0.0, 0.0, 0.1]), np.array([1.0, 0, 0, 0]), np.zeros(20)

    def health(self) -> dict[str, Any]:
        return {
            "quaternion": np.array([1.0, 0, 0, 0]),
            "gyroscope": np.zeros(3),
            "motor_speed": np.full(12, self.speed),
            "motor_temperature": np.zeros(12),
            "battery_percent": 0.0,
            "read_at": time.monotonic(),
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
    return DdsRuntime(manifest, bus=bus, pad=pad, sleep=lambda _s: None)


class TheDdsRuntime(unittest.TestCase):
    def test_stand_handover_and_stop_are_their_chords(self) -> None:
        pad = _Pad()
        rt = _dds(_Bus(), pad)
        rt.stand()
        rt.handover()
        rt.stop()
        chords = [e[1:] for e in pad.log if e[0] == "chord"]
        self.assertEqual(
            chords, [STAND_CHORD] * 2 + [HANDOVER_CHORD] * 2 + [PASSIVE_CHORD]
        )

    def test_reset_is_stand_then_handover(self) -> None:
        pad = _Pad()
        _dds(_Bus(), pad).reset()
        chords = [e[1:] for e in pad.log if e[0] == "chord"]
        self.assertEqual(chords, [STAND_CHORD] * 2 + [HANDOVER_CHORD] * 2)

    def test_their_stop_is_measured_and_the_stop_file_ends_the_walk(self) -> None:
        pad = _Pad()
        rt = _dds(_Bus(), pad)
        rt._clock = lambda: 0.0  # no pacing under test
        with tempfile.TemporaryDirectory() as tmp:
            stop_file = pf.request_stop(Path(tmp))
            out = pf.measure_dds_stop(
                rt, np.array([0.5, 0, 0]), stand_in=True, stop_file=stop_file
            )
        self.assertEqual(out["stopped_by"], pf.OPERATOR_STOP)
        self.assertEqual(pad.log[-1], ("chord", *PASSIVE_CHORD))
        self.assertAlmostEqual(out["max_body_fall_mps"], 0.2)

    def test_a_watchdog_ends_the_walk(self) -> None:
        pad = _Pad()
        rt = _dds(_Bus(speed=12.0), pad)
        rt._clock = lambda: 0.0
        out = pf.measure_dds_stop(rt, np.array([0.5, 0, 0]), stand_in=True)
        self.assertEqual(out["stopped_by"], "joint_velocity")


class TheDrawer(unittest.TestCase):
    def test_the_checks_and_the_measurements_become_tables(self) -> None:
        record = {
            "verdict": "passed 1/1",
            "basis": "simulation",
            "checks": [
                {
                    "name": "policy widths",
                    "passed": True,
                    "measured": "47",
                    "limit": "47",
                }
            ],
            "ramp_in": {
                "window_s": 1.0,
                "from": "lying",
                "with ramp": {
                    "first_tick_target_step_rad": 0.02,
                    "max_target_step_rad": 0.08,
                    "max_force_step_nm": 1.8,
                    "stood": True,
                },
            },
            "soft_stop": {"soft": {"seconds_to_damping": 1.0, "max_force_step_nm": 8}},
        }
        sections = preflight_sections(record)
        self.assertEqual(sections[0]["rows"][0][:2], ["policy widths", "passed"])
        self.assertEqual(len(sections[1]["rows"]), 2)


class TheDoors(unittest.TestCase):
    def test_the_action_spawns_the_tool_verbatim(self) -> None:
        with harness() as (actions, spawner):
            actions.preflight_deployment("final", project="/p", runtime="dds", seed=5)
            (argv, cwd), *_ = spawner.calls
            tool = argv[argv.index(str(TOOLS_DIR / "preflight-deployment.py")) :]
            for flag, value in (
                ("--project", "/p"),
                ("--name", "final"),
                ("--runtime", "dds"),
                ("--seed", "5"),
            ):
                self.assertEqual(tool[tool.index(flag) + 1], value)
            self.assertEqual(cwd, PIPELINE_DIR)

    def test_the_server_doors_refuse_by_name_and_the_stop_writes_its_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = create_project(Path(tmp) / "p", "p")
            os.environ[PROJECT_ENV] = str(project.root)
            try:
                for door in (server.preflight_deployment, server.stop_deployment):
                    out = door("ghost")
                    self.assertEqual(out["status"], "refused")
                    self.assertIn("ghost", out["reason"])
                out = server.preflight_deployment("ghost", runtime="webots")
                self.assertIn("webots", out["reason"])
                folder = project.folder("deploy") / "d"
                folder.mkdir(parents=True)
                (folder / "deploy.json").write_text("{}")
                out = server.stop_deployment("d")
                self.assertEqual(out["status"], "done")
                self.assertTrue((folder / pf.STOP_FILE).is_file())
            finally:
                os.environ.pop(PROJECT_ENV, None)


if __name__ == "__main__":
    unittest.main()
