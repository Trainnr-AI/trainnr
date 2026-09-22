"""The gate's course protocol (`deploy.course`): on a staged scene the
gate walks the course the scene's author laid out instead of holding a
random twist for twenty seconds, which on a scene drives a flat-ground
policy into its hurdles (docs/78 §8.3). The steering law is the one the
policy trained under (mjlab's heading pursuit), the speed is drawn per
trial, the budget is the path's length at that speed with slack, and
success is reaching the course's end upright."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from rq_pipeline.deploy.course import (
    COMMANDS_ALONG,
    COURSE_REACH_M,
    COURSE_SLACK,
    COURSE_SPEED_FRACTION,
    COURSE_STEER_GAIN,
    Course,
    CourseTrial,
    Steering,
    course_criterion_text,
    draw_speeds,
    run_course_trial,
    wrap_to_pi,
    yaw_of,
)
from rq_pipeline.deploy.gate import COMMANDS_DRAWN, gate
from rq_pipeline.deploy.manifest import MANIFEST_FILE, MANIFEST_SCHEMA, Key, Manifest

WAYPOINTS = [[1.0, 0.0, 0.3], [2.0, 0.5, 0.3]]  # a straight leg, then a 27° turn
START = [0.0, 0.0, 0.3]


def _raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
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
        "scene": {
            "file": "scene.xml",
            "start": START,
            "course": {"waypoints": WAYPOINTS, "source": "the test"},
        },
        "commands": {
            "twist": {
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1, 1],
                "ang_vel_z": [-0.5, 0.5],
            }
        },
    }
    raw.update(overrides)
    return raw


def _manifest(**overrides: Any) -> Manifest:
    return Manifest(root=Path("/nowhere"), raw=_raw(**overrides))


class _Walker:
    """A kinematic runtime: the base integrates the commanded twist
    exactly, so a steered walker reaches every waypoint."""

    instrument = "kinematic"
    command_limit = None

    def __init__(self, dt: float, *, moves: bool = True, falls_at: int = -1) -> None:
        self.dt, self.moves, self.falls_at = dt, moves, falls_at
        self.command = np.zeros(3, np.float32)
        self.xy = np.array(START[:2], float)
        self.yaw = 0.0
        self.ticks = 0

    def reset(self) -> None:
        self.xy, self.yaw, self.ticks = np.array(START[:2], float), 0.0, 0

    def observe(self) -> np.ndarray:
        return np.zeros(1, np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(1, np.float32)

    def apply(self, action: np.ndarray) -> None:
        if not self.moves:
            return
        vx, _, wz = (float(c) for c in self.command)
        self.xy += self.dt * vx * np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.yaw += self.dt * wz
        self.ticks += 1

    def base_velocity_b(self) -> np.ndarray:
        v = self.command.astype(float) if self.moves else np.zeros(3)
        return np.array([v[0], v[1], 0.0])

    def fell_over(self) -> bool:
        return self.ticks == self.falls_at

    def contact_points(self) -> np.ndarray | None:
        return None

    def pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        half = self.yaw / 2
        quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        return np.array([*self.xy, START[2]]), quat, np.zeros(1)


class TheAngles(unittest.TestCase):
    def test_yaw_reads_off_a_wxyz_quaternion(self) -> None:
        for yaw in (0.0, 0.7, -2.5, 3.0):
            half = yaw / 2
            quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
            self.assertAlmostEqual(yaw_of(quat), yaw, places=12)

    def test_wrapping_lands_in_the_half_open_turn(self) -> None:
        self.assertAlmostEqual(wrap_to_pi(math.pi + 0.1), -math.pi + 0.1)
        self.assertAlmostEqual(wrap_to_pi(-math.pi - 0.1), math.pi - 0.1)
        self.assertAlmostEqual(wrap_to_pi(0.3), 0.3)


class TheCourse(unittest.TestCase):
    def test_the_course_comes_from_the_manifest_with_the_start_first(self) -> None:
        course = Course.of_manifest(_manifest())
        assert course is not None
        np.testing.assert_allclose(course.start, START)
        self.assertEqual(course.waypoints.shape, (2, 3))
        self.assertEqual(course.path.shape, (3, 3))
        # the start's leg and the diagonal one
        self.assertAlmostEqual(course.length_m, 1.0 + math.sqrt(1.25))
        self.assertEqual(course.reach_m, COURSE_REACH_M)
        facts = course.describe()
        self.assertEqual(facts["waypoints"], 2)
        self.assertEqual(facts["reach_m"], COURSE_REACH_M)

    def test_a_plane_lays_out_no_course(self) -> None:
        self.assertIsNone(Course.of_manifest(_manifest(scene={"file": "scene.xml"})))
        empty = {"file": "scene.xml", "start": START, "course": {"waypoints": []}}
        self.assertIsNone(Course.of_manifest(_manifest(scene=empty)))

    def test_a_course_without_a_start_is_refused_by_name(self) -> None:
        scene = {"file": "scene.xml", "course": {"waypoints": WAYPOINTS}}
        with self.assertRaisesRegex(ValueError, "start"):
            Course.of_manifest(_manifest(scene=scene))

    def test_budget_is_the_path_at_that_speed_with_slack(self) -> None:
        course = Course.of_manifest(_manifest())
        assert course is not None
        self.assertAlmostEqual(
            course.budget_s(0.5), course.length_m / 0.5 * COURSE_SLACK
        )


class TheSteering(unittest.TestCase):
    def test_the_gain_is_the_manifests_when_it_names_one(self) -> None:
        steering = Steering.of_manifest(_manifest())
        self.assertEqual(steering.gain, COURSE_STEER_GAIN)
        self.assertIn("the protocol's own", steering.basis)
        named = _manifest(commands=_raw()["commands"] | {"heading_gain": 0.8})
        steering = Steering.of_manifest(named)
        self.assertEqual(steering.gain, 0.8)
        self.assertIn("manifest", steering.basis)

    def test_the_yaw_rate_is_proportional_and_clipped(self) -> None:
        steering = Steering(gain=0.5, yaw_rate=(-0.5, 0.5), basis="test")
        np.testing.assert_allclose(steering.command(0.7, 0.0), [0.7, 0.0, 0.0])
        np.testing.assert_allclose(
            steering.command(0.7, 0.4), [0.7 * math.cos(0.4), 0.0, 0.2], rtol=1e-6
        )
        # facing away: it stands and turns at the clip
        np.testing.assert_allclose(steering.command(0.7, 3.0), [0.0, 0.0, 0.5])
        np.testing.assert_allclose(steering.command(0.7, -3.0), [0.0, 0.0, -0.5])

    def test_a_runtimes_envelope_clips_the_yaw_rate_too(self) -> None:
        steering = Steering.of_manifest(_manifest(), limit=0.2)
        self.assertEqual(steering.yaw_rate, (-0.2, 0.2))


class TheSpeeds(unittest.TestCase):
    def test_speeds_are_seeded_in_the_upper_part_of_the_forward_range(self) -> None:
        m = _manifest()
        a = draw_speeds(m, 6, 3)
        np.testing.assert_array_equal(a, draw_speeds(m, 6, 3))
        self.assertEqual(a.shape, (6,))
        self.assertTrue((a >= COURSE_SPEED_FRACTION * 1.0).all())
        self.assertTrue((a <= 1.0).all())

    def test_a_policy_that_cannot_walk_forward_is_refused_by_name(self) -> None:
        commands = {
            "twist": {"lin_vel_x": [-1, 0], "lin_vel_y": [0, 0], "ang_vel_z": [0, 0]}
        }
        with self.assertRaisesRegex(ValueError, "forward"):
            draw_speeds(_manifest(commands=commands), 2, 1)

    def test_a_runtimes_envelope_caps_the_speed(self) -> None:
        limit = 0.6
        a = draw_speeds(_manifest(), 6, 3, limit=limit)
        self.assertTrue((a <= limit).all())


class TheTrial(unittest.TestCase):
    def _run(self, runtime: _Walker, speed: float = 0.8) -> CourseTrial:
        m = _manifest()
        course = Course.of_manifest(m)
        assert course is not None
        return run_course_trial(m, runtime, course, Steering.of_manifest(m), speed)

    def test_a_walker_that_follows_its_command_reaches_the_end(self) -> None:
        trial = self._run(_Walker(0.02))
        self.assertTrue(trial.finished)
        self.assertEqual((trial.reached, trial.of), (2, 2))
        self.assertTrue(trial.success)
        self.assertFalse(trial.fell)
        self.assertLess(trial.seconds, trial.budget_s)
        self.assertEqual(trial.speed, 0.8)
        # it tracked what it was told, exactly
        self.assertAlmostEqual(trial.err_ratio, 0.0)

    def test_a_walker_that_stands_still_runs_out_its_budget(self) -> None:
        trial = self._run(_Walker(0.02, moves=False))
        self.assertFalse(trial.finished)
        self.assertEqual(trial.reached, 0)
        self.assertFalse(trial.success)
        self.assertAlmostEqual(trial.seconds, trial.budget_s, delta=0.02)
        self.assertEqual(trial.steps, round(trial.budget_s / 0.02))

    def test_a_fall_ends_the_trial(self) -> None:
        trial = self._run(_Walker(0.02, falls_at=10))
        self.assertTrue(trial.fell)
        self.assertFalse(trial.success)
        self.assertEqual(trial.steps, 10)

    def test_the_record_row_carries_the_course_facts(self) -> None:
        trial = self._run(_Walker(0.02))
        row = trial.row()
        for key in ("speed", "reached", "of", "seconds", "budget_s", "finished"):
            self.assertIn(key, row)
        self.assertEqual(row["success"], True)


class TheGateChoosesByTheManifest(unittest.TestCase):
    def _deployment(self, tmp: Path, raw: dict[str, Any]) -> Path:
        (tmp / MANIFEST_FILE).write_text(json.dumps(raw))
        (tmp / "policy.onnx").write_bytes(b"\x00")
        (tmp / "scene.xml").write_text("<mujoco/>")
        return tmp

    def test_a_staged_manifest_walks_the_course(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._deployment(Path(tmp), _raw())
            record = gate(
                folder,
                assets_dir=None,
                trials=3,
                seed=1,
                open=lambda manifest, assets_dir=None: _Walker(
                    manifest.control.step_dt
                ),
            )
            protocol = record["protocol"]
            self.assertEqual(protocol["commands"], COMMANDS_ALONG)
            self.assertEqual(protocol["criterion"], course_criterion_text())
            self.assertEqual(protocol["course"]["waypoints"], 2)
            self.assertEqual(protocol["steer"]["gain"], COURSE_STEER_GAIN)
            self.assertEqual(record["successes"], 3)
            self.assertEqual(len(record["records"]), 3)
            self.assertTrue(all(r["finished"] for r in record["records"]))

    def test_a_plane_still_holds_drawn_twists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._deployment(Path(tmp), _raw(scene={"file": "scene.xml"}))
            record = gate(
                folder,
                assets_dir=None,
                trials=2,
                seed=1,
                open=lambda manifest, assets_dir=None: _Walker(
                    manifest.control.step_dt
                ),
            )
            self.assertEqual(record["protocol"]["commands"], COMMANDS_DRAWN)
            self.assertNotIn("course", record["protocol"])
            self.assertIn("command", record["records"][0])


class TheStudioPage(unittest.TestCase):
    """The deployment's detail page shows a gate in its protocol's own
    shape (`project.details.gate_trials`): a course gate's rows say how
    far and how fast; a twist gate's what was held."""

    def _course_record(self) -> dict[str, Any]:
        m = _manifest()
        course = Course.of_manifest(m)
        assert course is not None
        trial = run_course_trial(m, _Walker(0.02), course, Steering.of_manifest(m), 0.8)
        return {
            "protocol": {
                "commands": COMMANDS_ALONG,
                "criterion": course_criterion_text(),
                "course": course.describe(),
            },
            "records": [trial.row()],
        }

    def test_a_course_gate_tabulates_arrival(self) -> None:
        from rq_pipeline.project.details import (  # noqa: PLC0415
            COURSE_COLUMNS,
            gate_trials,
        )

        section = gate_trials(self._course_record(), "plain MuJoCo")
        self.assertEqual(section["columns"][:3], [label for label, _ in COURSE_COLUMNS])
        (row,) = section["rows"]
        self.assertEqual(row[0], 0.8)
        self.assertEqual(row[1], "2 / 2")
        self.assertEqual(row[-1], "success")
        self.assertIn(course_criterion_text(), section["note"])
        # every key the page reads is one the trial's row writes
        keys = set(CourseTrial.__dataclass_fields__) | {"finished", "success"}
        self.assertTrue({key for _, key in COURSE_COLUMNS} <= keys)

    def test_a_twist_gate_tabulates_the_held_command(self) -> None:
        from rq_pipeline.project.details import gate_trials  # noqa: PLC0415

        record = {
            "protocol": {"commands": COMMANDS_DRAWN},
            "records": [
                {
                    "command": [0.5, 0.0, 0.1],
                    "steps": 50,
                    "fell": False,
                    "err_ratio": 0.2,
                    "success": True,
                }
            ],
        }
        section = gate_trials(record, "plain MuJoCo")
        self.assertEqual(section["rows"][0][0], "0.50, 0.00, 0.10")
        self.assertEqual(section["rows"][0][1], 50)
        self.assertIsNone(section["note"])


class TheCommandsBlock(unittest.TestCase):
    def test_the_heading_gain_is_optional_and_read(self) -> None:
        self.assertIsNone(_manifest().commands.heading_gain)
        named = _manifest(commands=_raw()["commands"] | {"heading_gain": 0.5})
        self.assertEqual(named.commands.heading_gain, 0.5)
        self.assertEqual(named.raw[Key.COMMANDS]["heading_gain"], 0.5)


if __name__ == "__main__":
    unittest.main()


class TheCourseVerdict(unittest.TestCase):
    def test_a_course_gate_reports_and_never_judges_against_a_plane_evaluation(
        self,
    ) -> None:
        from rq_pipeline.deploy.gate import OTHER_PROTOCOL, _verdict  # noqa: PLC0415

        cert = {"successes": 38, "trials": 40}
        plane = _verdict(3, 4, 0.1, cert)
        self.assertIsNotNone(plane["passed"])
        course = _verdict(0, 4, 0.1, cert, along_course=True)
        self.assertIsNone(course["passed"])
        self.assertEqual(course["rule"], OTHER_PROTOCOL)
