"""A walk for any onboarded legged robot (#100): the spec names the robot
and the actuator, the rest is read off the model, and anything missing is
refused by name, the field to declare named with it."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._extras import needs_mcp, needs_sim

# A small biped (`fixtures/biped.xml`): a free-floating trunk with an IMU,
# two legs with a foot site and a foot collision geom each, a head joint,
# position actuators, a `stand` keyframe after a `zero` one, and a floor
# (an onboarded scene usually has one). The trainer's test reads it too.
BIPED = (Path(__file__).parent / "fixtures" / "biped.xml").read_text()


def biped_xml(*, feet: bool = True, imu: bool = True) -> str:
    xml = BIPED
    if not feet:
        xml = xml.replace("_foot_collision", "_sole")
    if not imu:
        xml = xml.replace('<velocimeter name="imu_lin_vel" site="imu"/>', "")
    return xml


class _InAProject(unittest.TestCase):
    """A project to onboard robots into (no tests of its own)."""

    def setUp(self) -> None:
        from trainnr.project import create_project  # noqa: PLC0415
        from trainnr.project.locate import PROJECT_ENV  # noqa: PLC0415

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.project = create_project(self.root / "p", "p")
        env = mock.patch.dict(os.environ, {PROJECT_ENV: str(self.project.root)})
        env.start()
        self.addCleanup(env.stop)

    def _onboard(self, name: str, xml: str) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        source = self.root / f"src-{name}"
        source.mkdir()
        (source / "scene.xml").write_text(xml)
        out = mcp_server.onboard_robot(str(source / "scene.xml"), name)
        self.assertEqual(out["status"], "done", out)


@needs_mcp
@needs_sim
class TheLeggedWalk(_InAProject):
    def test_everything_the_model_says_is_read_off_it(self) -> None:
        from trainnr.tasks.walks import LeggedWalkSpec, resolve_legged  # noqa: PLC0415

        self._onboard("biped", biped_xml())
        resolved, _, _ = resolve_legged(
            LeggedWalkSpec(robot="biped", actuator="model", pinned_joints=("neck.*",))
        )
        self.assertEqual(resolved["trunk"], "torso")
        self.assertEqual(resolved["foot_sites"], ["left_foot", "right_foot"])
        self.assertEqual(
            resolved["foot_geoms"], ["left_foot_collision", "right_foot_collision"]
        )
        self.assertEqual(resolved["stand_keyframe"], "stand")  # not the first
        self.assertEqual(resolved["actuator"], "model")
        self.assertTrue(resolved["robot"].startswith("biped@"))

    def test_what_is_missing_or_wrong_is_refused_by_name(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        self._onboard("biped", biped_xml())
        self._onboard("blind", biped_xml(imu=False))
        self._onboard("footless", biped_xml(feet=False))

        def declare(name: str, **settings: object) -> dict:
            return mcp_server.create_task("legged-walk", name, settings)

        self.assertIn("needs an actuator", declare("a", robot="biped")["reason"])
        self.assertIn(
            "onboard it first", declare("b", robot="ghost", actuator="model")["reason"]
        )
        self.assertIn(
            "sensor imu_lin_vel",
            declare("c", robot="blind", actuator="model")["reason"],
        )
        reason = declare("d", robot="footless", actuator="model")["reason"]
        self.assertIn("declare foot_geoms", reason)
        self.assertIn(
            "no actuator bundle", declare("e", robot="biped", actuator="x9")["reason"]
        )
        self.assertIn(
            "missing keyframe nope",
            declare("f", robot="biped", actuator="model", stand_keyframe="nope")[
                "reason"
            ],
        )
        self._onboard(
            "torqued",
            biped_xml().replace(
                '<position joint="left_hip" kp="5"/>', '<motor joint="left_hip"/>'
            ),
        )
        self.assertIn(
            "not position actuators",
            declare("g", robot="torqued", actuator="model")["reason"],
        )
        self.assertIn(
            "matches none",
            declare("h", robot="biped", actuator="model", pinned_joints=["tail.*"])[
                "reason"
            ],
        )
        self.assertIn(
            "is not a pattern",
            declare("i", robot="biped", actuator="model", pinned_joints=["("])[
                "reason"
            ],
        )
        self.assertIn(
            "nothing would walk",
            declare("j", robot="biped", actuator="model", pinned_joints=[".*"])[
                "reason"
            ],
        )
        keyless = biped_xml()
        keyless = keyless[: keyless.index("<keyframe>")] + "</mujoco>"
        self._onboard("keyless", keyless)
        self.assertIn(
            "no keyframe to stand in",
            declare("k", robot="keyless", actuator="model")["reason"],
        )
        ok = declare("walk", robot="biped", actuator="model", pinned_joints=["neck.*"])
        self.assertEqual(ok["status"], "done", ok)

    def test_a_stand_pose_outside_the_ranges_is_refused(self) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        bent = biped_xml().replace(
            'qpos="0 0 0.28 1 0 0 0 0 0.2 -0.4 0.2 -0.4"',
            'qpos="0 0 0.28 1 0 0 0 0 0.2 0.4 0.2 -0.4"',  # a knee bent forward
        )
        self._onboard("bent", bent)
        out = mcp_server.create_task(
            "legged-walk", "w", {"robot": "bent", "actuator": "model"}
        )
        self.assertIn("outside its joints' ranges: left_knee", out["reason"])

    def test_training_takes_the_resolved_spec_and_names_the_run_after_the_robot(
        self,
    ) -> None:
        from trainnr import mcp_server  # noqa: PLC0415

        self._onboard("biped", biped_xml())
        mcp_server.create_task(
            "legged-walk", "walk", {"robot": "biped", "actuator": "model"}
        )
        self.assertIn(
            "legged-walk", mcp_server.train_walk("smoke", robot="legged")["reason"]
        )
        calls: list[dict] = []

        def fake(
            _self: object, agent: str, envs: object, iterations: object, **kw: object
        ) -> dict:
            calls.append({"agent": agent, **kw})
            return {"job_id": "train-walk-x", "log": "", "pid": 1}

        with mock.patch("trainnr.mcp_actions.Actions.train_walk", fake):
            out = mcp_server.train_walk("full", task="walk", iterations=1)
        self.assertEqual(out["status"], "started", out)
        self.assertTrue(out["experiment"].startswith("biped-walk-"))
        [call] = calls
        self.assertEqual(call["robot"], "legged")
        spec = json.loads(Path(str(call["legged_spec"])).read_text())
        self.assertTrue(spec["name"].startswith("legged-walk@"))
        self.assertEqual(spec["trunk"], "torso")


@needs_mcp
@needs_sim
class TheLimitsInTheWalk(_InAProject):
    """A bundle's limits over several joints (#121) enter its walk: the
    stand pose must keep them, and what breaking one does is the task's."""

    def _with_limits(self, name: str, knee_above: float) -> None:
        from trainnr.robot.joint_limits import SCHEMA  # noqa: PLC0415

        source = self.root / f"src-{name}"
        source.mkdir()
        (source / "scene.xml").write_text(biped_xml())
        rule = {
            "name": "knees back",
            "form": "rule",
            "when": [{"joint": "left_knee", "above": knee_above}],
        }
        (source / "limits.json").write_text(
            json.dumps({"schema": SCHEMA, "limits": [rule]})
        )
        from trainnr import mcp_server  # noqa: PLC0415

        self.assertEqual(
            mcp_server.onboard_robot(str(source / "scene.xml"), name)["status"], "done"
        )

    def test_the_limits_ride_in_the_spec_and_the_stand_pose_must_keep_them(
        self,
    ) -> None:
        from trainnr.tasks.walks import LeggedWalkSpec, resolve_legged  # noqa: PLC0415

        self._with_limits("kept", knee_above=-0.1)  # stand: knee at -0.4 rad
        resolved, _, _ = resolve_legged(LeggedWalkSpec(robot="kept", actuator="model"))
        self.assertEqual(resolved["limits"]["mode"], "terminate")
        off, _, _ = resolve_legged(
            LeggedWalkSpec(robot="kept", actuator="model", limits="off")
        )
        self.assertIsNone(off["limits"])
        self._with_limits("broken", knee_above=-50)  # degrees: -0.4 rad is -22.9
        with self.assertRaisesRegex(
            ValueError, "'stand' breaks its limits: knees back"
        ):
            resolve_legged(LeggedWalkSpec(robot="broken", actuator="model"))

    def test_a_mode_asked_for_without_limits_is_refused(self) -> None:
        from trainnr.tasks.walks import LeggedWalkSpec, resolve_legged  # noqa: PLC0415

        self._onboard("plain", biped_xml())
        with self.assertRaisesRegex(ValueError, "declares no limits"):
            resolve_legged(
                LeggedWalkSpec(robot="plain", actuator="model", limits="cost")
            )
        with self.assertRaisesRegex(ValueError, "limits is one of"):
            resolve_legged(
                LeggedWalkSpec(robot="plain", actuator="model", limits="hold")
            )


class TheHeldJoints(unittest.TestCase):
    """A deployment whose action drives some joints holds the rest at the
    default pose (#100: a legged walk's pinned head)."""

    def test_the_target_moves_only_the_acting_joints(self) -> None:
        import numpy as np  # noqa: PLC0415

        from trainnr.deploy.manifest import Joints  # noqa: PLC0415
        from trainnr.deploy.runtime import Runtime  # noqa: PLC0415

        joints = Joints.of(
            {
                "policy_order": ["neck", "hip", "knee"],
                "action_to_ctrl": [0, 1, 2],
                "default_pos": [0.1, 0.2, -0.4],
                "action_joints": ["hip", "knee"],
            }
        )
        self.assertEqual(joints.acting, ("hip", "knee"))
        rt = object.__new__(Runtime)
        rt.default_pos = np.asarray(joints.default_pos)
        rt.scale = np.asarray([0.5, 0.5])
        rt.acted = np.array([joints.policy_order.index(j) for j in joints.acting])
        np.testing.assert_allclose(
            rt.target_of(np.array([1.0, -1.0])), [0.1, 0.7, -0.9]
        )


if __name__ == "__main__":
    unittest.main()
