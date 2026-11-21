"""A walk for any onboarded legged robot (#100), as the trainer builds it:
a declared spec becomes a `LeggedRobot` (every hinge driven unless pinned,
the stand keyframe the default pose, a scene's floor left to the terrain),
and the walk config and its identity follow; microduck stays its own."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

TRAINNR_TESTS = Path(__file__).resolve().parents[2] / "trainnr"


def _biped() -> str:
    """The small biped trainnr's own legged-walk test onboards."""
    return (TRAINNR_TESTS / "tests" / "fixtures" / "biped.xml").read_text()


class TheLeggedRobot(unittest.TestCase):
    def test_a_spec_becomes_the_robot_and_its_walk(self) -> None:
        import mujoco  # noqa: PLC0415
        from trainnr.bundles import locate  # noqa: PLC0415

        from trainnr_mjlab.legged_walk import (  # noqa: PLC0415
            MODEL_DR_BASIS,
            legged_robot,
            legged_walk_env_cfg,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "biped"
            bundle.mkdir()
            (bundle / "scene.xml").write_text(_biped())
            locate.add_search_root(Path(tmp))  # the trainer finds it by name
            self.addCleanup(locate._SEARCH_ROOTS.remove, Path(tmp))
            spec = {
                "name": "legged-walk@000000000000",
                "robot": "biped@000000000000",
                "model_file": "scene.xml",
                "actuator": "model",
                "stand_keyframe": "stand",
                "trunk": "torso",
                "foot_sites": ["left_foot", "right_foot"],
                "foot_geoms": ["left_foot_collision", "right_foot_collision"],
                "pinned_joints": ["neck.*"],
                "dr_span": 0.1,
            }
            robot = legged_robot(spec)
            self.assertEqual(robot.pinned, ("^neck_yaw$",))
            self.assertNotIn("^neck_yaw$", robot.pose_joints)
            assert robot.init_state is not None
            self.assertAlmostEqual(robot.init_state.pos[2], 0.28)
            self.assertAlmostEqual(robot.init_state.joint_pos["^left_knee$"], -0.4)
            self.assertEqual(robot.pose_std_standing["^left_knee$"], 0.15)
            cfg, identity = legged_walk_env_cfg(robot)
            spec_fn = cfg.scene.entities["robot"].spec_fn
            planes = [
                g
                for g in spec_fn().worldbody.geoms
                if g.type == mujoco.mjtGeom.mjGEOM_PLANE
            ]
            self.assertEqual(planes, [])  # the scene's floor left to mjlab's terrain
        self.assertEqual(identity["dr_basis"], MODEL_DR_BASIS)
        self.assertTrue(identity["actuator"].startswith("biped-model-actuators@"))
        self.assertEqual(identity["head"], "pinned")
        self.assertEqual(
            cfg.actions["joint_pos"].actuator_names, ("^(?!^neck_yaw$).*",)
        )

    def test_a_spec_key_it_does_not_read_is_refused(self) -> None:
        from trainnr_mjlab.legged_walk import legged_robot  # noqa: PLC0415

        with self.assertRaisesRegex(ValueError, "has no colour"):
            legged_robot({"colour": "red"})


if __name__ == "__main__":
    unittest.main()
