"""The walk on a project's Go2 (docs/77): mjlab's velocity task wired to
the Go2's names, the reference's constants as a declared basis, and the
identity recomputable from the project's bundle."""

from __future__ import annotations

import unittest
from pathlib import Path

from mjlab.entity.entity import Entity

from rq_mjlab.go2_walk import DEPLOYABLE_ACTOR, go2_robot_cfg, go2_walk_env_cfg
from rq_mjlab.walks import ROBOTS, use_project, walk_spec

PROJECT = Path(__file__).resolve().parents[2] / "projects" / "go2-walk"


@unittest.skipUnless(
    (PROJECT / "robots" / "go2" / "go2.xml").is_file(), "no go2-walk project"
)
class TheGo2Cfg(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        use_project(PROJECT)
        cls.cfg, cls.identity = go2_walk_env_cfg()

    def test_the_task_is_flat_with_the_go2s_names(self) -> None:
        self.assertEqual(self.cfg.scene.terrain.terrain_type, "plane")
        self.assertEqual(self.cfg.viewer.body_name, "base_link")
        self.assertEqual(
            self.cfg.events["base_com"].params["asset_cfg"].body_names, ("base_link",)
        )
        sensors = {s.name for s in self.cfg.scene.sensors}
        self.assertIn("feet_ground_contact", sensors)
        self.assertNotIn("terrain_scan", sensors)
        self.assertNotIn("height_scan", self.cfg.observations["actor"].terms)
        self.assertIn("fell_over", self.cfg.terminations)

    def test_the_actor_sees_what_the_real_go2_measures(self) -> None:
        actor = self.cfg.observations["actor"].terms
        self.assertEqual(tuple(actor), DEPLOYABLE_ACTOR)
        self.assertNotIn("base_lin_vel", actor)
        self.assertEqual(actor["phase"].params["period"], 0.6)
        self.assertIn("base_lin_vel", self.cfg.observations["critic"].terms)

    def test_the_studys_events_ride_on_the_declared_gains(self) -> None:
        self.assertIn("actuator_gains", self.cfg.events)
        self.assertIn("actuator_armature", self.cfg.events)
        self.assertEqual(
            self.cfg.events["actuator_gains"].params["kp_range"], (0.9, 1.1)
        )

    def test_the_identity_is_the_projects_bundle_and_a_declared_basis(self) -> None:
        self.assertTrue(self.identity["robot"].startswith("go2@"))
        self.assertTrue(
            self.identity["actuator"].startswith("unitree-go2-declared-pd@")
        )
        self.assertIn("declared", self.identity["dr_basis"])
        self.assertNotIn("derived", self.identity["dr_basis"])

    def test_the_entity_compiles_with_twelve_actuators_from_the_constants(self) -> None:
        robot = Entity(go2_robot_cfg())
        model = robot.spec.compile()
        self.assertEqual(model.nu, 12)
        self.assertEqual(model.nkey, 1)


class TheRegistry(unittest.TestCase):
    def test_go2_is_a_walk_and_needs_no_bundle(self) -> None:
        self.assertIn("go2", ROBOTS)
        spec = walk_spec("go2")
        self.assertEqual(spec.source_prefix, "go2-walk")
        with self.assertRaises(ValueError):
            spec.env_cfg(dr_span=0.1, pin_scale=None, bundle=Path("x"))
