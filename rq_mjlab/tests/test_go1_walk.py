"""The walk C1 on mjlab's own Go1 (2026-09-04): their flat task
unchanged, our span on their derived gains as real draws, the three
bases distinct, the identity recomputable — and the registry that
lets every walk tool take --robot."""

from __future__ import annotations

import unittest

from rq_mjlab.go1_walk import (
    ACTUATOR_DR_SPAN,
    SCALED,
    actuator_dr_events,
    derived_actuator_constants,
    go1_walk_env_cfg,
)
from rq_mjlab.walks import ROBOTS, walk_spec


class TheGo1Cfg(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, cls.identity = go1_walk_env_cfg()

    def test_their_task_is_untouched_except_our_two_events(self) -> None:
        from mjlab.tasks.velocity.config.go1.env_cfgs import (  # noqa: PLC0415
            unitree_go1_flat_env_cfg,
        )

        theirs = unitree_go1_flat_env_cfg()
        ours = set(self.cfg.events) - set(theirs.events)
        self.assertEqual(ours, {"actuator_gains", "actuator_armature"})
        for name in theirs.events:
            self.assertIn(name, self.cfg.events)  # pushes, friction, bias, CoM stay
        self.assertEqual(self.cfg.scene.terrain.terrain_type, "plane")

    def test_the_span_is_a_real_draw_on_their_pd_actuator(self) -> None:
        from mjlab.envs import mdp  # noqa: PLC0415

        gains = self.cfg.events["actuator_gains"]
        self.assertIs(gains.func, mdp.dr.pd_gains)
        self.assertEqual(gains.params["kp_range"], (0.9, 1.1))
        self.assertEqual(gains.params["kd_range"], (0.9, 1.1))
        self.assertEqual(gains.params["operation"], "scale")
        arm = self.cfg.events["actuator_armature"]
        self.assertIs(arm.func, mdp.dr.joint_armature)
        self.assertEqual(arm.params["ranges"], (0.9, 1.1))

    def test_the_identity_names_their_derivation(self) -> None:
        self.assertTrue(self.identity["robot"].startswith("unitree-go1@"))
        self.assertTrue(self.identity["actuator"].startswith("unitree-go1-derived-pd@"))
        self.assertIn(f"±{ACTUATOR_DR_SPAN:g}", self.identity["dr_basis"])
        constants = derived_actuator_constants()
        self.assertAlmostEqual(constants["damping_ratio"], 2.0)
        # stiffness = armature * w^2, the derivation a reader can redo
        hip = constants["hip"]
        self.assertAlmostEqual(
            hip["stiffness"],
            hip["armature"] * constants["natural_frequency_rad_s"] ** 2,
        )

    def test_the_three_arms_have_three_bases_and_one_identity(self) -> None:
        _, none = go1_walk_env_cfg(dr_span=None)
        _, zero = go1_walk_env_cfg(dr_span=0.0)
        _, wide = go1_walk_env_cfg(dr_span=0.30)
        _, pinned = go1_walk_env_cfg(pin_scale=0.7)
        self.assertEqual(none["dr_basis"], zero["dr_basis"])
        self.assertIn("none", none["dr_basis"])
        self.assertIn("0.3", wide["dr_basis"])
        self.assertIn("pinned", pinned["dr_basis"])
        self.assertEqual(
            {none["robot"], wide["robot"], pinned["robot"]}, {self.identity["robot"]}
        )
        self.assertEqual(
            {none["actuator"], wide["actuator"]}, {self.identity["actuator"]}
        )
        events, _ = actuator_dr_events(dr_span=None, pin_scale=0.7)
        self.assertEqual(events["actuator_gains"].params["kp_range"], (0.7, 0.7))
        self.assertEqual(SCALED, ("kp", "kd", "armature"))


class TheRegistry(unittest.TestCase):
    def test_every_robot_resolves_and_agrees_on_the_knobs(self) -> None:
        for robot in ROBOTS:
            spec = walk_spec(robot)
            self.assertEqual(spec.name, robot)
            self.assertGreater(spec.default_span, 0.0)
        with self.assertRaisesRegex(KeyError, "no walk"):
            walk_spec("spot")

    def test_go1_agent_takes_the_studys_iterations(self) -> None:
        agent = walk_spec("go1").agent(123)
        self.assertEqual(agent.max_iterations, 123)
        self.assertEqual(agent.experiment_name, "go1_velocity")
