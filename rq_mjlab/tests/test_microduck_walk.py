"""Flagship G2, the cfg (docs/e2e-research/63 §2.2): microduck's walk
recipe on mjlab 1.6 through the certified stack — construction facts,
the transcribed numbers spot-checked against the delta file, and the
linter refusing their two silent no-ops by name."""

from __future__ import annotations

import math
import unittest

from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from rq_mjlab.linter import SilentNoOp, lint
from rq_mjlab.microduck_walk import (
    LAW_DR_SPAN,
    microduck_walk_env_cfg,
)


class TheWalkCfg(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, cls.stamps = microduck_walk_env_cfg()

    def test_the_identity_travels_with_the_cfg(self) -> None:
        self.assertTrue(self.stamps["robot"].startswith("microduck@"))
        self.assertTrue(self.stamps["actuator"].startswith("xl330-m6@"))
        self.assertIn("declared", self.stamps["dr_basis"])
        self.assertIn(str(LAW_DR_SPAN), self.stamps["dr_basis"])

    def test_the_action_scale_is_one_explicitly(self) -> None:
        # The number their deploy flag contradicts (57 §5) — pinned.
        self.assertEqual(self.cfg.actions["joint_pos"].scale, 1.0)

    def test_their_reward_numbers_landed(self) -> None:
        rewards = self.cfg.rewards
        self.assertEqual(rewards["air_time"].weight, 3.0)
        self.assertEqual(rewards["air_time"].params["threshold_min"], 0.125)
        self.assertEqual(rewards["upright"].weight, 2.0)
        self.assertAlmostEqual(rewards["upright"].params["std"], math.sqrt(0.05))
        self.assertEqual(rewards["foot_slip"].weight, -0.1)
        self.assertEqual(rewards["self_collisions"].weight, -1.0)
        self.assertEqual(rewards["foot_clearance"].params["target_height"], 0.02)
        self.assertNotIn("soft_landing", rewards)

    def test_their_command_ranges_landed(self) -> None:
        ranges = self.cfg.commands["twist"].ranges
        self.assertEqual(ranges.lin_vel_x, (-0.4, 0.4))
        self.assertEqual(ranges.lin_vel_y, (-0.3, 0.3))
        self.assertEqual(ranges.ang_vel_z, (-1.0, 1.0))

    def test_the_actor_is_blind_where_the_critic_is_privileged(self) -> None:
        self.assertNotIn("base_lin_vel", self.cfg.observations["actor"].terms)
        self.assertIn("base_lin_vel", self.cfg.observations["critic"].terms)
        for group in ("actor", "critic"):
            self.assertNotIn("height_scan", self.cfg.observations[group].terms)

    def test_nan_guard_present_and_curricula_absent(self) -> None:
        self.assertIn("nan_state", self.cfg.terminations)
        # Mutation-based curricula are omitted so the cfg stays hashable
        # (docs/e2e-research/63 §2.2's named tension; G3 decides ramps).
        self.assertEqual(self.cfg.curriculum, {})

    def test_play_mode_is_a_mode_not_an_inversion(self) -> None:
        # play changes TWO things and says why: pushes come more often
        # (a viewer sees recovery) and observation corruption goes off
        # (a viewer wants the true state). The first cut only inverted
        # the push interval (review 2026-09-01).
        from rq_mjlab.microduck_walk import PLAY_PUSH_INTERVAL_S  # noqa: PLC0415

        play_cfg, _stamps = microduck_walk_env_cfg(play=True)
        self.assertEqual(
            play_cfg.events["push_robot"].interval_range_s, PLAY_PUSH_INTERVAL_S
        )
        self.assertFalse(play_cfg.observations["actor"].enable_corruption)
        self.assertTrue(self.cfg.observations["actor"].enable_corruption)

    def test_our_events_ride_along(self) -> None:
        for name in ("bam_expansion", "bam_param_dr", "mass_inertia", "armature"):
            self.assertIn(name, self.cfg.events)
        self.assertEqual(self.cfg.events["foot_friction"].params["ranges"], (0.7, 1.3))

    def test_their_silent_no_ops_are_refusals_here(self) -> None:
        # The showcase (63 §1): wiring DR to a field the BAM actuator
        # overwrites every step — their randomize_joint_friction /
        # randomize_joint_damping — is not a warning, it is a refusal.
        from mjlab.envs import mdp as envs_mdp  # noqa: PLC0415

        cfg, _stamps = microduck_walk_env_cfg()
        actuator = cfg.scene.entities["robot"].articulation.actuators[0]
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=envs_mdp.dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "ranges": (0.9, 1.1),
            },
        )
        with self.assertRaises(SilentNoOp) as ctx:
            lint(cfg.events, (actuator,))
        self.assertIn("randomize_joint_friction", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TheLawDrSpan(unittest.TestCase):
    """The walk C1's knob: a declared span around the point fit, or no
    law DR at all — and the basis says which (2026-09-04)."""

    def test_no_span_means_no_law_dr_and_says_so(self) -> None:
        from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415

        cfg, stamps = microduck_walk_env_cfg(law_dr_span=None)
        self.assertNotIn("bam_param_dr", cfg.events)
        self.assertIn("point fit", stamps["dr_basis"])
        zero_cfg, zero_stamps = microduck_walk_env_cfg(law_dr_span=0.0)
        self.assertNotIn("bam_param_dr", zero_cfg.events)
        self.assertEqual(zero_stamps["dr_basis"], stamps["dr_basis"])

    def test_a_wide_span_lands_on_the_basis(self) -> None:
        from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415

        cfg, stamps = microduck_walk_env_cfg(law_dr_span=0.30)
        self.assertIn("bam_param_dr", cfg.events)
        self.assertIn("0.3", stamps["dr_basis"])
        # Robot and actuator stamps do not move with the span.
        _base, base_stamps = microduck_walk_env_cfg()
        self.assertEqual(stamps["robot"], base_stamps["robot"])
        self.assertEqual(stamps["actuator"], base_stamps["actuator"])

    def test_a_pin_can_move_one_axis_and_refuses_an_unknown_one(self) -> None:
        from rq_mjlab.microduck_walk import (  # noqa: PLC0415
            PIN_AXES,
            microduck_walk_env_cfg,
        )

        _, kt = microduck_walk_env_cfg(law_pin_scale=0.8, law_pin_only=PIN_AXES["kt"])
        self.assertIn("pinned: kt at fit x 0.8", kt["dr_basis"])
        self.assertIn("the rest at the fit", kt["dr_basis"])
        _, friction = microduck_walk_env_cfg(
            law_pin_scale=0.8, law_pin_only=PIN_AXES["friction"]
        )
        self.assertIn("friction_base", friction["dr_basis"])
        self.assertNotIn("kt at", friction["dr_basis"])
        with self.assertRaisesRegex(ValueError, "pin_only names"):
            microduck_walk_env_cfg(law_pin_scale=0.8, law_pin_only=("armature",))

    def test_a_pinned_scale_is_a_degenerate_region_with_its_own_basis(self) -> None:
        from rq_mjlab.microduck_walk import microduck_walk_env_cfg  # noqa: PLC0415

        cfg, stamps = microduck_walk_env_cfg(law_pin_scale=0.5)
        self.assertIn("bam_param_dr", cfg.events)
        self.assertIn("pinned", stamps["dr_basis"])
        self.assertIn("0.5", stamps["dr_basis"])
        with self.assertRaisesRegex(ValueError, "positive"):
            microduck_walk_env_cfg(law_pin_scale=0.0)
