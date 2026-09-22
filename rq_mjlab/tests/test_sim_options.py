"""The line-search warning alone is switched off; every other overflow
still prints (E0, docs/78, 2026-09-22: 149,000 lines in a smoke train)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco_warp as mjwarp
from mjlab.sim.sim import SimulationCfg

from rq_mjlab.sim_options import QuietSimulationCfg, quiet
from rq_mjlab.walks import ROBOTS, walk_spec


class TheQuietConfig(unittest.TestCase):
    def test_only_the_line_search_bit_is_cleared(self) -> None:
        cfg = QuietSimulationCfg()
        opt = SimpleNamespace(
            warn_overflow=int(mjwarp.OverflowType.ALL),
            contact_sensor_maxmatch=0,
            broadphase=None,
            broadphase_filter=None,
        )
        cfg.apply_wp_opt(opt)
        cleared = int(mjwarp.OverflowType.ALL) & ~int(mjwarp.OverflowType.LS_ITERATIONS)
        self.assertEqual(opt.warn_overflow, cleared)
        self.assertTrue(opt.warn_overflow & int(mjwarp.OverflowType.NEFC))
        self.assertEqual(opt.contact_sensor_maxmatch, cfg.contact_sensor_maxmatch)

    def test_quiet_keeps_every_field_of_mjlabs(self) -> None:
        env = SimpleNamespace(sim=SimulationCfg(njmax=300, nconmax=None))
        env.sim.mujoco.ls_iterations = 20
        quiet(env)
        self.assertIsInstance(env.sim, QuietSimulationCfg)
        self.assertEqual(env.sim.njmax, 300)
        self.assertEqual(env.sim.mujoco.ls_iterations, 20)
        self.assertIs(quiet(env).sim, env.sim, "idempotent")

    def test_every_walk_builder_is_wrapped_and_keeps_its_signature(self) -> None:
        for robot in ROBOTS:
            with self.subTest(robot=robot):
                builder = walk_spec(robot).env_cfg
                self.assertTrue(hasattr(builder, "__wrapped__"))


if __name__ == "__main__":
    unittest.main()
