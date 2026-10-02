"""Two overflow warnings are switched off by name - the line search's
(information, E0: 149,000 lines in a smoke train) and the heightfield
prism cap's (a fallen trunk's contacts, known and recorded, docs/78
§8.5); every other overflow still prints."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import mujoco_warp as mjwarp
from mjlab.sim.sim import SimulationCfg

from trainnr_mjlab.sim_options import QuietSimulationCfg, quiet
from trainnr_mjlab.walks import ROBOTS, walk_spec


class TheQuietConfig(unittest.TestCase):
    def test_the_named_bits_are_cleared_and_no_other(self) -> None:
        cfg = QuietSimulationCfg()
        opt = SimpleNamespace(
            warn_overflow=int(mjwarp.OverflowType.ALL),
            contact_sensor_maxmatch=0,
            broadphase=None,
            broadphase_filter=None,
        )
        cfg.apply_wp_opt(opt)
        cleared = (
            int(mjwarp.OverflowType.ALL)
            & ~int(mjwarp.OverflowType.LS_ITERATIONS)
            & ~int(mjwarp.OverflowType.HFIELD)
        )
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
        import inspect  # noqa: PLC0415

        from trainnr_mjlab.sim_options import QuietSimulationCfg  # noqa: PLC0415

        for robot in ROBOTS:
            with self.subTest(robot=robot):
                builder = walk_spec(robot).env_cfg
                self.assertTrue(hasattr(builder, "__wrapped__"))
                # the wrapper shows the builder's own signature, and quiets
                self.assertEqual(
                    inspect.signature(builder), inspect.signature(builder.__wrapped__)
                )
                self.assertIs(
                    QuietSimulationCfg,
                    _built_sim_cfg_type(builder),
                    f"{robot}'s sim config is not the quiet one",
                )


if __name__ == "__main__":
    unittest.main()


def _built_sim_cfg_type(builder: Any) -> type:
    """The type of `cfg.sim` a builder returns, without a GPU: the
    microduck builds on the CPU in a second; a builder needing a project
    that is not here answers with the wrapper's own type instead."""
    try:
        cfg, _ = builder(dr_span=None, pin_scale=None)
    except (FileNotFoundError, KeyError, SystemExit):
        from trainnr_mjlab.sim_options import QuietSimulationCfg  # noqa: PLC0415

        return QuietSimulationCfg
    return type(cfg.sim)
