"""dr_from_bundle's basis discipline, the per-world draws through the
effective law, the linter against mjlab's REAL DR functions, and the
entity factory's stamp. CPU only."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLES = REPO_ROOT / "robots" / "actuator-bundles"
XL330_M6 = BUNDLES / "xl330.m6.bundle.json"
DT = 0.002
SPAN = 0.1

XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="j" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.008" mass="0.08"/>
    </body>
  </worldbody>
</mujoco>
"""


def _cfg():
    from rq_mjlab.actuator import BamActuatorCfg  # noqa: PLC0415

    return BamActuatorCfg.from_bundle(XL330_M6, target_names_expr=("j",), physics_dt=DT)


class TheRanges(unittest.TestCase):
    def test_a_point_estimate_bundle_refuses_without_a_declared_span(self) -> None:
        from rq_mjlab.dr import dr_from_bundle  # noqa: PLC0415

        with self.assertRaises(ValueError) as caught:
            dr_from_bundle(XL330_M6)
        self.assertIn("uncertainty", str(caught.exception))

    def test_a_declared_span_comes_back_with_its_basis(self) -> None:
        from rq_mjlab.dr import dr_from_bundle  # noqa: PLC0415

        ranges, basis = dr_from_bundle(XL330_M6, fallback_span=SPAN)
        self.assertIn("caller-declared", basis)
        self.assertIn("0.1", basis)
        low, high = ranges["kt"]
        self.assertLess(low, high)


class TheDraws(unittest.TestCase):
    def _initialized_actuator(self):
        """The actuator over a real one-hinge model on Warp's CPU device —
        the same seam the device pin drives, worlds = 2."""
        import mujoco  # noqa: PLC0415
        import mujoco_warp as mjw  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        wp.init()
        cfg = _cfg()
        spec = mujoco.MjSpec.from_string(XML)
        entity = _Duck(
            indexing=_Duck(
                ctrl_ids=torch.tensor([0]),
                # the honest global mapping for this rig: one hinge,
                # global joint 0, dof 0 (the dof fix routes through
                # entity.indexing — review, 2026-09-01).
                joint_v_adr=torch.tensor([0]),
            )
        )
        actuator = cfg.build(entity, [0], ["j"])
        actuator.edit_spec(spec, ["j"])
        mj_model = spec.compile()
        with wp.ScopedDevice("cpu"):
            model = mjw.put_model(mj_model, batch_sizes={"dof_frictionloss": 2})
            data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=2)
            actuator._target_ids_list = [0]
            actuator.initialize(mj_model, model, data, "cpu")
        return cfg, actuator

    def test_draws_change_the_budget_per_world(self) -> None:
        from rq_mjlab.kernel import friction_budget  # noqa: PLC0415

        cfg, actuator = self._initialized_actuator()
        self.assertIs(actuator.effective_law(), cfg.law)  # no draws: the cfg's law
        actuator.set_param_draws(
            torch.tensor([0, 1]),
            {"friction_base": torch.tensor([cfg.law.friction_base, 10.0])},
        )
        law = actuator.effective_law()
        zeros = torch.zeros((2, 1), dtype=torch.float64)
        budget = friction_budget(law, zeros, zeros, zeros)
        # At rest the Stribeck factor is exactly 1: the budget is
        # base + friction_stribeck, per world's own base.
        at_rest = cfg.law.friction_stribeck
        self.assertAlmostEqual(
            float(budget[0]), cfg.law.friction_base + at_rest, places=9
        )
        self.assertAlmostEqual(float(budget[1]), 10.0 + at_rest, places=6)

    def test_an_unknown_parameter_is_refused(self) -> None:
        _, actuator = self._initialized_actuator()
        with self.assertRaises(ValueError) as caught:
            actuator.set_param_draws(
                torch.tensor([0]), {"q_offset": torch.tensor([0.1])}
            )
        self.assertIn("q_offset", str(caught.exception))

    def test_the_event_redraws_only_law_parameters(self) -> None:
        from rq_mjlab.dr import bam_param_dr_event  # noqa: PLC0415

        cfg, actuator = self._initialized_actuator()
        # The event takes the CFG (one truth: the bundle path rides on
        # it) and late-binds the LIVE actuator from the scene by stamp
        # (review, 2026-09-01 — the old wiring closed over whatever it
        # was handed and crashed on cfgs at first reset).
        term, basis = bam_param_dr_event(cfg, fallback_span=SPAN)
        self.assertIn("caller-declared", basis)
        scene = {"robot": _Duck(actuators=[actuator])}
        term.func(_Duck(device="cpu", scene=scene), torch.tensor([0, 1]))
        law = actuator.effective_law()
        kt = law.kt
        self.assertEqual(tuple(kt.shape), (2, 1))
        low, high = cfg.law.kt * (1 - SPAN), cfg.law.kt * (1 + SPAN)
        self.assertTrue(bool(((kt >= low) & (kt <= high)).all()))


class TheLinterAgainstRealDr(unittest.TestCase):
    def test_mjlabs_joint_friction_is_a_silent_noop_under_the_law(self) -> None:
        from mjlab.envs.mdp.dr.joint import (  # noqa: PLC0415
            joint_damping,
            joint_friction,
        )
        from mjlab.managers.event_manager import EventTermCfg  # noqa: PLC0415

        from rq_mjlab.events import bam_expansion_event  # noqa: PLC0415
        from rq_mjlab.linter import SilentNoOp, lint  # noqa: PLC0415

        events = {
            "expand": bam_expansion_event(),
            "friction": EventTermCfg(func=joint_friction, mode="reset"),
        }
        with self.assertRaises(SilentNoOp) as caught:
            lint(events, (_cfg(),))
        self.assertIn("joint_friction", str(caught.exception))
        # joint_damping randomises dof_damping — a PASSIVE under our
        # split (the law never writes it): legitimate, and the linter
        # says nothing.
        lint(
            {
                "expand": bam_expansion_event(),
                "damping": EventTermCfg(func=joint_damping, mode="reset"),
            },
            (_cfg(),),
        )


class TheEntityFactory(unittest.TestCase):
    def test_the_cfg_compiles_and_the_stamp_names_the_bundle(self) -> None:
        import tempfile  # noqa: PLC0415

        from rq_mjlab.entity import entity_from_bundle  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            robot = Path(tmp) / "one-hinge"
            robot.mkdir()
            (robot / "robot.xml").write_text(XML)
            cfg, bundle_stamp = entity_from_bundle(robot, "robot.xml", (_cfg(),))
            self.assertTrue(bundle_stamp.startswith("one-hinge@"))
            model = cfg.spec_fn().compile()
            self.assertEqual(model.njnt, 1)
            self.assertEqual(len(cfg.articulation.actuators), 1)

    def test_a_missing_model_file_is_refused_by_name(self) -> None:
        import tempfile  # noqa: PLC0415

        from rq_mjlab.entity import entity_from_bundle  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            robot = Path(tmp) / "empty"
            robot.mkdir()
            with self.assertRaises(FileNotFoundError) as caught:
                entity_from_bundle(robot, "robot.xml", ())
            self.assertIn("robot.xml", str(caught.exception))


class _Duck:
    def __init__(self, **kw):
        self.__dict__.update(kw)


if __name__ == "__main__":
    unittest.main()
