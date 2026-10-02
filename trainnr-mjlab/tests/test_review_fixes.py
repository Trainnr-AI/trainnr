"""Pins for 2026-09-01's deep review — each test names the bug that
existed and would have been invisible without it."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from trainnr_mjlab.actuator import BamActuator, BamActuatorCfg
from trainnr_mjlab.dr import bam_param_dr_event

REPO = Path(__file__).resolve().parents[2]
XL330_BUNDLE = REPO / "robots" / "actuator-bundles" / "xl330.m6.bundle.json"


def xl330_cfg() -> BamActuatorCfg:
    return BamActuatorCfg.from_bundle(
        XL330_BUNDLE, target_names_expr=(".*",), physics_dt=0.005
    )


# A floating-base rig: freejoint (6 dofs) + two hinges. Global joint
# ids 1,2 -> dofs 6,7; LOCAL non-free ids 0,1. Indexing jnt_dofadr with
# the local ids yields dofs 0 and 6 — the base's linear dof and one
# servo, the exact bug.
FREEJOINT_XML = """
<mujoco>
  <worldbody>
    <body name="trunk">
      <freejoint name="root"/>
      <geom size="0.05"/>
      <body name="link_a" pos="0.1 0 0">
        <joint name="j_a" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
        <geom size="0.03"/>
        <body name="link_b" pos="0.1 0 0">
          <joint name="j_b" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
          <geom size="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class DofMapping(unittest.TestCase):
    def test_a_floating_base_rig_resolves_servo_dofs_not_the_base(self) -> None:
        # THE bug: local target ids were fed into mj_model.jnt_dofadr as
        # if global — on a freejoint robot the friction budget landed on
        # the base's linear dof and the last servo fell off. The first
        # "pin" recomputed the fix's own expression instead of calling
        # initialize, so the suite stayed green with the bug restored
        # (second review, 2026-09-01). This one drives the real path.
        import mujoco  # noqa: PLC0415
        import mujoco_warp as mjw  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415
        from mjlab.entity import EntityCfg  # noqa: PLC0415
        from mjlab.entity.entity import (  # noqa: PLC0415
            Entity,
            EntityArticulationInfoCfg,
        )

        wp.init()
        cfg = BamActuatorCfg.from_bundle(
            XL330_BUNDLE, target_names_expr=("j_a", "j_b"), physics_dt=0.005
        )
        entity = Entity(
            EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_string(FREEJOINT_XML),
                articulation=EntityArticulationInfoCfg(actuators=(cfg,)),
            )
        )
        mj_model = entity.spec.compile()
        with wp.ScopedDevice("cpu"):
            model = mjw.put_model(mj_model, batch_sizes={"dof_frictionloss": 1})
            data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=1)
            # mjlab's own indexing, computed from the REAL compiled
            # model (Entity.initialize does this then initializes its
            # actuators; the rest of that method needs the scene's
            # torch-viewed model, which a unit test has no business
            # rebuilding — the mapping under test is this one line).
            entity.indexing = entity._compute_indexing(mj_model, "cpu")
            actuator = entity.actuators[0]
            actuator.initialize(mj_model, model, data, "cpu")
        # dofs 6 and 7 — never 0 (the base) and never short.
        self.assertEqual(actuator._dof_ids.tolist(), [6, 7])


class CoercionRefusals(unittest.TestCase):
    """A generator drains inside a comprehension and leaves the checks
    empty — the same silent-coercion family as the dict, and still open
    after the first fix (second review, 2026-09-01)."""

    def test_entity_and_lint_refuse_generators_and_dicts(self) -> None:
        from trainnr_mjlab.entity import entity_from_bundle  # noqa: PLC0415
        from trainnr_mjlab.linter import lint  # noqa: PLC0415

        cfg = xl330_cfg()
        robot = REPO / "robots" / "microduck"
        for shape in ((c for c in (cfg,)), {"servos": cfg}):
            with self.assertRaises(TypeError):
                entity_from_bundle(robot, "robot_walk.xml", shape)
            with self.assertRaises(TypeError):
                lint({}, shape)


class _StubBam(BamActuator):
    """A stamp-carrying stand-in: records draws, skips initialize."""

    def __init__(self, cfg: BamActuatorCfg) -> None:  # noqa: super-init-not-called
        self.cfg = cfg
        self.calls: list[tuple[list[int], list[str]]] = []

    def set_param_draws(self, env_ids, draws) -> None:  # type: ignore[override]
        self.calls.append((list(env_ids.tolist()), sorted(draws)))


class _StubScene(dict):
    pass


class _StubEnv:
    def __init__(self, actuators) -> None:
        entity = type("E", (), {"actuators": actuators})()
        self.scene = _StubScene(robot=entity)
        self.device = "cpu"


class DrEventWiring(unittest.TestCase):
    def test_the_event_late_binds_the_live_actuator_by_stamp(self) -> None:
        # THE bug: the event closed over the CFG and called
        # set_param_draws on it — AttributeError at the first reset.
        # Now the closure resolves the live BamActuator from the scene.
        cfg = xl330_cfg()
        live = _StubBam(cfg)
        term, _basis = bam_param_dr_event(cfg, fallback_span=0.1)
        term.func(_StubEnv([live]), torch.tensor([0, 2]))
        [(env_ids, names)] = live.calls
        self.assertEqual(env_ids, [0, 2])
        self.assertIn("kt", names)

    def test_zero_or_two_matching_actuators_refuse_by_stamp(self) -> None:
        cfg = xl330_cfg()
        term, _basis = bam_param_dr_event(cfg, fallback_span=0.1)
        with self.assertRaises(RuntimeError) as ctx:
            term.func(_StubEnv([]), torch.tensor([0]))
        self.assertIn(cfg.stamp, str(ctx.exception))
        with self.assertRaises(RuntimeError):
            term.func(_StubEnv([_StubBam(cfg), _StubBam(cfg)]), torch.tensor([0]))

    def test_an_instance_or_stranger_is_refused_at_build(self) -> None:
        with self.assertRaises(TypeError):
            bam_param_dr_event(object(), fallback_span=0.1)  # type: ignore[arg-type]


class FromBundleDoors(unittest.TestCase):
    def test_an_unconsumed_fit_key_is_refused_by_name(self) -> None:
        # THE gap: params were cherry-picked by known name, so a typo'd
        # field silently zeroed a friction term.
        from trainnr.robot.actuator_bundle import _stamp  # noqa: PLC0415

        bundle = json.loads(XL330_BUNDLE.read_text())
        bundle["params"]["load_friction_motor_qad"] = 0.01  # the typo
        bundle["checks"] = __import__(
            "trainnr.robot.actuator_bundle", fromlist=["run_checks"]
        ).run_checks(bundle["params"])
        bundle["stamp"] = _stamp(bundle)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "xl330.m6.bundle.json"
            path.write_text(json.dumps(bundle))
            with self.assertRaises(ValueError) as ctx:
                BamActuatorCfg.from_bundle(
                    path, target_names_expr=(".*",), physics_dt=0.005
                )
        self.assertIn("load_friction_motor_qad", str(ctx.exception))

    def test_hand_construction_is_refused(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            BamActuatorCfg(target_names_expr=(".*",))
        self.assertIn("from_bundle", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
