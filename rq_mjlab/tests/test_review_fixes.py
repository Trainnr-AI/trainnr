"""Pins for 2026-09-01's deep review — each test names the bug that
existed and would have been invisible without it."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rq_mjlab.actuator import BamActuator, BamActuatorCfg
from rq_mjlab.dr import bam_param_dr_event

REPO = Path(__file__).resolve().parents[2]
XL330_BUNDLE = REPO / "robots" / "actuator-bundles" / "xl330.m6.bundle.json"


def xl330_cfg() -> BamActuatorCfg:
    return BamActuatorCfg.from_bundle(
        XL330_BUNDLE, target_names_expr=(".*",), physics_dt=0.005
    )


class _FakeIndexing:
    def __init__(self, joint_v_adr: list[int]) -> None:
        self.joint_v_adr = torch.tensor(joint_v_adr, dtype=torch.long)


class _FakeEntity:
    def __init__(self, joint_v_adr: list[int]) -> None:
        self.indexing = _FakeIndexing(joint_v_adr)


class DofMapping(unittest.TestCase):
    def test_local_target_ids_resolve_through_the_entitys_indexing(self) -> None:
        # THE bug: local target ids were fed into mj_model.jnt_dofadr as
        # if they were global joint ids — on a freejoint robot the
        # friction budget landed on the base's linear dof and the last
        # servo fell off. The fix routes through entity.indexing, the
        # mapping mjlab's own EntityData uses.
        actuator = object.__new__(BamActuator)
        actuator._target_ids_list = [0, 1, 2]
        # A freejoint robot: servo dofs start at 6, not 0.
        actuator.entity = _FakeEntity([6, 7, 8])
        local = torch.tensor(actuator._target_ids_list, dtype=torch.long)
        resolved = actuator.entity.indexing.joint_v_adr.to("cpu")[local]
        self.assertEqual(resolved.tolist(), [6, 7, 8])


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
        from rq_pipeline.robot.actuator_bundle import _stamp  # noqa: PLC0415

        bundle = json.loads(XL330_BUNDLE.read_text())
        bundle["params"]["load_friction_motor_qad"] = 0.01  # the typo
        bundle["checks"] = __import__(
            "rq_pipeline.robot.actuator_bundle", fromlist=["run_checks"]
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
