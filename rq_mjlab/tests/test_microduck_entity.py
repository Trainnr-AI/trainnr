"""Flagship G2, first brick (docs/e2e-research/63): microduck's walk
model from OUR stamped bundle, driven by the CERTIFIED xl330.m6 — every
identity carries a hash, which is exactly what their stack cannot say."""

from __future__ import annotations

import unittest
from pathlib import Path

from rq_mjlab import (
    BamActuatorCfg,
    bam_expansion_event,
    bam_param_dr_event,
    dr_from_bundle,
    entity_from_bundle,
    lint,
)

REPO = Path(__file__).resolve().parents[2]
ROBOT = REPO / "robots" / "microduck"
ACTUATOR_BUNDLE = REPO / "robots" / "actuator-bundles" / "xl330.m6.bundle.json"
PHYSICS_DT = 0.002
SERVOS = 14


class MicroduckEntity(unittest.TestCase):
    def _actuator(self) -> BamActuatorCfg:
        return BamActuatorCfg.from_bundle(
            ACTUATOR_BUNDLE, target_names_expr=[".*"], physics_dt=PHYSICS_DT
        )

    def test_builds_from_stamped_bundles_and_compiles(self) -> None:
        actuator = self._actuator()
        cfg, stamp = entity_from_bundle(ROBOT, "robot_walk.xml", (actuator,))
        self.assertTrue(stamp.startswith("microduck@"))
        self.assertTrue(actuator.stamp.startswith("xl330-m6@"))
        model = cfg.spec_fn().compile()
        self.assertEqual(model.nu, SERVOS)

    def test_a_dict_of_actuators_is_refused_not_silently_stringified(self) -> None:
        # mjlab's actuators field is an ordered tuple; a dict used to
        # become a tuple of its KEY STRINGS — an entity with no law at
        # all (caught 2026-09-01 by the walk cfg's linter showcase).
        with self.assertRaises(TypeError) as ctx:
            entity_from_bundle(ROBOT, "robot_walk.xml", {"servos": self._actuator()})
        self.assertIn("tuple of actuator cfgs", str(ctx.exception))

    def test_point_estimate_dr_refuses_without_a_declared_span(self) -> None:
        # The store's m6 is a point estimate (advisories say so); asking
        # for DR without declaring a span must refuse, never invent ±10%.
        with self.assertRaises(ValueError):
            dr_from_bundle(ACTUATOR_BUNDLE)

    def test_dr_with_declared_span_carries_its_basis(self) -> None:
        _ranges, basis = dr_from_bundle(ACTUATOR_BUNDLE, fallback_span=0.1)
        self.assertIn("declared", basis)
        # The event takes the CFG (the live actuator does not exist at
        # cfg-build time) and reads the bundle path FROM it — one truth.
        event, event_basis = bam_param_dr_event(self._actuator(), fallback_span=0.1)
        self.assertEqual(basis, event_basis)
        events = {"dr": event, "expand": bam_expansion_event()}
        lint(events, (self._actuator(),))  # no refusal: blessed setup

    def test_lint_refuses_a_dict_instead_of_passing_vacuously(self) -> None:
        # A dict iterates its KEY STRINGS: the old lint filtered them
        # out and blessed anything (review, 2026-09-01).
        with self.assertRaises(TypeError) as ctx:
            lint({}, {"servos": self._actuator()})
        self.assertIn("ordered tuple", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
