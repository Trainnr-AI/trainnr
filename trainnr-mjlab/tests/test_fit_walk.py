"""A walk trained under a measured fit (2026-09-25): the joints carry the
fit's estimates, each pinned parameter is drawn over its interval, each
unpinned one over the declared span and said so, and the run, the
certificate and the manifest name the fit by its own stamp."""

from __future__ import annotations

import unittest

import mujoco
from trainnr.deploy.manifest import Key
from trainnr.robot.fit_record import FitRecord, fit_stamp
from trainnr.robot.identify import IdentifiedParameter

from trainnr_mjlab.fit_walk import (
    NOT_PINNED,
    apply_to_spec,
    basis_text,
    dr_events,
    terms_of,
)
from trainnr_mjlab.walk_export import manifest_identity
from trainnr_mjlab.walk_view import require_same_identity
from trainnr_mjlab.walks import Identity

XML = """<mujoco><worldbody><body><freejoint/><geom size="0.1"/>
  <body><joint name="FL_hip_joint" type="hinge"/><geom size="0.05"/></body>
</body></worldbody></mujoco>"""


def fit(
    *params: tuple[str, float, float, bool], basis: str = "public log"
) -> FitRecord:
    return FitRecord(
        robot="go2",
        recording="iit-go2-chirp@645b846a7d38",
        anchor="PD command as torque",
        confidence=0.95,
        parameters=tuple(
            IdentifiedParameter(name, est, half, allowed_range=1.0, pinned=pinned)
            for name, est, half, pinned in params
        ),
        created_utc="2026-09-24T00:00:00+00:00",
        basis=basis,
    )


FIT = fit(
    ("FL_hip_joint.armature", 0.02, 0.008, False),
    ("FL_hip_joint.damping", 0.18, 0.15, True),
    ("FL_hip_joint.frictionloss", 0.30, 0.19, True),
)


class TheJoints(unittest.TestCase):
    def test_the_model_s_joint_carries_the_fit_s_estimates(self) -> None:
        spec = apply_to_spec(mujoco.MjSpec.from_string(XML), FIT)
        model = spec.compile()
        joint = model.joint("FL_hip_joint")
        self.assertAlmostEqual(float(model.dof_armature[joint.dofadr[0]]), 0.02)
        self.assertAlmostEqual(float(model.dof_damping[joint.dofadr[0]]), 0.18)
        self.assertAlmostEqual(float(model.dof_frictionloss[joint.dofadr[0]]), 0.30)

    def test_a_joint_the_model_lacks_is_refused_by_name(self) -> None:
        other = fit(("FR_calf_joint.damping", 0.1, 0.01, True))
        with self.assertRaisesRegex(ValueError, "FR_calf_joint"):
            apply_to_spec(mujoco.MjSpec.from_string(XML), other)


class TheSpans(unittest.TestCase):
    def test_pinned_terms_span_their_interval_the_rest_the_declared_span(self) -> None:
        by_term = {t.term: t for t in terms_of(FIT)}
        self.assertAlmostEqual(by_term["damping"].low, 0.03)
        self.assertAlmostEqual(by_term["damping"].high, 0.33)
        self.assertAlmostEqual(by_term["frictionloss"].low, 0.11)
        span = NOT_PINNED.relative_span
        self.assertAlmostEqual(by_term["armature"].low, 0.02 * (1 - span))
        self.assertAlmostEqual(by_term["armature"].high, 0.02 * (1 + span))

    def test_an_interval_reaching_below_zero_is_floored(self) -> None:
        wide = fit(("FL_hip_joint.damping", 0.05, 0.2, True))
        self.assertEqual(terms_of(wide)[0].low, 0.0)

    def test_one_reset_event_per_term_over_every_joint_absolute(self) -> None:
        events = dr_events(FIT)
        self.assertEqual(
            set(events), {"fit_armature", "fit_damping", "fit_frictionloss"}
        )
        damping = events["fit_damping"]
        self.assertEqual(damping.mode, "reset")
        self.assertEqual(damping.params["operation"], "abs")
        (low, high) = damping.params["ranges"]["^FL_hip_joint$"]
        self.assertAlmostEqual(low, 0.03)
        self.assertAlmostEqual(high, 0.33)

    def test_the_basis_names_the_fit_its_robot_and_what_is_not_pinned(self) -> None:
        text = basis_text(FIT)
        self.assertIn(fit_stamp(FIT), text)
        self.assertIn("public log", text)
        self.assertIn("2/3 joint terms drawn over their bootstrap intervals", text)
        self.assertIn(f"1 {NOT_PINNED.word}", text)


class TheIdentity(unittest.TestCase):
    def test_a_fit_trained_checkpoint_is_refused_in_a_world_without_it(self) -> None:
        trained = {Identity.ROBOT: "go2@x", Identity.FIT: fit_stamp(FIT)}
        with self.assertRaises(SystemExit):
            require_same_identity(trained, {Identity.ROBOT: "go2@x"})
        require_same_identity(
            trained, {Identity.ROBOT: "go2@x", Identity.FIT: fit_stamp(FIT)}
        )

    def test_a_checkpoint_from_before_fits_is_not_held_to_one(self) -> None:
        require_same_identity({Identity.ROBOT: "go2@x"}, {Identity.ROBOT: "go2@x"})

    def test_the_manifest_carries_the_fit_and_its_basis(self) -> None:
        trained = {
            Key.FIT: fit_stamp(FIT),
            Key.FIT_BASIS: "public log",
            Key.DR_BASIS: "b",
        }
        out = manifest_identity({Key.ROBOT: "go2@x"}, trained)
        self.assertEqual(out[Key.FIT], fit_stamp(FIT))
        self.assertEqual(out[Key.FIT_BASIS], "public log")

    def test_the_identity_and_the_manifest_spell_the_fit_alike(self) -> None:
        self.assertEqual((Identity.FIT, Identity.FIT_BASIS), (Key.FIT, Key.FIT_BASIS))

    def test_a_fit_s_stamp_is_its_content(self) -> None:
        self.assertTrue(fit_stamp(FIT).startswith("fit@"))
        moved = fit(("FL_hip_joint.damping", 0.19, 0.15, True))
        self.assertNotEqual(fit_stamp(moved), fit_stamp(FIT))


if __name__ == "__main__":
    unittest.main()
