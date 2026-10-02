"""The port's pins, CPU-side: the kernel's equations at both ends, the
refusals at every door, the linter's judgements. The GPU integration
(a batched env stepping under the law) is its own test file; nothing
here needs CUDA or a compiled scene beyond one joint."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLES = REPO_ROOT / "robots" / "actuator-bundles"
XL330_M6 = BUNDLES / "xl330.m6.bundle.json"
XL330_M1 = BUNDLES / "xl330.m1.bundle.json"
STS_M1 = BUNDLES / "feetech_sts3215_7_4V.m1.bundle.json"
DT = 0.002


def t(value: float) -> "torch.Tensor":
    """A (1, 1) float64 tensor: the tests pin equations, not float32."""
    return torch.tensor([[value]], dtype=torch.float64)


def _params(path: Path) -> dict:
    return json.loads(path.read_text())["params"]


class TheKernel(unittest.TestCase):
    def _law(self, path: Path, **overrides):
        from trainnr_mjlab.actuator import BamActuatorCfg  # noqa: PLC0415

        return BamActuatorCfg.from_bundle(
            path, target_names_expr=("j",), physics_dt=DT, **overrides
        ).law

    def test_m1_has_no_state_dependent_extra(self) -> None:
        from trainnr_mjlab.kernel import friction_budget  # noqa: PLC0415

        law = self._law(XL330_M1)
        base = _params(XL330_M1)["friction_base"]
        for tau_prev, tau_ext, qd in (
            (0.0, 0.0, 0.0),
            (0.4, -0.3, 2.0),
            (-1.0, 1.0, -5.0),
        ):
            budget = friction_budget(
                law,
                t(tau_prev),
                t(tau_ext),
                t(qd),
            )
            self.assertAlmostEqual(float(budget), base, places=12)

    def test_m6_budget_spelled_out_longhand(self) -> None:
        """The XL330 M6 numbers through the budget by hand: at rest
        (`qd = 0` so the Stribeck factor is exactly 1) with opposing
        torques `tau_prev = 0.1`, `tau_ext = -0.2` — the same hand case
        the pipeline's law was pinned with."""
        from trainnr_mjlab.kernel import friction_budget  # noqa: PLC0415

        law = self._law(XL330_M6)
        p = _params(XL330_M6)
        tau_prev, tau_ext = 0.1, -0.2
        gearbox = abs(
            tau_ext * p["load_friction_external"] - tau_prev * p["load_friction_motor"]
        )
        gearbox_s = abs(
            tau_ext * p["load_friction_external_stribeck"]
            - tau_prev * p["load_friction_motor_stribeck"]
        )
        # Opposing signs, |tau_ext| > |tau_prev|: the external side's term.
        quadratic = p["load_friction_motor_quad"] * tau_prev**2
        expected = (
            p["friction_base"]
            + gearbox
            + (p["friction_stribeck"] + gearbox_s + quadratic)
        )
        budget = friction_budget(
            law,
            t(tau_prev),
            t(tau_ext),
            t(0.0),
        )
        self.assertAlmostEqual(float(budget), expected, places=11)

    def test_equal_signs_pay_no_quadratic(self) -> None:
        from trainnr_mjlab.kernel import friction_budget  # noqa: PLC0415

        law = self._law(XL330_M6)
        same = friction_budget(law, t(0.1), t(0.2), t(0.0))
        opposing = friction_budget(law, t(0.1), t(-0.2), t(0.0))
        self.assertLess(float(same), float(opposing))

    def test_back_emf_and_the_current_window(self) -> None:
        from trainnr_mjlab.kernel import duty, torque  # noqa: PLC0415

        law = self._law(XL330_M6)
        # At the no-load speed (V = kt*qd) the torque is zero.
        qd = t(law.vin / law.kt)
        full = t(law.max_pwm)
        self.assertAlmostEqual(float(torque(law, qd, full)), 0.0, places=9)
        # At rest, the current limiter caps |duty| at R*I_max/vin.
        big_error = t(100.0)
        capped = duty(law, t(0.0), t(0.0), big_error)
        self.assertAlmostEqual(
            float(capped), law.R * law.max_current / law.vin, places=9
        )

    def test_the_pwm_ceiling_clips_last(self) -> None:
        """Without a current limiter the pwm ceiling is the only clip —
        a hand law, since every bundled Dynamixel carries the limiter."""
        from trainnr_mjlab.kernel import LawParams, duty  # noqa: PLC0415

        law = LawParams(
            kt=1.0, R=2.0, friction_base=0.0, kp=32.0, error_gain=0.166, max_pwm=0.97
        )
        d = duty(law, t(0.0), t(-50.0), t(100.0))
        self.assertAlmostEqual(float(d), law.max_pwm, places=12)


class TheDoors(unittest.TestCase):
    def test_a_verified_bundle_loads_with_its_stamp_and_advisories(self) -> None:
        from trainnr_mjlab.bundle import verified_bundle  # noqa: PLC0415

        bundle, advisories = verified_bundle(XL330_M6)
        self.assertTrue(bundle["stamp"].startswith("xl330-m6@"))
        # The store's bundles today carry no context/metrics/uncertainty
        # sections; the advisories say so and travel with the load.
        self.assertEqual(len(advisories), 3)
        self.assertTrue(any("uncertainty" in a for a in advisories))

    def test_a_tampered_bundle_is_refused_by_name(self) -> None:
        from trainnr_mjlab.bundle import BundleRefused, verified_bundle  # noqa: PLC0415

        data = json.loads(XL330_M6.read_text())
        data["params"]["kt"] *= 1.01
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xl330.m6.bundle.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(BundleRefused) as caught:
                verified_bundle(path)
        self.assertIn("stamp", str(caught.exception).lower())

    def test_the_cfg_carries_the_bundle_and_the_register_choices(self) -> None:
        from trainnr_mjlab.actuator import BamActuatorCfg  # noqa: PLC0415

        cfg = BamActuatorCfg.from_bundle(
            XL330_M6, target_names_expr=("j",), physics_dt=DT
        )
        p = _params(XL330_M6)
        self.assertTrue(cfg.stamp.startswith("xl330-m6@"))
        self.assertEqual(len(cfg.advisories), 3)
        self.assertEqual(cfg.law.kt, p["kt"])
        self.assertEqual(cfg.kp, 400.0)  # the firmware table's measured default
        self.assertEqual(cfg.armature, p["armature"])
        self.assertEqual(cfg.frictionloss, p["friction_base"])
        self.assertEqual(cfg.viscous_damping, p["friction_viscous"])
        expected_delay = math.ceil(round(p["command_delay"] / DT, 9))
        self.assertEqual(cfg.delay_min_lag, expected_delay)
        self.assertEqual(cfg.delay_max_lag, expected_delay)
        deployed = BamActuatorCfg.from_bundle(
            XL330_M6, target_names_expr=("j",), physics_dt=DT, kp=200.0
        )
        self.assertEqual(deployed.kp, 200.0)  # microduck's register value, declared
        self.assertEqual(deployed.law.kp, 200.0)

    def test_a_rate_limited_firmware_is_refused_for_now(self) -> None:
        from trainnr_mjlab.actuator import BamActuatorCfg  # noqa: PLC0415

        with self.assertRaises(NotImplementedError) as caught:
            BamActuatorCfg.from_bundle(STS_M1, target_names_expr=("j",), physics_dt=DT)
        self.assertIn("rate-limit", str(caught.exception))

    def test_friction_dof_is_mujocos_enum(self) -> None:
        import mujoco  # noqa: PLC0415

        from trainnr_mjlab.actuator import FRICTION_DOF  # noqa: PLC0415

        self.assertEqual(FRICTION_DOF, int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))


class TheSpecEdit(unittest.TestCase):
    def test_edit_spec_creates_the_motor_with_the_bundles_passives(self) -> None:
        import mujoco  # noqa: PLC0415

        from trainnr_mjlab.actuator import BamActuatorCfg  # noqa: PLC0415

        cfg = BamActuatorCfg.from_bundle(
            XL330_M6, target_names_expr=("j",), physics_dt=DT
        )
        spec = mujoco.MjSpec.from_string(
            '<mujoco><worldbody><body><joint name="j" axis="0 1 0"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.01" mass="0.1"/>'
            "</body></worldbody></mujoco>"
        )
        actuator = cfg.build(None, [0], ["j"])  # entity unused by edit_spec
        actuator.edit_spec(spec, ["j"])
        model = spec.compile()
        p = _params(XL330_M6)
        stall = cfg.law.stall_torque
        self.assertEqual(model.nu, 1)
        self.assertEqual(
            int(model.actuator_gaintype[0]), int(mujoco.mjtGain.mjGAIN_FIXED)
        )
        self.assertEqual(
            int(model.actuator_biastype[0]), int(mujoco.mjtBias.mjBIAS_NONE)
        )
        self.assertAlmostEqual(float(model.actuator_forcerange[0, 1]), stall, places=9)
        self.assertAlmostEqual(float(model.dof_armature[0]), p["armature"], places=12)
        self.assertAlmostEqual(
            float(model.dof_frictionloss[0]), p["friction_base"], places=12
        )
        self.assertAlmostEqual(
            float(model.dof_damping[0]), p["friction_viscous"], places=12
        )


class TheLinter(unittest.TestCase):
    def _cfg(self):
        from trainnr_mjlab.actuator import BamActuatorCfg  # noqa: PLC0415

        return BamActuatorCfg.from_bundle(
            XL330_M6, target_names_expr=("j",), physics_dt=DT
        )

    def test_a_dr_term_on_an_overwritten_field_is_refused(self) -> None:
        from mjlab.managers.event_manager import (  # noqa: PLC0415
            EventTermCfg,
            requires_model_fields,
        )

        from trainnr_mjlab.linter import SilentNoOp, lint  # noqa: PLC0415

        @requires_model_fields("dof_frictionloss")
        def randomize_friction(env, env_ids):  # pragma: no cover - never called
            del env, env_ids

        events = {"friction_dr": EventTermCfg(func=randomize_friction, mode="reset")}
        with self.assertRaises(SilentNoOp) as caught:
            lint(events, (self._cfg(),))
        self.assertIn("friction_dr", str(caught.exception))
        self.assertIn("dof_frictionloss", str(caught.exception))

    def test_a_cfg_without_the_expansion_event_is_refused(self) -> None:
        from trainnr_mjlab.linter import lint  # noqa: PLC0415

        with self.assertRaises(RuntimeError) as caught:
            lint({}, (self._cfg(),))
        self.assertIn("bam_expansion_event", str(caught.exception))

    def test_the_blessed_cfg_passes(self) -> None:
        from trainnr_mjlab.events import bam_expansion_event  # noqa: PLC0415
        from trainnr_mjlab.linter import lint  # noqa: PLC0415

        lint({"expand": bam_expansion_event()}, (self._cfg(),))

    def test_no_bam_actuators_means_no_opinion(self) -> None:
        from trainnr_mjlab.linter import lint  # noqa: PLC0415

        lint({}, ())


if __name__ == "__main__":
    unittest.main()
