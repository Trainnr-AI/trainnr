"""The certified actuator bundle, against the real vendored fits —
wrap/verify round-trips, the refusals, and the checks that fire on
BAM's own published numbers.
"""

import json
import tempfile
import unittest
from pathlib import Path

from trainnr.robot.actuator_bundle import (
    SCHEMA,
    _stamp,
    as_scales,
    dr_ranges,
    read_bundle,
    run_checks,
    sample_dynamics,
    verify,
    wrap,
    wrap_all,
    write_bundle,
)
from trainnr.robot.actuator_library import list_actuators, list_models


class Wrapping(unittest.TestCase):
    def test_the_inner_params_ride_verbatim(self) -> None:
        # The dialect contract (docs/e2e-research/58 §9): no key renamed,
        # BAM's identity keys untouched — byte-for-byte the published fit.
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        from trainnr.robot.actuator_library import ACTUATORS_ROOT  # noqa: PLC0415

        published = json.loads(
            (ACTUATORS_ROOT / "feetech_sts3215_7_4V" / "m6.json").read_text()
        )
        self.assertEqual(bundle["params"], published)
        self.assertEqual(bundle["schema"], SCHEMA)

    def test_the_stamp_is_content_derived_not_directory_derived(self) -> None:
        # BAM lets the directory slug disagree with the `actuator` key
        # (feetech_sts3215_7_4V vs "sts3215", 57 §3) — the stamp must
        # come from content alone or a renamed directory breaks identity.
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        self.assertTrue(bundle["stamp"].startswith("sts3215-m6@"))

    def test_the_wrap_date_does_not_change_the_identity(self) -> None:
        monday = wrap("feetech_sts3215_7_4V", "m6", wrapped_on="2026-08-31")
        friday = wrap("feetech_sts3215_7_4V", "m6", wrapped_on="2026-09-04")
        self.assertEqual(monday["stamp"], friday["stamp"])

    def test_an_unknown_tier_is_refused_naming_what_exists(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            wrap("feetech_sts3215_7_4V", "m9")
        self.assertIn("m6", str(ctx.exception))


class Checks(unittest.TestCase):
    def test_the_vendored_sts3215_m6_sits_near_the_alpha_rail(self) -> None:
        # Our own library's flagship fit carries alpha = 9.93 — within
        # 1% of BAM's observed search bound of 10. The check exists to
        # say exactly this out loud.
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        self.assertIn("alpha", bundle["checks"]["near_search_bound"])

    def test_floor_and_rail_heuristics_on_synthetic_values(self) -> None:
        checks = run_checks(
            {
                "model": "m6",
                "actuator": "test",
                "load_friction_motor": 3e-13,  # optimizer floor
                "alpha": 9.999,  # on the rail
                "friction_base": 0.05,  # honest value
            }
        )
        self.assertEqual(checks["at_floor"], ["load_friction_motor"])
        self.assertEqual(checks["near_search_bound"], ["alpha"])

    def test_a_mid_range_alpha_is_not_flagged(self) -> None:
        checks = run_checks({"model": "m6", "actuator": "t", "alpha": 5.0})
        self.assertEqual(checks["near_search_bound"], [])


class Verifying(unittest.TestCase):
    def test_a_fresh_wrap_verifies_with_honest_advisories(self) -> None:
        # Vendored fits are BAM point estimates: metrics, uncertainty
        # and context are legitimately absent — and must be SAID.
        advisories = verify(wrap("feetech_sts3215_7_4V", "m6"))
        text = " ".join(advisories)
        for word in ("metrics", "uncertainty", "context"):
            self.assertIn(word, text)

    def test_editing_a_fit_value_after_stamping_is_caught(self) -> None:
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        bundle["params"]["kt"] += 0.1
        with self.assertRaises(ValueError) as ctx:
            verify(bundle)
        self.assertIn("stamp mismatch", str(ctx.exception))

    def test_an_unknown_envelope_section_is_refused(self) -> None:
        # OUR discipline, the opposite of BAM's silently-tolerant
        # loader: a typo'd section never passes as metadata.
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        bundle["provenence"] = {"oops": True}
        with self.assertRaises(ValueError) as ctx:
            verify(bundle)
        self.assertIn("provenence", str(ctx.exception))

    def test_unknown_keys_inside_params_are_bams_business(self) -> None:
        # BAM's key set varies per motor by design (57 §3): a field this
        # verifier has never heard of is verbatim content, not an error.
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        bundle["params"]["error_gain_ratio_v2"] = 1.0
        bundle["stamp"] = _stamp(bundle)
        verify(bundle)  # must not raise

    def test_params_without_bam_identity_keys_are_refused(self) -> None:
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        del bundle["params"]["model"]
        with self.assertRaises(ValueError) as ctx:
            verify(bundle)
        self.assertIn("model", str(ctx.exception))


class Sampling(unittest.TestCase):
    def _with_uncertainty(self) -> dict:
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        bundle["uncertainty"] = {
            "kt": {"low": 1.2, "high": 1.35},
            "friction_base": {"low": 0.04, "high": 0.07},
        }
        bundle["stamp"] = _stamp(bundle)
        return bundle

    def test_a_point_estimate_bundle_refuses_to_invent_a_region(self) -> None:
        # The microduck lesson (57 §5): fitted-treated-as-exact beside a
        # hand-guessed span, with nothing marking which. Here the refusal
        # names the stamp and the way out.
        import numpy as np  # noqa: PLC0415

        bundle = wrap("feetech_sts3215_7_4V", "m6")
        with self.assertRaises(ValueError) as ctx:
            sample_dynamics(bundle, np.random.default_rng(0))
        self.assertIn(bundle["stamp"], str(ctx.exception))
        self.assertIn("fallback_span", str(ctx.exception))

    def test_identified_intervals_bound_every_draw(self) -> None:
        import numpy as np  # noqa: PLC0415

        bundle = self._with_uncertainty()
        rng = np.random.default_rng(7)
        for _ in range(50):
            dynamics, basis = sample_dynamics(bundle, rng)
            self.assertEqual(basis, "identified-interval")
            self.assertEqual(set(dynamics), {"kt", "friction_base"})
            for param, (low, high) in dr_ranges(bundle).items():
                self.assertTrue(low <= dynamics[param] <= high, param)

    def test_a_declared_fallback_span_says_so_in_its_basis(self) -> None:
        import numpy as np  # noqa: PLC0415

        bundle = wrap("feetech_sts3215_7_4V", "m6")
        dynamics, basis = sample_dynamics(
            bundle, np.random.default_rng(0), fallback_span=0.1
        )
        self.assertIn("caller-declared", basis)
        self.assertIn("0.1", basis)
        kt = bundle["params"]["kt"]
        self.assertTrue(kt * 0.9 <= dynamics["kt"] <= kt * 1.1)
        # The negative-parameter case that caught the inverted-endpoints
        # bug, on a MOTOR parameter (load_friction_motor is negative in
        # no shipped fit, so use the min/max property directly).
        for param, value in dynamics.items():
            point = float(bundle["params"][param])
            low, high = sorted((point * 0.9, point * 1.1))
            self.assertTrue(low <= value <= high, param)

    def test_the_rigs_own_parameters_are_never_sampled(self) -> None:
        # q_offset is the BENCH's mount bias and command_delay the BUS's
        # latency (BAM's docs, 57 §7); max_velocity/error_gain_ratio are
        # firmware registers. Jittering them is physically meaningless —
        # a declared span must not reach them (review 2026-09-01).
        import numpy as np  # noqa: PLC0415

        from trainnr.robot.actuator_bundle import (  # noqa: PLC0415
            RIG_AND_FIRMWARE_PARAMS,
        )

        bundle = wrap("feetech_sts3215_7_4V", "m6")
        dynamics, _basis = sample_dynamics(
            bundle, np.random.default_rng(0), fallback_span=0.1
        )
        self.assertIn("q_offset", bundle["params"])  # it IS in the fit
        self.assertFalse(set(dynamics) & RIG_AND_FIRMWARE_PARAMS)

    def test_deleting_the_checks_section_is_refused(self) -> None:
        # checks left the content hash, which opened a tamper: DELETE the
        # section and the rail flags vanish with the stamp still valid
        # (second review, 2026-09-01). Required now.
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        del bundle["checks"]
        with self.assertRaises(ValueError) as ctx:
            verify(bundle)
        self.assertIn("checks", str(ctx.exception))

    def test_identified_intervals_over_rig_params_are_not_sampled(self) -> None:
        # The exclusion must hold on BOTH paths: the first cut filtered
        # only the fallback span, so a bundle DECLARING an interval for
        # q_offset still had it sampled (second review, 2026-09-01).
        import numpy as np  # noqa: PLC0415

        bundle = wrap("feetech_sts3215_7_4V", "m6")
        bundle["uncertainty"] = {
            "kt": {"low": 1.2, "high": 1.35},
            "q_offset": {"low": -0.08, "high": -0.05},
        }
        bundle["stamp"] = _stamp(bundle)
        dynamics, basis = sample_dynamics(bundle, np.random.default_rng(0))
        self.assertEqual(basis, "identified-interval")
        self.assertNotIn("q_offset", dynamics)
        self.assertIn("kt", dynamics)

    def test_a_zero_point_estimate_refuses_scaling(self) -> None:
        # No committed bundle carries a zero param, so nothing exercised
        # this raise until now (second review, 2026-09-01).
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        bundle["params"]["friction_stribeck"] = 0.0
        bundle["checks"] = run_checks(bundle["params"])
        bundle["stamp"] = _stamp(bundle)
        with self.assertRaises(ValueError) as ctx:
            as_scales({"friction_stribeck": 0.01}, bundle)
        self.assertIn("zero", str(ctx.exception))

    def test_scales_are_multipliers_of_the_point_estimates(self) -> None:
        bundle = self._with_uncertainty()
        scales = as_scales({"kt": bundle["params"]["kt"] * 1.05}, bundle)
        self.assertAlmostEqual(scales["kt"], 1.05)


class Files(unittest.TestCase):
    def test_write_read_round_trip_prefers_the_voltage_bearing_name(self) -> None:
        bundle = wrap("feetech_sts3215_7_4V", "m6")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bundle(bundle, Path(tmp))
            self.assertEqual(path.name, "feetech_sts3215_7_4V.m6.bundle.json")
            self.assertEqual(read_bundle(path), bundle)

    def test_wrap_all_covers_every_vendored_fit(self) -> None:
        expected = sum(len(list_models(slug)) for slug in list_actuators())
        with tempfile.TemporaryDirectory() as tmp:
            paths = wrap_all(Path(tmp))
            self.assertEqual(len(paths), expected)
            for path in paths:
                read_bundle(path)  # every emitted file verifies


if __name__ == "__main__":
    unittest.main()
