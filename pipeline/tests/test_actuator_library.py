"""The vendored actuator library: every model loads, provenance is
enforced, and BAM's real STS3215 numbers land in the right fields —
run against the actual files under robots/actuators/, not fixtures.
"""

import unittest

from rq_pipeline.robot.actuator_library import (
    list_actuators,
    list_models,
    load_actuator,
)


class VendoredLibrary(unittest.TestCase):
    def test_all_eight_bam_actuators_are_present(self) -> None:
        # Pinned by name: a re-vendor that drops or renames one should
        # fail loudly here, not silently shrink the library.
        self.assertEqual(
            set(list_actuators()),
            {
                "erob80_50",
                "erob80_100",
                "feetech_sts3215_7_4V",
                "mx64",
                "mx106",
                "waveshare_st3025",
                "xl320",
                "xl330",
            },
        )

    def test_every_actuator_has_all_six_tiers(self) -> None:
        for slug in list_actuators():
            with self.subTest(slug=slug):
                self.assertEqual(
                    list_models(slug), ("m1", "m2", "m3", "m4", "m5", "m6")
                )

    def test_every_tier_of_every_actuator_loads(self) -> None:
        for slug in list_actuators():
            for tier in list_models(slug):
                with self.subTest(slug=slug, tier=tier):
                    model = load_actuator(slug, tier)
                    self.assertEqual(model.slug, slug)
                    self.assertEqual(model.tier, tier)
                    self.assertIsInstance(model.friction.friction_viscous, float)
                    self.assertEqual(model.provenance.source, "bam")

    def test_m1_has_no_stribeck_m6_has_everything(self) -> None:
        m1 = load_actuator("feetech_sts3215_7_4V", "m1")
        self.assertIsNone(m1.friction.friction_stribeck)
        self.assertIsNone(m1.friction.load_friction_motor_quad)

        m6 = load_actuator("feetech_sts3215_7_4V", "m6")
        self.assertIsNotNone(m6.friction.friction_stribeck)
        self.assertIsNotNone(m6.friction.load_friction_motor)
        self.assertIsNotNone(m6.friction.load_friction_motor_quad)

    def test_sts3215_m1_matches_the_published_torque_constant(self) -> None:
        # docs/e2e-research/53's cited value — pin-both-ends against a
        # published number, not just against our own loader's plumbing.
        model = load_actuator("feetech_sts3215_7_4V", "m1")
        self.assertAlmostEqual(model.servo.kt, 1.1776311631627974)

    def test_undirected_and_directional_load_terms_are_mutually_exclusive(self) -> None:
        m3 = load_actuator("feetech_sts3215_7_4V", "m3")
        self.assertIsNotNone(m3.friction.load_friction_base)
        self.assertIsNone(m3.friction.load_friction_motor)

        m5 = load_actuator("feetech_sts3215_7_4V", "m5")
        self.assertIsNone(m5.friction.load_friction_base)
        self.assertIsNotNone(m5.friction.load_friction_motor)


class Refusals(unittest.TestCase):
    def test_unknown_actuator_names_what_exists(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            load_actuator("no-such-servo")
        self.assertIn("feetech_sts3215_7_4V", str(ctx.exception))

    def test_unknown_tier_names_what_exists(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            load_actuator("feetech_sts3215_7_4V", "m9")
        self.assertIn("m1", str(ctx.exception))

    def test_a_directory_with_no_provenance_is_refused(self) -> None:
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from rq_pipeline.robot import actuator_library  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mystery_servo").mkdir()
            (root / "mystery_servo" / "m1.json").write_text(
                '{"friction_viscous": 0.1, "friction_base": 0.1}'
            )
            with mock.patch.object(actuator_library, "ACTUATORS_ROOT", root):
                self.assertEqual(actuator_library.list_actuators(), ())
                with self.assertRaises(KeyError):
                    actuator_library.load_actuator("mystery_servo")

    def test_provenance_with_unknown_source_is_refused(self) -> None:
        import json  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        from rq_pipeline.robot import actuator_library  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            servo = root / "guessed_servo"
            servo.mkdir()
            (servo / "m1.json").write_text(
                '{"friction_viscous": 0.1, "friction_base": 0.1}'
            )
            (servo / "PROVENANCE.json").write_text(
                json.dumps(
                    {"source": "vibes", "citation": "trust me", "license": "MIT"}
                )
            )
            with (
                mock.patch.object(actuator_library, "ACTUATORS_ROOT", root),
                self.assertRaises(ValueError),
            ):
                actuator_library.load_actuator("guessed_servo", "m1")


if __name__ == "__main__":
    unittest.main()
