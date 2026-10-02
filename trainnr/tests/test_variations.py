"""Variations drawn by trial, and the exact main-effects statistic — stdlib."""

import unittest

from trainnr.evaluate.variations import (
    Choice,
    Uniform,
    Variation,
    VariationKeys,
    describe,
    draw,
    draw_all,
    parse_variation,
    sensitivity_table,
)
from trainnr.stats.effects import (
    SplitLabel,
    Verdict,
    fisher_exact,
    format_table,
    main_effect,
    split_continuous,
)

PROTOCOL_HASH = "abc123"
LOW, HIGH = 0.8, 1.2
DAMPING = Variation(
    VariationKeys.JOINTS, VariationKeys.DAMPING_SCALE, Uniform((LOW,), (HIGH,))
)
CAMERA = Variation("top", "offset_m", Uniform((-0.03,) * 3, (0.03,) * 3))
LIGHT = Variation("light", "level", Choice(("dim", "nominal", "bright")))
OFF = Variation("cube", "mass_kg", Uniform((0.05,), (0.2,)), enabled=False)
MANY = 200
ALPHA, DELTA = 0.05, 0.2


class DrawByTrial(unittest.TestCase):
    def test_same_trial_same_value_across_policies_and_processes(self) -> None:
        self.assertEqual(
            draw(DAMPING, 3, PROTOCOL_HASH), draw(DAMPING, 3, PROTOCOL_HASH)
        )
        self.assertNotEqual(
            draw(DAMPING, 3, PROTOCOL_HASH), draw(DAMPING, 4, PROTOCOL_HASH)
        )
        # A different protocol is a different sweep.
        self.assertNotEqual(draw(DAMPING, 3, PROTOCOL_HASH), draw(DAMPING, 3, "other"))

    def test_uniform_stays_in_its_box_and_fills_it(self) -> None:
        values = [draw(DAMPING, t, PROTOCOL_HASH) for t in range(MANY)]
        self.assertTrue(all(LOW <= v <= HIGH for v in values))
        self.assertLess(min(values), 0.85)
        self.assertGreater(max(values), 1.15)
        vector = draw(CAMERA, 0, PROTOCOL_HASH)
        self.assertEqual(len(vector), 3)
        self.assertNotEqual(vector[0], vector[1])  # per-component draws

    def test_choice_is_balanced_round_robin(self) -> None:
        labels = [draw(LIGHT, t, PROTOCOL_HASH) for t in range(6)]
        self.assertEqual(labels, ["dim", "nominal", "bright"] * 2)

    def test_draw_all_skips_disabled_and_refuses_duplicate_keys(self) -> None:
        values = draw_all((DAMPING, LIGHT, OFF), 1, PROTOCOL_HASH)
        self.assertEqual(set(values), {"joints.damping_scale", "light.level"})
        with self.assertRaises(ValueError):
            draw_all((DAMPING, DAMPING), 0, PROTOCOL_HASH)
        self.assertIn("cube.mass_kg [off]", describe((OFF,))[0])

    def test_bad_samplers_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            Uniform((1.0,), (0.5,))
        with self.assertRaises(ValueError):
            Choice(("a", "a"))


class ParseAndTable(unittest.TestCase):
    def test_cli_forms(self) -> None:
        damping = parse_variation("joints.damping_scale=0.8:1.2")
        self.assertEqual(damping.key, "joints.damping_scale")
        self.assertEqual(damping.sampler, Uniform((0.8,), (1.2,)))
        offset = parse_variation("top.offset_m=-0.03,-0.03,-0.03:0.03,0.03,0.03")
        self.assertEqual(len(offset.sampler.low), 3)
        light = parse_variation("light.level=dim|bright")
        self.assertEqual(light.sampler, Choice(("dim", "bright")))
        for bad in ("nodot=1:2", "a.b=1", "a.b", "=1:2"):
            with self.assertRaises(ValueError):
                parse_variation(bad)

    def test_sensitivity_table_reads_recorded_draws(self) -> None:
        from trainnr.evaluate.records import EpisodeRecord  # noqa: PLC0415

        protocol = {"trials": 16, "steps": 10, "control_interval": 1, "home": None}

        def row(trial: int, damping: float, label: str, success: bool) -> EpisodeRecord:
            return EpisodeRecord(
                source="t@000000000000",
                policy="p",
                trial=trial,
                success=success,
                steps=10,
                instrument="mujoco-x",
                protocol=protocol,
                variations={"joints.damping_scale": damping, "light.level": label},
            )

        # Damping decides everything; light decides nothing.
        rows = [
            row(t, 0.85 if t % 2 == 0 else 1.15, ["dim", "bright"][t % 2], t % 2 == 0)
            for t in range(16)
        ]
        two_lights = Variation("light", "level", Choice(("dim", "bright")))
        table = sensitivity_table(rows, (DAMPING, two_lights), alpha=ALPHA, delta=DELTA)
        by_key = {e.key: e for e in table if e.side_b in (SplitLabel.HIGH, "dim")}
        self.assertEqual(by_key["joints.damping_scale"].verdict, Verdict.SENSITIVE)
        # Light and damping are confounded in this synthetic sweep (dim
        # rides with low damping), so light reads SENSITIVE too — the
        # table reports the data, the balanced design is what prevents
        # it in a real sweep (round-robin labels x hashed draws).
        self.assertEqual(by_key["light.level"].side_b, "dim")
        with self.assertRaises(ValueError):
            sensitivity_table(rows, (CAMERA,), alpha=ALPHA, delta=DELTA)  # no draws


class FisherExact(unittest.TestCase):
    def test_reference_tables(self) -> None:
        # Fisher's tea tasting: [[3, 1], [1, 3]] -> p = 0.4857 (two-sided).
        self.assertAlmostEqual(fisher_exact(3, 1, 1, 3), 0.48571, places=4)
        # Perfect separation of 8 vs 8: p = 2 / C(16, 8) = 0.000155.
        self.assertAlmostEqual(fisher_exact(8, 0, 0, 8), 2 / 12870, places=6)
        self.assertEqual(fisher_exact(0, 0, 0, 0), 1.0)
        self.assertEqual(fisher_exact(2, 2, 2, 2), 1.0)


class MainEffects(unittest.TestCase):
    def test_verdicts(self) -> None:
        sensitive = main_effect(
            "top.offset_m",
            "<mid",
            ">=mid",
            [True] * 8,
            [False] * 8,
            alpha=ALPHA,
            delta=DELTA,
        )
        self.assertEqual(sensitive.verdict, Verdict.SENSITIVE)
        self.assertAlmostEqual(sensitive.difference, -1.0)
        unresolved = main_effect(
            "joints.damping_scale",
            "<mid",
            ">=mid",
            [True, False],
            [False, True],
            alpha=ALPHA,
            delta=DELTA,
        )
        self.assertEqual(unresolved.verdict, Verdict.UNRESOLVED)
        insensitive = main_effect(
            "light.level",
            "dim",
            "bright",
            [True] * 200,
            [True] * 199 + [False],
            alpha=ALPHA,
            delta=DELTA,
        )
        self.assertEqual(insensitive.verdict, Verdict.INSENSITIVE)
        self.assertIn("SENSITIVE", format_table([sensitive]))
        with self.assertRaises(ValueError):
            main_effect("k", "a", "b", [], [True], alpha=ALPHA, delta=DELTA)

    def test_split_at_midpoint(self) -> None:
        below, above = split_continuous(
            [0.8, 1.0, 1.2, 0.9], [True, False, False, True], 1.0
        )
        self.assertEqual((below, above), ([True, True], [False, False]))


if __name__ == "__main__":
    unittest.main()
