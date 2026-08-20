"""Cross-task pooling: tighter when tasks agree, loud when they don't."""

import unittest
from math import factorial

from rq_pipeline.stats.pooling import pool_rank_correlations
from rq_pipeline.stats.ranking import fisher_rank_ci

# Seven policies, perfectly agreeing ranks — one task's worth.
SEVEN = list(range(7))
AGREE = (SEVEN, [float(v) for v in SEVEN])
DISAGREE = (SEVEN, [float(v) for v in reversed(SEVEN)])
# One adjacent swap: high but imperfect agreement.
SWAPPED = (SEVEN, [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0])


class PoolingTightens(unittest.TestCase):
    def test_seven_agreeing_tasks_beat_any_single_task(self) -> None:
        single_lower, _, _ = fisher_rank_ci(*AGREE)
        pooled = pool_rank_correlations({f"task{i}": AGREE for i in range(7)})
        self.assertGreater(pooled.lower, single_lower)
        # The Paper 2 design point: seven n=7 tasks in near-perfect
        # agreement support a pooled lower bound a single task cannot.
        self.assertGreater(pooled.lower, 0.9)
        self.assertEqual(pooled.task_count, 7)
        # Homogeneous by construction — the pooled number may lead.
        self.assertGreater(pooled.heterogeneity_p, 0.99)
        self.assertNotIn("HETEROGENEOUS", pooled.summary())

    def test_combined_exact_p_multiplies_evidence(self) -> None:
        pooled = pool_rank_correlations({f"task{i}": AGREE for i in range(3)})
        per_task = 1.0 / factorial(7)
        self.assertIsNotNone(pooled.combined_exact_p)
        # Three independent 1/5040 results are far stronger than one.
        self.assertLess(pooled.combined_exact_p, per_task)
        for entry in pooled.tasks:
            self.assertAlmostEqual(entry.exact_p, per_task)

    def test_nine_policies_drop_the_combined_exact_p_only(self) -> None:
        nine = list(range(9))
        pooled = pool_rank_correlations(
            {"small": AGREE, "big": (nine, [float(v) for v in nine])}
        )
        self.assertIsNone(pooled.combined_exact_p)
        self.assertGreater(pooled.rho, 0.9)


class HeterogeneityPolice(unittest.TestCase):
    def test_disagreeing_tasks_flag_the_pooled_number(self) -> None:
        pooled = pool_rank_correlations(
            {
                "agree1": AGREE,
                "agree2": AGREE,
                "reversed1": DISAGREE,
                "reversed2": DISAGREE,
            }
        )
        # The average is meaningless here, and the summary must lead
        # with that instead of the ~0 pooled rho.
        self.assertLess(pooled.heterogeneity_p, 0.001)
        self.assertIn("HETEROGENEOUS", pooled.summary())
        self.assertAlmostEqual(pooled.rho, 0.0, delta=0.05)

    def test_mild_variation_does_not_cry_wolf(self) -> None:
        pooled = pool_rank_correlations(
            {"perfect": AGREE, "swapped": SWAPPED, "perfect2": AGREE}
        )
        self.assertGreater(pooled.heterogeneity_p, 0.05)
        self.assertNotIn("HETEROGENEOUS", pooled.summary())


class Refusals(unittest.TestCase):
    def test_one_task_refused(self) -> None:
        with self.assertRaises(ValueError):
            pool_rank_correlations({"only": AGREE})

    def test_small_task_refused(self) -> None:
        three = ([1, 2, 3], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            pool_rank_correlations({"ok": AGREE, "tiny": three})

    def test_rank_dead_task_raises_rather_than_shrinks(self) -> None:
        dead = (SEVEN, [0.0] * 7)
        with self.assertRaises(ValueError):
            pool_rank_correlations({"ok": AGREE, "dead": dead})


if __name__ == "__main__":
    unittest.main()
