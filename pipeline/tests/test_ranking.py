"""Gate A statistics, including the lesson that created this module."""

import unittest
from math import factorial

from rq_pipeline.stats.ranking import (
    bootstrap_rank_ci,
    exact_spearman_p,
    fisher_rank_ci,
    spearman,
    top_pick_probability,
)

# Five policies, one adjacent swap in the real ranking: Spearman r = 0.9
# exactly (sum of squared rank differences = 2).
SIM_FIVE = [0.1, 0.2, 0.3, 0.4, 0.5]
REAL_FIVE_ONE_SWAP = [0.1, 0.2, 0.3, 0.5, 0.4]


class ExactPermutation(unittest.TestCase):
    def test_perfect_agreement_earns_exactly_one_over_n_factorial(self) -> None:
        # The honest version of what the bootstrap once mis-reported as
        # certainty: at n=5 a perfect ranking is evidence at p = 1/120,
        # no more — exactly one of 120 pairings ties the observed rho.
        p_value, count = exact_spearman_p(SIM_FIVE, [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(count, 5)
        self.assertAlmostEqual(p_value, 1.0 / factorial(5))

    def test_one_swap_at_five_policies_hand_enumerable(self) -> None:
        # rho = 0.9; permutations reaching rho >= 0.9 are the identity
        # (1.0) and the four single adjacent swaps (0.9 each): 5/120.
        p_value, _ = exact_spearman_p(SIM_FIVE, REAL_FIVE_ONE_SWAP)
        self.assertAlmostEqual(p_value, 5.0 / factorial(5))

    def test_reversal_is_no_evidence_of_positive_association(self) -> None:
        p_value, _ = exact_spearman_p([1, 2, 3, 4], [40, 30, 20, 10])
        self.assertEqual(p_value, 1.0)

    def test_ties_are_conditioned_on_not_refused(self) -> None:
        p_value, _ = exact_spearman_p([1, 2, 3, 4, 5], [1, 1, 2, 3, 4])
        self.assertGreater(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_beyond_enumeration_limit_refused(self) -> None:
        nine = list(range(9))
        with self.assertRaises(ValueError):
            exact_spearman_p(nine, nine)


class Spearman(unittest.TestCase):
    def test_perfect_agreement(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_perfect_reversal(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_ties_get_average_ranks(self) -> None:
        # Two tied sim scores against distinct real scores: |r| < 1 by
        # construction, and no exception.
        result = spearman([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        self.assertLess(result, 1.0)
        self.assertGreater(result, 0.5)

    def test_identical_scores_refused(self) -> None:
        with self.assertRaises(ValueError):
            spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


class FisherInterval(unittest.TestCase):
    def test_five_policies_is_wide_by_construction(self) -> None:
        # The defect that created this module (C1 in the defect register):
        # at n = 5, observed r = 0.9 has a 95% lower bound near zero. A
        # customer-facing gate must see this interval, not the 0.9.
        lower, upper, n = fisher_rank_ci(SIM_FIVE, REAL_FIVE_ONE_SWAP)
        self.assertEqual(n, 5)
        self.assertLess(lower, 0.3)
        self.assertGreater(upper, 0.9)

    def test_more_policies_tighten_the_bound(self) -> None:
        sim_fifteen = [i / 10 for i in range(15)]
        real_fifteen = list(sim_fifteen)
        real_fifteen[3], real_fifteen[4] = real_fifteen[4], real_fifteen[3]
        lower_small, _, _ = fisher_rank_ci(SIM_FIVE, REAL_FIVE_ONE_SWAP)
        lower_large, _, n = fisher_rank_ci(sim_fifteen, real_fifteen)
        self.assertEqual(n, 15)
        self.assertGreater(lower_large, lower_small)

    def test_too_few_policies_refused(self) -> None:
        with self.assertRaises(ValueError):
            fisher_rank_ci([1, 2, 3], [1, 2, 3])

    def test_perfect_agreement_capped_by_discreteness(self) -> None:
        # Second bug this suite caught: a fixed atanh clamp let the CLAMP
        # CONSTANT set the bound at perfect observed agreement (lower
        # bound 0.998 at n = 5). The cap is now the resolution of ranking
        # itself — one adjacent swap, r = 0.9 at n = 5 — so five perfectly
        # agreeing policies still report an honest, wide interval.
        perfect = [0.1, 0.2, 0.3, 0.4, 0.5]
        lower, _, n = fisher_rank_ci(perfect, list(perfect))
        self.assertEqual(n, 5)
        self.assertLess(lower, 0.3)


class BootstrapInterval(unittest.TestCase):
    def test_percentile_bootstrap_is_blind_past_observed_data(self) -> None:
        # THE reason the gate uses Fisher, demonstrated: a perfectly
        # monotone pair resamples to r = 1 in every draw, so the bootstrap
        # asserts certainty at n = 5 where the Fisher interval honestly
        # spans most of [0, 1]. Caught by this suite's first ever run.
        sim = [0.1, 0.3, 0.5, 0.7, 0.9]
        real = [0.15, 0.28, 0.55, 0.66, 0.88]  # observed r = 1.0
        lower, upper, _ = bootstrap_rank_ci(sim, real, seed=7)
        self.assertEqual((lower, upper), (1.0, 1.0))

    def test_visible_disagreement_widens_the_interval(self) -> None:
        lower, upper, n = bootstrap_rank_ci(SIM_FIVE, REAL_FIVE_ONE_SWAP, seed=7)
        self.assertEqual(n, 5)
        self.assertLessEqual(lower, 0.9)
        self.assertLessEqual(upper, 1.0)

    def test_too_few_policies_refused(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_rank_ci([1, 2, 3], [1, 2, 3])


class TopPickProbability(unittest.TestCase):
    def test_clear_winner_scores_near_one(self) -> None:
        probability = top_pick_probability(
            sim_scores=[0.9, 0.3, 0.2],
            real_successes=[45, 15, 10],
            real_trials=[50, 50, 50],
            seed=3,
        )
        self.assertGreater(probability, 0.95)

    def test_indistinguishable_policies_score_near_chance(self) -> None:
        probability = top_pick_probability(
            sim_scores=[0.51, 0.50, 0.49, 0.48],
            real_successes=[25, 25, 25, 25],
            real_trials=[50, 50, 50, 50],
            seed=3,
        )
        self.assertAlmostEqual(probability, 0.25, delta=0.05)

    def test_sim_picking_the_real_loser_scores_near_zero(self) -> None:
        probability = top_pick_probability(
            sim_scores=[0.9, 0.1],
            real_successes=[5, 45],
            real_trials=[50, 50],
            seed=3,
        )
        self.assertLess(probability, 0.05)

    def test_rejects_bad_counts(self) -> None:
        with self.assertRaises(ValueError):
            top_pick_probability([0.5, 0.6], [10, 60], [50, 50])


if __name__ == "__main__":
    unittest.main()


class TheDiscretenessCap(unittest.TestCase):
    def test_five_policies_resolve_one_swap_at_point_nine(self) -> None:
        from rq_pipeline.stats.ranking import max_resolvable_rho  # noqa: PLC0415

        self.assertAlmostEqual(max_resolvable_rho(5), 0.9)
        self.assertLess(max_resolvable_rho(3), max_resolvable_rho(10))
