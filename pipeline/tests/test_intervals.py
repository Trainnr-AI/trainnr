"""Intervals tested against closed-form truth and one field reference."""

import unittest

from rq_pipeline.stats.intervals import (
    clopper_pearson,
    regularized_incomplete_beta,
    wilson,
)


class RegularizedIncompleteBeta(unittest.TestCase):
    def test_uniform_case_is_identity(self) -> None:
        # Beta(1, 1) is the uniform distribution: I_x(1, 1) = x.
        for x in (0.0, 0.1, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(
                regularized_incomplete_beta(1.0, 1.0, x), x, places=9
            )

    def test_symmetry(self) -> None:
        # I_x(a, b) = 1 - I_{1-x}(b, a), exercised across both fraction
        # branches.
        for a, b, x in ((2.0, 5.0, 0.3), (7.5, 1.5, 0.8), (0.5, 0.5, 0.05)):
            self.assertAlmostEqual(
                regularized_incomplete_beta(a, b, x),
                1.0 - regularized_incomplete_beta(b, a, 1.0 - x),
                places=9,
            )

    def test_median_of_symmetric_beta(self) -> None:
        self.assertAlmostEqual(
            regularized_incomplete_beta(2.0, 2.0, 0.5), 0.5, places=9
        )


class ClopperPearson(unittest.TestCase):
    def test_field_reference_vector(self) -> None:
        # NVIDIA's evaluation-methodology worked example: 90% observed on
        # 70 rollouts (63/70) -> 95% CI of 80.5%-95.9%.
        lower, upper = clopper_pearson(63, 70)
        self.assertAlmostEqual(lower, 0.805, delta=0.005)
        self.assertAlmostEqual(upper, 0.959, delta=0.005)

    def test_zero_and_full_success_edges(self) -> None:
        lower, upper = clopper_pearson(0, 20)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)
        lower, upper = clopper_pearson(20, 20)
        self.assertLess(lower, 1.0)
        self.assertEqual(upper, 1.0)

    def test_exact_contains_wilson(self) -> None:
        # Clopper-Pearson is conservative: it should contain the Wilson
        # interval. A broken quantile breaks this relationship first.
        for successes, trials in ((7, 10), (45, 50), (1, 30), (63, 70)):
            cp_lower, cp_upper = clopper_pearson(successes, trials)
            w_lower, w_upper = wilson(successes, trials)
            self.assertLessEqual(cp_lower, w_lower + 1e-9)
            self.assertGreaterEqual(cp_upper, w_upper - 1e-9)

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            clopper_pearson(5, 0)
        with self.assertRaises(ValueError):
            clopper_pearson(11, 10)
        with self.assertRaises(ValueError):
            clopper_pearson(5, 10, confidence=1.0)


if __name__ == "__main__":
    unittest.main()


class TheChiSquaredAndGammaPrimitives(unittest.TestCase):
    """The honesty layer's pooling (Cochran's Q, Fisher's method) and the
    Fisher-z ranking bounds rest on these; reference values from tables."""

    def test_normal_quantile(self) -> None:
        from rq_pipeline.stats.intervals import normal_quantile  # noqa: PLC0415

        self.assertAlmostEqual(normal_quantile(0.975), 1.959964, places=5)
        self.assertAlmostEqual(normal_quantile(0.5), 0.0, places=9)

    def test_chi_squared_survival_at_the_five_percent_points(self) -> None:
        from rq_pipeline.stats.intervals import chi_squared_survival  # noqa: PLC0415

        self.assertAlmostEqual(chi_squared_survival(3.841, 1), 0.05, places=3)
        self.assertAlmostEqual(chi_squared_survival(5.991, 2), 0.05, places=3)

    def test_regularized_lower_gamma_on_both_sides_of_its_branch(self) -> None:
        from rq_pipeline.stats.intervals import regularized_lower_gamma  # noqa: PLC0415

        self.assertAlmostEqual(
            regularized_lower_gamma(1.0, 1.0), 1 - 1 / 2.718281828, places=6
        )
        self.assertAlmostEqual(regularized_lower_gamma(10.0, 20.0), 0.99500, places=3)
