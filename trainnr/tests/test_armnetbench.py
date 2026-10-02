"""The committed ArmnetBench aggregate: pinned totals, honest semantics.

These tests pin the artifact's ground truth so a silent re-aggregation
cannot drift: 2,099 rollout episodes, 7 policies, 8 tasks, and the two
quirks worth remembering — pi0 lost one episode on eye_drops_to_shelf
(29 trials), and cable_clip defeated every policy (0/… across the
board), which makes it rank-uninformative and the stats layer must say
so rather than manufacture a correlation from it.
"""

import json
import tempfile
import unittest
from pathlib import Path

from trainnr.evaluate.armnetbench import (
    load_benchmark,
    real_outcomes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS = REPO_ROOT / "data" / "armnetbench-v01-so101-counts.json"

POLICY_COUNT = 7
TASK_COUNT = 8
TOTAL_ROLLOUTS = 2099


class CommittedArtifact(unittest.TestCase):
    def test_totals_are_pinned(self) -> None:
        benchmark = load_benchmark(COUNTS)
        self.assertEqual(len(benchmark.tasks), TASK_COUNT)
        for task, policies in benchmark.tasks.items():
            self.assertEqual(len(policies), POLICY_COUNT, task)
        self.assertEqual(benchmark.total_episodes, TOTAL_ROLLOUTS)
        # Provenance must travel with the numbers.
        self.assertIn(
            "armnet/armnetbench_v01_robometer", benchmark.provenance["dataset"]
        )
        self.assertEqual(benchmark.provenance["revision"], "v1.0")

    def test_known_quirks_survive_reaggregation(self) -> None:
        benchmark = load_benchmark(COUNTS)
        self.assertEqual(benchmark.tasks["eye_drops_to_shelf"]["pi0"].trials, 29)
        clip = benchmark.tasks["cable_clip"]
        self.assertTrue(all(c.successful == 0 for c in clip.values()))

    def test_strict_versus_lenient_outcomes(self) -> None:
        benchmark = load_benchmark(COUNTS)
        strict = real_outcomes(benchmark, "ring_insert")
        lenient = real_outcomes(
            benchmark, "ring_insert", count_suboptimal_as_success=True
        )
        self.assertEqual(len(strict), POLICY_COUNT)
        for policy in strict:
            s_succ, s_trials = strict[policy]
            l_succ, l_trials = lenient[policy]
            self.assertEqual(s_trials, l_trials)
            self.assertGreaterEqual(l_succ, s_succ)

    def test_cable_clip_is_rank_uninformative_and_stats_refuse_it(self) -> None:
        # The correlation machinery must refuse a task where every policy
        # scored zero — identical real scores carry no ranking signal.
        from trainnr.stats.ranking import spearman  # noqa: PLC0415

        benchmark = load_benchmark(COUNTS)
        outcomes = real_outcomes(benchmark, "cable_clip")
        rates = [succ / trials for succ, trials in outcomes.values()]
        with self.assertRaises(ValueError):
            spearman(list(range(len(rates))), rates)

    def test_unknown_task_refused(self) -> None:
        benchmark = load_benchmark(COUNTS)
        with self.assertRaises(ValueError):
            real_outcomes(benchmark, "block_stack_v2")


class ReconciliationWithThePaper(unittest.TestCase):
    def test_pooled_rates_agree_in_magnitude_with_the_paper_table(self) -> None:
        """R10: our aggregate versus ArmnetBench's own headline numbers.

        Exact agreement is NOT expected and would be suspicious: the
        paper's table spans all 12 tasks (4 bimanual, absent here), and
        its suboptimal handling is unstated. What must hold is magnitude
        agreement on the single-arm-dominant picture — if our pooled
        strict rates drifted far from the paper's, the aggregation
        (dedupe, demo exclusion, run pooling) would be suspect. The
        release's run names include `eval_debug`, so the release may
        pool runs the paper's core table does not; that residual risk is
        why the tolerance is 4 points and not 1.
        """
        benchmark = load_benchmark(COUNTS)
        pooled = {}
        for policies in benchmark.tasks.values():
            for policy, counts in policies.items():
                s_, t_ = pooled.get(policy, (0, 0))
                pooled[policy] = (s_ + counts.successful, t_ + counts.trials)
        paper_reported = {"pi0.5": 0.476, "act": 0.192}
        for policy, reported in paper_reported.items():
            successes, trials = pooled[policy]
            ours = successes / trials
            self.assertLess(
                abs(ours - reported),
                0.04,
                f"{policy}: ours {ours:.3f} vs paper {reported:.3f}",
            )


class Validation(unittest.TestCase):
    def test_malformed_top_level_refused(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(json.dumps({"counts": {}}))
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                load_benchmark(path)
        finally:
            path.unlink()

    def test_unknown_label_refused(self) -> None:
        raw = {
            "provenance": {"dataset": "x", "revision": "r"},
            "counts": {"t": {"p": {"successful": 1, "excellent": 2}}},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(json.dumps(raw))
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                load_benchmark(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
