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

from rq_pipeline.evaluate.armnetbench import (
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
        from rq_pipeline.stats.ranking import spearman  # noqa: PLC0415

        benchmark = load_benchmark(COUNTS)
        outcomes = real_outcomes(benchmark, "cable_clip")
        rates = [succ / trials for succ, trials in outcomes.values()]
        with self.assertRaises(ValueError):
            spearman(list(range(len(rates))), rates)

    def test_unknown_task_refused(self) -> None:
        benchmark = load_benchmark(COUNTS)
        with self.assertRaises(ValueError):
            real_outcomes(benchmark, "block_stack_v2")


class Validation(unittest.TestCase):
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
