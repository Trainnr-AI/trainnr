"""The per-trial record and its fold: stdlib only, runs in the fast suite."""

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.evaluate.records import (
    EpisodeRecord,
    SimScore,
    append_records,
    fold,
    from_eval_info,
    read_records,
)

PROTOCOL = {"trials": 4, "steps": 4000, "control_interval": 10, "home": "neutral"}
SOURCE = "aloha2-kitting@000000000000"


def record(policy: str, trial: int, success: bool) -> EpisodeRecord:
    return EpisodeRecord(
        source=SOURCE,
        policy=policy,
        trial=trial,
        success=success,
        steps=4000,
        instrument="mujoco-3.11.0",
        protocol=PROTOCOL,
    )


class Fold(unittest.TestCase):
    def test_counts_per_policy_in_first_seen_order(self) -> None:
        rows = [record("a", 0, True), record("a", 1, False), record("b", 0, False)]
        rows.append(record("b", 1, False))
        self.assertEqual(fold(rows), (SimScore("a", 1, 2), SimScore("b", 0, 2)))

    def test_unpaired_policies_are_refused_with_the_culprit_named(self) -> None:
        rows = [record("a", 0, True), record("a", 1, True), record("b", 0, True)]
        with self.assertRaises(ValueError) as caught:
            fold(rows)
        self.assertIn("'b'", str(caught.exception))

    def test_duplicate_trial_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fold([record("a", 0, True), record("a", 0, True)])
        with self.assertRaises(ValueError):
            fold([])

    def test_unstamped_source_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            EpisodeRecord(
                source="unstamped",
                policy="a",
                trial=0,
                success=True,
                steps=1,
                instrument="mujoco-3.11.0",
                protocol=PROTOCOL,
            )


class RoundTrip(unittest.TestCase):
    def test_jsonl_append_and_read(self) -> None:
        rows = [record("a", 0, True), record("a", 1, False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            append_records(path, rows[:1])
            append_records(path, rows[1:])  # appended, not overwritten
            lines = path.read_text().splitlines()
            back = read_records(path)
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["policy"], "a")
        self.assertEqual(back, tuple(rows))

    def test_lerobot_eval_info_folds_by_seed_order(self) -> None:
        # LeRobot 0.6.1: successes as a list in episode order, no seed.
        eval_info = {
            "per_task": [
                {
                    "task_group": "robotiq",
                    "task_id": 0,
                    "metrics": {"successes": [False, True, False, False]},
                }
            ],
            "overall": {"pc_success": 25.0, "n_episodes": 4},
        }
        rows = from_eval_info(
            eval_info,
            policy="act-kitting-20000",
            source=SOURCE,
            instrument="mujoco-3.12.0",
            protocol=PROTOCOL,
            start_seed=1000,
            trials=4,
            steps=4000,
        )
        self.assertEqual([r.seed for r in rows], [1000, 1001, 1002, 1003])
        self.assertEqual([r.trial for r in rows], [0, 1, 2, 3])
        self.assertEqual(fold(rows), (SimScore("act-kitting-20000", 1, 4),))
        with self.assertRaises(ValueError):
            from_eval_info(
                {"per_task": []},
                policy="p",
                source=SOURCE,
                instrument="x",
                protocol=PROTOCOL,
                start_seed=0,
                trials=4,
                steps=1,
            )


if __name__ == "__main__":
    unittest.main()
