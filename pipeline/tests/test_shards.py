"""Sharding a press (docs/66 D4): the plan is disjoint and seeded, a
shard's record round-trips, the merge states an exact keep rate, and
the datasheet says so instead of refusing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rq_pipeline.collect.datasheet import render, summarize
from rq_pipeline.collect.press import DemoBatch
from rq_pipeline.collect.shards import (
    MergedRate,
    ShardFlags,
    ShardRecord,
    ShardSpec,
    merge_records,
    plan_shards,
    read_shard_records,
    run_shards,
    shard_argv,
)


class ThePlan(unittest.TestCase):
    def test_ranges_are_disjoint_contiguous_and_balanced(self) -> None:
        plan = plan_shards(10, 3, seed=100, first_episode=5)
        self.assertEqual(
            [(s.first_episode, s.episodes, s.seed) for s in plan],
            [(5, 4, 100), (9, 3, 101), (12, 3, 102)],
        )
        self.assertEqual(plan[-1].last_episode, 14)

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            plan_shards(2, 3, seed=1)
        with self.assertRaises(ValueError):
            plan_shards(2, 0, seed=1)


class TheMerge(unittest.TestCase):
    def test_records_sum_to_an_exact_rate(self) -> None:
        a = ShardRecord(0, 1, 0, 3, 3, 4, "e@1", "t@1")
        b = ShardRecord(1, 2, 3, 3, 3, 8, "e@1", "t@1")
        self.assertEqual(merge_records([a, b]), MergedRate(6, 12, 2))
        self.assertAlmostEqual(merge_records([a, b]).rate, 0.5)

    def test_a_record_is_written_from_a_batch_and_read_back_in_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ShardSpec(1, 4, 4, 11)
            batch = DemoBatch(root, 4, 4, 9, 4, "e@1", "t@1")
            ShardRecord.of(spec, batch).write_to(root)
            ShardRecord(0, 10, 0, 4, 4, 5, "e@1", "t@1").write_to(root)
            records = read_shard_records(root)
            self.assertEqual([r.index for r in records], [0, 1])
            self.assertEqual(records[1].attempts, 9)


class TheDatasheet(unittest.TestCase):
    def test_shard_records_turn_the_refusal_into_an_exact_rate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, seed in ((0, 7), (1, 8)):
                episode = root / f"episode_{index:04d}"
                episode.mkdir()
                (episode / "manifest.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "attempt": 1,
                            "task": "t@1",
                            "expert": "e@1",
                            "instrument": "mujoco-3.11.0+test",
                            "dynamics": {"friction": 1.0},
                            "dynamics_basis": "b",
                            "draws": {},
                            "retries": [],
                            "control_hz": 50,
                            "frame_every_control_ticks": 1,
                        }
                    )
                )
            ShardRecord(0, 7, 0, 1, 1, 3, "e@1", "t@1").write_to(root)
            ShardRecord(1, 8, 1, 1, 1, 1, "e@1", "t@1").write_to(root)
            summary = summarize(root)
            self.assertEqual(summary.shards, 2)
            self.assertEqual(summary.merged, MergedRate(2, 4, 2))
            self.assertEqual(summary.warnings, ())
            self.assertIn(
                "keep rate **50%** exactly: 2 kept of 4 attempts", render(summary)
            )


class TheDatasheetUnderARace(unittest.TestCase):
    def test_an_episode_still_being_written_is_not_counted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / "episode_0000"
            done.mkdir()
            (done / "manifest.json").write_text(
                json.dumps(
                    {
                        "seed": 7,
                        "attempt": 2,
                        "task": "t@1",
                        "expert": "e@1",
                        "instrument": "i",
                        "dynamics": {"friction": 1.0},
                        "dynamics_basis": "b",
                        "draws": {},
                        "retries": [],
                        "control_hz": 50,
                        "frame_every_control_ticks": 1,
                    }
                )
            )
            (root / "episode_0001").mkdir()  # the other shard, mid-write
            self.assertEqual(summarize(root).episodes, 1)


class TheRunner(unittest.TestCase):
    def test_runs_every_shard_and_keeps_plan_order(self) -> None:
        plan = plan_shards(6, 3, seed=1)
        self.assertEqual(
            run_shards(plan, lambda s: s.index * 10, parallel=2), [0, 10, 20]
        )

    def test_shard_argv_re_aims_the_tool_at_one_shard(self) -> None:
        argv = [
            "planner-demos.py",
            "lift",
            "out",
            "--shards",
            "3",
            "--parallel=2",
            "--episodes",
            "9",
        ]
        self.assertEqual(
            shard_argv(argv, ShardSpec(2, 6, 3, 19), ShardFlags()),
            [
                "planner-demos.py",
                "lift",
                "out",
                "--episodes",
                "9",
                "--first-episode",
                "6",
                "--episodes",
                "3",
                "--seed",
                "19",
                "--shard-index",
                "2",
            ],
        )


if __name__ == "__main__":
    unittest.main()
