"""The trainer's own event file becomes the training record: every
series it logs, named so the cards and views keep reading the same
five totals, sampled like the console record."""

from __future__ import annotations

import unittest

from trainnr.envs.rsl_rl_log import TrainingRecord
from trainnr.envs.tfevents import (
    column_name,
    curve_groups,
    events_file,
    record_from_events,
    record_from_scalars,
)
from trainnr.project.locate import projects_dir

# The certified Go2 run: on the box only (docs/77 §5).
GO2_RUN = projects_dir() / "go2-walk" / "runs" / "go2-c1"


class TheColumns(unittest.TestCase):
    def test_the_console_names_survive_and_terms_take_their_group(self) -> None:
        self.assertEqual(column_name("Train/mean_reward"), "reward")
        self.assertEqual(column_name("Loss/value"), "value_loss")
        self.assertEqual(column_name("Perf/total_fps"), "steps_per_second")
        self.assertEqual(column_name("Episode_Reward/pose"), "reward/pose")
        self.assertEqual(column_name("Loss/surrogate"), "loss/surrogate")
        self.assertEqual(
            column_name("Curriculum/command_vel/lin_vel_x_max"),
            "curriculum/command_vel/lin_vel_x_max",
        )
        self.assertIsNone(column_name("Train/mean_reward/time"))

    def test_groups_for_the_viewer(self) -> None:
        groups = curve_groups(
            [
                "iteration",
                "reward",
                "entropy",
                "reward/pose",
                "reward/upright",
                "loss/surrogate",
            ]
        )
        self.assertEqual(groups["reward"], ["reward"])
        self.assertEqual(groups["reward terms"], ["reward/pose", "reward/upright"])
        self.assertEqual(groups["loss terms"], ["loss/surrogate"])


class TheRecord(unittest.TestCase):
    def test_from_scalars_keeps_facts_and_samples(self) -> None:
        scalars = {
            "Train/mean_reward": [(i, float(i)) for i in range(1000)],
            "Episode_Reward/pose": [(i, 0.5) for i in range(1000)],
            "Train/mean_reward/time": [(i, 0.0) for i in range(1000)],
        }
        facts = TrainingRecord(iterations=8000, envs=4096, device="cuda:0")
        record = record_from_scalars(scalars, facts=facts, max_points=100)
        assert record is not None
        self.assertEqual(record.columns, ["iteration", "reward", "reward/pose"])
        self.assertEqual(record.iterations, 8000)
        self.assertEqual(record.envs, 4096)
        self.assertEqual(record.iterations_logged, 1000)
        self.assertEqual(record.best_reward, 999.0)
        self.assertEqual(record.final["reward"], 999.0)
        self.assertLessEqual(len(record.curve), 101)
        self.assertEqual(record.curve[-1][0], 999.0)

    def test_without_the_total_there_is_no_record(self) -> None:
        self.assertIsNone(record_from_scalars({"Loss/value": [(0, 1.0)]}))

    @unittest.skipUnless(GO2_RUN.is_dir(), "the Go2 run lives on the box")
    def test_the_go2_run_s_file_reads(self) -> None:
        path = events_file(GO2_RUN)
        assert path is not None
        record = record_from_events(path)
        assert record is not None
        self.assertIn("reward/track_linear_velocity", record.columns)
        self.assertIn("curriculum/command_vel/lin_vel_x_max", record.columns)
        self.assertEqual(record.iterations_logged, 8000)


class ThePlannedCount(unittest.TestCase):
    def test_the_file_never_invents_the_planned_iterations(self) -> None:
        scalars = {"Train/mean_reward": [(i, 1.0) for i in range(10)]}
        record = record_from_scalars(scalars)
        assert record is not None
        self.assertIsNone(record.iterations)  # the console said nothing
        self.assertEqual(record.iterations_logged, 10)
