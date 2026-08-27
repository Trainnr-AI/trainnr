"""LeRobot's training log, read back: big numbers, the metrics line, the
in-loop summary, the run manifest, and a tail that returns only what
was appended."""

import tempfile
import unittest
from pathlib import Path

from rq_pipeline.envs.lerobot_train_log import (
    FileFollower,
    RunManifest,
    is_stage_line,
    parse_big_number,
    parse_eval_line,
    parse_train_line,
    watch_dir_for,
)

# Verbatim shapes from lerobot 0.6.1 on 2026-08-27 (the B200 run).
METRICS_LINE = (
    "INFO 2026-08-27 17:20:03 ot_train.py:597 step:1.0K smpl:64K ep:45 epch:1.58 "
    "loss:0.281 grdn:12.345 lr:3.0e-05 updt_s:0.081 data_s:0.001 smp/s:24 mem_gb:2.12"
)
EVAL_LINE = (
    "INFO 2026-08-27 17:43:27 ot_train.py:738 Suite overall aggregated: "
    "{'avg_sum_reward': 0.5, 'avg_max_reward': 0.5, 'pc_success': 50.0, "
    "'n_episodes': 2, 'eval_s': 85.53975558280945, 'eval_ep_s': 42.76987}"
)


class BigNumbers(unittest.TestCase):
    def test_lerobot_suffixes_read_back(self) -> None:
        self.assertEqual(parse_big_number("450"), 450)
        self.assertEqual(parse_big_number("1.0K"), 1000)
        self.assertEqual(parse_big_number("10K"), 10000)
        self.assertEqual(parse_big_number("1.2M"), 1_200_000)
        with self.assertRaises(ValueError):
            parse_big_number("step")


class TheMetricsLine(unittest.TestCase):
    def test_counts_and_every_metric(self) -> None:
        line = parse_train_line(METRICS_LINE)
        self.assertIsNotNone(line)
        self.assertEqual(line.step, 1000)  # not 1: the K is read
        self.assertEqual(line.samples, 64000)
        self.assertEqual(line.episodes, 45)
        self.assertAlmostEqual(line.epochs, 1.58)
        self.assertAlmostEqual(line.metrics["loss"], 0.281)
        self.assertAlmostEqual(line.metrics["grdn"], 12.345)
        self.assertAlmostEqual(line.metrics["lr"], 3e-5)
        self.assertAlmostEqual(line.metrics["updt_s"], 0.081)
        self.assertNotIn("step", line.metrics)
        # Not metrics: the logger's `ot_train.py:597` and the clock's `17:20:03`
        # (the first dashboard logged `train/py` = 597, 2026-08-28).
        self.assertEqual(
            sorted(line.metrics),
            ["data_s", "grdn", "loss", "lr", "mem_gb", "smp/s", "updt_s"],
        )
        self.assertEqual(line.metrics["smp/s"], 24.0)

    def test_other_lines_are_not_metrics(self) -> None:
        self.assertIsNone(parse_train_line("INFO Start offline training"))
        self.assertIsNone(parse_train_line("Training:  43%| 4267/10000 [09:22<10:34]"))
        # tqdm's progress prefix glued in front of the logger's line
        glued = "Training:  12%|█▎ | 50/400 [00:16<00:50,  6.96step/s]" + METRICS_LINE
        self.assertEqual(parse_train_line(glued).step, 1000)
        self.assertIsNone(parse_eval_line(METRICS_LINE))

    def test_the_in_loop_summary(self) -> None:
        summary = parse_eval_line(EVAL_LINE)
        self.assertEqual(summary.pc_success, 50.0)
        self.assertEqual(summary.n_episodes, 2)
        self.assertAlmostEqual(summary.eval_s, 85.54, places=2)


class TheStageLines(unittest.TestCase):
    def test_the_chains_narration_is_recognised(self) -> None:
        for line in (
            "=== demos: 1 kept at +-30% DR -> runs/x-demos (00:21:05) ===",
            "--- 91 s",
            "attempt 1: KEEP (damping x1.21, gain x0.93, retries 0)",
            "kept 1/1 episodes -> runs/x-demos (episodes 0..0)",
            "provenance: aloha2-nominal@80ee6fd7ef99, expert kitting-expert@2d",
            "play it: train-watch --play runs/x-act/checkpoints/000300/pretrained",
        ):
            self.assertTrue(is_stage_line(line), line)
        for line in (
            METRICS_LINE,
            "Training:  12%|  | 50/400",
            "$ lerobot-train …",
            "",
        ):
            self.assertFalse(is_stage_line(line), line)


class TheRunManifest(unittest.TestCase):
    def test_round_trip_and_the_table(self) -> None:
        manifest = RunManifest(
            name="t5-cloud",
            policy="act",
            steps=10000,
            batch_size=64,
            checkpoint_every=5000,
            inloop_episodes=2,
            eval_episodes=10,
            workers=16,
            learning_rate=3e-5,
            device="cuda",
            dataset_root="runs/t5-cloud-lerobot",
            dataset_repo_id="rq-pipeline/aloha2-kitting-t5-cloud",
            command="lerobot-train --steps=10000",
            started="2026-08-27T17:18:00Z",
            provenance={
                "bundle": "aloha2-nominal@80ee6fd7ef99",
                "expert": "kitting-expert@2daa0fcfba8a",
                "episodes": 29,
            },
            task="kitting",
            scale="cloud",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest.write(Path(tmp))
            self.assertEqual(RunManifest.read(Path(tmp)), manifest)
        table = manifest.as_markdown()
        for needle in (
            "| batch size | 64 |",
            "| learning rate | 3e-05 |",
            "kitting-expert@2daa0fcfba8a",
            "| dataset episodes | 29 |",
        ):
            self.assertIn(needle, table)


class TheWatchDir(unittest.TestCase):
    def test_beside_the_run_never_inside(self) -> None:
        self.assertEqual(
            watch_dir_for(Path("runs/t5-cloud-act")), Path("runs/t5-cloud-watch")
        )
        self.assertEqual(watch_dir_for(Path("runs/other")), Path("runs/other-watch"))


class TheFollower(unittest.TestCase):
    def test_only_what_was_appended_whole_lines_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.log"
            follower = FileFollower(path)
            self.assertEqual(follower.new_lines(), [])  # not there yet
            path.write_text("one\ntwo\nthr", encoding="utf-8")
            self.assertEqual(follower.new_lines(), ["one", "two"])
            with path.open("a", encoding="utf-8") as handle:
                handle.write("ee\nfour\n")
            self.assertEqual(follower.new_lines(), ["three", "four"])
            self.assertEqual(follower.new_lines(), [])
            path.write_text("fresh\n", encoding="utf-8")  # truncated and rewritten
            self.assertEqual(follower.new_lines(), ["fresh"])


if __name__ == "__main__":
    unittest.main()
