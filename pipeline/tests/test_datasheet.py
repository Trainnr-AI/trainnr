"""The datasheet over synthetic sidecars of both schemas — the summary
facts, the honesty (bounds and unrecorded stamps), the warnings, and
the rendered page. No simulator, no numpy.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rq_pipeline.collect.datasheet import (
    UNRECORDED,
    render,
    summarize,
    write_datasheet,
)


def write_manifest(root: Path, index: int, raw: dict) -> None:
    episode = root / f"episode_{index:04d}"
    episode.mkdir(parents=True)
    (episode / "manifest.json").write_text(json.dumps(raw))


def press_manifest(attempt: int, friction: float, basis: str = "identified-interval"):
    return {
        "seed": 7,
        "attempt": attempt,
        "task": "demo-task@abc123",
        "expert": "demo-expert@def456",
        "instrument": "mujoco-3.11.0+test",
        "dynamics": {"friction": friction},
        "dynamics_basis": basis,
        "draws": {},
        "retries": [],
        "control_hz": 50,
        "frame_every_control_ticks": 1,
    }


def legacy_manifest(attempt: int) -> dict:
    return {
        "seed": 7,
        "attempt": attempt,
        "draws": {},
        "dr_span": 0.3,
        "damping_scale": 1.1,
        "gain_scale": 0.9,
        "retries": [["left", 100, 0.02]],
        "control_hz": 50,
        "frame_every_control_ticks": 1,
        "expert": "kitting-expert@aaa111",
    }


class Summaries(unittest.TestCase):
    def test_a_clean_press_batch_summarizes_without_warnings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root, 0, press_manifest(2, 0.95))
            write_manifest(root, 1, press_manifest(5, 1.05))
            summary = summarize(root)
            self.assertEqual(summary.episodes, 2)
            self.assertEqual(summary.max_attempt, 5)
            self.assertAlmostEqual(summary.keep_rate_bound, 0.4)
            self.assertEqual(summary.tasks, ("demo-task@abc123",))
            self.assertEqual(summary.bases, ("identified-interval",))
            spread = summary.dynamics["friction"]
            self.assertEqual((spread.low, spread.high), (0.95, 1.05))
            self.assertEqual(summary.warnings, ())

    def test_legacy_manifests_normalize_with_absences_named(self) -> None:
        # The kitting task's committed batches: scales become a dynamics
        # dict, the span becomes a basis, and the stamps the schema
        # never carried are reported as unrecorded — not invented.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root, 0, legacy_manifest(3))
            summary = summarize(root)
            self.assertEqual(summary.tasks, (UNRECORDED,))
            self.assertIn("±0.3", summary.bases[0])
            self.assertEqual(set(summary.dynamics), {"damping", "gain"})
            self.assertEqual(summary.retries_total, 1)
            self.assertTrue(any("task stamp unrecorded" in w for w in summary.warnings))

    def test_mixed_stamps_and_bases_are_warned_out_loud(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root, 0, press_manifest(1, 1.0))
            other = press_manifest(2, 1.0, basis="caller-declared span ±0.1")
            other["instrument"] = "mjx-warp-3.12.0+gpu"
            write_manifest(root, 1, other)
            summary = summarize(root)
            text = " ".join(summary.warnings)
            self.assertIn("mixed instrument stamps", text)
            self.assertIn("mixed dynamics bases", text)


class ShardsAndAbsences(unittest.TestCase):
    def test_shards_refuse_a_single_keep_rate_bound(self) -> None:
        # Two seeds in one directory: attempt counters restart per
        # shard, so episodes/max_attempt read "400%" once (review
        # 2026-09-01).
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = press_manifest(1, 1.0)
            second = press_manifest(1, 1.0)
            second["seed"] = 99
            write_manifest(root, 0, first)
            write_manifest(root, 1, second)
            summary = summarize(root)
            self.assertEqual(summary.shards, 2)
            page = render(summary)
            self.assertIn("2 shards in this directory", page)
            self.assertNotIn("keep rate ≤", page)

    def test_same_seed_shards_are_caught_by_the_attempt_restart(self) -> None:
        # Two runs with the DEFAULT seed and disjoint episode ranges —
        # the documented sharding pattern — collapse the seed signal;
        # the attempt counter's restart is the boundary that survives
        # (second review, 2026-09-01).
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root, 0, press_manifest(3, 1.0))  # shard A ends at 3
            write_manifest(root, 1, press_manifest(1, 1.0))  # shard B restarts
            summary = summarize(root)
            self.assertEqual(summary.shards, 2)

    def test_an_empty_basis_is_unstated_not_called_legacy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root, 0, press_manifest(1, 1.0, basis=""))
            summary = summarize(root)
            self.assertIn("unstated", summary.bases[0])
            self.assertNotIn("legacy", summary.bases[0])


class Rendering(unittest.TestCase):
    def test_the_page_states_the_facts_and_writes_beside_the_batch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(root, 0, press_manifest(4, 1.02))
            page = render(summarize(root))
            for needle in (
                "episodes kept: **1**",
                "keep rate ≤ 25%",
                "demo-task@abc123",
                "identified-interval",
                "| friction |",
            ):
                self.assertIn(needle, page)
            path = write_datasheet(root)
            self.assertEqual(path.name, "datasheet.md")
            self.assertEqual(path.read_text(), page)


if __name__ == "__main__":
    unittest.main()
