"""The Studio's cloud feed, pinned at its parsers: every rsl-rl row
lands somewhere, clock rows never become scalars, the layout follows
what the data reveals, and one unreadable record line cannot kill an
overnight feed.

The module under test is a tool (tools/studio-cloud-feed.py), loaded by
path with tools/ on sys.path first for its `_lab` bootstrap. rerun never
enters: main() defers the import, so a FakeRr double stands in for the
whole SDK surface the parsers touch.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
_SPEC = importlib.util.spec_from_file_location(
    "studio_cloud_feed", REPO / "tools" / "studio-cloud-feed.py"
)
assert _SPEC is not None and _SPEC.loader is not None
feed = importlib.util.module_from_spec(_SPEC)
sys.modules["studio_cloud_feed"] = feed
_SPEC.loader.exec_module(feed)

# One real rsl-rl console block, ANSI-bold header and aligned rows with
# their leading and trailing spaces intact (an mjlab run, 2026-09-01).
RSL_BLOCK = (
    "\x1b[1m                          Learning iteration 3477/8000"
    "                           \x1b[0m\n"
    "\n"
    "                            Total steps: 341901312 \n"
    "                       Steps per second: 139905 \n"
    "                        Collection time: 0.640s \n"
    "                        Mean value loss: 0.0466\n"
    "                            Mean reward: 111.84\n"
    "                    Mean episode length: 1000.00\n"
    "               Metrics/peak_height_mean: 0.0237\n"
    "        Episode_Reward/upright: 1.8666\n"
    "          Episode_Termination/fell_over: 0.0000\n"
    "                         Iteration time: 0.70s\n"
    "                           Time elapsed: 0:40:04\n"
    "                                    ETA: 0:52:06\n"
)
RSL_SCALAR_PATHS = {
    "g3/rl/total_steps",
    "g3/rl/steps_per_second",
    "g3/rl/collection_time",
    "g3/rl/value_loss",
    "g3/rl/reward",
    "g3/rl/episode_length",
    "g3/rl/metrics/peak_height_mean",
    "g3/rl/reward_terms/upright",
    "g3/rl/terminations/fell_over",
    "g3/rl/iteration_time",
}
GPU_LINE = "72 %, 16741 MiB"
MIB_PER_GIB = 1024.0


class FakeRr:
    """Just enough rerun: log/set_time/reset_time recorded, payload
    classes that keep their argument."""

    class Scalars:
        def __init__(self, value):
            self.arg = value

    class TextLog:
        def __init__(self, text):
            self.arg = text

    class TextDocument:
        def __init__(self, text, media_type=None):
            self.arg = text
            self.media_type = media_type

    class MediaType:
        MARKDOWN = "text/markdown"

    def __init__(self):
        self.calls = []  # one (path, payload, kwargs) per rr.log
        self.times = []  # ("wall"/"iteration"/…, kwargs); ("reset", {})

    def log(self, path, payload, **kwargs):
        self.calls.append((path, payload, kwargs))

    def set_time(self, name, **kwargs):
        self.times.append((name, kwargs))

    def reset_time(self):
        self.times.append(("reset", {}))


def scalars_by_path(rr: FakeRr) -> dict:
    return {
        path: payload.arg
        for path, payload, _ in rr.calls
        if isinstance(payload, FakeRr.Scalars)
    }


class TheRslParser(unittest.TestCase):
    def setUp(self) -> None:
        feed.MORE_SEEN.clear()

    def test_a_fresh_block_becomes_series_and_a_status(self) -> None:
        rr = FakeRr()
        seen: set[int] = set()
        status = feed.parse_rsl(rr, "g3", RSL_BLOCK, seen)
        self.assertEqual(
            status,
            feed.RslStatus(3477, 8000, reward=111.84, elapsed="0:40:04", eta="0:52:06"),
        )
        self.assertEqual(seen, {3477})
        by_path = scalars_by_path(rr)
        self.assertEqual(set(by_path), RSL_SCALAR_PATHS)
        self.assertEqual(by_path["g3/rl/total_steps"], 341901312.0)
        self.assertEqual(by_path["g3/rl/reward"], 111.84)
        self.assertEqual(by_path["g3/rl/collection_time"], 0.640)
        self.assertEqual(by_path["g3/rl/reward_terms/upright"], 1.8666)
        self.assertEqual(by_path["g3/rl/terminations/fell_over"], 0.0)
        # Both clocks set once, at the unseen header.
        self.assertEqual(rr.times[0][0], "wall")
        self.assertEqual(rr.times[1], ("iteration", {"sequence": 3477}))

    def test_clock_rows_never_become_scalars(self) -> None:
        # The H:MM:SS regression, 2026-09-01: the numeric-row regex
        # reads "ETA: 0:52:06" as the bare 0 — RSL_CLOCK_RE must claim
        # the clock rows first, so no eta/elapsed series ever appears.
        rr = FakeRr()
        feed.parse_rsl(rr, "g3", RSL_BLOCK, set())
        clockish = [
            path for path in scalars_by_path(rr) if "eta" in path or "elapsed" in path
        ]
        self.assertEqual(clockish, [])
        self.assertEqual(feed.MORE_SEEN, set())  # nothing fell to more/

    def test_a_seen_iteration_emits_nothing_but_refreshes_the_card(self) -> None:
        rr = FakeRr()
        status = feed.parse_rsl(rr, "g3", RSL_BLOCK, {3477})
        self.assertEqual(status.reward, 111.84)
        self.assertEqual(status.elapsed, "0:40:04")
        self.assertEqual(status.eta, "0:52:06")
        self.assertEqual(rr.calls, [])  # no series re-logged
        self.assertEqual(rr.times, [])  # no clock touched

    def test_a_log_without_rsl_blocks_yields_none(self) -> None:
        rr = FakeRr()
        self.assertIsNone(feed.parse_rsl(rr, "g3", "step:100 loss:0.5\n", set()))
        self.assertEqual(rr.calls, [])


class TheRowRouter(unittest.TestCase):
    def setUp(self) -> None:
        feed.MORE_SEEN.clear()

    def test_an_unrecognised_row_lands_under_more_and_is_remembered(self) -> None:
        # 2026-09-01: the whitelist was silently dropping terminations
        # and timing rows; now everything unrecognised goes to rl/more/.
        rr = FakeRr()
        feed.route_rsl_row(rr, "g3", "Mean KL divergence", 0.01)
        path, payload, _ = rr.calls[0]
        self.assertEqual(path, "g3/rl/more/mean_kl_divergence")
        self.assertEqual(payload.arg, 0.01)
        self.assertEqual(feed.MORE_SEEN, {"mean_kl_divergence"})

    def test_known_rows_never_touch_the_catch_all(self) -> None:
        rr = FakeRr()
        feed.route_rsl_row(rr, "g3", "Mean reward", 1.0)
        feed.route_rsl_row(rr, "g3", "Episode_Termination/fell_over", 0.0)
        self.assertEqual(
            [call[0] for call in rr.calls],
            ["g3/rl/reward", "g3/rl/terminations/fell_over"],
        )
        self.assertEqual(feed.MORE_SEEN, set())


class TheRecordingId(unittest.TestCase):
    def test_a_log_streams_into_one_recording_across_restarts(self) -> None:
        a = feed.recording_id_for("root@h:1", "/workspace/x.log")
        self.assertEqual(a, feed.recording_id_for("root@h:1", "/workspace/x.log"))
        self.assertNotEqual(a, feed.recording_id_for("root@h:1", "/workspace/y.log"))


class TheEngineCounters(unittest.TestCase):
    def test_kept_counts_distinct_attempts_even_when_the_poll_repeats_them(
        self,
    ) -> None:
        # Every poll carries the whole-log digest AND the tail, so a line
        # can appear twice; the card once read "84 kept of 48 attempts".
        lines = ["attempt 1: KEEP (x)", "attempt 2: discard", "attempt 3: KEEP (y)"]
        raw = "\n".join([*lines, *lines])
        status = feed.parse_engine(FakeRr(), "c", raw, set())
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual((status.kept, status.attempts), (2, 3))


class TheLayoutChooser(unittest.TestCase):
    def setUp(self) -> None:
        feed.MORE_SEEN.clear()

    def test_no_signal_keeps_whatever_was_sent(self) -> None:
        with mock.patch.object(feed, "send_layout") as sender:
            self.assertIsNone(
                feed.choose_layout(FakeRr(), "g3", None, (False, False, False))
            )
            self.assertEqual(
                feed.choose_layout(
                    FakeRr(), "g3", ("rl", False), (False, False, False)
                ),
                ("rl", False),
            )
        sender.assert_not_called()

    def test_the_first_rl_data_sends_the_rl_layout(self) -> None:
        with mock.patch.object(feed, "send_layout") as sender:
            sent = feed.choose_layout(FakeRr(), "g3", None, (True, False, False))
        self.assertEqual(sent, ("rl", False))
        sender.assert_called_once()
        self.assertEqual(sender.call_args.args[2], "rl")

    def test_a_more_row_earns_exactly_one_resend(self) -> None:
        sent = ("rl", False)
        with mock.patch.object(feed, "send_layout") as sender:
            self.assertEqual(
                feed.choose_layout(FakeRr(), "g3", sent, (True, False, False)), sent
            )
            sender.assert_not_called()  # nothing on screen should change
            feed.MORE_SEEN.add("mean_kl_divergence")
            self.assertEqual(
                feed.choose_layout(FakeRr(), "g3", sent, (True, False, False)),
                ("rl", True),
            )
            sender.assert_called_once()  # the "more" pane earned its place

    def test_an_engine_log_gets_its_own_panes_never_the_rl_ones(self) -> None:
        # The campaign feed once wore the RL layout: eight empty panes
        # (the operator's screenshot, 2026-09-03).
        self.assertEqual(
            feed.choose_layout(FakeRr(), "g3", None, (False, False, True)),
            ("engine", False),
        )

    def test_the_engine_layout_grows_with_the_campaign(self) -> None:
        with mock.patch.object(feed, "send_layout") as sender:
            sent = feed.choose_layout(FakeRr(), "c", None, (False, False, True))
            self.assertEqual(sent, ("engine", feed.ENGINE_PRESS))
            sent = feed.choose_layout(
                FakeRr(),
                "c",
                sent,
                (False, True, True),
                engine_level=feed.ENGINE_TRAINING,
            )
            self.assertEqual(sent, ("engine", feed.ENGINE_TRAINING))
            self.assertEqual(sender.call_count, 2)  # the trainer panes earned theirs
            self.assertEqual(sender.call_args.kwargs["level"], feed.ENGINE_TRAINING)

    def test_a_lerobot_run_gets_the_train_eval_layout(self) -> None:
        with mock.patch.object(feed, "send_layout") as sender:
            self.assertEqual(
                feed.choose_layout(FakeRr(), "g3", None, (False, True, False)),
                ("lerobot", False),
            )
        sender.assert_called_once()
        self.assertEqual(sender.call_args.args[2], "lerobot")


class TheRlCard(unittest.TestCase):
    def test_the_card_carries_iteration_reward_clocks_and_gpu(self) -> None:
        rr = FakeRr()
        status = feed.RslStatus(
            3477, 8000, reward=111.84, elapsed="0:40:04", eta="0:52:06"
        )
        feed.write_rl_card(rr, "g3", status, GPU_LINE)
        path, doc, kwargs = rr.calls[0]
        self.assertEqual(path, "g3/status")
        self.assertEqual(kwargs, {"static": True})  # "right now", any scrub
        self.assertIsInstance(doc, FakeRr.TextDocument)
        self.assertEqual(doc.media_type, FakeRr.MediaType.MARKDOWN)
        self.assertIn("iteration **3477 / 8000**", doc.arg)
        self.assertIn("mean reward **111.84**", doc.arg)
        self.assertIn("elapsed 0:40:04, ETA **0:52:06**", doc.arg)
        self.assertIn(f"GPU {GPU_LINE}", doc.arg)

    def test_a_bare_first_block_still_makes_a_card(self) -> None:
        rr = FakeRr()
        feed.write_rl_card(rr, "g3", feed.RslStatus(0, 8000), "?")
        card = rr.calls[0][1].arg
        self.assertIn("iteration **0 / 8000**", card)
        self.assertNotIn("reward", card)
        self.assertNotIn("ETA", card)


class TheGpuLine(unittest.TestCase):
    def test_the_sentinel_line_becomes_two_series(self) -> None:
        rr = FakeRr()
        feed.log_gpu(rr, "g3", GPU_LINE)
        by_path = scalars_by_path(rr)
        self.assertEqual(by_path["g3/gpu/utilization"], 72.0)
        self.assertAlmostEqual(by_path["g3/gpu/memory_gb"], 16741 / MIB_PER_GIB)
        self.assertEqual(rr.times[0][0], "wall")

    def test_a_placeholder_without_percent_logs_nothing(self) -> None:
        rr = FakeRr()
        feed.log_gpu(rr, "g3", "?")
        self.assertEqual(rr.calls, [])
        self.assertEqual(rr.times, [])


class TheEvalRecords(unittest.TestCase):
    def test_unreadable_lines_are_refused_not_raised(self) -> None:
        # 2026-09-01: a JSON array line and {"trial": null} both raise
        # TypeError, which the old (ValueError, KeyError) catch let
        # through to kill an overnight feed.
        rr = FakeRr()
        self.assertFalse(feed.emit_record(rr, "g3", "g2", "[1, 2, 3]", set()))
        self.assertFalse(feed.emit_record(rr, "g3", "g2", '{"trial": null}', set()))
        self.assertFalse(feed.emit_record(rr, "g3", "g2", "not json", set()))
        self.assertFalse(feed.emit_record(rr, "g3", "g2", '{"success": true}', set()))
        self.assertEqual(rr.calls, [])

    def test_a_record_emits_once_on_its_own_trial_clock(self) -> None:
        rr = FakeRr()
        seen: set[tuple[str, int]] = set()
        line = '{"trial": 3, "success": true}'
        self.assertTrue(feed.emit_record(rr, "g3", "g2", line, seen))
        self.assertEqual(seen, {("g2", 3)})
        # reset FIRST, then only the trial clock: set_time persists on
        # the thread, and a leaked wall/train_step stamped the NEXT
        # metric line with this record's clocks (review 2026-09-01).
        self.assertEqual(rr.times, [("reset", {}), ("trial", {"sequence": 3})])
        by_path = {path: payload for path, payload, _ in rr.calls}
        self.assertEqual(by_path["g3/eval/g2/success"].arg, 1.0)
        self.assertEqual(by_path["g3/stage"].arg, "eval g2 trial 3: KEEP")

    def test_a_duplicate_is_acknowledged_but_not_re_logged(self) -> None:
        rr = FakeRr()
        seen = {("g2", 3)}
        line = '{"trial": 3, "success": false}'
        self.assertTrue(feed.emit_record(rr, "g3", "g2", line, seen))
        self.assertEqual(rr.calls, [])


class ThePollParser(unittest.TestCase):
    def test_sentinel_lines_route_to_arm_gpu_eval_and_records(self) -> None:
        rr = FakeRr()
        raw = (
            "RQARM == training left\n"
            f"RQGPU {GPU_LINE}\n"
            "RQEVAL Stepping through eval batches 3/10\n"
            "RQREC /workspace/runs/g2-brick-records.jsonl\n"
            '{"trial": 0, "success": false}\n'
            "RQPOLL_OK\n"
        )
        arm, gpu, latest, eval_progress, last_step = feed.parse_poll(
            rr, "g3", raw, (set(), set(), {})
        )
        self.assertEqual(arm, "left")
        self.assertEqual(gpu, GPU_LINE)
        self.assertIsNone(latest)  # no train metric lines in this poll
        self.assertEqual(eval_progress, "Stepping through eval batches 3/10")
        self.assertEqual(last_step, -1)
        by_path = scalars_by_path(rr)
        self.assertEqual(by_path["g3/eval/g2-brick/success"], 0.0)


if __name__ == "__main__":
    unittest.main()
