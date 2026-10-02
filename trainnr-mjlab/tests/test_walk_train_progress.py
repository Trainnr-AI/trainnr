"""A training run says where it is in the Studio's Running now panel:
rsl_rl's console, read line by line into the job's progress (2026-09-25)."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from trainnr.mcp_jobs import JOBS_DIR_NAME, STATUS_SUFFIX, read_status, track

from trainnr_mjlab.walk_train import TRAIN_KIND, Tee, TrainingTicker

# Two iterations of rsl_rl's console as it prints them: a bold banner,
# then padded `key: value` lines.
RSL_RL_CONSOLE = (
    "\x1b[1m Learning iteration 340/1500 \x1b[0m\n"
    "                       Computation: 51234 steps/s\n"
    "                       Mean reward: 41.23\n"
    "               Mean episode length: 998.00\n"
    "\x1b[1m Learning iteration 341/1500 \x1b[0m\n"
    "                       Mean reward: 42.5\n"
)


class TheTicker(unittest.TestCase):
    def test_rsl_rl_s_lines_become_iteration_and_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / JOBS_DIR_NAME
            with track(TRAIN_KIND, jobs_dir=table, name="go2-smoke") as run:
                console = io.StringIO()
                tee = Tee(console, None, on_line=TrainingTicker(run).line)
                # printed in pieces, the way a stream is written
                for piece in (RSL_RL_CONSOLE[:40], RSL_RL_CONSOLE[40:]):
                    tee.write(piece)
                status = read_status(table / f"{run.job_id}{STATUS_SUFFIX}")
            assert status is not None
            self.assertEqual((status.done, status.total), (341, 1500))
            self.assertEqual(status.unit, "iterations")
            self.assertEqual(status.stage, "iteration 341 of 1500, reward 42.5")
            self.assertEqual(console.getvalue(), RSL_RL_CONSOLE)

    def test_a_tee_without_a_file_or_a_listener_is_the_console(self) -> None:
        console = io.StringIO()
        Tee(console, None).write("hello\n")
        self.assertEqual(console.getvalue(), "hello\n")


if __name__ == "__main__":
    unittest.main()
