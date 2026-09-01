"""The locomotion certificate's judgment: the declared criterion,
spelled a second time so the tool cannot drift from what the
certificate claims (G4, docs/e2e-research/63 §2.4)."""

from __future__ import annotations

import unittest

from rq_mjlab.walk_verdict import ERR_FLOOR, ERR_RATIO_BOUND, EpisodeOutcome


class TheCriterion(unittest.TestCase):
    def test_success_needs_both_milestones(self) -> None:
        walked = EpisodeOutcome(steps=1000, fell=False, mean_err=0.05, mean_cmd=0.3)
        self.assertTrue(walked.survived and walked.tracked and walked.success)
        fell = EpisodeOutcome(steps=412, fell=True, mean_err=0.05, mean_cmd=0.3)
        self.assertFalse(fell.success)
        drifted = EpisodeOutcome(steps=1000, fell=False, mean_err=0.2, mean_cmd=0.3)
        self.assertFalse(drifted.tracked or drifted.success)

    def test_the_ratio_is_the_standing_still_gap_longhand(self) -> None:
        # err_ratio = mean_err / max(mean_cmd, ERR_FLOOR); tracked means
        # the policy closes at least half the gap standing still leaves.
        outcome = EpisodeOutcome(steps=1000, fell=False, mean_err=0.12, mean_cmd=0.3)
        self.assertAlmostEqual(outcome.err_ratio, 0.12 / 0.3)
        self.assertTrue(outcome.err_ratio < ERR_RATIO_BOUND)

    def test_a_gentle_command_floors_the_denominator(self) -> None:
        # A near-zero command must not make any wobble an automatic fail
        # (or a trivial pass): the denominator floors at ERR_FLOOR m/s.
        idle = EpisodeOutcome(steps=1000, fell=False, mean_err=0.04, mean_cmd=0.01)
        self.assertAlmostEqual(idle.err_ratio, 0.04 / ERR_FLOOR)
        self.assertTrue(idle.tracked)
        wobbly = EpisodeOutcome(steps=1000, fell=False, mean_err=0.06, mean_cmd=0.01)
        self.assertFalse(wobbly.tracked)


if __name__ == "__main__":
    unittest.main()
