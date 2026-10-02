"""The per-trial record and its fold: stdlib only, runs in the fast suite."""

import json
import tempfile
import unittest
from pathlib import Path

from trainnr.evaluate.harness import EpisodeProtocol, events_for
from trainnr.evaluate.records import (
    EpisodeRecord,
    SimScore,
    append_records,
    disagreements,
    fold,
    funnel,
    milestones,
    passes,
    read_records,
)

STAGES = ["moved", "lifted", "placed"]
TWO, FIVE = 2, 5  # the synthetic chain's thresholds
PROTOCOL = {
    "trials": 4,
    "steps": 4000,
    "control_interval": 10,
    "home": "neutral",
    "milestones": STAGES,
}
SOURCE = "aloha2-kitting@000000000000"


def record(policy: str, trial: int, success: bool, reached: int = 0) -> EpisodeRecord:
    events = tuple(
        {"index": index, "name": STAGES[index], "step": 100 * (index + 1)}
        for index in range(reached)
    )
    return EpisodeRecord(
        source=SOURCE,
        policy=policy,
        trial=trial,
        success=success,
        steps=4000,
        instrument="mujoco-3.11.0",
        protocol=PROTOCOL,
        events=events,
    )


class Milestones(unittest.TestCase):
    def _protocol(self) -> EpisodeProtocol:
        # A one-dimensional "episode": states[t] = (t,), the chain reads it.
        return EpisodeProtocol(
            trials=1,
            steps=10,
            control_interval=1,
            perturb=lambda _trial, home: home,
            success=lambda _states, _sensors: True,
            milestones=(
                ("past_two", lambda states, _s, t: states[t][0] > TWO),
                ("past_five", lambda states, _s, t: states[t][0] > FIVE),
                ("never", lambda _states, _s, _t: False),
            ),
        )

    def test_walker_fires_each_milestone_once_in_order(self) -> None:
        states = [(t,) for t in range(10)]
        events = events_for(self._protocol(), states, None)
        self.assertEqual(
            events,
            (
                {"index": 0, "name": "past_two", "step": 3},
                {"index": 1, "name": "past_five", "step": 6},
            ),
        )

    def test_a_later_milestone_cannot_fire_before_an_earlier_one(self) -> None:
        # Only the CURRENT milestone is evaluated: a state past five at
        # step 0 still has to pass "past_two" first, and both fire on
        # successive steps, never the same one.
        states = [(9,)] * 4
        events = events_for(self._protocol(), states, None)
        self.assertEqual([e["name"] for e in events], ["past_two", "past_five"])
        self.assertEqual([e["step"] for e in events], [0, 1])

    def test_duplicate_milestone_names_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            EpisodeProtocol(
                trials=1,
                steps=1,
                control_interval=1,
                perturb=lambda _t, h: h,
                success=lambda _s, _x: True,
                milestones=(("a", lambda *_: True), ("a", lambda *_: True)),
            )


class Funnel(unittest.TestCase):
    def test_counts_reached_per_stage_per_policy(self) -> None:
        rows = [
            record("expert", 0, True, reached=3),
            record("expert", 1, True, reached=3),
            record("grip", 0, False, reached=2),
            record("grip", 1, False, reached=1),
            record("limp", 0, False, reached=0),
            record("limp", 1, False, reached=0),
        ]
        self.assertEqual(
            funnel(rows), {"expert": [2, 2, 2], "grip": [2, 1, 0], "limp": [0, 0, 0]}
        )
        with self.assertRaises(ValueError):
            funnel([])

    def test_disagreements_flag_verdict_versus_chain(self) -> None:
        rows = [
            record("a", 0, True, reached=3),  # agree
            record("a", 1, False, reached=3),  # complete chain, late drop
            record("a", 2, True, reached=1),  # referee looser than the chain
            record("a", 3, False, reached=0),  # agree
        ]
        self.assertEqual([r.trial for r in disagreements(rows)], [1, 2])


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

    def test_passes_split_a_file_several_evaluations_appended_to(self) -> None:
        """The trainer's in-loop eval appends one pass per checkpoint under
        one policy name; a pass ends where a (policy, trial) recurs, and
        each pass folds on its own (the first cloud run, 2026-08-27)."""
        rows = [
            record("inloop", 0, False),
            record("inloop", 1, False),
            record("inloop", 0, True),
            record("inloop", 1, False),
        ]
        chunks = passes(rows)
        self.assertEqual([len(c) for c in chunks], [2, 2])
        self.assertEqual(fold(chunks[0]), (SimScore("inloop", 0, 2),))
        self.assertEqual(fold(chunks[1]), (SimScore("inloop", 1, 2),))
        self.assertEqual(passes([]), ())
        self.assertEqual(len(passes(rows[:2])), 1)

    def test_milestones_are_read_from_the_records(self) -> None:
        self.assertEqual(milestones([record("a", 0, True)]), list(STAGES))
        with self.assertRaises(ValueError):
            milestones([])

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
            lines = path.read_text(encoding="utf-8").splitlines()
            back = read_records(path)
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["policy"], "a")
        self.assertEqual(back, tuple(rows))

    def test_lerobot_eval_info_folds_by_seed_order(self) -> None:
        from trainnr.envs.lerobot_info import from_eval_info  # noqa: PLC0415

        # LeRobot 0.6.1: successes as a list in episode order, no seed.
        eval_info = {
            "per_task": [
                {
                    "task_group": "trainnr",
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
