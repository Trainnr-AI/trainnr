"""The replay shards: the paper's transition, stored per episode,
sampled across episodes, never half-written."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rq_pipeline.rl.replay import FIELDS, ReplayShards, Transition

LAST = 2  # the transition that ends the episode


def _transition(k: int, n: int = 2, nu: int = 3) -> Transition:
    ones = np.ones
    return Transition(
        state=ones(4) * k,
        reference=ones((2 * n, nu)) * k,
        committed=ones((n, nu)) * k,
        execution=ones((n, nu)) * k,
        target=ones((2 * n, nu)) * k,
        reward=float(k),
        next_state=ones(4) * (k + 1),
        next_reference=ones((2 * n, nu)),
        next_committed=ones((n, nu)),
        done=k == LAST,
    )


class TheShards(unittest.TestCase):
    def test_episodes_stack_per_field_and_sample_across_shards(self) -> None:
        with TemporaryDirectory() as tmp:
            replay = ReplayShards(Path(tmp))
            replay.append([_transition(0), _transition(1)])
            replay.append([_transition(2)])
            self.assertEqual(len(list(replay.shards())), 2)
            table = replay.load()
            self.assertEqual(set(table), set(FIELDS))
            self.assertEqual(table["reward"].tolist(), [0.0, 1.0, 2.0])
            self.assertEqual(table["execution"].shape, (3, 2, 3))
            self.assertEqual(table["done"].tolist(), [False, False, True])
            batch = replay.sample(8, np.random.default_rng(0))
            self.assertEqual(batch["state"].shape, (8, 4))
            self.assertTrue(set(batch["reward"].tolist()) <= {0.0, 1.0, 2.0})

    def test_an_empty_episode_and_an_empty_replay_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            replay = ReplayShards(Path(tmp))
            with self.assertRaises(ValueError):
                replay.append([])
            with self.assertRaises(ValueError):
                replay.sample(1, np.random.default_rng(0))

    def test_no_half_written_shard_is_visible(self) -> None:
        with TemporaryDirectory() as tmp:
            replay = ReplayShards(Path(tmp))
            replay.append([_transition(0)])
            self.assertEqual([p.suffix for p in Path(tmp).iterdir()], [".npz"])


if __name__ == "__main__":
    unittest.main()
