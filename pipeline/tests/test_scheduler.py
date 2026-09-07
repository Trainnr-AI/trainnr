"""The action scheduler: our horizon, every fetch checked, nothing leaks."""

import unittest

from rq_pipeline.evaluate.scheduler import ActionScheduler, ChunkPolicy, Delayed
from rq_pipeline.protocol import EpisodeProtocol, protocol_fields

NU = 3
HORIZON = 5
EXECUTED = 2


class Counting:
    """A chunk policy that numbers its chunks so the schedule is visible."""

    def __init__(self, horizon: int = HORIZON, width: int = NU) -> None:
        self.calls = 0
        self.resets = 0
        self.horizon, self.width = horizon, width

    def predict(self, observation):
        del observation
        self.calls += 1
        import numpy as np  # noqa: PLC0415

        chunk = np.zeros((self.horizon, self.width))
        chunk[:, 0] = self.calls  # which fetch
        chunk[:, 1] = np.arange(self.horizon)  # which row of it
        return chunk

    def reset(self) -> None:
        self.resets += 1

    def policy(self) -> ChunkPolicy:
        return ChunkPolicy("counting", self.predict, self.reset)


class TheSchedule(unittest.TestCase):
    def test_replans_every_executed_horizon_ticks(self) -> None:
        counting = Counting()
        scheduler = ActionScheduler(counting.policy(), executed_horizon=EXECUTED, nu=NU)
        scheduler.reset()
        rows = [tuple(scheduler.act({})[:2]) for _ in range(5)]
        # fetch 1 rows 0,1; fetch 2 rows 0,1; fetch 3 row 0
        self.assertEqual(rows, [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)])
        self.assertEqual(counting.calls, 3)
        self.assertEqual(scheduler.fetches, 3)

    def test_a_latency_budget_skips_the_committed_rows(self) -> None:
        # docs/e2e-research/71: asked at tick 2, a chunk takes over at
        # tick 3 with its row 1 - row 0 was the committed region the
        # previous chunk supplied while this one was being inferred.
        counting = Counting()
        scheduler = ActionScheduler(
            counting.policy(), executed_horizon=EXECUTED, nu=NU, latency=1
        )
        scheduler.reset()
        rows = [tuple(scheduler.act({})[:2]) for _ in range(6)]
        self.assertEqual(rows, [(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1)])
        self.assertEqual(scheduler.fetches, 3)

    def test_a_budget_beyond_the_horizon_becomes_the_execution_window(self) -> None:
        # latency 4 with 2 executed rows: one request in flight at a time,
        # so each chunk runs rows [4, 8) - the paper's [n, 2n) window.
        counting = Counting(horizon=12)
        scheduler = ActionScheduler(
            counting.policy(), executed_horizon=EXECUTED, nu=NU, latency=4
        )
        scheduler.reset()
        rows = [tuple(scheduler.act({})[:2]) for _ in range(14)]
        self.assertEqual(rows[:6], [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)])
        self.assertEqual(rows[6:10], [(2, 4), (2, 5), (2, 6), (2, 7)])
        self.assertEqual(rows[10:14], [(3, 4), (3, 5), (3, 6), (3, 7)])
        self.assertEqual(scheduler.fetches, 4)  # asked at 0, 2, 6, 10

    def test_zero_latency_is_the_synchronous_loop(self) -> None:
        counting = Counting()
        scheduler = ActionScheduler(
            counting.policy(), executed_horizon=EXECUTED, nu=NU, latency=0
        )
        scheduler.reset()
        rows = [tuple(scheduler.act({})[:2]) for _ in range(5)]
        self.assertEqual(rows, [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)])

    def test_a_chunk_must_cover_the_wait_and_the_window(self) -> None:
        scheduler = ActionScheduler(
            Counting(horizon=2).policy(), executed_horizon=EXECUTED, nu=NU, latency=1
        )
        scheduler.reset()
        with self.assertRaises(ValueError):
            scheduler.act({})
        with self.assertRaises(ValueError):
            ActionScheduler(
                Counting().policy(), executed_horizon=EXECUTED, nu=NU, latency=-1
            )

    def test_reset_drops_the_held_chunk_and_resets_the_policy(self) -> None:
        counting = Counting()
        scheduler = ActionScheduler(counting.policy(), executed_horizon=EXECUTED, nu=NU)
        scheduler.reset()
        scheduler.act({})
        scheduler.reset()
        self.assertEqual(tuple(scheduler.act({})[:2]), (2, 0))  # a fresh fetch, row 0
        self.assertEqual(counting.resets, 2)

    def test_every_fetch_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            ActionScheduler(Counting().policy(), executed_horizon=0, nu=NU)
        one_d = ChunkPolicy("flat", lambda _o: [0.0] * NU, lambda: None)
        with self.assertRaises(ValueError) as caught:
            ActionScheduler(one_d, executed_horizon=1, nu=NU).act({})
        self.assertIn("flat", str(caught.exception))
        narrow = Counting(width=NU - 1)
        with self.assertRaises(ValueError) as caught:
            ActionScheduler(narrow.policy(), executed_horizon=1, nu=NU).act({})
        self.assertIn("actuators", str(caught.exception))
        short = Counting(horizon=EXECUTED - 1)
        with self.assertRaises(ValueError) as caught:
            ActionScheduler(short.policy(), executed_horizon=EXECUTED, nu=NU).act({})
        self.assertIn("executed_horizon", str(caught.exception))


class TheProtocolField(unittest.TestCase):
    def _protocol(self, **extra):
        return EpisodeProtocol(
            trials=1,
            steps=5,
            control_interval=1,
            perturb=lambda _t, home: home,
            success=lambda _s, _r: True,
            **extra,
        )

    def test_the_horizon_is_hashed_only_when_declared(self) -> None:
        self.assertNotIn("executed_horizon", protocol_fields(self._protocol()))
        self.assertEqual(
            protocol_fields(self._protocol(executed_horizon=EXECUTED))[
                "executed_horizon"
            ],
            EXECUTED,
        )
        with self.assertRaises(ValueError):
            self._protocol(executed_horizon=0)


if __name__ == "__main__":
    unittest.main()


class TheDelay(unittest.TestCase):
    def test_actions_arrive_ticks_later_and_the_first_holds(self) -> None:
        seen = Delayed(lambda observation: observation, ticks=2)
        self.assertEqual([seen(t) for t in range(6)], [0, 0, 0, 1, 2, 3])

    def test_zero_delay_is_the_policy_itself(self) -> None:
        self.assertEqual(
            [Delayed(lambda o: o * 10, 0)(t) for t in range(3)], [0, 10, 20]
        )
        with self.assertRaises(ValueError):
            Delayed(lambda o: o, -1)
