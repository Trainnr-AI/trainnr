"""The harness honours the dataset's cadence: one policy action, N
control ticks (2026-09-04: hold 1 scored 0/3, hold 5 scored 3/3 on
the same checkpoint)."""

from __future__ import annotations

import unittest

import gymnasium as gym

from tests._extras import needs_train_sim


class _Counting(gym.Env):
    """A minimal env: counts steps, terminates at `stop`."""

    def __init__(self, stop: int) -> None:
        super().__init__()
        self.steps = 0
        self.stop = stop
        self.observation_space = gym.spaces.Discrete(1000)
        self.action_space = gym.spaces.Discrete(1)

    def step(self, action):
        self.steps += 1
        return self.steps, 1.0, self.steps >= self.stop, False, {"n": self.steps}

    def reset(self, **kw):
        return 0, {}


class TheHold(unittest.TestCase):
    def test_holds_n_ticks_sums_reward_and_stops_at_the_end(self) -> None:
        from trainnr.envs.hold import ActionHold  # noqa: PLC0415

        env = ActionHold(_Counting(stop=100), 5)
        obs, reward, term, _trunc, _info = env.step(0)
        self.assertEqual((obs, reward, term), (5, 5.0, False))
        env = ActionHold(_Counting(stop=3), 5)
        obs, reward, term, _t, _i = env.step(0)
        self.assertEqual((obs, reward, term), (3, 3.0, True))  # stopped early
        with self.assertRaises(ValueError):
            ActionHold(_Counting(stop=1), 0)


@needs_train_sim  # the plugin's config imports lerobot
class TheFlag(unittest.TestCase):
    def test_frame_every_sets_fps_and_refuses_a_non_divisor(self) -> None:
        from trainnr.envs.lerobot_plugin import RobotiqEnvConfig  # noqa: PLC0415

        cfg = RobotiqEnvConfig(task="lift-study", frame_every=5)
        self.assertEqual(cfg.fps, 10)
        self.assertEqual(RobotiqEnvConfig(task="lift-study").fps, 50)
        with self.assertRaisesRegex(ValueError, "frame_every"):
            RobotiqEnvConfig(task="lift-study", frame_every=7)
        self.assertIn(
            "--env.frame_every=5",
            RobotiqEnvConfig.cli_flags("lift-study", frame_every=5),
        )


if __name__ == "__main__":
    unittest.main()
