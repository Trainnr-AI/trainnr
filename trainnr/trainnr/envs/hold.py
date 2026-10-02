"""Hold each policy action for `ticks` control steps.

A dataset pressed with `frame_every=5` is a 10 Hz recording of a 50 Hz
controller; a policy trained on it emits one action per FRAME. The
harness stepped the env once per action — every chunk played five
times too fast — and a policy that reproduced its training actions to
0.005 rad scored 0/80 (2026-09-04; hold 1: 0/3, hold 5: 3/3 on the
same checkpoint). The walk learned the same lesson first (campaign 1,
docs/07 2026-09-02): cadence is the dataset's, and the env must honour
it. This wrapper is that honour: one action in, `ticks` control steps
out, reward summed, the episode's end respected.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class ActionHold(gym.Wrapper):
    """Repeat each action for `ticks` env steps (1 = no hold)."""

    def __init__(self, env: gym.Env, ticks: int) -> None:
        if ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {ticks}")
        super().__init__(env)
        self.ticks = ticks

    def step(self, action: Any) -> tuple:
        total = 0.0
        for _ in range(self.ticks):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info
