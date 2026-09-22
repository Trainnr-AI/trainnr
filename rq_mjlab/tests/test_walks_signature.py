"""Every registered walk takes the keywords the doors pass: the trainer
and the verdict hand each walk's env builder the same set, and a walk
added without one of them breaks both doors only when a GPU run starts
(the Go2 without `head`, found 2026-09-22 by E0's re-certification)."""

from __future__ import annotations

import inspect
import unittest

from rq_mjlab.walks import ROBOTS, walk_spec

# What walk_train and walk_verdict pass to `spec.env_cfg`.
DOOR_KEYWORDS = (
    "play",
    "dr_span",
    "pin_scale",
    "pin_axis",
    "bundle",
    "head",
    "scene",
    "cameras",
)


class EveryWalkTakesTheDoorsKeywords(unittest.TestCase):
    def test_each_builder_accepts_them(self) -> None:
        for robot in ROBOTS:
            with self.subTest(robot=robot):
                params = inspect.signature(walk_spec(robot).env_cfg).parameters
                missing = [k for k in DOOR_KEYWORDS if k not in params]
                self.assertEqual(missing, [], f"{robot} lacks {missing}")


if __name__ == "__main__":
    unittest.main()
