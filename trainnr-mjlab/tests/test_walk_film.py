"""The film's pure parts: the one command it holds, and how a filmed
checkpoint is kept. The rollout and the render need the GPU venv and are
exercised by filming a run (#137)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trainnr_mjlab.walk_film import (
    PLAY_FILE,
    VIDEO_FILE,
    _already,
    film_command,
    iteration_of,
)


def _cfg() -> SimpleNamespace:
    ranges = SimpleNamespace(
        lin_vel_x=(-0.2, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
    )
    twist = SimpleNamespace(
        ranges=ranges,
        heading_command=True,
        rel_standing_envs=0.1,
        rel_world_envs=0.2,
        rel_forward_envs=0.3,
        init_velocity_prob=0.5,
    )
    return SimpleNamespace(
        commands={"twist": twist}, curriculum={"command_vel": object(), "other": 1}
    )


class TheHeldCommand(unittest.TestCase):
    def test_forward_at_the_top_of_the_range_unless_told(self) -> None:
        cfg = _cfg()
        self.assertEqual(film_command(cfg, None), [0.4, 0.0, 0.0])
        ranges = cfg.commands["twist"].ranges
        self.assertEqual(ranges.lin_vel_x, (0.4, 0.4))
        self.assertEqual(ranges.lin_vel_y, (0.0, 0.0))
        self.assertEqual(ranges.ang_vel_z, (0.0, 0.0))

    def test_every_other_draw_is_off(self) -> None:
        cfg = _cfg()
        self.assertEqual(film_command(cfg, [0.1, -0.2, 0.5]), [0.1, -0.2, 0.5])
        twist = cfg.commands["twist"]
        self.assertFalse(twist.heading_command)
        for share in ("rel_standing_envs", "rel_world_envs", "rel_forward_envs"):
            self.assertEqual(getattr(twist, share), 0.0)
        self.assertEqual(twist.init_velocity_prob, 0.0)
        self.assertEqual(cfg.curriculum, {"other": 1})  # the command curriculum gone


class AFilmedCheckpoint(unittest.TestCase):
    def test_kept_only_when_filmed_the_same_way(self) -> None:
        folder = Path(tempfile.mkdtemp())
        record = folder / PLAY_FILE
        wanted = {"command": [0.4, 0.0, 0.0], "seconds_asked": 10.0, "seed": 1000}
        self.assertIsNone(_already(record, wanted))
        record.write_text(json.dumps({**wanted, "fell": False}))
        self.assertIsNone(_already(record, wanted))  # no video beside it
        (folder / VIDEO_FILE).write_bytes(b"v")
        self.assertEqual(_already(record, wanted)["fell"], False)
        self.assertIsNone(_already(record, {**wanted, "seed": 7}))
        self.assertIsNone(_already(record, {**wanted, "command": [0.2, 0.0, 0.0]}))

    def test_the_iteration_comes_from_the_name(self) -> None:
        self.assertEqual(iteration_of(Path("model_2500.pt")), 2500)
        self.assertIsNone(iteration_of(Path("policy.pt")))


if __name__ == "__main__":
    unittest.main()
