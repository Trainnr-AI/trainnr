"""A certificate and a manifest describe the command envelope the
checkpoint trained under, not the curriculum's first stage."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from rq_mjlab.envelope import (
    checkpoint_iteration,
    pin_command_envelope,
    stage_reached,
)

STAGES = [
    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    {"step": 120000, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
    {"step": 240000, "lin_vel_x": (-2.0, 3.0)},
]
STEPS = 24


def _cfg(with_curriculum: bool = True) -> SimpleNamespace:
    ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=None,
    )
    curriculum = (
        {"command_vel": SimpleNamespace(params={"velocity_stages": STAGES})}
        if with_curriculum
        else {}
    )
    return SimpleNamespace(
        commands={"twist": SimpleNamespace(ranges=ranges)}, curriculum=curriculum
    )


class TheEnvelope(unittest.TestCase):
    def test_iteration_from_a_checkpoint_name(self) -> None:
        self.assertEqual(checkpoint_iteration("model_7999"), 7999)
        self.assertEqual(checkpoint_iteration("model_0.pt"), 0)
        self.assertIsNone(checkpoint_iteration("best.pt"))

    def test_stage_reached(self) -> None:
        self.assertEqual(stage_reached(STAGES, 0), 0)
        self.assertEqual(stage_reached(STAGES, 119999), 0)
        self.assertEqual(stage_reached(STAGES, 120000), 1)
        self.assertEqual(stage_reached(STAGES, 300000), 2)

    def test_the_go2_run_s_two_certified_checkpoints(self) -> None:
        cfg = _cfg()
        early = pin_command_envelope(cfg, 1400, STEPS)
        self.assertEqual(early["commands"]["lin_vel_x"], [-1.0, 1.0])
        self.assertIn("stage 1 of 3", early["basis"])
        cfg = _cfg()
        final = pin_command_envelope(cfg, 7999, STEPS)
        self.assertEqual(final["commands"]["lin_vel_x"], [-1.5, 2.0])
        self.assertEqual(final["commands"]["ang_vel_z"], [-0.7, 0.7])
        self.assertEqual(final["commands"]["lin_vel_y"], [-1.0, 1.0])
        self.assertIn("stage 2 of 3, reached by iteration 7999", final["basis"])
        # The curriculum is gone, so a fresh env cannot move the ranges back.
        self.assertEqual(cfg.curriculum, {})

    def test_beyond_the_last_stage_and_without_a_curriculum(self) -> None:
        cfg = _cfg()
        self.assertEqual(
            pin_command_envelope(cfg, 12000, STEPS)["commands"]["lin_vel_x"],
            [-2.0, 3.0],
        )
        cfg = _cfg(with_curriculum=False)
        out = pin_command_envelope(cfg, 7999, STEPS)
        self.assertEqual(out["commands"]["lin_vel_x"], [-1.0, 1.0])
        self.assertIn("no curriculum", out["basis"])
        cfg = _cfg()
        self.assertIn(
            "iteration unknown", pin_command_envelope(cfg, None, STEPS)["basis"]
        )
