"""The tracking export's pure parts (docs/87 §8): the manifest's tracking
section read from the task's own config, and the start-up draws with the
feet's geoms resolved. The export itself needs the checkpoint and the
reference, and ran on the court dance (the record
g1-court-dance-gated-2026-11-23)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from trainnr_mjlab.tracking_export import randomization_of, tracking_block


def _event(params: dict, **kw: object) -> NS:
    return NS(params=params, **kw)


def _cfgs() -> tuple[NS, NS]:
    events = {
        "encoder_bias": _event({"bias_range": (-0.01, 0.01)}),
        "base_com": _event(
            {
                "asset_cfg": NS(body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 2: (-0.05, 0.05)},
            }
        ),
        "foot_friction": _event(
            {
                "asset_cfg": NS(geom_names=r"^(left|right)_foot[1-2]_collision$"),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            }
        ),
    }
    motion = NS(
        anchor_body_name="torso_link",
        body_names=("pelvis", "torso_link", "left_wrist"),
        joint_position_range=(-0.1, 0.1),
        pose_range={},
        velocity_range={},
    )
    terminations = {
        "anchor_pos": NS(params={"threshold": 0.25}),
        "anchor_ori": NS(params={"threshold": 0.8}),
        "ee_body_pos": NS(params={"threshold": 0.25, "body_names": ("left_wrist",)}),
    }
    play = NS(commands={"motion": motion}, terminations=terminations, events=events)
    trained_motion = NS(
        pose_range={"x": (-0.05, 0.05)}, velocity_range={"z": (-0.2, 0.2)}
    )
    noise = NS(n_min=-0.5, n_max=0.5)
    training = NS(
        commands={"motion": trained_motion},
        events={
            "push_robot": NS(
                interval_range_s=(1.0, 3.0),
                params={"velocity_range": {"x": (-0.5, 0.5)}},
            )
        },
        observations={
            "actor": NS(terms={"base_lin_vel": NS(noise=noise), "actions": NS()})
        },
    )
    return play, training


GEOMS = [
    "floor",
    "left_foot1_collision",
    "left_foot2_collision",
    "right_foot1_collision",
    "left_foot3_collision",
]


class TheSection(unittest.TestCase):
    def test_every_number_comes_from_the_task(self) -> None:
        play, training = _cfgs()
        command = NS(motion=NS(time_step_total=1095))
        block = tracking_block(play, training, command, 0.9, GEOMS, lag_ms=20.0)
        self.assertEqual(block["frames"], 1095)
        self.assertEqual(block["terminations"]["end_effectors"], ["left_wrist"])
        self.assertEqual(
            block["start"]["pose_range"], {"x": [-0.05, 0.05]}
        )  # training's
        self.assertEqual(block["start"]["soft_limit_factor"], 0.9)
        self.assertEqual(block["pushes"]["interval_s"], [1.0, 3.0])
        self.assertEqual(block["noise"], {"base_lin_vel": [-0.5, 0.5]})
        self.assertEqual(block["lag_ms"], 20.0)

    def test_the_feet_are_resolved_and_a_draw_the_gate_cannot_make_is_refused(
        self,
    ) -> None:
        play, _ = _cfgs()
        draws = randomization_of(play, GEOMS)
        self.assertEqual(
            draws["foot_friction"]["geoms"],
            ["left_foot1_collision", "left_foot2_collision", "right_foot1_collision"],
        )
        self.assertEqual(
            draws["body_com"]["ranges"], {"0": [-0.025, 0.025], "2": [-0.05, 0.05]}
        )
        play.events["foot_friction"].params["shared_random"] = False
        with self.assertRaises(SystemExit):
            randomization_of(play, GEOMS)


if __name__ == "__main__":
    unittest.main()
