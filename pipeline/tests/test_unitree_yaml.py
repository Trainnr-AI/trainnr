"""Our manifest as Unitree's deploy.yaml, and the refusal that is the
deployability check."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from rq_pipeline.deploy.manifest import Manifest
from rq_pipeline.deploy.unitree_yaml import (
    NotDeployableError,
    unitree_deploy,
    unitree_observations,
)

GO2_DEPLOY = Path("/home/prakhar-pc/robotiq/projects/go2-walk/deploy/go2-c1-deploy")
THEIR_TOP_KEYS = {
    "joint_ids_map",
    "step_dt",
    "stiffness",
    "damping",
    "default_joint_pos",
    "commands",
    "actions",
    "observations",
}


def _manifest(observations: list[dict]) -> Manifest:
    raw = {
        "control": {"control_hz": 50, "decimation": 4, "episode_length_s": 20.0},
        "joints": {
            "sdk_order_map": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
            "stiffness": [20.0] * 12,
            "damping": [1.0] * 12,
            "default_pos": [0.0] * 12,
        },
        "action": {"scale": [0.25] * 12, "offset": [0.0] * 12, "clip": None},
        "commands": {
            "twist": {
                "lin_vel_x": [-1.5, 2.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_z": [-0.7, 0.7],
                "heading": None,
            }
        },
        "observations": observations,
    }
    return Manifest(root=Path("/nowhere"), raw=raw)


DEPLOYABLE = [
    {"name": "base_ang_vel", "width": 3, "scale": 1.0},
    {"name": "projected_gravity", "width": 3, "scale": 1.0},
    {"name": "command", "width": 3, "scale": 1.0},
    {"name": "phase", "width": 2, "scale": 1.0, "params": {"period": 0.6}},
    {"name": "joint_pos", "width": 12, "scale": 1.0},
    {"name": "joint_vel", "width": 12, "scale": 0.05},
    {"name": "actions", "width": 12, "scale": 1.0},
]


class TheWriter(unittest.TestCase):
    def test_the_deployable_actor_becomes_their_file(self) -> None:
        out = unitree_deploy(_manifest(DEPLOYABLE))
        self.assertEqual(set(out), THEIR_TOP_KEYS)
        self.assertEqual(out["step_dt"], 0.02)
        self.assertEqual(
            list(out["observations"]),
            [
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "gait_phase",
                "joint_pos_rel",
                "joint_vel_rel",
                "last_action",
            ],
        )
        self.assertEqual(out["observations"]["gait_phase"]["params"], {"period": 0.6})
        self.assertEqual(
            out["observations"]["velocity_commands"]["params"],
            {"command_name": "base_velocity"},
        )
        self.assertEqual(out["observations"]["joint_vel_rel"]["scale"], [0.05] * 12)
        self.assertEqual(
            out["commands"]["base_velocity"]["ranges"]["lin_vel_x"], [-1.5, 2.0]
        )

    def test_a_term_their_runtime_lacks_is_refused_by_name(self) -> None:
        with self.assertRaises(NotDeployableError) as caught:
            unitree_observations(_manifest([{"name": "base_lin_vel", "width": 3}]))
        self.assertIn("base_lin_vel", str(caught.exception))

    @unittest.skipUnless(GO2_DEPLOY.is_dir(), "the Go2 deployment lives on the box")
    def test_the_first_certified_go2_policy_is_not_deployable_there(self) -> None:
        raw = json.loads((GO2_DEPLOY / "deploy.json").read_text())
        with self.assertRaises(NotDeployableError):
            unitree_deploy(Manifest(root=GO2_DEPLOY, raw=raw))
