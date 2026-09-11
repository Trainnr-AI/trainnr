"""Our manifest as Unitree's deploy.yaml, and the refusal that is the
deployability check: terms matched on what they compute, never on a
name."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from rq_pipeline.deploy.manifest import (
    SOURCE_COMMAND_TWIST,
    SOURCE_GAIT_PHASE,
    SOURCE_IMU_ANG_VEL,
    SOURCE_IMU_LIN_VEL,
    SOURCE_JOINT_POS_REL,
    SOURCE_JOINT_VEL_REL,
    SOURCE_LAST_ACTION,
    SOURCE_PROJECTED_GRAVITY,
    Manifest,
)
from rq_pipeline.deploy.unitree_yaml import (
    UNITREE_TERMS,
    NotDeployableError,
    deployable_actor_terms,
    unitree_deploy,
    unitree_observations,
)

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


def _manifest(observations: list[dict[str, Any]], **overrides: object) -> Manifest:
    raw: dict[str, Any] = {
        "control": {
            "physics_timestep_s": 0.005,
            "decimation": 4,
            "control_hz": 50,
            "episode_length_s": 20.0,
        },
        "joints": {
            "policy_order": [f"j{i}" for i in range(12)],
            "action_to_ctrl": list(range(12)),
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
    raw.update(overrides)
    return Manifest(root=Path("/nowhere"), raw=raw)


DEPLOYABLE = [
    {"name": "base_ang_vel", "width": 3, "source": SOURCE_IMU_ANG_VEL},
    {"name": "projected_gravity", "width": 3, "source": SOURCE_PROJECTED_GRAVITY},
    {"name": "command", "width": 3, "source": SOURCE_COMMAND_TWIST},
    {
        "name": "phase",
        "width": 2,
        "source": SOURCE_GAIT_PHASE,
        "params": {"period": 0.6, "command_name": "twist"},
    },
    {"name": "joint_pos", "width": 12, "source": SOURCE_JOINT_POS_REL},
    {"name": "joint_vel", "width": 12, "source": SOURCE_JOINT_VEL_REL, "scale": 0.05},
    {"name": "actions", "width": 12, "source": SOURCE_LAST_ACTION},
]


class TheWriter(unittest.TestCase):
    def test_the_deployable_actor_becomes_their_file(self) -> None:
        out = unitree_deploy(_manifest(DEPLOYABLE))
        self.assertEqual(set(out), THEIR_TOP_KEYS)
        self.assertEqual(out["step_dt"], 0.02)
        self.assertEqual(list(out["observations"]), [t.theirs for t in UNITREE_TERMS])
        self.assertEqual(out["observations"]["gait_phase"]["params"], {"period": 0.6})
        self.assertEqual(
            out["observations"]["velocity_commands"]["params"],
            {"command_name": "base_velocity"},
        )
        self.assertEqual(out["observations"]["joint_vel_rel"]["scale"], [0.05] * 12)
        self.assertEqual(
            out["commands"]["base_velocity"]["ranges"]["lin_vel_x"], [-1.5, 2.0]
        )
        self.assertIsNone(out["commands"]["base_velocity"]["ranges"]["heading"])
        self.assertIsNone(out["actions"]["JointPositionAction"]["clip"])

    def test_the_actor_order_is_theirs(self) -> None:
        self.assertEqual(
            deployable_actor_terms(),
            (
                "base_ang_vel",
                "projected_gravity",
                "command",
                "phase",
                "joint_pos",
                "joint_vel",
                "actions",
            ),
        )

    def test_a_term_their_runtime_lacks_is_refused_by_source(self) -> None:
        with self.assertRaises(NotDeployableError) as caught:
            unitree_observations(
                _manifest(
                    [{"name": "base_lin_vel", "width": 3, "source": SOURCE_IMU_LIN_VEL}]
                )
            )
        self.assertIn("base_lin_vel", str(caught.exception))
        self.assertIn(SOURCE_IMU_LIN_VEL, str(caught.exception))
        # A term merely NAMED like theirs is refused too: the source decides.
        with self.assertRaises(NotDeployableError):
            unitree_observations(
                _manifest([{"name": "phase", "width": 2, "source": SOURCE_IMU_LIN_VEL}])
            )

    def test_a_missing_parameter_is_refused_by_name(self) -> None:
        with self.assertRaises(NotDeployableError) as caught:
            unitree_observations(
                _manifest([{"name": "phase", "width": 2, "source": SOURCE_GAIT_PHASE}])
            )
        self.assertIn("period", str(caught.exception))

    def test_a_robot_without_an_sdk_order_is_refused(self) -> None:
        manifest = _manifest(
            DEPLOYABLE,
            joints={
                "policy_order": ["j"],
                "action_to_ctrl": [0],
                "default_pos": [0.0],
            },
        )
        with self.assertRaises(NotDeployableError) as caught:
            unitree_deploy(manifest)
        self.assertIn("joint_ids_map", str(caught.exception))
