"""A robot's limits over several joints (#121) in torch: the same answers
as `trainnr.robot.joint_limits` (the firmware's reading), on every world
at once."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from trainnr.robot.joint_limits import SCHEMA, JointLimits

from trainnr_mjlab.limits_term import TorchLimits

LIMITS = {
    "schema": SCHEMA,
    "limits": [
        {
            "name": "interlock",
            "form": "rule",
            "when": [
                {"joint": "yaw", "outside": [-1, 1]},
                {"joint": "roll", "outside": [-1, 1], "sign": -1},
            ],
        },
        {
            "name": "table",
            "form": "table",
            "joint": "ankle",
            "over": "knee",
            "from": 30,
            "minimum_by": [[40, -60], [50, -45], [68, 0]],
            "maximum_by": [[68, 80]],
            "signs": {"knee": -1},
        },
        {
            "name": "grid",
            "form": "grid",
            "axes": ["knee", "roll", "yaw"],
            "values": {"knee": [-40, 0, 40, 68], "roll": [-10, 0, 10], "yaw": [0]},
            "forbidden": [[1, 2, 0], [3, 0, 0]],
        },
    ],
}
JOINTS = ["yaw", "roll", "knee", "ankle"]


class TheTorchLimits(unittest.TestCase):
    def test_it_agrees_with_the_reference_on_random_poses(self) -> None:
        limits = JointLimits.from_dict(LIMITS)
        rng = np.random.default_rng(3)
        deg = rng.uniform(-90, 90, size=(20000, 4))
        # poses on grid values and table rows too, where the rules turn
        deg[:5000, 2] = rng.choice([-40, 0, 40, 68, 30, 50], 5000)
        deg[:5000, 1] = rng.choice([-10, 0, 10, 1, -1], 5000)
        q = np.radians(deg)
        reference = limits.violated(q, JOINTS)
        checker = TorchLimits(limits, JOINTS, "cpu")
        mine = checker.violated(torch.tensor(q, dtype=torch.float32)).numpy()
        # float32 against float64 may differ only at a boundary to the
        # last bit: none in practice; a handful would say so
        self.assertLessEqual(int((mine != reference).sum()), 3)
        self.assertGreater(int(reference.any(axis=1).sum()), 1000)

    def test_a_joint_the_robot_lacks_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(KeyError, "lacks: ankle"):
            TorchLimits(JointLimits.from_dict(LIMITS), ["yaw", "roll", "knee"], "cpu")


if __name__ == "__main__":
    unittest.main()
