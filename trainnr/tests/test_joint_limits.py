"""Limits over several joints (#121), read as the Microduck Nano firmware
reads its own (`safety.cpp`): rules, step tables and grids, a right leg in
the left leg's signs, the stricter neighbour between grid values."""

from __future__ import annotations

import unittest

import numpy as np

from trainnr.robot.joint_limits import SCHEMA, JointLimits


def _limits(*items: dict) -> JointLimits:
    return JointLimits.from_dict({"schema": SCHEMA, "limits": list(items)})


def _rad(**deg: float) -> dict[str, float]:
    return {k: float(np.radians(v)) for k, v in deg.items()}


INTERLOCK = {
    "name": "interlock",
    "form": "rule",
    "when": [
        {"joint": "yaw", "outside": [-1, 1]},
        {"joint": "roll", "outside": [-1, 1]},
    ],
}
KNEE_ANKLE = {
    "name": "knee ankle",
    "form": "table",
    "joint": "ankle",
    "over": "knee",
    "from": 30,
    "minimum_by": [[40, -60], [50, -45], [68, 0]],
}


class TheForms(unittest.TestCase):
    def test_a_rule_holds_only_when_every_condition_does(self) -> None:
        limits = _limits(INTERLOCK)
        self.assertEqual(limits.violations(_rad(yaw=0.5, roll=10)), [])
        self.assertEqual(limits.violations(_rad(yaw=5, roll=0.5)), [])
        self.assertEqual(limits.violations(_rad(yaw=5, roll=10)), ["interlock"])

    def test_a_table_reads_the_first_row_at_or_above(self) -> None:
        limits = _limits(KNEE_ANKLE)
        self.assertEqual(limits.violations(_rad(knee=30, ankle=-80)), [])  # not past 30
        self.assertEqual(limits.violations(_rad(knee=35, ankle=-61)), ["knee ankle"])
        self.assertEqual(limits.violations(_rad(knee=40, ankle=-60)), [])
        self.assertEqual(limits.violations(_rad(knee=41, ankle=-50)), ["knee ankle"])
        self.assertEqual(limits.violations(_rad(knee=80, ankle=-1)), ["knee ankle"])

    def test_signs_put_a_right_leg_in_the_left_legs_signs(self) -> None:
        right = dict(KNEE_ANKLE, name="right", signs={"knee": -1, "ankle": -1})
        limits = _limits(right)
        self.assertEqual(limits.violations(_rad(knee=-45, ankle=50)), ["right"])
        self.assertEqual(limits.violations(_rad(knee=-45, ankle=-50)), [])

    def test_a_grid_takes_the_stricter_neighbour_and_its_ends(self) -> None:
        grid = {
            "name": "grid",
            "form": "grid",
            "axes": ["a", "b"],
            "values": {"a": [0, 10, 20], "b": [0, 10]},
            "forbidden": [[1, 1]],
        }
        limits = _limits(grid)
        self.assertEqual(limits.violations(_rad(a=10, b=10)), ["grid"])
        self.assertEqual(limits.violations(_rad(a=5, b=5)), ["grid"])  # between
        self.assertEqual(limits.violations(_rad(a=0, b=5)), [])  # on a free value
        self.assertEqual(
            limits.violations(_rad(a=10.0005, b=10)), ["grid"]
        )  # within eps
        self.assertEqual(limits.violations(_rad(a=20.5, b=50)), [])  # beyond: the end
        self.assertEqual(limits.violations(_rad(a=15, b=50)), ["grid"])

    def test_a_batch_of_poses_reads_in_one_call(self) -> None:
        limits = _limits(INTERLOCK, KNEE_ANKLE)
        q = np.radians([[0, 0, 0, 0], [5, 10, 35, -61]])
        hit = limits.violated(q, ["yaw", "roll", "knee", "ankle"])
        self.assertEqual(hit.tolist(), [[False, False], [True, True]])

    def test_a_malformed_file_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "form is one of"):
            _limits({"name": "x", "form": "polygon"})
        with self.assertRaisesRegex(ValueError, "one of above"):
            _limits({"name": "x", "form": "rule", "when": [{"joint": "a"}]})
        with self.assertRaisesRegex(ValueError, "outside the"):
            _limits(
                {
                    "name": "g",
                    "form": "grid",
                    "axes": ["a"],
                    "values": {"a": [0, 1]},
                    "forbidden": [[2]],
                }
            )
        with self.assertRaisesRegex(KeyError, "lack: knee"):
            _limits(KNEE_ANKLE).violated(np.zeros((1, 1)), ["ankle"])


if __name__ == "__main__":
    unittest.main()
