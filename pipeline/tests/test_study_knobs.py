"""The study's two knobs the pods found missing (2026-09-04): a camera
subset for the press, refused by name when unknown, and a lift-study
that can declare any number of paired trials."""

from __future__ import annotations

import unittest
from dataclasses import replace

from tests._extras import needs_sim


class _Cam:
    def __init__(self, key: str) -> None:
        self.key = key


class CameraSubset(unittest.TestCase):
    def test_none_means_every_declared_camera(self) -> None:
        from rq_pipeline.collect.scripted_demos import select_cameras  # noqa: PLC0415

        declared = (_Cam("front"), _Cam("top"))
        self.assertEqual(select_cameras(declared, None), declared)

    def test_a_subset_keeps_the_asked_order_and_refuses_unknown(self) -> None:
        from rq_pipeline.collect.scripted_demos import select_cameras  # noqa: PLC0415

        declared = (_Cam("front"), _Cam("top"), _Cam("wrist"))
        chosen = select_cameras(declared, ["wrist", "front"])
        self.assertEqual([c.key for c in chosen], ["wrist", "front"])
        with self.assertRaisesRegex(ValueError, "ghost"):
            select_cameras(declared, ["front", "ghost"])


@needs_sim
class SizableLiftStudy(unittest.TestCase):
    def test_trials_come_from_the_spec_and_change_the_stamp(self) -> None:
        from rq_pipeline.tasks.so101 import (  # noqa: PLC0415
            LIFT_STUDY_SPEC,
            build_lift_study,
        )

        default = build_lift_study()
        eighty = build_lift_study(spec=replace(LIFT_STUDY_SPEC, trials=80))
        self.assertEqual(default.protocol.trials, 40)
        self.assertEqual(eighty.protocol.trials, 80)
        self.assertNotEqual(default.stamp, eighty.stamp)

    def test_the_env_sizes_the_study_to_the_evaluation(self) -> None:
        from rq_pipeline.envs.robotiq import make_env  # noqa: PLC0415

        env = make_env("lift-study", trials=6)
        try:
            self.assertEqual(env.protocol.trials, 6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
