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
        from trainnr.collect.scripted_demos import select_cameras  # noqa: PLC0415

        declared = (_Cam("front"), _Cam("top"))
        self.assertEqual(select_cameras(declared, None), declared)

    def test_a_subset_keeps_the_asked_order_and_refuses_unknown(self) -> None:
        from trainnr.collect.scripted_demos import select_cameras  # noqa: PLC0415

        declared = (_Cam("front"), _Cam("top"), _Cam("wrist"))
        chosen = select_cameras(declared, ["wrist", "front"])
        self.assertEqual([c.key for c in chosen], ["wrist", "front"])
        with self.assertRaisesRegex(ValueError, "ghost"):
            select_cameras(declared, ["front", "ghost"])


@needs_sim
class SizableLiftStudy(unittest.TestCase):
    def test_trials_come_from_the_spec_and_change_the_stamp(self) -> None:
        from trainnr.tasks.so101 import (  # noqa: PLC0415
            LIFT_STUDY_SPEC,
            build_lift_study,
        )

        default = build_lift_study()
        eighty = build_lift_study(spec=replace(LIFT_STUDY_SPEC, trials=80))
        self.assertEqual(default.protocol.trials, 40)
        self.assertEqual(eighty.protocol.trials, 80)
        self.assertNotEqual(default.stamp, eighty.stamp)

    def test_the_env_sizes_the_study_to_the_evaluation(self) -> None:
        from trainnr.envs.gymnasium_env import make_env  # noqa: PLC0415

        env = make_env("lift-study", trials=6)
        try:
            self.assertEqual(env.protocol.trials, 6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()


class TheArmSubset(unittest.TestCase):
    """--arms lets a second pod take the list from the other end; a name
    the study does not have is refused, not silently skipped."""

    def test_subset_and_refusal(self) -> None:
        import importlib.util  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        tool = Path(__file__).resolve().parents[2] / "tools" / "study.py"
        if str(tool.parent) not in sys.path:  # study.py imports its _lab sibling
            sys.path.insert(0, str(tool.parent))
        spec_ = importlib.util.spec_from_file_location("study_tool", tool)
        module = importlib.util.module_from_spec(spec_)
        assert spec_.loader is not None
        spec_.loader.exec_module(module)
        selected_arms = module.selected_arms

        spec = {
            "arms": {"n8": {"episodes": 8}, "n16": {"episodes": 16}},
            "replicates": 2,
        }
        every = selected_arms(spec, None)
        self.assertEqual(sorted(every), ["n16#1", "n16#2", "n8#1", "n8#2"])
        self.assertEqual(
            list(selected_arms(spec, ["n16#2", "n8#1"])), ["n16#2", "n8#1"]
        )
        with self.assertRaisesRegex(KeyError, "unknown arms"):
            selected_arms(spec, ["n32#1"])
