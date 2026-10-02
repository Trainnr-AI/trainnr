"""The one task shape: what every builder must say about its rig."""

import unittest
from pathlib import Path

from trainnr.evaluate.vision import ARMNETBENCH_CAMERAS
from trainnr.protocol import EpisodeProtocol
from trainnr.tasks.task import Task

BUNDLE = Path("robots") / "so101-nominal"
PROTOCOL = EpisodeProtocol(
    trials=1,
    steps=1,
    control_interval=1,
    perturb=lambda _trial, home: home,
    success=lambda _states, _sensors: True,
)


class TaskShape(unittest.TestCase):
    def test_a_task_carries_its_rig_and_its_sentence(self) -> None:
        task = Task(
            name="reach",
            spec=None,
            protocol=PROTOCOL,
            cameras=ARMNETBENCH_CAMERAS,
            state_width=6,
            instruction="reach the target",
            bundle_dir=BUNDLE,
        )
        self.assertEqual(task.state_width, 6)
        self.assertIsNone(task.target)
        self.assertEqual(task.bundle_dir.name, "so101-nominal")
        self.assertEqual(PROTOCOL.control_ticks, 1)

    def test_missing_facts_are_refused(self) -> None:
        for kwargs in (
            {"state_width": 0, "instruction": "x", "cameras": ARMNETBENCH_CAMERAS},
            {"state_width": 6, "instruction": "  ", "cameras": ARMNETBENCH_CAMERAS},
            {"state_width": 6, "instruction": "x", "cameras": ()},
        ):
            with self.assertRaises(ValueError):
                Task(
                    name="t", spec=None, protocol=PROTOCOL, bundle_dir=BUNDLE, **kwargs
                )


if __name__ == "__main__":
    unittest.main()
