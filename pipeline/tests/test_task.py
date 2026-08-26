"""The one task shape: what every builder must say about its rig."""

import unittest

from rq_pipeline.evaluate.harness import EpisodeProtocol
from rq_pipeline.evaluate.vision import ARMNETBENCH_CAMERAS
from rq_pipeline.tasks.task import Task

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
        )
        self.assertEqual(task.state_width, 6)
        self.assertIsNone(task.target)

    def test_missing_facts_are_refused(self) -> None:
        for kwargs in (
            {"state_width": 0, "instruction": "x", "cameras": ARMNETBENCH_CAMERAS},
            {"state_width": 6, "instruction": "  ", "cameras": ARMNETBENCH_CAMERAS},
            {"state_width": 6, "instruction": "x", "cameras": ()},
        ):
            with self.assertRaises(ValueError):
                Task(name="t", spec=None, protocol=PROTOCOL, **kwargs)


if __name__ == "__main__":
    unittest.main()
