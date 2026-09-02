"""The rollout->dataset writer's pure parts (D2): the failure row, the
state names off an observation manager, the checkpoint stamp, the
export descriptor - everything a batch says about itself, pinned
without a GPU."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from rq_pipeline.collect.demo_export import ExportSpec
from rq_pipeline.collect.kitting_export import DemoLayout

from rq_mjlab.walk_press import (
    ACTION_SEMANTICS,
    CAMERA_KEY,
    checkpoint_stamp,
    failure_row,
    state_names,
)
from rq_mjlab.walk_verdict import EpisodeOutcome, WorldEpisode


class _Manager:
    """The two observation-manager attributes state_names reads."""

    active_terms: ClassVar = {"actor": ["base_ang_vel", "joint_pos", "command"]}
    group_obs_term_dim: ClassVar = {"actor": [(3,), (14,), (3,)]}


class StateNames(unittest.TestCase):
    def test_one_name_per_component_in_term_order(self) -> None:
        names = state_names(_Manager())
        self.assertEqual(len(names), 3 + 14 + 3)
        self.assertEqual(
            names[:3], ["base_ang_vel[0]", "base_ang_vel[1]", "base_ang_vel[2]"]
        )
        self.assertEqual(names[3], "joint_pos[0]")
        self.assertEqual(names[-1], "command[2]")


class FailureRows(unittest.TestCase):
    def test_a_discard_keeps_its_command_and_why(self) -> None:
        fell = WorldEpisode(
            EpisodeOutcome(steps=412, fell=True, mean_err=0.05, mean_cmd=0.3),
            command=[0.3, 0.0, 0.1],
        )
        row = failure_row(77, 4, fell)
        self.assertEqual((row["batch_seed"], row["world"]), (77, 4))
        self.assertTrue(row["fell"])
        self.assertEqual(row["command"], [0.3, 0.0, 0.1])
        # JSON-clean: the file is one row per line, nothing numpy.
        json.dumps(row)


class TheExpertStamp(unittest.TestCase):
    def test_the_checkpoint_bytes_are_the_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "model_10.pt"
            a.write_bytes(b"weights-a")
            first = checkpoint_stamp(a)
            self.assertTrue(first.startswith("model_10@"))
            a.write_bytes(b"weights-b")
            self.assertNotEqual(first, checkpoint_stamp(a))


class TheExportDescriptor(unittest.TestCase):
    def test_round_trips_and_names_its_file(self) -> None:
        spec = ExportSpec(
            state_width=2,
            state_names=["a", "b"],
            cameras=[CAMERA_KEY],
            instruction="walk",
            bundle="microduck",
            action_semantics=ACTION_SEMANTICS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = spec.write_to(Path(tmp))
            self.assertEqual(path.name, DemoLayout.EXPORT_FILE)
            self.assertEqual(ExportSpec.read_from(Path(tmp)), spec)

    def test_a_batch_without_one_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                ExportSpec.read_from(Path(tmp))
            self.assertIn(DemoLayout.EXPORT_FILE, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
