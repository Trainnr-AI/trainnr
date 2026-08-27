"""Roundtrip: rig recording → aligned frames → LeRobot dataset → reload.

Runs only when the `train` extra is installed (`uv sync --extra train`);
the fast pre-commit gate skips it, the full verify should not.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.bundles.profile import load_profile
from rq_pipeline.collect.frames import AlignedEpisode, align
from rq_pipeline.collect.lerobot_export import PROVENANCE_FILE, export_episode
from rq_pipeline.collect.wire import parse_recording
from tests._extras import needs_train

REPO_ROOT = Path(__file__).resolve().parents[2]
CHASE_RECORDING = REPO_ROOT / "recordings" / "chase-arm-2026-08-17.wire"
RIG = load_profile(REPO_ROOT / "robots" / "rig-drivetrain")

EXPECTED_FRAMES = 27
ACTION_DIMENSIONS = 4


@needs_train
class ExportRoundtrip(unittest.TestCase):
    def test_chase_recording_roundtrips_through_lerobot(self) -> None:
        # In-test import is load-bearing: a top-level lerobot import would
        # fail at DISCOVERY time when the extra is absent, before skipUnless
        # can fire.
        from lerobot.datasets.lerobot_dataset import (  # noqa: PLC0415
            LeRobotDataset,
        )

        recording = parse_recording(CHASE_RECORDING)
        episode = align(recording, stamp("chase", CHASE_RECORDING))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            export_episode(episode, root, RIG)

            provenance = json.loads(
                (root / PROVENANCE_FILE).read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["source"], episode.source)
            self.assertEqual(provenance["robot_profile"], RIG.name)
            self.assertEqual(provenance["frames"], EXPECTED_FRAMES)

            reloaded = LeRobotDataset("rq-pipeline/rig", root=root)
            self.assertEqual(reloaded.num_frames, EXPECTED_FRAMES)
            self.assertEqual(reloaded.num_episodes, 1)
            item = reloaded[0]
            self.assertEqual(tuple(item["action"].shape), (ACTION_DIMENSIONS,))
            # Channel-first torch convention: (3, height, width).
            self.assertEqual(item["observation.images.front"].shape[0], 3)
            # The real wire clock survived the synthetic-fps compromise.
            self.assertAlmostEqual(item["wire_timestamp_s"].item(), 70.5, delta=0.01)

    def test_empty_episode_refused(self) -> None:
        empty = AlignedEpisode(
            source="empty@000000000000", frames=(), dropped_incomplete=0
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(ValueError),
        ):
            export_episode(empty, Path(directory) / "dataset", RIG)


if __name__ == "__main__":
    unittest.main()
