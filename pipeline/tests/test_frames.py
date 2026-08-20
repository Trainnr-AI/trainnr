"""Alignment of recordings into training frames, against the real fixture."""

import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.bundles.profile import load_profile
from rq_pipeline.collect.frames import align, rgb565_to_rgb888
from rq_pipeline.collect.wire import parse_recording

REPO_ROOT = Path(__file__).resolve().parents[2]
CHASE_RECORDING = REPO_ROOT / "recordings" / "chase-arm-2026-08-17.wire"

# The robot's numbers come from the robot's bundle, even in tests.
RIG = load_profile(REPO_ROOT / "robots" / "rig-drivetrain")


class Rgb565Expansion(unittest.TestCase):
    def test_extremes_stay_full_scale(self) -> None:
        # Bit replication, not zero-padding: white must stay 255, not 248.
        self.assertEqual(rgb565_to_rgb888(0xFFFF), (255, 255, 255))
        self.assertEqual(rgb565_to_rgb888(0x0000), (0, 0, 0))
        self.assertEqual(rgb565_to_rgb888(0xF800), (255, 0, 0))
        self.assertEqual(rgb565_to_rgb888(0x07E0), (0, 255, 0))
        self.assertEqual(rgb565_to_rgb888(0x001F), (0, 0, 255))


class AlignChaseRecording(unittest.TestCase):
    def test_every_image_aligns_and_time_advances(self) -> None:
        # Fixture ground truth, established 2026-08-21: all 27 images pair
        # with a following status and servo command; the status seq at
        # 50 Hz gives strictly increasing timestamps spanning ~104 s.
        recording = parse_recording(CHASE_RECORDING)
        episode = align(recording, stamp("chase", CHASE_RECORDING))
        self.assertEqual(len(episode.frames), 27)
        self.assertEqual(episode.dropped_incomplete, 0)
        timestamps = [frame.timestamp_s for frame in episode.frames]
        self.assertTrue(all(a < b for a, b in pairwise(timestamps)))
        self.assertAlmostEqual(timestamps[-1] - timestamps[0], 104.0, delta=1.0)
        for frame in episode.frames:
            self.assertEqual(len(frame.rgb888), frame.width * frame.height * 3)
            pan, tilt, grip, duty = frame.action
            # Servo pulses live inside the firmware's clamp band, which
            # the bundle profile carries; duty is a percentage.
            for pulse in (pan, tilt, grip):
                self.assertTrue(
                    RIG.servo_pulse_floor_us <= pulse <= RIG.servo_pulse_ceiling_us
                )
            self.assertTrue(0.0 <= duty <= RIG.duty_ceiling_percent)

    def test_unstamped_source_refused(self) -> None:
        recording = parse_recording(CHASE_RECORDING)
        with self.assertRaises(ValueError):
            align(recording, "chase")


class DroppedFrames(unittest.TestCase):
    def test_image_with_no_following_pair_is_dropped_not_half_filled(
        self,
    ) -> None:
        # An image completed at end-of-file has no status or servo after
        # it: the frame is dropped and counted, never emitted half-empty.
        content = "\n".join(
            [
                "n=100 pose x=+0.0 y=+0.0 th=+0.0 ticks L=0 R=0 errL=0 errR=0 duty=0%",
                "# IMG 2 1 rgb565",
                "# FFFF0000",
            ]
        )
        with tempfile.NamedTemporaryFile("w", suffix=".wire", delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        recording = parse_recording(path)
        episode = align(recording, stamp("tail", path))
        path.unlink()
        self.assertEqual(len(recording.images), 1)
        self.assertEqual(len(episode.frames), 0)
        self.assertEqual(episode.dropped_incomplete, 1)


if __name__ == "__main__":
    unittest.main()
