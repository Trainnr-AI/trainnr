"""The wire parser, including cross-language conformance against the Rust gate."""

import tempfile
import unittest
from pathlib import Path

from trainnr.collect.wire import (
    parse_camera_note,
    parse_image_header,
    parse_image_row,
    parse_recording,
    parse_servo_note,
    parse_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHASE_RECORDING = REPO_ROOT / "recordings" / "chase-arm-2026-08-17.wire"

STATUS_LINE = (
    "n=3325 pose x=+0.000 y=+0.105 th=+1.620  ticks L=-36 R=5494  "
    "errL=104 errR=49  duty=7%"
)


class LineParsers(unittest.TestCase):
    def test_status_roundtrip(self) -> None:
        frame = parse_status(STATUS_LINE)
        assert frame is not None
        self.assertEqual(frame.seq, 3325)
        self.assertEqual(frame.ticks_left, -36)
        self.assertEqual(frame.ticks_right, 5494)
        self.assertEqual(frame.duty_percent, 7)
        self.assertFalse(frame.stalled)

    def test_stalled_banner_sets_flag(self) -> None:
        frame = parse_status(
            STATUS_LINE + "  *** STALLED: commanded but not moving ***"
        )
        assert frame is not None
        self.assertTrue(frame.stalled)

    def test_missing_field_rejects_line_whole(self) -> None:
        # Half a pose treated as complete is worse than a dropped frame.
        truncated = STATUS_LINE.replace("errR=49  duty=7%", "errR=")
        self.assertIsNone(parse_status(truncated))

    def test_image_header_and_row(self) -> None:
        self.assertEqual(parse_image_header("# IMG 30 20 rgb565"), (30, 20))
        self.assertIsNone(parse_image_header("# IMG 0 20 rgb565"))
        self.assertEqual(parse_image_row("# FFFF0000F800", 3), (0xFFFF, 0, 0xF800))
        self.assertIsNone(parse_image_row("# FFFF0000F800", 4))  # wrong width
        self.assertIsNone(parse_image_row("# servo us 1 2 3", 3))  # prose

    def test_servo_note(self) -> None:
        self.assertEqual(
            parse_servo_note("servo us 1500 1500 1500"), (1500, 1500, 1500)
        )
        self.assertIsNone(parse_servo_note("servo us 1500 1500"))

    def test_camera_note_blob(self) -> None:
        note = parse_camera_note(
            "camera pid=0x76 OK | 4 fps | blob x-0.38 y+0.23 area 2726"
        )
        assert note is not None and note.blob is not None
        self.assertAlmostEqual(note.blob[0], -0.38)
        self.assertAlmostEqual(note.blob[1], 0.23)
        self.assertEqual(note.blob[2], 2726)
        no_blob = parse_camera_note("camera pid=0x76 OK | 4 fps | no blob")
        assert no_blob is not None
        self.assertIsNone(no_blob.blob)


class TornImages(unittest.TestCase):
    def test_header_while_open_and_eof_both_tear(self) -> None:
        content = "\n".join(
            [
                "# IMG 2 2 rgb565",
                "# FFFF0000",
                "# IMG 2 2 rgb565",  # tears the first
                "# FFFF0000",
                "# 0000FFFF",  # completes the second
                "# IMG 2 2 rgb565",
                "# FFFF0000",  # EOF tears the third
            ]
        )
        with tempfile.NamedTemporaryFile("w", suffix=".wire", delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        recording = parse_recording(path)
        path.unlink()
        self.assertEqual(len(recording.images), 1)
        self.assertEqual(recording.torn_images, 2)


class RustGateConformance(unittest.TestCase):
    def test_reproduces_rig_replay_counts_exactly(self) -> None:
        # Two parsers, two languages, one committed fixture. The Rust gate
        # (rig_replay, pinned in tools/verify.sh) printed this exact line
        # for this exact file on 2026-08-17. If either implementation
        # drifts from the format, this equality breaks first.
        recording = parse_recording(CHASE_RECORDING)
        self.assertEqual(
            recording.summary(),
            "5523 status (21 stalled), 27 images (0 torn), "
            "215 servo, 36 notes, 1 unparsable",
        )

    def test_the_recording_tells_the_logged_story(self) -> None:
        # the log records this session's shape; the parser must see it too:
        # the arm genuinely tracked (many distinct pulse triples), and all
        # stalls latched at the very end (battery switch-off, not a fault).
        recording = parse_recording(CHASE_RECORDING)
        self.assertGreater(len(set(recording.servo_pulses)), 100)
        stalled_seqs = [s.seq for s in recording.statuses if s.stalled]
        final_seq = recording.statuses[-1].seq
        self.assertTrue(all(seq > final_seq - 25 for seq in stalled_seqs))


if __name__ == "__main__":
    unittest.main()
