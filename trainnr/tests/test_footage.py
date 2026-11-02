"""Footage (`trainnr.video.footage`, docs/86 §5.1): a video lands as it is,
with ffprobe's numbers and its first frame, stamped; refusals write
nothing; the index lists it and proves no loop state with it."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trainnr.project.index import index_project
from trainnr.project.kinds import Kind, detect, stamp_kind
from trainnr.project.locate import create_project
from trainnr.video.footage import FIRST_FRAME, FOOTAGE_FILE, SCHEMA, ingest_footage

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def clip(path: Path, seconds: int = 1, fps: int = 10, codec: str = "mpeg4") -> Path:
    """ffmpeg's own test pattern: 64x48, `fps`, `seconds` long, in `codec`."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate={fps}:duration={seconds}",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg and ffprobe")
class IngestFootage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.video = clip(self.tmp / "walk.mp4")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lands_the_video_its_numbers_and_its_first_frame(self) -> None:
        out = ingest_footage(self.tmp / "footage", self.video, origin="filmed")
        record = json.loads((out / FOOTAGE_FILE).read_text())
        self.assertEqual(out.name, "walk")
        self.assertEqual(record["schema"], SCHEMA)
        self.assertEqual((record["width"], record["height"]), (64, 48))
        self.assertEqual(record["fps"], 10.0)
        self.assertAlmostEqual(record["duration_s"], 1.0, places=1)
        self.assertEqual(record["codec"], "mpeg4")
        self.assertEqual(record["origin"], "filmed")
        self.assertEqual((out / record["video"]).read_bytes(), self.video.read_bytes())
        self.assertTrue((out / FIRST_FRAME).is_file())
        self.assertIs(detect(out), Kind.FOOTAGE)
        self.assertTrue(stamp_kind(Kind.FOOTAGE, out).startswith("walk@"))

    def test_refusals_write_nothing(self) -> None:
        footage = self.tmp / "footage"
        ingest_footage(footage, self.video)
        not_a_video = self.tmp / "notes.mp4"
        not_a_video.write_text("not a video")
        cases = (
            (FileExistsError, self.video, None),
            (ValueError, self.video, "a/b"),
            (ValueError, not_a_video, None),
            (FileNotFoundError, self.tmp / "missing.mp4", None),
        )
        for error, source, name in cases:
            with (
                self.subTest(error=error.__name__, source=source.name, name=name),
                self.assertRaises(error),
            ):
                ingest_footage(footage, source, name=name)
        self.assertEqual(sorted(p.name for p in footage.iterdir()), ["walk"])


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg and ffprobe")
class FootageInAProject(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_is_listed_and_proves_no_state(self) -> None:
        project = create_project(self.tmp / "p", "p")
        ingest_footage(project.footage, clip(self.tmp / "dance.mp4", seconds=2))
        index = index_project(project)
        (footage,) = index.by_kind(Kind.FOOTAGE)
        self.assertEqual(footage.path, "footage/dance")
        self.assertEqual(footage.summary["size"], "64x48")
        self.assertEqual(footage.summary["rate"], "10 fps")
        self.assertFalse(any(s.present for s in index.states))


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg and ffprobe")
class FootageInTheViewer(unittest.TestCase):
    """Show in viewer: Rerun's player for what it decodes, frames for AV1
    (the Studio's viewer has no AV1 decoder), and the first frame, with
    the reason, for a file nothing can play."""

    def setUp(self) -> None:
        try:
            import rerun as rr  # noqa: PLC0415
        except ImportError:  # pragma: no cover - the viz extra is absent
            self.skipTest("rerun not installed")
        self.rr = rr
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.project = create_project(self.tmp / "p", "p")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def shown(self, codec: str, *, damage: bool = False) -> tuple[str, str]:
        """The view a `codec` clip gets, and the record panel's text."""
        from trainnr.project import present  # noqa: PLC0415

        try:
            source = clip(self.tmp / f"{codec}.mp4", codec=codec)
        except subprocess.CalledProcessError:
            self.skipTest(f"this ffmpeg cannot encode {codec}")
        out = ingest_footage(self.project.footage, source)
        if damage:
            (out / "video.mp4").write_bytes(b"not a video any more")
        (artifact,) = index_project(self.project).by_kind(Kind.FOOTAGE)
        stream = self.rr.RecordingStream("trainnr-test")
        stream.memory_recording()
        records: list[str] = []
        original = present._doc

        def doc(markdown: str) -> object:
            records.append(markdown)
            return original(markdown)

        with mock.patch.object(present, "_doc", doc):
            shown = present._PRESENTERS[Kind.FOOTAGE](self.project, artifact, stream)
        return shown["view"], " ".join(records)

    def test_av1_comes_as_frames(self) -> None:
        self.assertEqual(self.shown("libaom-av1")[0], "frames at 15 fps")

    def test_h264_plays_in_the_viewer(self) -> None:
        self.assertEqual(self.shown("libx264")[0], "video")

    def test_an_unplayable_file_shows_its_first_frame_and_why(self) -> None:
        view, record = self.shown("libx264", damage=True)
        self.assertEqual(view, "first frame")
        self.assertIn("could not play it", record)


if __name__ == "__main__":
    unittest.main()
