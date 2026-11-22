"""A camera as a live capture source (#130): which addresses it takes,
that a password never reaches a record, ffmpeg's one read with two
outputs, the preview's JPEG splitting, and a real capture of a local
MJPEG stream into a project as stamped footage."""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from trainnr.robots.camera_capture import (
    CameraCapture,
    checked_url,
    ffmpeg_argv,
    jpegs,
    shown_url,
)
from trainnr.robots.capture import FOOTAGE, INGESTED, LISTENING, sources

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class Addresses(unittest.TestCase):
    def test_streams_and_devices_are_cameras_and_nothing_else_is(self) -> None:
        for url in (
            "http://10.0.0.2:8080/stream",
            "rtsp://cam.local/live",
            "https://h/x",
        ):
            self.assertEqual(checked_url(url), url)
        for url in ("file:///etc/passwd", "/etc/passwd", "ftp://h/x", "http:///x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                checked_url(url)
        if sys.platform.startswith("linux"):
            with self.assertRaisesRegex(FileNotFoundError, "no video device"):
                checked_url("/dev/video97")

    def test_a_password_never_reaches_a_record(self) -> None:
        self.assertEqual(
            shown_url("rtsp://admin:hunter2@10.0.0.2:554/live?x=1"),
            "rtsp://10.0.0.2:554/live?x=1",
        )
        self.assertEqual(shown_url("/dev/video0"), "/dev/video0")
        listener = CameraCapture(
            Path("f"), Path("s"), "look", url="http://u:p@10.0.0.2/stream"
        )
        self.assertEqual(listener.state.source, "camera:http://10.0.0.2/stream")
        self.assertEqual(listener.state.lands, FOOTAGE)

    def test_the_registry_offers_the_camera_with_its_url(self) -> None:
        (camera,) = [s for s in sources() if s["name"] == "camera"]
        self.assertEqual(camera["options"], ["url"])
        self.assertEqual(camera["lands"], FOOTAGE)


class OneReadTwoOutputs(unittest.TestCase):
    def test_the_window_caps_the_read_and_both_outputs_follow(self) -> None:
        argv = ffmpeg_argv("http://h/stream", Path("raw.mp4"), 60)
        self.assertLess(argv.index("-t"), argv.index("-i"))  # an input option
        self.assertIn("-use_wallclock_as_timestamps", argv)  # MJPEG has no clock
        self.assertEqual(argv.count("-map"), 2)
        self.assertIn("raw.mp4", argv)
        self.assertEqual(argv[-1], "pipe:1")
        self.assertNotIn("-nostdin", argv)  # `q` closes the footage
        self.assertIn("-rtsp_transport", ffmpeg_argv("rtsp://h/x", Path("r"), 1))
        device = ffmpeg_argv("/dev/video0", Path("r"), 1)
        self.assertEqual(device[device.index("-f") + 1], "v4l2")

    def test_pictures_split_across_reads_come_out_whole(self) -> None:
        one, two = b"\xff\xd8one\xff\xd9", b"\xff\xd8two\xff\xd9"
        stream = io.BytesIO(b"junk" + one + two)
        self.assertEqual(list(jpegs(stream, chunk=3)), [one, two])


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg and ffprobe")
class ALiveCapture(unittest.TestCase):
    """ffmpeg serves a test pattern as MJPEG over HTTP, the way a robot's
    board does; the capture shows it (no Studio here: the viewer port is
    one nothing listens on) and lands it as stamped footage."""

    def test_a_stream_lands_in_the_project_as_footage(self) -> None:
        from trainnr.project.index import index_project  # noqa: PLC0415
        from trainnr.project.ingest import capture  # noqa: PLC0415
        from trainnr.project.kinds import Kind  # noqa: PLC0415
        from trainnr.project.locate import create_project  # noqa: PLC0415
        from trainnr.video.footage import FOOTAGE_FILE, Footage  # noqa: PLC0415
        from trainnr.viz import VIEWER_BIND_ENV  # noqa: PLC0415

        port = _free_port()
        server = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=20",
                "-f", "mpjpeg", "-listen", "1", f"http://127.0.0.1:{port}/cam",
            ],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # fmt: skip
        self.addCleanup(server.kill)
        time.sleep(1.0)  # the server binds
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {VIEWER_BIND_ENV: f"127.0.0.1:{_free_port()}"}),
        ):
            project = create_project(Path(tmp) / "p", "p")
            listener = capture(
                project,
                "look",
                source="camera",
                url=f"http://who:pw@127.0.0.1:{port}/cam",
                python=[sys.executable],
            )
            state = listener.start(30)
            self.assertEqual(state.state, LISTENING)
            time.sleep(3)
            (row,) = index_project(project).in_progress
            self.assertEqual(row["kind"], Kind.FOOTAGE.value)
            self.assertIn("320x240", row["stage"])
            state = listener.stop()
            self.assertEqual(state.state, INGESTED, state.error)
            self.assertEqual(state.measured["width"], 320)
            self.assertGreater(state.measured["fps"], 10)
            record = Footage.read(project.footage / "look" / FOOTAGE_FILE)
            self.assertEqual(record.codec, "h264")
            self.assertGreater(record.duration_s, 1.0)
            self.assertNotIn("pw", record.origin)
            (artifact,) = index_project(project).by_kind(Kind.FOOTAGE)
            self.assertEqual(state.stamp, artifact.stamp)
            index_dir = project.root / ".index"
            self.assertEqual(sorted(p.name for p in index_dir.glob("capture-*")), [])
            written = (
                json.dumps(state.__dict__) + (index_dir / "capture.json").read_text()
            )
            self.assertNotIn("pw@", written)


if __name__ == "__main__":
    unittest.main()
