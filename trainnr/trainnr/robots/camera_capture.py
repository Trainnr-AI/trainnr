"""A camera as a live capture source (#130): its picture in the Studio now,
saved as footage when the capture stops.

The camera is a network stream from the robot (MJPEG over HTTP, which
most small boards serve, or RTSP) or a local video device. ffmpeg reads
it once and writes two things: the footage, H.264 in a fragmented MP4
(still playable if the feed is cut off), and a JPEG preview at the
camera's own rate. The preview goes to the Studio's viewer with the
measured frame rate and size beside it. Delay is not measured: an MJPEG
stream or a device carries no capture time to compare with arrival.

Two halves, so the MCP server needs neither Rerun nor a decoder:

- `CameraCapture`, the listener the server holds (`robots.capture`'s
  registry names it), starts the feed as a child process with the `viz`
  extra and reads the feed's numbers from a small state file. On `stop()`
  it stops the feed and hands the MP4 to the footage ingest the project
  layer gives it, the same stamping as `ingest_video`.
- `main()`, the feed: `python -m trainnr.robots.camera_capture --url ...`.

A URL's user and password never reach a record, a state file or the
Studio: only the address with them taken out is written.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from trainnr.robots.capture import (
    DEFAULT_WINDOW_S,
    FAILED,
    FOOTAGE,
    INGESTED,
    LISTENING,
    CaptureState,
    Ingest,
    OnState,
)

NETWORK_SCHEMES = ("http", "https", "rtsp", "rtsps")
DEVICE = re.compile(r"^/dev/video\d+$")
FEED_MODULE = "trainnr.robots.camera_capture"
APP_PREFIX = "trainnr-capture-"
# How long `start` waits for the first picture before it refuses (uv
# starting the feed's environment takes a few seconds of it).
FIRST_FRAME_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 15.0
POLL_S = 1.0
STATE_EVERY_S = 0.5
RATE_WINDOW_S = 2.0
PREVIEW_WIDTH = 640
PREVIEW_FPS = 15.0  # pictures sent to the viewer at most; every one is counted
LOG_TAIL = 12
JPEG_START, JPEG_END = b"\xff\xd8", b"\xff\xd9"
STREAM_LINE = re.compile(r"Stream #0:\d+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})")
DELAY_NOTE = "not measured: the stream carries no capture time"


def checked_url(url: str) -> str:
    """`url` when it names a camera trainnr reads: an http(s) or rtsp(s)
    stream with a host, or a video device; refused by name otherwise (a
    local file or any other scheme is not a camera)."""
    if DEVICE.match(url):
        if not sys.platform.startswith("linux"):
            raise RuntimeError(f"a video device ({url}) is read on Linux only")
        if not Path(url).exists():
            raise FileNotFoundError(f"no video device at {url}")
        return url
    parts = urlsplit(url)
    if parts.scheme not in NETWORK_SCHEMES or not parts.hostname:
        raise ValueError(
            f"a camera is an http://, https://, rtsp:// or rtsps:// stream with a "
            f"host, or a device like /dev/video0; got {shown_url(url)!r}"
        )
    return url


def shown_url(url: str) -> str:
    """The address with any user and password taken out, for every record."""
    parts = urlsplit(url)
    if not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


class CameraCapture:
    """The listener: the feed started for a window, stopped into footage.
    `python` is the interpreter command that has the `viz` extra (the
    server passes its uv line); this one's own Python when not given."""

    def __init__(  # noqa: PLR0913 - the listener's knobs, each named
        self,
        footage: Path,
        state_dir: Path,
        name: str,
        *,
        url: str,
        python: list[str] | None = None,
        ingest: Ingest | None = None,
        on_state: OnState | None = None,
    ) -> None:
        self.footage = Path(footage)
        self.state_dir = Path(state_dir)
        self.name = name
        self.url = url
        self.python = list(python) if python else [sys.executable]
        self.ingest = ingest
        self.on_state = on_state
        self.raw = self.state_dir / f"capture-{name}.mp4"
        self.feed_state = self.state_dir / f"capture-{name}.feed.json"
        self.log = self.state_dir / f"capture-{name}.log"
        self.state = CaptureState(
            name=name,
            source=f"camera:{shown_url(url)}",
            raw_file=str(self.raw),
            lands=FOOTAGE,
        )
        self._child: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, window_s: float = DEFAULT_WINDOW_S) -> CaptureState:
        if (self.footage / self.name).exists():
            raise FileExistsError(f"footage named {self.name!r} already exists")
        checked_url(self.url)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        for stale in (self.raw, self.feed_state):
            stale.unlink(missing_ok=True)
        argv = [
            *self.python, "-m", FEED_MODULE, "--url", self.url,
            "--out", str(self.raw), "--state", str(self.feed_state),
            "--app", f"{APP_PREFIX}{self.name}", "--window", f"{window_s:g}",
            "--parent", str(os.getpid()),
        ]  # fmt: skip
        with self.log.open("wb") as log:
            self._child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # a stop signals the feed, not the server
            )
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
        while self._read_feed().get("frames", 0) == 0:
            ended = self._child.poll() is not None
            if ended or time.monotonic() > deadline:
                self._end_child()
                why = (
                    self._log_tail() or "no picture arrived"
                    if ended
                    else f"nothing within {FIRST_FRAME_TIMEOUT_S:g} s"
                )
                self._cleanup()
                raise RuntimeError(
                    f"the camera at {shown_url(self.url)} gave no picture: {why}"
                )
            time.sleep(0.2)
        self.state.state = LISTENING
        self.state.started = time.time()
        self._update()
        self._thread = threading.Thread(
            target=self._watch, daemon=True, name=f"capture-{self.name}"
        )
        self._thread.start()
        return self.state

    def _read_feed(self) -> dict[str, Any]:
        with contextlib.suppress(OSError, ValueError):
            return dict(json.loads(self.feed_state.read_text(encoding="utf-8")))
        return {}

    def _log_tail(self) -> str:
        with contextlib.suppress(OSError):
            lines = self.log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = " | ".join(
                line.strip() for line in lines[-LOG_TAIL:] if line.strip()
            )
            return tail.replace(self.url, shown_url(self.url))  # never the password
        return ""

    def _update(self) -> None:
        feed = self._read_feed()
        self.state.datagrams = int(feed.get("frames", 0))
        self.state.last_datagram = feed.get("last_frame")
        self.state.measured = {
            k: feed[k] for k in ("fps", "width", "height", "codec") if k in feed
        } | {"delay": DELAY_NOTE}
        if feed.get("ended"):
            ended = str(feed["ended"]).replace(self.url, shown_url(self.url))
            note = f"the stream ended: {ended}"
            if note not in self.state.notes:
                self.state.notes.append(note)
        self._write_state()

    def _watch(self) -> None:
        while not self._stop.wait(POLL_S):
            self._update()

    def _end_child(self) -> None:
        child = self._child
        if child is None or child.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)  # the feed closes ffmpeg's file
        try:
            child.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()

    def _cleanup(self) -> None:
        for leftover in (self.raw, self.feed_state):
            leftover.unlink(missing_ok=True)

    def stop(self) -> CaptureState:
        """Stop the feed and land the footage."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(POLL_S + 1.0)
        self._end_child()
        self._update()
        if self.state.datagrams == 0 or not self.raw.is_file():
            self.state.state = FAILED
            self.state.error = "no picture arrived — nothing to save"
        elif self.ingest is None:
            self.state.state = FAILED
            self.state.error = "no footage ingest was given"
        else:
            try:
                record = self.ingest(
                    self.footage,
                    self.raw,
                    name=self.name,
                    origin=f"filmed live from {shown_url(self.url)}",
                )
                self.state.stamp = record.get("stamp")
                self.state.state = INGESTED
                self.state.notes += list(record.get("notes", []))
                self.log.unlink(missing_ok=True)  # kept only when it went wrong
            except Exception as why:
                self.state.state = FAILED
                self.state.error = str(why)
        self._cleanup()
        self._write_state()
        return self.state

    def _write_state(self) -> None:
        self.state.write(self.state_dir)
        if self.on_state is not None:
            self.on_state(self.state)


# -- the feed: ffmpeg's two outputs, the preview into the Studio ---------------


def ffmpeg_argv(
    url: str, out: Path, window_s: float, ffmpeg: str = "ffmpeg"
) -> list[str]:
    """One read of the camera, two outputs: the footage and the preview."""
    if DEVICE.match(url):
        source = ["-f", "v4l2"]
    elif urlsplit(url).scheme.startswith("rtsp"):
        source = ["-rtsp_transport", "tcp"]
    else:  # MJPEG over HTTP carries no clock: the arrival is the time
        source = ["-use_wallclock_as_timestamps", "1"]
    # `q` on stdin closes the footage cleanly; -t on the input caps both outputs
    return [
        ffmpeg, "-hide_banner", "-loglevel", "info", *source,
        "-t", f"{window_s:g}", "-i", url,
        "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(out),
        "-map", "0:v:0", "-vf", f"scale='min({PREVIEW_WIDTH},iw)':-2",
        "-c:v", "mjpeg", "-q:v", "5", "-f", "image2pipe", "pipe:1",
    ]  # fmt: skip


def jpegs(stream: Any, chunk: int = 1 << 16) -> Any:
    """Each JPEG in ffmpeg's image pipe, as bytes."""
    buffer = b""
    while True:
        data = stream.read1(chunk) if hasattr(stream, "read1") else stream.read(chunk)
        if not data:
            return
        buffer += data
        while True:
            start = buffer.find(JPEG_START)
            end = buffer.find(JPEG_END, start + 2) if start >= 0 else -1
            if end < 0:
                if start > 0:
                    buffer = buffer[start:]
                break
            yield buffer[start : end + 2]
            buffer = buffer[end + 2 :]


class Feed:
    """The numbers the feed measures and writes for the listener."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.frames = 0
        self.last_frame: float | None = None
        self.arrivals: deque[float] = deque()
        self.stream: dict[str, Any] = {}
        self.ended = ""
        self._written = 0.0

    def frame(self, now: float) -> None:
        self.frames += 1
        self.last_frame = time.time()
        self.arrivals.append(now)
        while self.arrivals and now - self.arrivals[0] > RATE_WINDOW_S:
            self.arrivals.popleft()

    @property
    def fps(self) -> float | None:
        if len(self.arrivals) < 2:  # noqa: PLR2004 - a rate needs two
            return None
        span = self.arrivals[-1] - self.arrivals[0]
        return round((len(self.arrivals) - 1) / span, 1) if span > 0 else None

    def write(self, now: float, *, force: bool = False) -> None:
        if not force and now - self._written < STATE_EVERY_S:
            return
        record = {"frames": self.frames, "last_frame": self.last_frame, **self.stream}
        if self.fps is not None:
            record["fps"] = self.fps
        if self.ended:
            record["ended"] = self.ended
        staging = self.state.with_suffix(".tmp")
        staging.write_text(json.dumps(record), encoding="utf-8")
        staging.replace(self.state)
        self._written = now


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _viewer(app: str) -> Any:
    """The Studio's stream for the preview, or None (no viz extra, no
    Studio listening): the footage is saved either way."""
    try:
        import rerun as rr  # noqa: PLC0415
    except ImportError:
        print("[camera] no rerun-sdk: the picture is saved, not shown", flush=True)
        return None
    from trainnr.viz import entry_name, sinks  # noqa: PLC0415

    chosen = sinks(rr, wanted=False)
    if not chosen:
        print("[camera] no Studio listens: the picture is saved, not shown", flush=True)
        return None
    stream = rr.RecordingStream(entry_name(app))
    stream.set_sinks(*chosen)
    return stream


def _about(feed: Feed, url: str) -> str:
    size = (
        f"{feed.stream['width']}x{feed.stream['height']}"
        if "width" in feed.stream
        else "not known yet"
    )
    rate = f"{feed.fps:g} fps" if feed.fps is not None else "measuring"
    return "\n".join(
        [
            f"# camera: {shown_url(url)}",
            "",
            f"- frame rate: {rate} (measured as the pictures arrive)",
            f"- size: {size}",
            f"- codec: {feed.stream.get('codec', 'not known yet')}",
            f"- pictures: {feed.frames}",
            f"- delay: {DELAY_NOTE}",
        ]
    )


def main() -> int:  # noqa: PLR0915 - one feed, each step named
    parser = argparse.ArgumentParser(description="A camera's live feed (#130).")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--app", required=True)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument(
        "--parent", type=int, default=0, help="the server: the feed ends with it"
    )
    args = parser.parse_args()

    from trainnr.video.footage import tool  # noqa: PLC0415

    url = checked_url(args.url)
    feed = Feed(args.state)
    viewer = _viewer(args.app)
    ffmpeg = subprocess.Popen(
        ffmpeg_argv(url, args.out, args.window, str(tool("ffmpeg"))),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stop = threading.Event()
    tail: deque[str] = deque(maxlen=LOG_TAIL)

    def on_signal(*_: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def read_log() -> None:
        assert ffmpeg.stderr is not None
        for raw in ffmpeg.stderr:
            line = raw.decode(errors="replace").rstrip()
            tail.append(line)
            match = STREAM_LINE.search(line)
            if match and not feed.stream:
                codec, w, h = match.groups()
                feed.stream = {"codec": codec, "width": int(w), "height": int(h)}

    def read_pictures() -> None:
        import rerun as rr  # noqa: PLC0415 - only reached with a viewer

        assert ffmpeg.stdout is not None
        shown_at = 0.0
        about_at = 0.0
        for picture in jpegs(ffmpeg.stdout):
            now = time.monotonic()
            feed.frame(now)
            feed.write(now)
            if viewer is None:
                continue
            if now - shown_at >= 1.0 / PREVIEW_FPS:
                viewer.set_time("time", timestamp=time.time())
                viewer.log(
                    "camera/picture",
                    rr.EncodedImage(contents=picture, media_type="image/jpeg"),
                )
                shown_at = now
            if now - about_at >= 1.0:
                if feed.fps is not None:
                    viewer.log("camera/fps", rr.Scalars(feed.fps))
                viewer.log(
                    "camera/about",
                    rr.TextDocument(
                        _about(feed, url), media_type=rr.MediaType.MARKDOWN
                    ),
                )
                about_at = now

    def read_pictures_quietly() -> None:
        if viewer is None:
            assert ffmpeg.stdout is not None
            for _ in jpegs(ffmpeg.stdout):
                now = time.monotonic()
                feed.frame(now)
                feed.write(now)
            return
        read_pictures()

    threads = [
        threading.Thread(target=read_log, daemon=True),
        threading.Thread(target=read_pictures_quietly, daemon=True),
    ]
    for thread in threads:
        thread.start()
    while not stop.is_set() and ffmpeg.poll() is None:
        stop.wait(0.2)
        if args.parent and not _alive(args.parent):
            stop.set()  # the server that holds the capture is gone
    if ffmpeg.poll() is None:  # asked to stop: ffmpeg closes the footage itself
        with contextlib.suppress(BrokenPipeError, OSError):
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(b"q")
            ffmpeg.stdin.flush()
        try:
            ffmpeg.wait(STOP_TIMEOUT_S - 3)
        except subprocess.TimeoutExpired:
            ffmpeg.terminate()
            ffmpeg.wait()
    elif not stop.wait(0.5):  # it ended on its own, not by our stop
        feed.ended = " | ".join(line for line in tail if line.strip())[-500:] or (
            f"ffmpeg exited with {ffmpeg.returncode}"
        )
    for thread in threads:
        thread.join(2.0)
    feed.write(time.monotonic(), force=True)
    if viewer is not None:
        viewer.flush()
    print(f"[camera] {feed.frames} pictures, {feed.fps or 0:g} fps", flush=True)
    if feed.frames == 0:
        print("[camera] " + " | ".join(tail), flush=True)
    return 0 if feed.frames else 1


if __name__ == "__main__":
    sys.exit(main())
