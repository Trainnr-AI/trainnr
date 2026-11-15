"""Footage: a video brought into a project as it is, the first artifact of
the video-to-robot path (docs/86 §5.1).

A footage folder holds the video, copied byte for byte, a `footage.json`
(`trainnr-footage/1`) with what ffprobe says about it, and its first frame
as a still for the cards. The stamp covers the folder's files like any
bundle's. Nothing is estimated here: a motion estimated from footage is a
recording and proves the Telemetry state; footage alone proves none, so a
template's "Turn a video into motion" is not done the moment a video lands.

ffprobe and ffmpeg are found on PATH, the way the scene capture finds them
(`scenes/tooling.py`), and refused by name with the install line when absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from trainnr.bundles.json_record import JsonRecord
from trainnr.scenes.tooling import MissingToolError, install_hint

FOOTAGE_FILE = "footage.json"
SCHEMA = "trainnr-footage/1"
FIRST_FRAME = "first-frame.jpg"
UNRECORDED = "unrecorded"
# Plain words: the folder is the footage's name.
NAME_FORBIDDEN = '/\\@:<>"|?*'
PROBE_TIMEOUT_S = 60
QUARTER_TURN = 90  # degrees: a phone held upright stores its frames turned


@dataclass(frozen=True)
class Footage(JsonRecord):
    """What one video is: its own numbers, and where it came from."""

    name: str
    video: str  # the file beside this record
    duration_s: float
    fps: float
    width: int  # as displayed: a turned phone video's sides swapped back
    height: int
    codec: str
    frames: int | None  # what the container counts; None when it does not
    rotation_deg: int
    source: str  # the file it was copied from
    origin: str  # where the video came from: filmed, or a URL
    licence: str
    ingested: str
    schema: str = SCHEMA


def tool(name: str) -> Path:
    found = shutil.which(name)
    if found is None:
        raise MissingToolError(f"footage needs {name} ({install_hint(name)})")
    return Path(found)


def probe(video: Path) -> dict[str, Any]:
    """The first video stream's numbers, as `Footage` takes them. Refused
    by name when ffprobe finds no video stream in the file."""
    out = subprocess.run(
        [
            str(tool("ffprobe")),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_S,
        check=False,
    )
    if out.returncode != 0:
        said = out.stderr.strip().splitlines()
        raise ValueError(
            f"ffprobe cannot read {video.name}: {said[-1] if said else out.returncode}"
        )
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise ValueError(f"{video.name} has no video stream")
    stream = streams[0]
    # A turned phone video: a display matrix in newer ffprobe, a tag in older.
    rotation = int((stream.get("tags") or {}).get("rotate", 0))
    for side in stream.get("side_data_list") or []:
        rotation = int(side.get("rotation", rotation))
    width, height = int(stream["width"]), int(stream["height"])
    if abs(rotation) % (2 * QUARTER_TURN) == QUARTER_TURN:
        width, height = height, width
    rate = Fraction(stream.get("avg_frame_rate") or "0/1")
    frames = str(stream.get("nb_frames") or "")
    return {
        "duration_s": round(float((info.get("format") or {}).get("duration", 0.0)), 3),
        "fps": round(float(rate), 3) if rate.denominator else 0.0,
        "width": width,
        "height": height,
        "codec": str(stream.get("codec_name", UNRECORDED)),
        "frames": int(frames) if frames.isdigit() else None,
        "rotation_deg": rotation,
    }


def ingest_footage(
    footage: Path,
    source: Path,
    *,
    name: str | None = None,
    origin: str = UNRECORDED,
    licence: str = UNRECORDED,
) -> Path:
    """Copy `source` into `footage/<name>` with its record and first frame;
    the folder, ready to stamp. Refuses a missing file, a name that is not a
    plain word, a name already taken, and a file with no video stream,
    before anything is written."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"no video at {source}")
    label = name or source.stem
    if not label or any(c in label for c in NAME_FORBIDDEN):
        raise ValueError(f"footage names are plain words, got {label!r}")
    out = Path(footage) / label
    if out.exists():
        raise FileExistsError(f"footage named {label!r} already exists at {out}")
    numbers = probe(source)
    out.mkdir(parents=True)
    try:
        video = out / f"video{source.suffix.lower()}"
        shutil.copy2(source, video)
        subprocess.run(
            [
                str(tool("ffmpeg")),
                "-v",
                "error",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(out / FIRST_FRAME),
            ],
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
            check=True,
        )
        Footage(
            name=label,
            video=video.name,
            source=source.name,
            origin=origin,
            licence=licence,
            ingested=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **numbers,
        ).write(out / FOOTAGE_FILE)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out
