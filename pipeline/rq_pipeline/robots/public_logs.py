"""Public recordings of real robots — the logs that exercise the
telemetry stage before the operator's own robot exists.

A registry: every entry names where the bytes come from, how many there
are and their digest (a download that differs is refused by name, never
read), which adapter reads them, which robot they are from, when they
were recorded, and the licence STATE of the data — "unlabelled" is a
state, and it is written down rather than guessed. Fetching lands under
the cache (`runs/public-logs/<name>/`, or `$RQ_PUBLIC_LOGS_DIR`), never
in the repository; a test that needs one skips by name when it is
absent. Ingesting one stamps the recording with a provenance block whose
basis is BASIS_PUBLIC, so the loop's telemetry stage says
"public log" and never "met": it is a real Go2, and not ours.

The one entry today is the 12-minute Go2 walk YibinWu/leg-odometry
published with its release (2025-02-19, 499.8 Hz measured, docs/78
§0 row 1 of the market read): the only public rosbag2 of a Go2's full
`LowState` found on 2026-09-24. The DFKI (Zenodo 19336009, CC-BY-4.0)
and QuadSLAM sets are the next entries once their message layouts are
tables here.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.paths import checkout
from rq_pipeline.robots.recording import BASIS_PUBLIC

CACHE_ENV = "RQ_PUBLIC_LOGS_DIR"
CACHE_RELATIVE = ("runs", "public-logs")
CHUNK = 1 << 20
LICENCE_UNLABELLED = "unlabelled"


@dataclass(frozen=True)
class PublicLog:
    """One public recording: enough to fetch it, check it, read it, and
    say whose it is."""

    name: str
    robot: str
    url: str
    bytes: int
    sha256: str
    member: str  # the directory inside the archive the adapter reads
    adapter: str
    source: str  # repository and release, for a human
    recorded: str  # ISO 8601 date the source states
    licence: str  # the data's licence STATE, as found
    notes: tuple[str, ...] = ()

    # Whose robot: every entry here is someone else's real robot.
    basis: str = BASIS_PUBLIC

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "robot": self.robot,
            "source": self.source,
            "url": self.url,
            "recorded": self.recorded,
            "licence": self.licence,
            "sha256": self.sha256,
        }


PUBLIC_LOGS: dict[str, PublicLog] = {
    "go2-leg-odometry": PublicLog(
        name="go2-leg-odometry",
        robot="go2",
        url=(
            "https://github.com/YibinWu/leg-odometry/releases/download/"
            "test_v1.0/rosbag2_2025_02_19-22_46_51.zip"
        ),
        bytes=132_552_214,
        sha256="bb8c41206bad2f35cbb48eb5539623ec397933bcf9c75cfe9a076825597e074c",
        member="rosbag2_2025_02_19-22_46_51",
        adapter="rosbag2",
        source="github.com/YibinWu/leg-odometry, release test_v1.0 (2025-10-14)",
        recorded="2025-02-19",
        licence=(
            f"{LICENCE_UNLABELLED}: the repository is MIT, the bag carries "
            "no licence of its own"
        ),
        notes=(
            "a real Go2 walking indoors for 12 minutes: /lowstate and "
            "/sportmodestate at 500 Hz nominal; no LowCmd on the bag",
        ),
    ),
}


def cache_root() -> Path:
    named = os.environ.get(CACHE_ENV, "").strip()
    if named:
        return Path(named).expanduser()
    return checkout().joinpath(*CACHE_RELATIVE)


def resolve(name: str) -> PublicLog:
    entry = PUBLIC_LOGS.get(name)
    if entry is None:
        raise KeyError(f"unknown public log {name!r}; known: {sorted(PUBLIC_LOGS)}")
    return entry


def locate(name: str, cache: Path | None = None) -> Path | None:
    """The fetched log's source directory, or None when not in the cache."""
    entry = resolve(name)
    path = (cache or cache_root()) / entry.name / entry.member
    return path if path.is_dir() else None


def fetch(
    name: str, cache: Path | None = None, *, opener: Any = urllib.request.urlopen
) -> Path:
    """Download, check the byte count and the digest, unpack; returns the
    directory the adapter reads. Idempotent: an unpacked log is returned
    as is. A download of the wrong size or digest is deleted and refused
    by name."""
    entry = resolve(name)
    root = (cache or cache_root()) / entry.name
    unpacked = root / entry.member
    if unpacked.is_dir():
        return unpacked
    root.mkdir(parents=True, exist_ok=True)
    archive = root / Path(entry.url).name
    _download(entry, archive, opener)
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    if not unpacked.is_dir():
        shutil.rmtree(root, ignore_errors=True)
        raise ValueError(f"{entry.name}: the archive holds no {entry.member!r}")
    archive.unlink()
    return unpacked


def _download(entry: PublicLog, archive: Path, opener: Any) -> None:
    digest = hashlib.sha256()
    size = 0
    with opener(entry.url, timeout=600) as response, archive.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != entry.bytes or digest.hexdigest() != entry.sha256:
        archive.unlink(missing_ok=True)
        raise ValueError(
            f"{entry.name}: downloaded {size} bytes, sha256 "
            f"{digest.hexdigest()[:12]}…; the registry expects {entry.bytes} "
            f"bytes, {entry.sha256[:12]}… — "
            "the release changed or the download broke; nothing was read"
        )


def describe(entry: PublicLog, cache: Path | None = None) -> dict[str, Any]:
    """One registry row for a listing door: the facts plus whether it
    is fetched."""
    return {
        "name": entry.name,
        "robot": entry.robot,
        "adapter": entry.adapter,
        "source": entry.source,
        "recorded": entry.recorded,
        "licence": entry.licence,
        "bytes": entry.bytes,
        "fetched": locate(entry.name, cache) is not None,
        "notes": list(entry.notes),
    }


def listing(cache: Path | None = None) -> list[dict[str, Any]]:
    return [describe(e, cache) for e in PUBLIC_LOGS.values()]


def provenance_of(name: str) -> Mapping[str, Any]:
    return resolve(name).provenance()
