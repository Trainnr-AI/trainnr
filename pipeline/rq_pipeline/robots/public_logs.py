"""Public recordings of real robots — the logs that exercise the
telemetry and identification stages before the operator's own robot
exists.

A registry: every entry names where the bytes come from, what lands in
the cache and each piece's size and digest (a download that differs is
refused by name, never read), which adapter reads the result, which
robot it is from, when it was recorded, and the licence STATE of the
data — "unlabelled" is a state, and it is written down rather than
guessed. Fetching lands under the cache (`runs/public-logs/<name>/`, or
`$RQ_PUBLIC_LOGS_DIR`), never in the repository; a test that needs one
skips by name when it is absent. Ingesting one stamps the recording
with its provenance and the basis BASIS_PUBLIC, so the loop's stages say
"public log" and never "met": a real Go2, and not ours.

How the bytes arrive is a FETCHER, named per entry and kept in a table:
a whole archive unpacked (`archive`), members of a remote zip read by
byte range so 48 MB is fetched out of 3.5 GB (`zip-members`, the offsets
read once from the zip's central directory and dated here; a changed
archive fails the digest, never silently), or one file (`file`).

The finds of 2026-09-24 (docs/e2e-research/78 and docs/07): three are
entries — the leg-odometry walk bag, DFKI's field201 bag, IIT's in-air
chirp. Three more are real Go2 or Go1 logs no adapter here reads (ROS 1
bags); they are listed in `UNREADABLE` so a request for one is refused by
name with the reason, not with "unknown".
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq_pipeline.paths import checkout
from rq_pipeline.robots.recording import BASIS_PUBLIC

CACHE_ENV = "RQ_PUBLIC_LOGS_DIR"
CACHE_RELATIVE = ("runs", "public-logs")
CHUNK = 1 << 20
TIMEOUT_S = 600
LICENCE_UNLABELLED = "unlabelled"
LICENCE_NONE_STATED = "none stated"  # the source names no licence at all

FETCH_ARCHIVE = "archive"
FETCH_ZIP_MEMBERS = "zip-members"
FETCH_FILE = "file"

# A zip's local file header: signature, then 26 bytes whose last four are
# the name and extra-field lengths (APPNOTE 4.3.7).
ZIP_LOCAL_SIGNATURE = b"PK\x03\x04"
ZIP_LOCAL_HEADER = 30
ZIP_STORED = 0
ZIP_DEFLATED = 8
RAW_DEFLATE = -15  # zlib's window bits for deflate with no zlib header


@dataclass(frozen=True)
class ZipSpan:
    """Where one member sits inside a remote zip (from its central
    directory): the local header's offset, the compressed size, the
    method."""

    offset: int
    compressed: int
    method: int


@dataclass(frozen=True)
class Piece:
    """One checked download: the path it lands at under the entry's cache
    folder, its size once there (inflated), its digest, and — for a zip
    member read by range — where it sits in the remote archive."""

    path: str
    bytes: int
    sha256: str
    span: ZipSpan | None = None


@dataclass(frozen=True)
class PublicLog:
    """One public recording: enough to fetch it, check it, read it, and
    say whose it is."""

    name: str
    robot: str
    url: str
    fetch: str  # a key of FETCHERS
    pieces: tuple[Piece, ...]
    member: str  # the path, under the entry's cache folder, the adapter reads
    adapter: str
    source: str  # repository or record and release, for a human
    recorded: str  # ISO 8601 date the source states
    licence: str  # the data's licence STATE, as found
    notes: tuple[str, ...] = ()
    # Whose robot: every entry here is someone else's real robot.
    basis: str = BASIS_PUBLIC

    @property
    def bytes(self) -> int:
        """What lands in the cache, summed over the pieces."""
        return sum(p.bytes for p in self.pieces)

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "robot": self.robot,
            "source": self.source,
            "url": self.url,
            "recorded": self.recorded,
            "licence": self.licence,
            # One digest per piece, as one line a drawer can show.
            "sha256": (
                self.pieces[0].sha256
                if len(self.pieces) == 1
                else ", ".join(f"{Path(p.path).name} {p.sha256}" for p in self.pieces)
            ),
        }


@dataclass(frozen=True)
class Unreadable:
    """A real robot's public log found and NOT an entry: why, so a request
    for it is refused with the reason."""

    name: str
    robot: str
    url: str
    format: str
    licence: str
    why: str


DFKI_URL = "https://zenodo.org/api/records/19336009/files/outdoors.zip/content"
DFKI_FIELD201 = "outdoors/field201/rosbag2_2024_06_16-08_50_53"
IIT_COMMIT = "b73d6a2c4988dc9b3453369a2cdd29a307cb418e"

PUBLIC_LOGS: dict[str, PublicLog] = {
    "go2-leg-odometry": PublicLog(
        name="go2-leg-odometry",
        robot="go2",
        url=(
            "https://github.com/YibinWu/leg-odometry/releases/download/"
            "test_v1.0/rosbag2_2025_02_19-22_46_51.zip"
        ),
        fetch=FETCH_ARCHIVE,
        pieces=(
            Piece(
                "rosbag2_2025_02_19-22_46_51.zip",
                132_552_214,
                "bb8c41206bad2f35cbb48eb5539623ec397933bcf9c75cfe9a076825597e074c",
            ),
        ),
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
    "dfki-go2-field201": PublicLog(
        name="dfki-go2-field201",
        robot="go2",
        url=DFKI_URL,
        fetch=FETCH_ZIP_MEMBERS,
        # Offsets from outdoors.zip's central directory, read 2026-09-24
        # (3,505,492,727 bytes, 258 members, 54 bags); two members fetched
        # by range: 47.7 MB of the archive.
        pieces=(
            Piece(
                f"{DFKI_FIELD201}/rosbag2_2024_06_16-08_50_53_0.db3",
                150_773_760,
                "d8bfcdec690fe46771c07aed193d7f7f6fb56b995210e919308f52db880d1aa8",
                ZipSpan(offset=29_755, compressed=47_735_472, method=ZIP_DEFLATED),
            ),
            Piece(
                f"{DFKI_FIELD201}/metadata.yaml",
                10_464,
                "7da028ab3dd1dfa87cd306fb9ca525efafac60988498bcd6322f7dbd5ca8e846",
                ZipSpan(offset=47_765_384, compressed=900, method=ZIP_DEFLATED),
            ),
        ),
        member=DFKI_FIELD201,
        adapter="rosbag2",
        source=(
            "Zenodo record 19336009 (DFKI RIC, published 2026-03-30), "
            "outdoors.zip, field201 on Vulcano Island"
        ),
        recorded="2024-06-16",
        licence="CC-BY-4.0 (the Zenodo record's licence)",
        notes=(
            "a real Go2 under DFKI's own MPC/WBC controller, not Unitree's: "
            "/joint_states and /joint_cmd (with kp, kd) at 1 kHz, the state "
            "estimator's /quad_state, /imu_measurement; 150.8 MB inflated",
        ),
    ),
    "iit-go2-chirp": PublicLog(
        name="iit-go2-chirp",
        robot="go2",
        url=(
            "https://raw.githubusercontent.com/iit-DLSLab/"
            f"sim2real-robot-identification/{IIT_COMMIT}/datasets/go2/traj_0.pt"
        ),
        fetch=FETCH_FILE,
        pieces=(
            Piece(
                "traj_0.pt",
                939_662,
                "3a98be68c018d212d27f7fa2427d6b1591024d07da2dfa0380efc922072726d0",
            ),
        ),
        member="traj_0.pt",
        adapter="pt-dict",
        source=(
            "github.com/iit-DLSLab/sim2real-robot-identification @ b73d6a2 "
            "(2026-09-19), datasets/go2/traj_0.pt"
        ),
        # The source states no recording date; this is the commit that added
        # the file ('first run identification').
        recorded="2026-03-19",
        licence="BSD-3-Clause (the repository's LICENSE; the data carries none)",
        notes=(
            "a 23.9 s chirp at 200 Hz with the base fixed in the air: joint "
            "position, velocity, their commands, kp 20 and kd 1.5; no torque; "
            "the recording date is the commit's, the source states none",
            "the collection script has a simulation flag, so the real-robot "
            "origin is the maintainers' word",
        ),
    ),
}

ROS1_WHY = (
    "a ROS 1 bag (rosbag v2): no adapter here reads ROS 1, only rosbag2 and "
    "MCAP; out of scope until a ROS 1 reader is a registry entry"
)
UNREADABLE: dict[str, Unreadable] = {
    "quadslam": Unreadable(
        name="quadslam",
        robot="go2",
        url="https://github.com/EN3D-Lab/Quadruped-SLAM-dataset",
        format="ROS 1 .bag (22 sequences, ~10 GB each, /low_state at 500 Hz)",
        licence=LICENCE_NONE_STATED,
        why=ROS1_WHY,
    ),
    "doglegs": Unreadable(
        name="doglegs",
        robot="go2",
        url="https://github.com/YibinWu/DogLegs",
        format="ROS 1 .bag (5 field sequences; q, dq, foot forces; no torque)",
        licence=LICENCE_NONE_STATED,
        why=ROS1_WHY,
    ),
    "legkilo": Unreadable(
        name="legkilo",
        robot="go1",
        url="https://github.com/ouguangjun/legkilo-dataset",
        format="ROS 1 .bag (HighState, 50 Hz effective)",
        licence=LICENCE_NONE_STATED,
        why=ROS1_WHY,
    ),
}


def cache_root() -> Path:
    named = os.environ.get(CACHE_ENV, "").strip()
    if named:
        return Path(named).expanduser()
    return checkout().joinpath(*CACHE_RELATIVE)


def resolve(name: str) -> PublicLog:
    entry = PUBLIC_LOGS.get(name)
    if entry is not None:
        return entry
    unreadable = UNREADABLE.get(name)
    if unreadable is not None:
        raise KeyError(f"public log {name!r} ({unreadable.format}): {unreadable.why}")
    raise KeyError(f"unknown public log {name!r}; known: {sorted(PUBLIC_LOGS)}")


# Written LAST, inside the entry's folder, once every piece checked: a
# folder without it is a download that broke and is never served.
FETCHED_MARKER = ".fetched"
STAGING_PREFIX = "."  # the entry's staging folder: hidden, beside the slot
STAGING_SUFFIX = ".part"


def _complete(root: Path, entry: PublicLog) -> bool:
    """The slot holds a finished fetch: the marker, and the member."""
    return (root / FETCHED_MARKER).is_file() and (root / entry.member).exists()


def _adopt_verified(root: Path, entry: PublicLog) -> bool:
    """A slot fetched before the marker existed: when every piece is still
    on disk (a file or zip-member fetch keeps them) and checks, it is
    marked complete; an archive's pieces are gone, so it is fetched again."""
    if entry.fetch == FETCH_ARCHIVE or not (root / entry.member).exists():
        return False
    for piece in entry.pieces:
        path = root / piece.path
        if not path.is_file() or path.stat().st_size != piece.bytes:
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != piece.sha256:
            return False
    (root / FETCHED_MARKER).write_text(entry.name + "\n", encoding="utf-8")
    return True


def locate(name: str, cache: Path | None = None) -> Path | None:
    """The fetched log's source path, or None when it is not in the cache
    COMPLETE (a download that broke is not a fetched log)."""
    entry = resolve(name)
    root = (cache or cache_root()) / entry.name
    if _complete(root, entry) or _adopt_verified(root, entry):
        return root / entry.member
    return None


Opener = Callable[..., Any]


def fetch(
    name: str, cache: Path | None = None, *, opener: Opener = urllib.request.urlopen
) -> Path:
    """Download by the entry's fetcher, check every piece's byte count and
    digest; returns the path the adapter reads. Idempotent: a COMPLETE
    fetch is returned as is. Everything lands in a hidden staging folder
    first and is moved into place, marked complete, only when every piece
    checked; a broken download (a piece of the wrong size or digest, a
    network drop, an interrupt) leaves nothing that `fetch` or `locate`
    would serve, and is refused by name."""
    entry = resolve(name)
    base = cache or cache_root()
    root = base / entry.name
    found = locate(name, base)
    if found is not None:
        return found
    staging = base / f"{STAGING_PREFIX}{entry.name}{STAGING_SUFFIX}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        FETCHERS[entry.fetch](entry, staging, opener)
        if not (staging / entry.member).exists():
            raise ValueError(f"{entry.name}: the download holds no {entry.member!r}")
        (staging / FETCHED_MARKER).write_text(entry.name + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(root, ignore_errors=True)  # a slot a broken fetch left
    staging.replace(root)
    return root / entry.member


def _fetch_archive(entry: PublicLog, root: Path, opener: Opener) -> None:
    """One archive: stream it, check it, unpack it, drop it."""
    (piece,) = entry.pieces
    archive = root / piece.path
    _stream_checked(entry, piece, archive, opener)
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    archive.unlink()


def _fetch_file(entry: PublicLog, root: Path, opener: Opener) -> None:
    """Each piece is a file at the entry's url, streamed and checked."""
    for piece in entry.pieces:
        _stream_checked(entry, piece, root / piece.path, opener)


def _fetch_zip_members(entry: PublicLog, root: Path, opener: Opener) -> None:
    """Each piece is a member of the remote zip at the entry's url: its
    local header, then its compressed bytes, by range; inflated, checked,
    written."""
    for piece in entry.pieces:
        span = piece.span
        if span is None:
            raise ValueError(f"{entry.name}: {piece.path} has no span in the zip")
        head = _range(entry.url, span.offset, ZIP_LOCAL_HEADER, opener)
        if head[:4] != ZIP_LOCAL_SIGNATURE:
            raise ValueError(
                f"{entry.name}: no zip member header at byte {span.offset} of "
                f"{entry.url} — the archive changed; nothing was read"
            )
        name_len, extra_len = struct.unpack("<HH", head[26:30])
        start = span.offset + ZIP_LOCAL_HEADER + name_len + extra_len
        raw = _range(entry.url, start, span.compressed, opener)
        if span.method == ZIP_DEFLATED:
            data = zlib.decompress(raw, RAW_DEFLATE)
        elif span.method == ZIP_STORED:
            data = raw
        else:
            raise ValueError(f"{entry.name}: zip method {span.method} is not read")
        _check(entry, piece, len(data), hashlib.sha256(data).hexdigest())
        target = root / piece.path
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + STAGING_SUFFIX)
        part.write_bytes(data)
        part.replace(target)


FETCHERS: dict[str, Callable[[PublicLog, Path, Opener], None]] = {
    FETCH_ARCHIVE: _fetch_archive,
    FETCH_ZIP_MEMBERS: _fetch_zip_members,
    FETCH_FILE: _fetch_file,
}


def _stream_checked(entry: PublicLog, piece: Piece, out: Path, opener: Opener) -> None:
    """Stream to `<out>.part`, check, then move into place: a stream that
    breaks midway never leaves a file at `out`."""
    digest = hashlib.sha256()
    size = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + STAGING_SUFFIX)
    try:
        with opener(entry.url, timeout=TIMEOUT_S) as response, part.open("wb") as sink:
            while chunk := response.read(CHUNK):
                sink.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        _check(entry, piece, size, digest.hexdigest())
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(out)


def _range(url: str, start: int, length: int, opener: Opener) -> bytes:
    """`length` bytes from `start`, and never more: a server that ignores
    the Range header would otherwise send the whole archive."""
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{start + length - 1}"}
    )
    with opener(request, timeout=TIMEOUT_S) as response:
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError(
            f"{url}: asked for {length} bytes at {start}, got {len(data)} "
            "(a server that ignores ranges, or a truncated reply); nothing was read"
        )
    return data


def _check(entry: PublicLog, piece: Piece, size: int, sha256: str) -> None:
    if size != piece.bytes or sha256 != piece.sha256:
        raise ValueError(
            f"{entry.name}: {piece.path} came as {size} bytes, sha256 "
            f"{sha256[:12]}…; the registry expects {piece.bytes} bytes, "
            f"{piece.sha256[:12]}… — the source changed or the download broke; "
            "nothing was read"
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
        "fetch": entry.fetch,
        "readable": True,
        "fetched": locate(entry.name, cache) is not None,
        "notes": list(entry.notes),
    }


def describe_unreadable(entry: Unreadable) -> dict[str, Any]:
    return {
        "name": entry.name,
        "robot": entry.robot,
        "url": entry.url,
        "format": entry.format,
        "licence": entry.licence,
        "readable": False,
        "why": entry.why,
    }


def listing(cache: Path | None = None) -> list[dict[str, Any]]:
    """Every public log found: the readable entries first, then the ones
    refused, each with its licence state."""
    return [describe(e, cache) for e in PUBLIC_LOGS.values()] + [
        describe_unreadable(u) for u in UNREADABLE.values()
    ]


def provenance_of(name: str) -> Mapping[str, Any]:
    return resolve(name).provenance()
