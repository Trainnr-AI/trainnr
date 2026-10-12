#!/usr/bin/env python3
"""The engine's runtime: the part of the repository the desktop app needs
to run trainnr's tools without a checkout, as one archive per release.

    python3 tools/runtime-archive.py 0.1.0 --out dist

The Studio downloaded from a release finds no checkout, and every job
runs a script or a package from one (`mcp_actions`: `tools/`,
`trainnr/`, `trainnr-mjlab/`; the robot library under `robots/`). This
archive is exactly those folders as git tracks them, less their tests,
with the licences: about 9 MB. The app downloads it the first time an
action needs the engine, checks it against the SHA-256 written beside it
here, unpacks it, and then runs as from a checkout (docs/85, the engine,
2026-10-11).

The archive is a tar.gz with sorted members, zero ids, every time the
commit's own time, and a gzip header without a name or time: the same
commit gives the same bytes. Stdlib only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# What a job, the presenter or the simulator stream reads from a checkout.
FOLDERS = ("trainnr", "trainnr-mjlab", "tools", "robots")
# What travels with them: the licence, its notices, the third parties'.
FILES = ("LICENSE", "NOTICE")
LICENCE_FOLDER = "LICENSES"
# Never shipped: the suites stay in the repository.
EXCLUDED_PARTS = ("tests",)
NAME = "trainnr-runtime-v{version}"


def tracked(paths: tuple[str, ...]) -> list[str]:
    """Every file git tracks under `paths`, sorted."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return sorted(p for p in listed.split("\0") if p)


def members() -> list[str]:
    """The runtime's files: the folders less their tests, and the licences."""
    files = tracked((*FOLDERS, *FILES, LICENCE_FOLDER))
    return [f for f in files if not set(Path(f).parts) & set(EXCLUDED_PARTS)]


def commit_time() -> int:
    stamp = subprocess.run(
        ["git", "log", "-1", "--format=%ct"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    return int(stamp)


def build(version: str, out: Path) -> Path:
    """Write `trainnr-runtime-v<version>.tar.gz` and its `.sha256` in `out`."""
    name = NAME.format(version=version)
    mtime = commit_time()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for relative in members():
            source = ROOT / relative
            if source.is_symlink() or not source.is_file():
                raise SystemExit(f"{relative}: not a plain file")
            info = tarfile.TarInfo(f"{name}/{relative}")
            info.size = source.stat().st_size
            info.mtime = mtime
            info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
            with source.open("rb") as handle:
                tar.addfile(info, handle)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{name}.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as gz,
    ):
        gz.write(buffer.getvalue())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the release's version, without the v")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    archive = build(args.version, args.out.resolve())
    size = archive.stat().st_size
    print(f"{archive}  {size} bytes  ({len(members())} files)")
    print((archive.parent / f"{archive.name}.sha256").read_text().strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
