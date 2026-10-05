"""Files a robot bundle does not carry, fetched from their publisher on
first use: a bundle's `FETCH.json` names the repository, the pinned
commit and each file's git blob id, and the files land in the bundle
folder, checked byte for byte, before the model is loaded.

Why a bundle would not carry a file: its licence does not let this
repository redistribute it under its own terms. The microduck's meshes
are "licensed under Creative Commons BY-SA-NC" by Pollen Robotics while
the model and code are Apache-2.0; the meshes are fetched from Pollen
Robotics' repository instead (2026-10-05). The fetched bytes are the ones
the bundle's stamp was taken over, so the stamp is unchanged once they
are in place, and the records citing it still resolve.

A fetch is a network call: it happens when a bundle is loaded for use,
says what it fetches and under which licence, and is refused with the
manual line when `TRAINNR_NO_ROBOT_FETCH=1`.

A bundle can come from someone else's project, so a manifest is data: it
names a GitHub repository and a full commit id by their shapes only, and
every file lands inside the bundle's own folder, never through a link
out of it (2026-10-05, a path like `../../.bashrc` was written as given).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from trainnr import safe_write

FETCH_FILE = "FETCH.json"
FETCH_SCHEMA = "trainnr-bundle-fetch/1"
NO_FETCH_ENV = "TRAINNR_NO_ROBOT_FETCH"
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
OBJECT_ID = re.compile(r"[0-9a-f]{40}")


def _plain_relative(path: str) -> bool:
    """A forward-slash path with no root, drive, `..`, `.` or empty part."""
    parts = path.split("/")
    return (
        bool(path)
        and "\\" not in path
        and ":" not in path
        and all(part not in ("", ".", "..") for part in parts)
    )


def _checked(manifest: dict[str, Any], where: Path) -> dict[str, Any]:
    """The manifest, or ValueError naming the first entry out of shape."""
    repository, commit = manifest.get("repository"), manifest.get("commit")
    if not isinstance(repository, str) or not REPOSITORY.fullmatch(repository):
        raise ValueError(f"{where}: repository {repository!r} is not owner/name")
    if not isinstance(commit, str) or not OBJECT_ID.fullmatch(commit):
        raise ValueError(f"{where}: commit {commit!r} is not a full commit id")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{where}: files is not a table")
    for rel, entry in files.items():
        if not _plain_relative(rel):
            raise ValueError(f"{where}: {rel!r} is not a plain path inside the bundle")
        upstream = entry.get("upstream") if isinstance(entry, dict) else None
        blob = entry.get("blob") if isinstance(entry, dict) else None
        if not isinstance(upstream, str) or not _plain_relative(upstream):
            raise ValueError(
                f"{where}: {rel}'s upstream {upstream!r} is not a plain path"
            )
        if not isinstance(blob, str) or not OBJECT_ID.fullmatch(blob):
            raise ValueError(f"{where}: {rel}'s blob {blob!r} is not a git blob id")
    return manifest


def fetch_manifest(bundle_dir: Path) -> dict[str, Any] | None:
    """The bundle's fetch manifest, or None when it carries every file."""
    path = Path(bundle_dir) / FETCH_FILE
    if not path.is_file():
        return None
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FETCH_SCHEMA:
        raise ValueError(f"{path}: schema is not {FETCH_SCHEMA!r}")
    return _checked(manifest, path)


def missing_files(bundle_dir: Path) -> list[str]:
    """The manifest's files not yet in the bundle folder."""
    manifest = fetch_manifest(bundle_dir)
    if manifest is None:
        return []
    return [rel for rel in manifest["files"] if not (Path(bundle_dir) / rel).is_file()]


def ensure_fetched(bundle_dir: Path) -> list[Path]:
    """Fetch the bundle's missing files, each checked against its blob id;
    return the paths written ([] when nothing was missing)."""
    from trainnr.robot.asset_fetch import fetch_file  # noqa: PLC0415 - stdlib only

    bundle_dir = Path(bundle_dir)
    missing = missing_files(bundle_dir)
    if not missing:
        return []
    manifest = fetch_manifest(bundle_dir) or {}
    repo, commit = manifest["repository"], manifest["commit"]
    if os.environ.get(NO_FETCH_ENV) == "1":
        raise FileNotFoundError(
            f"{bundle_dir.name}: {len(missing)} files are fetched from {repo} on "
            f"first use, and {NO_FETCH_ENV}=1 forbids it; unset it, or run "
            f"`python -m trainnr.bundles.fetch {bundle_dir}`"
        )
    print(
        f"trainnr: fetching {len(missing)} files of {bundle_dir.name} from "
        f"{repo}@{commit[:7]} ({manifest.get('licence', 'see its repository')}); "
        "they are not redistributed with trainnr",
        file=sys.stderr,
    )
    written = []
    root = bundle_dir.resolve()
    for rel in missing:
        entry = manifest["files"][rel]
        target = bundle_dir / rel
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError(f"{bundle_dir}: {rel} would be written outside the bundle")
        data = fetch_file(repo, commit, entry["upstream"], blob=entry["blob"])
        written.append(safe_write.write_bytes(target, data, root))
    return written


def unavailable(bundle_dir: Path) -> str | None:
    """Why the bundle cannot be had whole right now, or None once it is:
    its files are fetched first, unless `TRAINNR_NO_ROBOT_FETCH=1` forbids
    it. A test skips by this reason; a failed download still raises."""
    bundle_dir = Path(bundle_dir)
    missing = missing_files(bundle_dir)
    if not missing:
        return None
    if os.environ.get(NO_FETCH_ENV) == "1":
        return (
            f"{bundle_dir.name}: {len(missing)} files are fetched on first use "
            f"and {NO_FETCH_ENV}=1 forbids it"
        )
    ensure_fetched(bundle_dir)
    return None


def main(argv: list[str] | None = None) -> int:
    """`python -m trainnr.bundles.fetch <bundle dir>...`: fetch ahead."""
    folders = sys.argv[1:] if argv is None else argv
    if not folders:
        print("usage: python -m trainnr.bundles.fetch <bundle dir>...", file=sys.stderr)
        return 2
    for folder in folders:
        written = ensure_fetched(Path(folder))
        left = len(missing_files(Path(folder)))
        print(f"{folder}: {len(written)} fetched, {left} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
