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
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

FETCH_FILE = "FETCH.json"
FETCH_SCHEMA = "trainnr-bundle-fetch/1"
NO_FETCH_ENV = "TRAINNR_NO_ROBOT_FETCH"


def fetch_manifest(bundle_dir: Path) -> dict[str, Any] | None:
    """The bundle's fetch manifest, or None when it carries every file."""
    path = Path(bundle_dir) / FETCH_FILE
    if not path.is_file():
        return None
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FETCH_SCHEMA:
        raise ValueError(f"{path}: schema is not {FETCH_SCHEMA!r}")
    return manifest


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
    for rel in missing:
        entry = manifest["files"][rel]
        data = fetch_file(repo, commit, entry["upstream"], blob=entry["blob"])
        target = bundle_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".part")
        staging.write_bytes(data)
        staging.replace(target)
        written.append(target)
    return written


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
