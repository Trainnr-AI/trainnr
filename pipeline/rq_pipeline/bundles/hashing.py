"""Content hashes for artifact bundles.

The hash covers file bytes AND relative paths, sorted — so renaming a
file inside a bundle changes its identity, while the bundle's own
location on disk does not.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 12 hex characters: short enough for a filename, long enough that a
# collision inside one project's artifact store is not a realistic event.
STAMP_LENGTH = 12


def bundle_hash(root: Path, exclude: tuple[str, ...] = ()) -> str:
    """SHA-256 over every file under `root`, in sorted relative-path order.
    `exclude` names files that live inside the root and are not its
    content (glob patterns on the POSIX relative path; an artifact's run
    records, `project.kinds.RUN_RECORDS`). Paths enter the digest in POSIX
    form so a bundle hashes the same on Windows as on Linux and macOS
    (identical bytes to `str(path)` on the latter, so no stamp moves)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"bundle root does not exist: {root}")
    digest = hashlib.sha256()
    if root.is_file():
        digest.update(root.name.encode())
        digest.update(root.read_bytes())
        return digest.hexdigest()
    # Hidden files are not bundle content: a Finder `.DS_Store` or an
    # editor's `.swp` would otherwise give the same bundle a different
    # identity on a different machine, and no error would say why.
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and not any(
            fnmatch.fnmatch(path.relative_to(root).as_posix(), pattern)
            for pattern in exclude
        )
    )
    if not files:
        what = "once its excluded files are left out" if exclude else ""
        raise ValueError(f"bundle is empty {what}: {root}".replace("  ", " "))
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


STAMP_SEPARATOR = "@"

# The folder a bundle's fit records live in (one spelling; before
# 2026-09-25 it was spelled in four modules).
FITS_DIR = "fits"
# The file a bundle's importer audit lives in (`bundles.bundle`).
AUDIT_FILE = "audit.json"
# Records ABOUT a bundle, not its content: identifying the robot or
# auditing its import must not change which robot it is. Until
# 2026-09-25 a fit or an audit written into the bundle moved its stamp,
# and every checkpoint certified on the old stamp was refused by the
# identity gate (go2@5003bf617b5f -> go2@c699dc1b0772 on 2026-09-24).
BUNDLE_RECORDS: tuple[str, ...] = (f"{FITS_DIR}/*", AUDIT_FILE)


def stamp(name: str, root: Path) -> str:
    """`name@hash` identity for an artifact, e.g. `scene-lab@3fa9c12ab45d`.
    A directory's records about itself (`BUNDLE_RECORDS`) are not its
    content and never move its stamp."""
    if STAMP_SEPARATOR in name:
        raise ValueError(f"artifact name must not contain '{STAMP_SEPARATOR}': {name}")
    digest = bundle_hash(root, exclude=() if Path(root).is_file() else BUNDLE_RECORDS)
    return f"{name}{STAMP_SEPARATOR}{digest[:STAMP_LENGTH]}"


def is_stamp(value: str) -> bool:
    return STAMP_SEPARATOR in value


def require_stamp(value: str, what: str = "source") -> str:
    """The one rule every consumer of an identity applies: nothing is
    nameable without its hash. Until 2026-08-26 six modules each spelled
    the check (and three different messages); this is the only copy."""
    if not is_stamp(value):
        raise ValueError(
            f"{what} must be a name{STAMP_SEPARATOR}hash stamp, got {value!r} — "
            "stamp it with rq_pipeline.bundles.stamp first"
        )
    return value


def fields_hash(fields: Mapping[str, Any]) -> str:
    """STAMP_LENGTH hex digits over a JSON-serialisable mapping, keys
    sorted: the content half of a stamp for anything that is data — a
    protocol's fields, a task spec. Non-JSON values are stringified."""
    encoded = json.dumps(dict(fields), sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:STAMP_LENGTH]


def content_stamp(name: str, fields: Mapping[str, Any]) -> str:
    """`name@hash` over `fields` — the same shape as a bundle's stamp, so
    a task spec is nameable the way a bundle is (`require_stamp` accepts it)."""
    return f"{name}{STAMP_SEPARATOR}{fields_hash(fields)}"
