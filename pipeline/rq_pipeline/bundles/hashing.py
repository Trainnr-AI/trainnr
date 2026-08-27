"""Content hashes for artifact bundles.

The hash covers file bytes AND relative paths, sorted — so renaming a
file inside a bundle changes its identity, while the bundle's own
location on disk does not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 12 hex characters: short enough for a filename, long enough that a
# collision inside one project's artifact store is not a realistic event.
STAMP_LENGTH = 12


def bundle_hash(root: Path) -> str:
    """SHA-256 over every file under `root`, in sorted relative-path order."""
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
    )
    if not files:
        raise ValueError(f"bundle is empty: {root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


STAMP_SEPARATOR = "@"


def stamp(name: str, root: Path) -> str:
    """`name@hash` identity for an artifact, e.g. `scene-lab@3fa9c12ab45d`."""
    if STAMP_SEPARATOR in name:
        raise ValueError(f"artifact name must not contain '{STAMP_SEPARATOR}': {name}")
    return f"{name}{STAMP_SEPARATOR}{bundle_hash(root)[:STAMP_LENGTH]}"


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
