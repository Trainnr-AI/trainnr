"""Content hashes for artifact bundles.

The hash covers file bytes AND relative paths, sorted — so renaming a
file inside a bundle changes its identity, while the bundle's own
location on disk does not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"bundle is empty: {root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def stamp(name: str, root: Path) -> str:
    """`name@hash` identity for an artifact, e.g. `scene-lab@3fa9c12ab45d`."""
    if "@" in name:
        raise ValueError(f"artifact name must not contain '@': {name}")
    return f"{name}@{bundle_hash(root)[:STAMP_LENGTH]}"
