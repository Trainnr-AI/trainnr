"""Where robot bundles live — one answer, overridable, loud when wrong.

The built-in bundles ship in the repository's `robots/` directory; a
wheel installed into site-packages has no such neighbour. Until the
bundles ship as package data, `TRAINNR_ROBOTS_DIR` names the directory (a
checkout's `robots/`, or wherever a deployment keeps them), and a
missing bundle file fails with that name in the message instead of a
MuJoCo parse error three frames down.
"""

from __future__ import annotations

import os
from pathlib import Path

from trainnr.paths import checkout

ROBOTS_DIR_ENV = "TRAINNR_ROBOTS_DIR"
_CHECKOUT_ROBOTS = checkout() / "robots"


def robots_dir() -> Path:
    """`$TRAINNR_ROBOTS_DIR`, else the checkout's `robots/` beside `trainnr/`:
    the robot LIBRARY — the rigs the repository ships."""
    override = os.environ.get(ROBOTS_DIR_ENV)
    return Path(override).expanduser() if override else _CHECKOUT_ROBOTS


# Directories searched before the library, most recent first: a project's
# own `robots/` (registered by the project layer when a project is
# current), so a robot onboarded into a project builds tasks like a
# library rig. Higher layers register; this module only searches.
_SEARCH_ROOTS: list[Path] = []


def add_search_root(root: Path, *, replace: bool = False) -> None:
    """Search `root` for bundles before the library (idempotent; the
    latest registration is searched first). `replace=True` makes it the
    ONLY project root: a long-lived server that switches projects must
    not keep listing the previous project's robots."""
    root = Path(root)
    if replace:
        _SEARCH_ROOTS.clear()
    elif root in _SEARCH_ROOTS:
        _SEARCH_ROOTS.remove(root)
    _SEARCH_ROOTS.insert(0, root)


def search_roots() -> list[Path]:
    """Every directory a bundle may live in, in search order."""
    return [*_SEARCH_ROOTS, robots_dir()]


def find_bundle(name: str) -> Path | None:
    """The directory of the bundle called `name`, project first."""
    for root in search_roots():
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def bundle_dirs() -> dict[str, Path]:
    """Every bundle by name across the search roots; a project's shadows
    the library's of the same name."""
    found: dict[str, Path] = {}
    for root in reversed(search_roots()):
        if root.is_dir():
            for entry in root.iterdir():
                if entry.is_dir():
                    found[entry.name] = entry
    return found


def bundle_file(bundle: str, filename: str) -> Path:
    """A file inside a named bundle — the path only; `require_bundle_file`
    checks it exists when a builder is about to load it. A bundle that
    exists nowhere resolves into the library, so the error names it."""
    root = find_bundle(bundle) or (robots_dir() / bundle)
    return root / filename


def require_bundle_file(path: Path) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"bundle file {path} does not exist — set {ROBOTS_DIR_ENV} to the "
            "directory holding the robot bundles (a checkout's robots/)"
        )
    return Path(path)
