"""Where robot bundles live — one answer, overridable, loud when wrong.

The built-in bundles ship in the repository's `robots/` directory; a
wheel installed into site-packages has no such neighbour. Until the
bundles ship as package data, `RQ_ROBOTS_DIR` names the directory (a
checkout's `robots/`, or wherever a deployment keeps them), and a
missing bundle file fails with that name in the message instead of a
MuJoCo parse error three frames down.
"""

from __future__ import annotations

import os
from pathlib import Path

ROBOTS_DIR_ENV = "RQ_ROBOTS_DIR"
_CHECKOUT_ROBOTS = Path(__file__).resolve().parents[3] / "robots"


def robots_dir() -> Path:
    """`$RQ_ROBOTS_DIR`, else the checkout's `robots/` beside `pipeline/`."""
    override = os.environ.get(ROBOTS_DIR_ENV)
    return Path(override).expanduser() if override else _CHECKOUT_ROBOTS


def bundle_file(bundle: str, filename: str) -> Path:
    """A file inside a named bundle — the path only; `require_bundle_file`
    checks it exists when a builder is about to load it."""
    return robots_dir() / bundle / filename


def require_bundle_file(path: Path) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"bundle file {path} does not exist — set {ROBOTS_DIR_ENV} to the "
            "directory holding the robot bundles (a checkout's robots/)"
        )
    return Path(path)
