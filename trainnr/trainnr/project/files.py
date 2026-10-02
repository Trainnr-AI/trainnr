"""Reading and writing the project's JSON records — one door each, UTF-8
every time, an atomic replace on every write.

Every module of the project layer read `json.loads(path.read_text())`
its own way before 2026-09-12: half with the encoding named, half
without, some tolerating a missing file and some not. A record is read
here and nowhere else; a caller that needs "empty when absent" says
so, and one that needs the file refuses by name through `json.load`'s
own errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ENCODING = "utf-8"
STAGING_SUFFIX = ".tmp"


def read_json(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    """The JSON object in `path`. With `missing_ok`, an absent, unreadable
    or non-object file reads as `{}`; without it, the error names the
    file."""
    path = Path(path)
    if missing_ok:
        try:
            raw = json.loads(path.read_text(encoding=ENCODING))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}
    raw = json.loads(path.read_text(encoding=ENCODING))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a JSON object was expected")
    return raw


def read_text(path: Path, *, errors: str = "strict") -> str:
    """The file's text, UTF-8."""
    return Path(path).read_text(encoding=ENCODING, errors=errors)


def write_json(path: Path, record: Any, *, default: Any = None) -> Path:
    """Write `record` as indented JSON with a trailing newline, through a
    staging file replaced in one step, so a reader never sees half of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + STAGING_SUFFIX)
    staging.write_text(
        json.dumps(record, indent=1, default=default) + "\n", encoding=ENCODING
    )
    staging.replace(path)
    return path


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=ENCODING)
    return path
