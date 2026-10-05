"""Writing inside a folder that may come from someone else, never through
a link it carries.

A project (or a robot bundle) is data, and a shared one can hold a link
where trainnr writes: `.index/project.json.tmp` pointing at a file in the
user's home, `.index/commands` at a folder of theirs. Writing through it
overwrote or deleted files outside the project (the security review of
2026-10-05 reproduced both with one `describe_project` call). Every write
the project layer, the job runner and the bundles make goes through here:

- the target, and every folder between it and its root (the project that
  holds it, or the root a caller names, such as a bundle's own folder),
  must not be a link; the root itself and the folders above it may be
  (a user's projects home is often a link, and macOS's /tmp is one);
- a file is written whole into a fresh staging file created exclusively
  (`O_EXCL`, `O_NOFOLLOW`), then renamed over the target, which replaces
  a name and never follows one;
- a log is opened for appending without following a link.

Stdlib only: the lowest layers use it.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from pathlib import Path
from typing import BinaryIO

PROJECT_MARKER = "project.json"
STAGING_SUFFIX = ".tmp"
ENCODING = "utf-8"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_BINARY = getattr(os, "O_BINARY", 0)  # Windows: no newline translation
FILE_MODE = 0o666  # less the user's umask, as a plain open would


class LinkRefusedError(OSError):
    """A write would have gone through a link."""


def project_root_of(path: Path) -> Path | None:
    """The nearest folder above `path` that holds a project's marker."""
    for parent in Path(path).absolute().parents:
        if (parent / PROJECT_MARKER).is_file():
            return parent
    return None


def refuse_links(path: Path, root: Path | None = None, *, target: bool = True) -> None:
    """LinkRefusedError when `path` (unless `target` is False), or a folder
    between it and `root`, is a link. `root` defaults to the project
    holding `path`; outside a project only the target itself is checked."""
    path = Path(path).absolute()
    chain = [path] if target else []
    stop = Path(root).absolute() if root is not None else project_root_of(path)
    if stop is not None and stop in path.parents:
        for parent in path.parents:
            if parent == stop:
                break
            chain.append(parent)
    for part in chain:
        if part.is_symlink():
            raise LinkRefusedError(
                f"{path}: {part} is a link; trainnr does not write through links"
            )


def write_bytes(path: Path, data: bytes, root: Path | None = None) -> Path:
    """Write `data` to `path` in one step, never through a link."""
    path = Path(path)
    refuse_links(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{secrets.token_hex(6)}{STAGING_SUFFIX}")
    fd = os.open(
        staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _BINARY, FILE_MODE
    )
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(staging, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(staging)
        raise
    return path


def write_text(path: Path, text: str, root: Path | None = None) -> Path:
    """`write_bytes` of UTF-8 text."""
    return write_bytes(path, text.encode(ENCODING), root)


def open_append(path: Path, root: Path | None = None) -> BinaryIO:
    """The file opened for appending in binary, created if absent, never
    through a link."""
    path = Path(path)
    refuse_links(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _BINARY, FILE_MODE
    )
    return os.fdopen(fd, "ab")


def remove(path: Path, root: Path | None = None) -> None:
    """Delete a file (absent is fine) in a folder that is not a link."""
    path = Path(path)
    refuse_links(path, root, target=False)  # unlinking a link removes only it
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
