#!/usr/bin/env python3
"""One version for the whole product, written everywhere it appears.

    python3 tools/release.py check            # every copy agrees (CI runs this)
    python3 tools/release.py bump 0.2.0       # write the new version everywhere
    python3 tools/release.py show             # print the version

The source of truth is `trainnr/pyproject.toml`. The Python packages, the
Studio, the Claude Code plugin and its marketplace entry, the MCP Registry
entry and the citation file each carry a copy, because each is read by a
different tool; `check` fails the build when any copy disagrees, and
`bump` rewrites them all and, for a release (not a pre-release), dates
the CHANGELOG's *Unreleased* section under a fresh one.

The version follows Semantic Versioning with the pre-1.0 rule: a minor
bump (0.2.0) may break the public API, a patch bump (0.1.1) never does.
The public API is the MCP tools' names and arguments, the `trainnr`
command line, the record schemas (`trainnr-*/1`) and the Python import
paths (GOVERNANCE.md).
"""

from __future__ import annotations

import datetime
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "trainnr" / "pyproject.toml"
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
CHANGELOG = REPO / "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"


@dataclass(frozen=True)
class Copy:
    """One place the version is written: a file and the pattern around it.
    The pattern's single group is the version; at least `count` copies
    must be there (an entry such as server.json's package block may come
    and go), and every copy found must agree."""

    path: str
    pattern: str
    count: int = 1


COPIES: tuple[Copy, ...] = (
    Copy("trainnr/pyproject.toml", r'(?m)^version = "([^"]+)"'),
    Copy("trainnr/trainnr/__init__.py", r'(?m)^__version__ = "([^"]+)"'),
    Copy("trainnr-mjlab/pyproject.toml", r'(?m)^version = "([^"]+)"'),
    Copy(
        "trainnr-mjlab/src/trainnr_mjlab/__init__.py", r'(?m)^__version__ = "([^"]+)"'
    ),
    Copy("crates/trainnr-studio/Cargo.toml", r'(?m)^version = "([^"]+)"'),
    Copy(".claude-plugin/plugin.json", r'(?m)^  "version": "([^"]+)"'),
    Copy(".claude-plugin/marketplace.json", r'"version": "([^"]+)"', count=2),
    Copy("server.json", r'"version": "([^"]+)"'),
    Copy("CITATION.cff", r"(?m)^version: (\S+)$"),
)


def found(copy: Copy) -> list[str]:
    text = (REPO / copy.path).read_text(encoding="utf-8")
    return re.findall(copy.pattern, text)


def source_version() -> str:
    versions = found(COPIES[0])
    if len(versions) != 1:
        raise SystemExit(f"{SOURCE}: expected one version line, found {len(versions)}")
    return versions[0]


def check() -> int:
    want = source_version()
    problems = []
    for copy in COPIES:
        versions = found(copy)
        if len(versions) < copy.count:
            problems.append(
                f"{copy.path}: expected {copy.count} version(s), found {len(versions)}"
            )
        problems += [
            f"{copy.path}: {v} (the source says {want})" for v in versions if v != want
        ]
    if problems:
        print("version copies disagree:", *problems, sep="\n  ")
        print("fix with: python3 tools/release.py bump", want)
        return 1
    print(
        f"version {want}: {sum(c.count for c in COPIES)} copies "
        f"in {len(COPIES)} files agree"
    )
    return 0


def date_changelog(new: str) -> str:
    """A release (not a pre-release) closes the changelog's *Unreleased*
    section: it becomes `## [X.Y.Z] - YYYY-MM-DD` under a fresh, empty
    *Unreleased* heading, so the release notes are that section."""
    if "-" in new:
        return "a pre-release: the changelog's Unreleased section stays open"
    text = CHANGELOG.read_text(encoding="utf-8")
    if UNRELEASED not in text:
        return "CHANGELOG.md has no Unreleased section; nothing dated"
    today = datetime.date.today().isoformat()
    dated = f"{UNRELEASED}\n\n## [{new}] - {today}"
    CHANGELOG.write_text(text.replace(UNRELEASED, dated, 1), encoding="utf-8")
    return f"CHANGELOG.md: Unreleased is now [{new}] - {today}"


def bump(new: str) -> int:
    if not SEMVER.match(new):
        print(f"not a semantic version: {new!r}")
        return 1
    for copy in COPIES:
        path = REPO / copy.path
        text = path.read_text(encoding="utf-8")

        def put(match: re.Match[str]) -> str:
            whole, start = match.group(0), match.start(1) - match.start(0)
            return whole[:start] + new + whole[start + len(match.group(1)) :]

        updated, n = re.subn(copy.pattern, put, text)
        if n < copy.count:
            print(
                f"{copy.path}: expected {copy.count} version(s), found {n}; "
                "nothing written"
            )
            return 1
        path.write_text(updated, encoding="utf-8")
    print(
        f"version {new} written to {len(COPIES)} files; the Studio's Cargo.lock "
        "updates on its next build (`cargo build`), the uv lockfiles on `uv lock`"
    )
    print(date_changelog(new))
    return check()


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"]:
        return check()
    if argv[:1] == ["show"]:
        print(source_version())
        return 0
    if argv[:1] == ["bump"] and len(argv[1:]) == 1:
        return bump(argv[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
