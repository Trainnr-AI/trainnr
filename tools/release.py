#!/usr/bin/env python3
"""One version for the whole product, written everywhere it appears.

    python3 tools/release.py check            # every copy agrees (CI runs this)
    python3 tools/release.py bump 0.2.0       # write the new version everywhere
    python3 tools/release.py show             # print the version

The source of truth is `trainnr/pyproject.toml`. The Python packages and
their uv lockfiles, the Studio and its Cargo.lock, the Claude Code plugin
and its marketplace entry, the MCP Registry entry and the citation file
each carry a copy, because each is read by a different tool; `check`
fails the build when any copy disagrees, and `bump` rewrites them all
(nothing is written unless every copy is found). A lockfile holds its
project's own version, so a bump that left one behind would fail
`uv run --locked` and `cargo build --locked` in CI.

For a release (not a pre-release), `bump` also dates the CHANGELOG's
*Unreleased* section under a fresh one and writes the same date as the
citation's `date-released`; `check` holds the two to agree, and refuses a
`date-released` for a version the CHANGELOG has not dated.

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
# The pre-releases a Python lockfile can carry: uv writes PEP 440's normal
# form (0.2.0-rc.1 -> 0.2.0rc1), so those copies are compared in it.
PRERELEASE = re.compile(r"^(\d+\.\d+\.\d+)(?:-(alpha|a|beta|b|rc)\.?(\d+))?$")
PEP440_TAG = {"alpha": "a", "a": "a", "beta": "b", "b": "b", "rc": "rc"}
CHANGELOG = REPO / "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"
CITATION = REPO / "CITATION.cff"
DATE_RELEASED = re.compile(r'(?m)^date-released: "?(\d{4}-\d{2}-\d{2})"?\n')
CITATION_VERSION = re.compile(r"(?m)^version: \S+\n")


@dataclass(frozen=True)
class Copy:
    """One place the version is written: a file and the pattern around it.
    The pattern's single group is the version; at least `count` copies
    must be there (an entry such as server.json's package block may come
    and go), and every copy found must agree. A `python` copy is written
    in PEP 440's normal form, as uv writes its lockfiles."""

    path: str
    pattern: str
    count: int = 1
    python: bool = False


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
    # The lockfiles' entries for the projects themselves (the editable
    # packages), which uv and cargo compare against the manifests.
    Copy(
        "trainnr/uv.lock",
        r'(?m)^name = "trainnr"\nversion = "([^"]+)"\nsource = \{ editable = "\." \}',
        python=True,
    ),
    Copy(
        "trainnr-mjlab/uv.lock",
        r'(?m)^name = "trainnr(?:-mjlab)?"\nversion = "([^"]+)"\n'
        r'source = \{ editable = "(?:\.|\.\./trainnr)" \}',
        count=2,
        python=True,
    ),
    Copy(
        "crates/trainnr-studio/Cargo.lock",
        r'(?m)^name = "trainnr-studio"\nversion = "([^"]+)"$',
    ),
)


def python_form(version: str) -> str | None:
    """The version as PEP 440 normalises it (what uv writes), or None for a
    pre-release tag Python packaging cannot carry."""
    m = PRERELEASE.match(version)
    if m is None:
        return None
    base, tag, number = m.groups()
    return base if tag is None else f"{base}{PEP440_TAG[tag]}{number}"


def expected(copy: Copy, version: str) -> str:
    return (python_form(version) or version) if copy.python else version


def found(copy: Copy) -> list[str]:
    text = (REPO / copy.path).read_text(encoding="utf-8")
    return re.findall(copy.pattern, text)


def source_version() -> str:
    versions = found(COPIES[0])
    if len(versions) != 1:
        raise SystemExit(f"{SOURCE}: expected one version line, found {len(versions)}")
    return versions[0]


def changelog_date(version: str) -> str | None:
    """The date the CHANGELOG gives a released version, or None."""
    text = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.is_file() else ""
    m = re.search(
        r"(?m)^## \[" + re.escape(version) + r"\] - (\d{4}-\d{2}-\d{2})\s*$", text
    )
    return m.group(1) if m else None


def citation_date_problems(version: str) -> list[str]:
    """CITATION.cff's date-released is the CHANGELOG's date for the version,
    and absent while the version is unreleased."""
    m = DATE_RELEASED.search(CITATION.read_text(encoding="utf-8"))
    cited, dated = (m.group(1) if m else None), changelog_date(version)
    if cited == dated:
        return []
    if dated is None:
        return [
            f"CITATION.cff: date-released {cited} for {version}, which the "
            "CHANGELOG has not released (bump dates both)"
        ]
    return [
        f"CITATION.cff: date-released {cited}, the CHANGELOG dates {version} {dated}"
    ]


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
            f"{copy.path}: {v} (the source says {want})"
            for v in versions
            if v != expected(copy, want)
        ]
    dates = citation_date_problems(want)
    if problems:
        print("version copies disagree:", *problems, sep="\n  ")
        print("fix with: python3 tools/release.py bump", want)
    if dates:
        print("release dates disagree:", *dates, sep="\n  ")
    if problems or dates:
        return 1
    released = changelog_date(want)
    print(
        f"version {want}: {sum(c.count for c in COPIES)} copies "
        f"in {len(COPIES)} files agree; "
        + (f"released {released}" if released else "not yet released")
    )
    return 0


def date_changelog(new: str, today: str) -> str:
    """A release (not a pre-release) closes the changelog's *Unreleased*
    section: it becomes `## [X.Y.Z] - YYYY-MM-DD` under a fresh, empty
    *Unreleased* heading, so the release notes are that section."""
    if "-" in new:
        return "a pre-release: the changelog's Unreleased section stays open"
    text = CHANGELOG.read_text(encoding="utf-8")
    if UNRELEASED not in text:
        return "CHANGELOG.md has no Unreleased section; nothing dated"
    dated = f"{UNRELEASED}\n\n## [{new}] - {today}"
    CHANGELOG.write_text(text.replace(UNRELEASED, dated, 1), encoding="utf-8")
    return f"CHANGELOG.md: Unreleased is now [{new}] - {today}"


def date_citation(new: str, today: str) -> str:
    """The citation's date-released follows the CHANGELOG: the release's
    date, and no date for a pre-release or an undated version."""
    text = CITATION.read_text(encoding="utf-8")
    text = DATE_RELEASED.sub("", text)
    if changelog_date(new) != today:
        CITATION.write_text(text, encoding="utf-8")
        return "CITATION.cff: no date-released until the version is released"
    m = CITATION_VERSION.search(text)
    if m is None:
        return "CITATION.cff has no version line; nothing dated"
    text = text[: m.end()] + f'date-released: "{today}"\n' + text[m.end() :]
    CITATION.write_text(text, encoding="utf-8")
    return f"CITATION.cff: date-released {today}"


def bump(new: str) -> int:
    if not SEMVER.match(new):
        print(f"not a semantic version: {new!r}")
        return 1
    if python_form(new) is None:
        print(
            f"{new!r}: a pre-release must be -alpha.N, -beta.N or -rc.N, "
            "the tags Python packaging can carry"
        )
        return 1
    # Every copy is found before any file is written: a half-bumped tree
    # is worse than none.
    updates: dict[Path, str] = {}
    for copy in COPIES:
        path = REPO / copy.path
        text = updates.get(path) or path.read_text(encoding="utf-8")
        value = expected(copy, new)

        def put(match: re.Match[str], value: str = value) -> str:
            whole, start = match.group(0), match.start(1) - match.start(0)
            return whole[:start] + value + whole[start + len(match.group(1)) :]

        updated, n = re.subn(copy.pattern, put, text)
        if n < copy.count:
            print(
                f"{copy.path}: expected {copy.count} version(s), found {n}; "
                "nothing written"
            )
            return 1
        updates[path] = updated
    for path, text in updates.items():
        path.write_text(text, encoding="utf-8")
    print(f"version {new} written to {len(updates)} files")
    today = datetime.date.today().isoformat()
    print(date_changelog(new, today))
    print(date_citation(new, today))
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
