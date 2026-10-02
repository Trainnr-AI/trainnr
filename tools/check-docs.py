#!/usr/bin/env python3
"""Check that the docs describe code that exists.

Docs rot silently. A method gets deleted, the paragraph explaining it
stays, and the next person goes looking for something that is not there —
which is worse than no documentation, because it costs them a search
before they stop trusting the page.

Checks four things, all mechanically:

  1. every `path/to/file.ext` in backticks resolves
  2. every markdown link to a local file resolves
  3. every backticked `OurType::member` exists in the source
  4. the progress log's dates run newest-first, as its header promises

Only types WE define are checked. `Pull::Up` and `Level::High` belong to
embassy, `ModelFormat::MLProgram` to CoreML, and this script has no
business asserting anything about them — the first version did, and every
one of those was a false alarm. A checker that cries wolf gets ignored,
and then it is worth nothing when it is right.

`docs/07-progress-log.md` is exempt entirely: it is a dated history, and
it is *supposed* to name things that were later removed.

Usage:  python3 tools/check-docs.py
"""

import posixpath
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = "07-progress-log.md"
# Placeholders in usage examples, not files that should exist.
PLACEHOLDERS = {
    "chase.perc",
    "run.wire",
    "mine.wire",
    "chase.pero",
    "x.rrd",
    "run.perc",
}

docs = sorted(
    list((ROOT / "docs").rglob("*.md"))
    + list(ROOT.glob("*.md"))
    # The two READMEs that name tools and modules by file — ungated until
    # 2026-08-27, when a review found rows describing tools that had moved on.
    + [ROOT / "tools" / "README.md", ROOT / "trainnr" / "README.md"]
)

# The universe of files a doc may name is what GIT TRACKS, not what this
# laptop's filesystem holds. The filesystem version passed locally while
# failing in CI the day a doc named a file inside a git-ignored clone —
# a reference that was dead for every machine but one. (Caught by CI's
# first ever run.)
tracked = set(
    subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.splitlines()
)
tracked_basenames = {Path(p).name for p in tracked}

source = subprocess.run(
    ["git", "grep", "-h", "", "--", "*.rs", "*.toml", "*.sh", "*.py", "*.ts"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=True,
).stdout
if not source:  # a silent empty grep would pass check 3 vacuously
    raise SystemExit("check-docs: git grep returned no source text")

# Types this workspace defines. Anything else in a `Foo::bar` is somebody
# else's crate and not ours to police.
OURS = set(
    re.findall(
        r"\b(?:pub\s+)?(?:struct|enum|trait|type)\s+([A-Z][A-Za-z0-9_]*)", source
    )
)

problems = []

for doc in docs:
    rel = doc.relative_to(ROOT)
    if doc.name == HISTORY:
        continue
    body = doc.read_text(encoding="utf-8")
    # An archived doc (restored from a retired branch, banner in its
    # first lines) describes code that never merged: its paths are not
    # promises about this tree. Links and identifiers are still checked.
    archived = "> **Archived" in body[:1200]

    # ---- 1. file paths ----
    for path in set(
        re.findall(r"`([A-Za-z0-9_./-]+\.(?:rs|toml|sh|py|md|ts|uf2))`", body)
    ):
        if archived or path in PLACEHOLDERS or path in tracked:
            continue
        # a bare basename mentioned in prose is fine if git tracks it anywhere
        if Path(path).name in tracked_basenames:
            continue
        problems.append(f"{rel}: path does not exist: {path}")

    # ---- 2. markdown links to local files ----
    for link in set(
        re.findall(r"\]\(([A-Za-z0-9_./-]+\.(?:md|rs|sh|py|toml))\)", body)
    ):
        from_root = posixpath.normpath(link)
        from_doc = posixpath.normpath(str(rel.parent / link))
        if from_root in tracked or from_doc in tracked:
            continue
        problems.append(f"{rel}: link goes nowhere: {link}")

    # ---- 3. code identifiers, ours only ----
    for ident in set(
        re.findall(r"`([A-Z][A-Za-z0-9_]*::[a-zA-Z_][A-Za-z0-9_]*)`", body)
    ):
        ty, member = ident.split("::", 1)
        if ty not in OURS:
            continue
        # A method, a const, an associated type, or an enum variant —
        # any definition of that name counts. Deliberately loose: the goal
        # is catching DELETIONS, not enforcing which namespace it sits in.
        defined = re.search(
            rf"\b(?:fn|const|struct|enum|type|static)\s+{re.escape(member)}\b", source
        ) or re.search(rf"^\s*{re.escape(member)}\s*[ ,({{]", source, re.M)
        if not defined:
            problems.append(f"{rel}: no such item in the source: {ident}")

# ---- 4. the progress log is ordered ----
#
# It says "Newest entries first" at the top, and an entry once landed in
# the newest slot while belonging eight entries down — a scripted insert
# whose anchor did not land where intended. Nothing noticed, because
# nothing was looking. Dates only: same-day ordering ("late", "night",
# "very late") is a human judgement and stays one.
log = ROOT / "docs" / HISTORY
if log.exists():
    body = log.read_text(encoding="utf-8")
    dates = re.findall(r"^## (\d{4}-\d{2}-\d{2})", body, re.M)
    # A heading that misses the date shape would fall out of the ordering
    # check below without a word; name it instead.
    for heading in re.findall(r"^## (.*)$", body, re.M):
        if not re.match(r"\d{4}-\d{2}-\d{2}\b", heading):
            problems.append(
                f"docs/{HISTORY}: heading '{heading[:40]}' does not start with a date"
            )
    # pairwise by index — zip's strict kwarg needs Python 3.10 and the
    # Mac's system python3 is 3.9; the gate must run on both boxes.
    for index in range(1, len(dates)):
        older, newer = dates[index], dates[index - 1]
        if older > newer:
            problems.append(
                f"docs/{HISTORY}: {older} appears below {newer}, but the log "
                f"is newest-first"
            )

for p in sorted(problems):
    print(f"  {p}")
print(f"\n{len(problems)} problem(s) in {len(docs)} documents")
sys.exit(1 if problems else 0)
