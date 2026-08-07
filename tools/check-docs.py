#!/usr/bin/env python3
"""Check that the docs describe code that exists.

Docs rot silently. A method gets deleted, the paragraph explaining it
stays, and the next person goes looking for something that is not there —
which is worse than no documentation, because it costs them a search
before they stop trusting the page.

Checks three things, all mechanically:

  1. every `path/to/file.ext` in backticks resolves
  2. every markdown link to a local file resolves
  3. every backticked `OurType::member` exists in the source

Only types WE define are checked. `Pull::Up` and `Level::High` belong to
embassy, `ModelFormat::MLProgram` to CoreML, and this script has no
business asserting anything about them — the first version did, and every
one of those was a false alarm. A checker that cries wolf gets ignored,
and then it is worth nothing when it is right.

`docs/07-progress-log.md` is exempt entirely: it is a dated history, and
it is *supposed* to name things that were later removed.

Usage:  python3 tools/check-docs.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = "07-progress-log.md"
# Placeholders in usage examples, not files that should exist.
PLACEHOLDERS = {"chase.perc", "run.wire", "mine.wire", "chase.pero", "x.rrd", "run.perc"}

docs = sorted(list((ROOT / "docs").rglob("*.md")) + list(ROOT.glob("*.md")))
source = subprocess.run(
    ["git", "grep", "-h", "", "--", "*.rs", "*.toml", "*.sh", "*.py", "*.ts"],
    cwd=ROOT, capture_output=True, text=True,
).stdout

# Types this workspace defines. Anything else in a `Foo::bar` is somebody
# else's crate and not ours to police.
OURS = set(re.findall(r"\b(?:pub\s+)?(?:struct|enum|trait|type)\s+([A-Z][A-Za-z0-9_]*)", source))

problems = []

for doc in docs:
    rel = doc.relative_to(ROOT)
    if doc.name == HISTORY:
        continue
    body = doc.read_text()

    # ---- 1. file paths ----
    for path in set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:rs|toml|sh|py|md|ts|uf2))`", body)):
        if path in PLACEHOLDERS or (ROOT / path).exists():
            continue
        # a bare basename mentioned in prose is fine if it exists anywhere
        if list(ROOT.rglob(Path(path).name)):
            continue
        problems.append(f"{rel}: path does not exist: {path}")

    # ---- 2. markdown links to local files ----
    for link in set(re.findall(r"\]\(([A-Za-z0-9_./-]+\.(?:md|rs|sh|py|toml))\)", body)):
        if (ROOT / link).exists() or (doc.parent / link).exists():
            continue
        problems.append(f"{rel}: link goes nowhere: {link}")

    # ---- 3. code identifiers, ours only ----
    for ident in set(re.findall(r"`([A-Z][A-Za-z0-9_]*::[a-zA-Z_][A-Za-z0-9_]*)`", body)):
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

for p in sorted(problems):
    print(f"  {p}")
print(f"\n{len(problems)} problem(s) in {len(docs)} documents")
sys.exit(1 if problems else 0)
