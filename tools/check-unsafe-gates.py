#!/usr/bin/env python3
"""Is unsafe still forbidden in every crate?

There is no unsafe code in this repo and the compiler is what keeps it
that way: `unsafe_code = "forbid"` in each crate's manifest, which cannot
be switched off locally by an `#[allow]`.

That protection is only as durable as the line declaring it, and deleting
a line from a manifest is silent. This checks the guard rather than
grepping for the thing the guard already prevents. The rig's crates and
firmware, which this gate used to cover too, moved to their own
repository with the same gate (https://github.com/Trainnr-AI/rig,
2026-10-02).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
manifests = sorted((ROOT / "crates").glob("*/Cargo.toml"))
if not manifests:  # a vacuous pass would be worse than a failure
    raise SystemExit("check-unsafe-gates: no crates/*/Cargo.toml found")
missing = []
for manifest in manifests:
    body = manifest.read_text(encoding="utf-8")
    rel = manifest.relative_to(ROOT)
    inherits = re.search(r"^\[lints\]\s*\nworkspace\s*=\s*true", body, re.M)
    own = re.search(r'^unsafe_code\s*=\s*"forbid"', body, re.M)
    if inherits and not own:
        missing.append(f"{rel}: inherits lints from a workspace this repo lacks")
    elif not own:
        missing.append(f"{rel}: no unsafe gate")

for m in missing:
    print(m)
print(f"unsafe gate: {len(missing)} problem(s) in {len(manifests)} manifest(s)")
sys.exit(1 if missing else 0)
