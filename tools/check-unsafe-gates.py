#!/usr/bin/env python3
"""Is unsafe still forbidden in every crate?

There is no unsafe code in this repo and the compiler is what keeps it
that way — `unsafe_code = "forbid"` in the manifests, and
`#![forbid(unsafe_code)]` in each firmware crate, which cannot be switched
off locally by an `#[allow]`.

That protection is only as durable as the lines declaring it, and deleting
a line from a manifest is silent. This checks the guard rather than
grepping for the thing the guard already prevents.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
missing = []

for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
    body = manifest.read_text()
    inherits = re.search(r"^\[lints\]\s*\nworkspace\s*=\s*true", body, re.M)
    own = re.search(r'^unsafe_code\s*=\s*"forbid"', body, re.M)
    if not (inherits or own):
        missing.append(f"{manifest.relative_to(ROOT)}: no unsafe gate")

# The workspace block the inheriting crates rely on.
if not re.search(
    r'^unsafe_code\s*=\s*"forbid"', (ROOT / "Cargo.toml").read_text(), re.M
):
    missing.append("Cargo.toml: [workspace.lints.rust] no longer forbids unsafe")

# Firmware is outside the workspace, so each declares it in source.
#
# ⚠️ Both `main.rs` AND `lib.rs`. This used to glob only `main.rs`, and on
# 2026-08-11 `firmware/support` was added as a library — code compiled
# into EVERY firmware binary, and therefore running on the robot, that
# this check could not see. It happened to carry the attribute. Nothing
# would have noticed if it stopped.
#
# A crate root is enough: `#![forbid(unsafe_code)]` covers every module in
# the crate, so `support/src/usb.rs` needs no line of its own.
#
# `build.rs` is deliberately NOT checked — build scripts run on the
# laptop at compile time and never reach the chip.
firmware_roots = sorted(
    p
    for pattern in ("*/src/main.rs", "*/src/lib.rs")
    for p in (ROOT / "firmware").glob(pattern)
)
for root in firmware_roots:
    text = root.read_text()
    # Must be a real attribute, not the words inside a doc comment.
    if not re.search(r"^#!\[forbid\(unsafe_code\)\]", text, re.M):
        missing.append(f"{root.relative_to(ROOT)}: no #![forbid(unsafe_code)]")

for m in missing:
    print(f"  {m}")
print(f"{len(missing)} crate(s) unprotected")
sys.exit(1 if missing else 0)
