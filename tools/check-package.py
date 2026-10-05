#!/usr/bin/env python3
"""The wheels a user installs from PyPI carry the licence and the notices.

    python3 tools/check-package.py

A redistribution carries the licence (FSL-1.1-ALv2's Redistribution
clause), and the third-party code under Apache-2.0 carries that licence's
text and the NOTICE (its 4(a) and 4(d)). Each package's pyproject lists
`LICENSE`, `LICENSES/Apache-2.0.txt` and `NOTICE` as licence files; trainnr-mjlab's
NOTICE and trainnr's are copies of the root one (each wheel ships the
third-party code NOTICE lists), and both packages' LICENSE and
LICENSES/Apache-2.0.txt are copies of the root's.
This checks the copies are identical, then builds both wheels with
`uv build` into a scratch directory and refuses one without LICENSE or
NOTICE under its `.dist-info/licenses/`. CI runs it as the required
`package` job (gates.yml); tools/verify.sh runs it too.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("trainnr", "trainnr-mjlab")
# (copy, original): files that must be byte-identical.
COPIES = (
    ("trainnr-mjlab/NOTICE", "NOTICE"),
    ("trainnr/NOTICE", "NOTICE"),
    ("trainnr/LICENSE", "LICENSE"),
    ("trainnr-mjlab/LICENSE", "LICENSE"),
    ("trainnr/LICENSES/Apache-2.0.txt", "LICENSES/Apache-2.0.txt"),
    ("trainnr-mjlab/LICENSES/Apache-2.0.txt", "LICENSES/Apache-2.0.txt"),
)
LICENCE_FILES = ("LICENSE", "LICENSES/Apache-2.0.txt", "NOTICE")


def main() -> int:
    failed = 0
    for copy, original in COPIES:
        if not filecmp.cmp(REPO / copy, REPO / original, shallow=False):
            print(f"{copy} differs from {original}: copy it again")
            failed = 1
    with tempfile.TemporaryDirectory() as dist:
        for package in PACKAGES:
            build = subprocess.run(
                ["uv", "build", "-q", "--out-dir", dist],
                cwd=REPO / package,
                capture_output=True,
                text=True,
                check=False,
            )
            if build.returncode != 0:
                print(f"{package}: the wheel did not build\n{build.stderr}")
                failed = 1
        wheels = sorted(Path(dist).glob("*.whl"))
        for wheel in wheels:
            names = zipfile.ZipFile(wheel).namelist()
            missing = [
                f
                for f in LICENCE_FILES
                if not any(n.endswith(".dist-info/licenses/" + f) for n in names)
            ]
            if missing:
                print(f"{wheel.name}: no {' or '.join(missing)}")
                failed = 1
            else:
                print(f"{wheel.name}: {', '.join(LICENCE_FILES)} present")
        if len(wheels) != len(PACKAGES):
            print(f"expected {len(PACKAGES)} wheels, built {len(wheels)}")
            failed = 1
    return failed


if __name__ == "__main__":
    sys.exit(main())
