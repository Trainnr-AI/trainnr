#!/usr/bin/env python3
"""The dependencies, checked: known vulnerabilities in every locked Python
set (pip-audit against PyPI's advisory database), and the Studio's Rust
dependencies through cargo-deny (advisories, licences, sources).

    python3 tools/supply-chain.py            # Python audit, then cargo-deny
    python3 tools/supply-chain.py --python   # the Python audit only

CI runs it on every pull request and weekly (gates.yml). A known
advisory fails the run unless it is listed in EXCEPTIONS below with the
reason no fixed version is reachable and the date it was judged; the
Rust exceptions live in `crates/trainnr-studio/deny.toml`. Needs `uv`,
network access, and `cargo-deny` on PATH for the Rust half.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Each locked Python set the product installs: a project and its extras.
# trainnr's `usd` extra conflicts with `mjx` (pyproject's declared
# conflicts), so it is audited on its own.
SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("trainnr", ("--all-extras", "--no-extra", "usd")),
    ("trainnr", ("--extra", "usd")),
    ("trainnr-mjlab", ("--all-extras",)),
)

# Advisory id -> why it stays, and when that was judged. Only for an
# advisory with no reachable fix; review at every release.
EXCEPTIONS: dict[str, str] = {
    "PYSEC-2026-3716": "datasets <5, held by LeRobot 0.6.1 (train); 2026-10-03",
    "PYSEC-2026-3929": "transformers 5.5, held by LeRobot 0.6.1 (train); 2026-10-03",
    "PYSEC-2026-4174": "transformers, no fixed version published; 2026-10-03",
    "PYSEC-2026-3447": "setuptools 81, held by Open3D and LeRobot; 2026-10-03",
    "PYSEC-2025-194": "torch 2.11: the training stack moves by a deliberate re-run, "
    "like MuJoCo; 2026-10-03",
}


def python_audit() -> int:
    failed = 0
    with tempfile.TemporaryDirectory() as scratch:
        for project, flags in SETS:
            requirements = Path(scratch) / f"{project}{'-'.join(flags)}.txt"
            export = subprocess.run(
                [
                    "uv",
                    "export",
                    "--frozen",
                    "--no-hashes",
                    "--no-emit-project",
                    "-q",
                    *flags,
                ],
                cwd=REPO / project,
                capture_output=True,
                text=True,
                check=False,
            )
            if export.returncode != 0:
                print(f"{project} {' '.join(flags)}: uv export failed\n{export.stderr}")
                failed = 1
                continue
            # Git-sourced packages cannot be looked up by version; the audit
            # covers every pinned one.
            pinned = [
                line
                for line in export.stdout.splitlines()
                if line and not line.startswith(("#", "-e", " ")) and " @ " not in line
            ]
            requirements.write_text("\n".join(pinned) + "\n")
            ignores = [arg for vuln in EXCEPTIONS for arg in ("--ignore-vuln", vuln)]
            audit = subprocess.run(
                [
                    "uvx",
                    "pip-audit",
                    "--no-deps",
                    "--disable-pip",
                    "--progress-spinner",
                    "off",
                    "-r",
                    str(requirements),
                    *ignores,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            label = f"{project} {' '.join(flags)}"
            if audit.returncode == 0:
                print(
                    f"{label}: {len(pinned)} packages, no known vulnerabilities "
                    f"outside the {len(EXCEPTIONS)} listed exceptions"
                )
            else:
                print(f"{label}: known vulnerabilities\n{audit.stdout}{audit.stderr}")
                failed = 1
    return failed


def rust_audit() -> int:
    if shutil.which("cargo-deny") is None:
        print("cargo-deny is not installed; skipping the Rust half")
        return 1
    run = subprocess.run(
        ["cargo", "deny", "--log-level", "error", "check"],
        cwd=REPO / "crates" / "trainnr-studio",
        check=False,
    )
    return run.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--python", action="store_true", help="the Python audit only")
    args = parser.parse_args()
    failed = python_audit()
    if not args.python:
        failed |= rust_audit()
    return failed


if __name__ == "__main__":
    sys.exit(main())
