#!/usr/bin/env python3
"""The dependencies, checked: known vulnerabilities, the Studio's crate
policy, and the licences of the installed Python packages.

    python3 tools/supply-chain.py --advisories      # pip-audit over every locked
                                                    # Python set, cargo-deny's
                                                    # advisories (network)
    python3 tools/supply-chain.py --policy          # the Studio's crates: licences,
                                                    # sources, bans (cargo-deny)
    python3 tools/supply-chain.py --licences [VENV ...]
                                                    # installed Python packages
                                                    # against the allow-list (offline)
    python3 tools/supply-chain.py                   # all three

Two kinds of check, run in different CI jobs (gates.yml):

- Deterministic: the same lockfiles give the same answer. `--policy`
  runs in the required `supply-chain` job; `--licences` runs in
  `fast-gates` over the environment its tests install, and weekly over
  every set.
- Advisory: `--advisories` reads databases that change without any
  commit here, so a newly published advisory would fail every pull
  request at once, including a stranger's typo fix. It runs in the
  `advisories` job, on every change and weekly, and is not required.

An advisory fails the run unless it is listed in EXCEPTIONS below with the
reason no fixed version is reachable and the date it was judged; the Rust
exceptions live in `crates/trainnr-studio/deny.toml`. A licence outside
the allow-list fails unless LICENCE_EXCEPTIONS names the package and why.
Needs `uv` and network access for the Python audit, `cargo-deny` on PATH
for the Rust half; the licence check needs only installed environments.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The auditor, pinned so a release of it cannot move the gate.
PIP_AUDIT = "pip-audit==2.10.1"

# Each locked Python set the product installs: a project with its extras
# and dependency groups. trainnr's `usd` extra conflicts with `mjx`
# (pyproject's declared conflicts), so it is audited on its own.
SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("trainnr", ("--all-extras", "--no-extra", "usd", "--all-groups")),
    ("trainnr", ("--extra", "usd")),
    ("trainnr-mjlab", ("--all-extras", "--all-groups")),
)

# Advisory id -> why it stays, and when that was judged. Only for an
# advisory with no reachable fix; review at every release. Each claim was
# checked against the locks and the advisory's fixed version.
EXCEPTIONS: dict[str, str] = {
    "PYSEC-2026-3716": "datasets 4.8.5; fixed in 5.0.1, LeRobot 0.6.1 (train) "
    "requires datasets<5; 2026-10-03",
    "PYSEC-2026-3929": "transformers 5.5.4; fixed in 5.10.0, LeRobot 0.6.1 (train) "
    "requires transformers<5.6; 2026-10-03",
    "PYSEC-2026-4174": "transformers 5.5.4; fixed from 5.9 (5.8.1 is the last "
    "affected), LeRobot 0.6.1 (train) requires transformers<5.6; 2026-10-04",
    "PYSEC-2026-3447": "setuptools 81 in trainnr's lock; fixed in 83.0.0, which "
    "trainnr-mjlab's lock has (84.0.0); only LeRobot 0.6.1 (train) caps it, "
    "setuptools<82; 2026-10-04",
    "PYSEC-2025-194": "torch 2.11 in trainnr's lock; fixed in 2.13.0, which "
    "trainnr-mjlab's lock has; LeRobot 0.6.1 (train) requires torch<2.12; "
    "2026-10-04",
}

# The licences a dependency may carry without a written reason: permissive
# ones, and MPL-2.0's file-level copyleft (using an unmodified MPL package
# is compatible with Apache-2.0). Matched against the SPDX expression, the
# licence classifiers and the free-text field, whichever a package has.
REFUSED = re.compile(
    r"\bA?GPL(v\d)?\b|\bLGPL|GNU (Affero |Lesser |Library )?General Public|"
    r"\bEPL\b|Eclipse Public|\bEUPL\b|\bSSPL\b|\bBUSL\b|Business Source|"
    r"Commons Clause|non-?commercial|\bCC-BY-NC|\bCC BY-NC|proprietary",
    re.IGNORECASE,
)
# Package -> why it is accepted although its licence is refused above, and
# when that was judged. A prefix ends with "*".
LICENCE_EXCEPTIONS: dict[str, str] = {
    "torchrunx": "GPL-3.0; mjlab declares it, and imports it only to launch "
    "multi-GPU training; trainnr imports it nowhere and redistributes "
    "nothing of it; 2026-10-04",
    "paramiko": "LGPL-2.1; via torchrunx's fabric (trainnr-mjlab), used "
    "unmodified as an installed library; 2026-10-04",
    "num2words": "LGPL; via LeRobot's smolvla extra (train), used unmodified "
    "as an installed library; 2026-10-04",
    "cyclonedds": "EPL-2.0 or BSD-3-Clause; the dds dependency group, which "
    "is not published with the package and is installed only from a "
    "checkout; 2026-10-04",
    "nvidia-*": "NVIDIA's CUDA libraries under NVIDIA's own licence, "
    "installed from NVIDIA's wheels by the user's torch or Warp install; "
    "never redistributed by trainnr; 2026-10-04",
    "cuda-*": "NVIDIA's CUDA toolkit metapackages (no licence metadata), as "
    "nvidia-*; 2026-10-04",
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# --- advisories ------------------------------------------------------------


def export(project: str, flags: tuple[str, ...]) -> tuple[list[list[str]], list[str]]:
    """The set's pins as requirement files pip-audit reads in full, and the
    git-sourced requirements it cannot look up.

    pip-audit evaluates an environment marker against the interpreter it
    runs on and silently skips a requirement whose marker does not match
    (a Windows-only package, a pin for another Python), so the markers are
    stripped. A package locked at two versions (one per Python or per
    platform) would then appear twice, which a requirements file cannot
    hold, so the n-th pin of each package goes into the n-th file."""
    run = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-hashes",
            "--no-emit-project",
            "--no-header",
            "-q",
            *flags,
        ],
        cwd=REPO / project,
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        raise RuntimeError(f"uv export failed\n{run.stderr}")
    files: list[list[str]] = []
    seen: dict[str, set[str]] = {}
    unauditable: list[str] = []
    for line in run.stdout.splitlines():
        if not line or line.startswith(("#", "-", " ")):
            continue  # comments, `-e` paths, uv's `# via` annotations
        requirement = line.split(";", 1)[0].strip()
        if " @ " in requirement:
            unauditable.append(requirement)
            continue
        name = canonical(re.split(r"[=<>!~\[ ]", requirement, maxsplit=1)[0])
        pins = seen.setdefault(name, set())
        if requirement in pins:
            continue
        pins.add(requirement)
        while len(files) < len(pins):
            files.append([])
        files[len(pins) - 1].append(requirement)
    return files, unauditable


def python_audit() -> int:
    failed = 0
    ignores = [arg for vuln in EXCEPTIONS for arg in ("--ignore-vuln", vuln)]
    with tempfile.TemporaryDirectory() as scratch:
        for n, (project, flags) in enumerate(SETS):
            label = f"{project} {' '.join(flags)}"
            try:
                files, unauditable = export(project, flags)
            except RuntimeError as err:
                print(f"{label}: {err}")
                failed = 1
                continue
            for requirement in unauditable:
                print(f"{label}: not audited (a git source): {requirement}")
            reports = []
            for i, pins in enumerate(files):
                path = Path(scratch) / f"set{n}-{i}.txt"
                path.write_text("\n".join(pins) + "\n")
                audit = subprocess.run(
                    [
                        "uvx",
                        "--from",
                        PIP_AUDIT,
                        "pip-audit",
                        "--no-deps",
                        "--disable-pip",
                        "--progress-spinner",
                        "off",
                        "-r",
                        str(path),
                        *ignores,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if audit.returncode != 0:
                    reports.append(audit.stdout + audit.stderr)
            count = sum(len(pins) for pins in files)
            if reports:
                print(f"{label}: known vulnerabilities\n" + "\n".join(reports))
                failed = 1
            else:
                print(
                    f"{label}: {count} pins in {len(files)} file(s), no known "
                    f"vulnerabilities outside the {len(EXCEPTIONS)} listed exceptions"
                )
    return failed


def cargo_deny(*which: str) -> int:
    if shutil.which("cargo-deny") is None:
        print(f"cargo-deny is not installed; cannot check {', '.join(which)}")
        return 1
    run = subprocess.run(
        ["cargo", "deny", "--log-level", "error", "--locked", "check", *which],
        cwd=REPO / "crates" / "trainnr-studio",
        check=False,
    )
    return run.returncode


# --- licences --------------------------------------------------------------


def site_packages(venv: Path) -> list[Path]:
    """The environment's package directories (POSIX and Windows layouts)."""
    return sorted(venv.glob("lib/python3*/site-packages")) + sorted(
        venv.glob("Lib/site-packages")
    )


def licence_of(meta: importlib.metadata.PackageMetadata) -> str:
    """Everything the package says about its licence, in one line."""
    parts = [meta.get("License-Expression") or ""]
    parts += [
        c.split(" :: ")[-1]
        for c in (meta.get_all("Classifier") or [])
        if c.startswith("License ::")
    ]
    free_text = (meta.get("License") or "").strip()  # sometimes the whole text
    parts.append(free_text.splitlines()[0] if free_text else "")
    return " | ".join(p.strip() for p in parts if p and p.strip())


def excepted(name: str) -> str | None:
    for key, reason in LICENCE_EXCEPTIONS.items():
        if (key.endswith("*") and name.startswith(key[:-1])) or name == key:
            return reason
    return None


def licence_check(venvs: Iterable[Path]) -> int:
    failed = 0
    for venv in venvs:
        paths = site_packages(venv)
        if not paths:
            print(f"{venv}: no installed environment there")
            failed = 1
            continue
        refused, accepted, total = [], [], 0
        for dist in importlib.metadata.distributions(path=[str(p) for p in paths]):
            name = canonical(dist.metadata["Name"] or "")
            total += 1
            licence = licence_of(dist.metadata)
            if licence and not REFUSED.search(licence):
                continue
            reason = excepted(name)
            shown = f"{name} {dist.version}: {licence or 'no licence metadata'}"
            if reason is None:
                refused.append(shown)
            else:
                accepted.append(shown)
        if total == 0:
            print(f"{venv}: no packages installed")
            failed = 1
            continue
        print(
            f"{venv}: {total} packages; {len(accepted)} accepted by a written "
            f"exception, {len(refused)} refused"
        )
        for line in refused:
            print(f"  REFUSED {line}")
        failed |= bool(refused)
    return int(failed)


def default_venvs() -> list[Path]:
    return [
        REPO / "trainnr" / ".venv",
        REPO / "trainnr" / ".venv-usd",
        REPO / "trainnr-mjlab" / ".venv",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--advisories",
        action="store_true",
        help="known vulnerabilities: pip-audit and cargo-deny advisories",
    )
    parser.add_argument(
        "--python", action="store_true", help="the pip-audit half of --advisories"
    )
    parser.add_argument(
        "--policy",
        action="store_true",
        help="the Studio's crates: cargo-deny licences, sources, bans",
    )
    parser.add_argument(
        "--licences",
        nargs="*",
        type=Path,
        metavar="VENV",
        help="installed Python packages against the licence allow-list "
        "(default: the repository's environments that exist)",
    )
    args = parser.parse_args()
    everything = not (args.advisories or args.python or args.policy) and (
        args.licences is None
    )
    failed = 0
    if args.advisories or args.python or everything:
        failed |= python_audit()
    if args.advisories or everything:
        failed |= cargo_deny("advisories")
    if args.policy or everything:
        failed |= cargo_deny("licenses", "bans", "sources")
    if args.licences is not None or everything:
        venvs = args.licences or [v for v in default_venvs() if v.is_dir()]
        if not venvs:
            print("no Python environment installed here; licences not checked")
        failed |= licence_check(venvs)
    return failed


if __name__ == "__main__":
    sys.exit(main())
