#!/usr/bin/env python3
"""A software bill of materials (CycloneDX JSON) for each thing a release
ships: both Python packages and the Studio.

    python3 tools/sbom.py --out dist/sbom 0.1.0

- A Python package's SBOM lists every component its lock file pins for
  any install it allows: every extra, with each group of conflicting
  extras (`[tool.uv] conflicts`, such as trainnr's `mjx` and `usd`)
  exported one member at a time and the pins united. `uv export` reads the
  lock; `cyclonedx-py` (cyclonedx-bom, pinned below) writes the SBOM, with
  the package itself as its subject.
- The Studio's lists every crate its Cargo.lock compiles in, written by
  `cargo cyclonedx` (cargo-cyclonedx, which the release workflow installs
  at the pinned version; a local run needs it on PATH).

The release attaches each SBOM and attests it (`actions/attest-sbom`), so
`gh attestation verify --predicate-type https://cyclonedx.org/bom` proves
which workflow wrote it.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

# Python 3.11+ (tomllib): the release workflow's runner has 3.12.

REPO = Path(__file__).resolve().parents[1]
PYTHON_PACKAGES = ("trainnr", "trainnr-mjlab")
STUDIO_CRATE = REPO / "crates" / "trainnr-studio"
CYCLONEDX_BOM = "cyclonedx-bom==7.5.0"
SPEC_VERSION = "1.5"  # the newest CycloneDX both writers produce


def extra_sets(pyproject: Path) -> list[list[str]]:
    """Each set of extras an install may take together: all of them, less
    all but one member of every conflict group."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = list(data["project"].get("optional-dependencies", {}))
    groups = [
        [item["extra"] for item in group if "extra" in item]
        for group in data.get("tool", {}).get("uv", {}).get("conflicts", [])
    ]
    if not groups:
        return [extras]
    sets = []
    for chosen in itertools.product(*groups):
        left_out = {e for group in groups for e in group} - set(chosen)
        sets.append([e for e in extras if e not in left_out])
    return sets


def pins(package_dir: Path) -> list[str]:
    """Every `name==version` line the lock gives any allowed install."""
    lines: set[str] = set()
    for extras in extra_sets(package_dir / "pyproject.toml"):
        argv = ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project"]
        argv += ["--quiet", *(arg for extra in extras for arg in ("--extra", extra))]
        out = subprocess.run(
            argv, cwd=package_dir, capture_output=True, text=True, check=True
        ).stdout
        for raw in out.splitlines():
            line = raw.strip()
            if line and not line.startswith(("#", "-e", "./", "../")):
                lines.add(line)
    return sorted(lines)


def python_sbom(package: str, version: str, out_dir: Path) -> Path:
    package_dir = REPO / package
    target = out_dir / f"{package}-{version}.cdx.json"
    with tempfile.TemporaryDirectory() as tmp:
        reqs = Path(tmp) / "requirements.txt"
        reqs.write_text("\n".join(pins(package_dir)) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "uvx",
                "--from",
                CYCLONEDX_BOM,
                "cyclonedx-py",
                "requirements",
                "--pyproject",
                "pyproject.toml",
                "--spec-version",
                SPEC_VERSION,
                "--output-format",
                "JSON",
                "--output-file",
                str(target),
                str(reqs),
            ],
            cwd=package_dir,
            check=True,
        )
    return target


def studio_sbom(version: str, out_dir: Path) -> Path:
    if shutil.which("cargo-cyclonedx") is None:
        raise SystemExit(
            "cargo-cyclonedx is not on PATH (cargo install cargo-cyclonedx --locked)"
        )
    name = f"trainnr-studio-{version}"
    subprocess.run(
        [
            "cargo",
            "cyclonedx",
            "--format",
            "json",
            "--spec-version",
            SPEC_VERSION,
            "--override-filename",
            name,
        ],
        cwd=STUDIO_CRATE,
        check=True,
    )
    made = STUDIO_CRATE / f"{name}.json"
    target = out_dir / f"{name}.cdx.json"
    shutil.move(made, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the release's version, as in the file names")
    parser.add_argument("--out", type=Path, default=REPO / "dist" / "sbom")
    parser.add_argument("--skip-studio", action="store_true")
    args = parser.parse_args(argv)
    # Absolute: the writers run inside each package's folder, and a path
    # relative to the repository root could not be opened from there (the
    # first release candidate, 2026-10-06).
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    made = [python_sbom(p, args.version, args.out) for p in PYTHON_PACKAGES]
    if not args.skip_studio:
        made.append(studio_sbom(args.version, args.out))
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
