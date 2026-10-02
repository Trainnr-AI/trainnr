"""Vendor new or changed actuators from a BAM source into robots/actuators/.

    cd trainnr && uv run python ../tools/sync-bam-actuators.py <source> \
        [--version X.Y.Z] [--dry-run]

`<source>` is either a BAM git checkout (a directory containing
`bam/params/`) or a repomix XML pack of the BAM repo (the format the
operator has supplied all session — a `<file path="bam/params/...">`
tree in one file).

Refuses, rather than silently overwriting, a params file whose content
changed without `--version` naming a new BAM release: a changed number
under an unchanged version is exactly the two-copies-of-one-fact drift
this repo's tests exist to catch, and vendored actuator data gets no
exception. `--dry-run` reports what would change without writing.

New actuators (a `params/<slug>/` this repo has never seen) are always
accepted — there is nothing to silently overwrite yet — and get a
fresh PROVENANCE.json. `robots/actuators/README.md` explains the two
other ways to grow this library: our own bench fit, or a datasheet-only
guess, neither of which goes through this tool.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TRAINNR_ROOT = Path(__file__).resolve().parent.parent
ACTUATORS_ROOT = TRAINNR_ROOT / "robots" / "actuators"
PARAMS_FILE_RE = re.compile(r"^bam/params/([^/]+)/(m[1-6])\.json$")
CITATION = (
    "Duclusaud, M., Passault, G., Padois, V., Ly, O. (2025). "
    "Extended Friction Models for the Physics Simulation of Servo "
    "Actuators. 2025 IEEE International Conference on Robotics and "
    "Automation (ICRA), pp. 12091-12097."
)


def _params_from_checkout(root: Path) -> dict[str, dict[str, str]]:
    """slug -> {tier: raw JSON text}, from a `bam/params/` tree on disk."""
    found: dict[str, dict[str, str]] = {}
    params_dir = root / "bam" / "params"
    for path in sorted(params_dir.glob("*/m[1-6].json")):
        slug, tier = path.parent.name, path.stem
        found.setdefault(slug, {})[tier] = path.read_text()
    return found


def _params_from_repomix_xml(pack: Path) -> dict[str, dict[str, str]]:
    """slug -> {tier: raw JSON text}, from a repomix XML pack of BAM."""
    text = pack.read_text(errors="replace")
    pattern = re.compile(
        r'<file path="(bam/params/[^"]+/m[1-6]\.json)">\n(.*?)\n</file>', re.DOTALL
    )
    found: dict[str, dict[str, str]] = {}
    for match in pattern.finditer(text):
        rel_path, content = match.group(1), match.group(2)
        m = PARAMS_FILE_RE.match(rel_path)
        if m:
            found.setdefault(m.group(1), {})[m.group(2)] = content + "\n"
    return found


def discover(source: Path) -> dict[str, dict[str, str]]:
    if source.is_dir():
        found = _params_from_checkout(source)
        if not found:
            raise ValueError(
                f"{source} has no bam/params/*/m[1-6].json — not a BAM checkout"
            )
        return found
    found = _params_from_repomix_xml(source)
    if not found:
        raise ValueError(
            f"{source} has no bam/params/ file entries — not a BAM repomix pack"
        )
    return found


def sync(source: Path, *, version: str | None, dry_run: bool) -> int:
    incoming = discover(source)
    changed = 0
    for slug, tiers in sorted(incoming.items()):
        directory = ACTUATORS_ROOT / slug
        provenance_path = directory / "PROVENANCE.json"
        is_new = not directory.exists()
        for tier, content in sorted(tiers.items()):
            dest = directory / f"{tier}.json"
            if dest.exists() and dest.read_text() == content:
                continue
            if dest.exists() and not is_new and version is None:
                raise ValueError(
                    f"{dest} would change but no --version was given — a params "
                    "file that changed without a stated BAM release is exactly "
                    "the drift this tool exists to refuse. Pass --version to "
                    "confirm this is a real re-fit, not accidental content skew."
                )
            print(f"{'would write' if dry_run else 'writing'} {dest}")
            changed += 1
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
        if is_new and not dry_run:
            provenance_path.write_text(
                json.dumps(
                    {
                        "slug": slug,
                        "source": "bam",
                        "source_repo": "https://github.com/Rhoban/bam",
                        "source_version": version or "unknown",
                        "license": "Apache-2.0",
                        "citation": CITATION,
                        "vendored_via": f"tools/sync-bam-actuators.py from {source}",
                        "models_available": sorted(tiers),
                        "notes": (
                            "Identified on BAM's own pendulum test bench, not "
                            "on a robot in this repo. See docs/e2e-research/"
                            "53-bam-actuator-identification.md."
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"{'would write' if dry_run else 'writing'} {provenance_path}")
        elif (
            not is_new
            and version is not None
            and not dry_run
            and provenance_path.exists()
        ):
            provenance = json.loads(provenance_path.read_text())
            provenance["source_version"] = version
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "source", type=Path, help="a BAM checkout dir or repomix XML pack"
    )
    parser.add_argument("--version", help="the BAM release this sync is from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = sync(args.source, version=args.version, dry_run=args.dry_run)
    print(f"{changed} file(s) {'would change' if args.dry_run else 'changed'}")
    if changed == 0:
        sys.exit(0)


if __name__ == "__main__":
    main()
