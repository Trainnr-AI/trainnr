#!/usr/bin/env python3
"""What the importer changed: a bundle's compiled model against the
description it came from (robot/import_audit; docs/e2e-research/78 §1).
The same audit every onboarding door runs, for a bundle already on
disk — one onboarded before the audit existed, or one whose source has
moved:

    python ../tools/audit-bundle.py ../robots/so101-nominal \\
        ../robots/so101-nominal/so101.xml
    python ../tools/audit-bundle.py ../robots/robotiq-2f85-isaac \\
        path/to/Robotiq_2F_85.usda --variant Physics=Newton_compliant --json

Prints every change with the reader's explanation, or UNEXPLAINED;
`--json` prints the record the door would write; `--write` puts it on
the bundle's `bundle.json` (which moves the bundle's stamp — say so
where the stamp is cited). Exit 1 on an unexplained change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.bundles.bundle import (  # noqa: E402
    AUDIT_KEY,
    amend_bundle_record,
    model_file_of,
)
from rq_pipeline.robot.import_audit import audit_bundle  # noqa: E402


def parse_variant(text: str) -> tuple[str, str]:
    name, sep, choice = text.partition("=")
    if not sep or not name or not choice:
        raise argparse.ArgumentTypeError(f"a variant is Set=Choice, got {text!r}")
    return name, choice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bundle", type=Path, help="the bundle directory")
    parser.add_argument("source", type=Path, help="the description it came from")
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        default=[],
        metavar="SET=CHOICE",
        help="a USD variant selection the import used",
    )
    parser.add_argument("--json", action="store_true", help="print the record")
    parser.add_argument(
        "--write", action="store_true", help="put the record on bundle.json"
    )
    args = parser.parse_args(argv)
    model_file = model_file_of(args.bundle)
    if model_file is None:
        print(f"no MJCF under {args.bundle}", file=sys.stderr)
        return 2
    options = {"variants": dict(args.variant)} if args.variant else {}
    audit = audit_bundle(args.source, model_file, options)
    if args.json:
        print(json.dumps(audit.to_record(), indent=1, sort_keys=True))
    else:
        print(
            f"{args.bundle.name}: {audit.format} via mujoco {audit.mujoco}; "
            f"changed: {audit.summary()}"
        )
        for change in audit.changes:
            print("  " + change.line())
        for advisory in audit.advisories:
            print("  advisory: " + advisory)
    if args.write:
        amend_bundle_record(args.bundle, AUDIT_KEY, audit.to_record())
        print(f"written to {args.bundle / 'bundle.json'}")
    return 1 if audit.unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
