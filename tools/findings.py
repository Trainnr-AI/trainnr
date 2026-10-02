"""Render the findings ledger: docs/findings/*.json -> docs/68-findings.md.

    python3 tools/findings.py [--check]

`--check` exits 1 when the page on disk differs from the records (the
commit gate's use): the ledger is generated, never hand-edited.
"""

import argparse
import sys

from _lab import REPO, bootstrap

bootstrap()

from trainnr.evaluate.findings import load_findings, render_ledger  # noqa: E402

LEDGER = REPO / "docs" / "68-findings.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    page = render_ledger(load_findings(REPO))
    if args.check:
        current = LEDGER.read_text() if LEDGER.exists() else ""
        if current != page:
            print(f"{LEDGER} is stale: run tools/findings.py", file=sys.stderr)
            return 1
        return 0
    LEDGER.write_text(page)
    print(f"ledger -> {LEDGER} ({len(load_findings(REPO))} findings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
