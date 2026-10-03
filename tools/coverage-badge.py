#!/usr/bin/env python3
"""Turn coverage.py's JSON report into a shields.io endpoint badge.

    python3 tools/coverage-badge.py coverage.json badge-coverage.json

CI runs it after the test suite; the README's coverage badge reads the
output from the repository's `badges` branch. The colour bands are the
common ones: red under 60 %, yellow under 75 %, green under 90 %,
bright green from 90 %.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BANDS = ((90.0, "brightgreen"), (75.0, "green"), (60.0, "yellow"), (0.0, "red"))


def badge(percent: float) -> dict[str, object]:
    colour = next(name for floor, name in BANDS if percent >= floor)
    return {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{percent:.0f}%",
        "color": colour,
    }


def main() -> int:
    report, out = Path(sys.argv[1]), Path(sys.argv[2])
    percent = float(json.loads(report.read_text())["totals"]["percent_covered"])
    out.write_text(json.dumps(badge(percent)) + "\n")
    print(f"coverage {percent:.1f}% -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
