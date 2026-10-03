#!/usr/bin/env python3
"""Turn coverage.py's JSON report into the README's coverage badge.

    python3 tools/coverage-badge.py coverage.json badge-coverage.json

Writes the shields.io endpoint JSON and, beside it, the badge itself as
an SVG (`badge-coverage.svg`). CI publishes both to the repository's
`badges` branch; the README shows the SVG through github.com, which
serves a repository's files to anyone allowed to read it, so the badge
renders while the repository is private as well as after (an external
badge service cannot read a private repository). The colour bands are the
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


COLOURS = {
    "brightgreen": "#4c1",
    "green": "#97ca00",
    "yellow": "#dfb317",
    "red": "#e05d44",
}
CHAR_PX = 7  # Verdana 11 px, the flat badge's width per character


def svg(label: str, message: str, colour: str) -> str:
    """A flat two-part badge in the shields.io style, self-contained."""
    lw, mw = 10 + CHAR_PX * len(label), 10 + CHAR_PX * len(message)
    width = lw + mw
    fill = COLOURS[colour]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" '
        f'role="img" aria-label="{label}: {message}"><title>{label}: {message}</title>'
        '<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" '
        'stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>'
        f'<clipPath id="r"><rect width="{width}" height="20" rx="3" '
        'fill="#fff"/></clipPath>'
        f'<g clip-path="url(#r)"><rect width="{lw}" height="20" fill="#555"/>'
        f'<rect x="{lw}" width="{mw}" height="20" fill="{fill}"/>'
        f'<rect width="{width}" height="20" fill="url(#s)"/></g>'
        '<g fill="#fff" text-anchor="middle" '
        'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        'font-size="11">'
        f'<text x="{lw / 2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>'
        f'<text x="{lw / 2}" y="14">{label}</text>'
        f'<text x="{lw + mw / 2}" y="15" fill="#010101" '
        f'fill-opacity=".3">{message}</text>'
        f'<text x="{lw + mw / 2}" y="14">{message}</text></g></svg>\n'
    )


def main() -> int:
    report, out = Path(sys.argv[1]), Path(sys.argv[2])
    percent = float(json.loads(report.read_text())["totals"]["percent_covered"])
    data = badge(percent)
    out.write_text(json.dumps(data) + "\n")
    out.with_suffix(".svg").write_text(
        svg(str(data["label"]), str(data["message"]), str(data["color"]))
    )
    print(f"coverage {percent:.1f}% -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
