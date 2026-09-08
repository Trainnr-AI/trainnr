"""Draw (or redraw) the figure of one or more finding records, and write
the figure paths onto each record.

    python3 tools/finding-figure.py <record-id> [<record-id> ...]

Dispatch by the record's shape (`evaluate/figures.render_any`): arms →
the success-rate figure, a bootstrap interval → the interval figure.
The record file keeps every field it carries (status, revised, …):
only its `artifacts.figure.*` entries change - a fold that rebuilds a
record from scratch would drop a `revised` note; this does not.
"""

from __future__ import annotations

import argparse
import json
import sys

from _lab import REPO, bootstrap

bootstrap()

from rq_pipeline.evaluate import figures as fx  # noqa: E402
from rq_pipeline.evaluate.findings import FINDINGS_DIR, Finding  # noqa: E402

CAPTIONS = REPO / "docs" / "paper" / "captions.json"


def paper_title(record_id: str) -> str | None:
    """The short title the paper gives this figure (docs/paper/captions.json)."""
    if not CAPTIONS.is_file():
        return None
    entry = json.loads(CAPTIONS.read_text()).get(record_id)
    return entry.get("title") if isinstance(entry, dict) else None


def redraw(record_id: str) -> str:
    path = REPO / FINDINGS_DIR / f"{record_id}.json"
    if not path.is_file():
        raise SystemExit(f"no record {path.relative_to(REPO)}")
    written = fx.render_any(Finding.read(path), REPO, title=paper_title(record_id))
    text = path.read_text()
    raw = json.loads(text)
    raw.setdefault("artifacts", {}).update(
        {f"figure.{k}": v for k, v in written.items()}
    )
    path.write_text(json.dumps(raw, indent=1) + ("\n" if text.endswith("\n") else ""))
    return written["svg"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("records", nargs="+", help="record ids under docs/findings/")
    args = parser.parse_args()
    for rid in args.records:
        print(f"{rid} -> {redraw(rid)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
