"""Public recordings of real robots: list them, fetch one, ingest one.

    python3 tools/public-log.py list
    python3 tools/public-log.py fetch go2-leg-odometry
    python3 tools/public-log.py ingest go2-leg-odometry --project projects/go2-walk

The registry (`rq_pipeline.robots.public_logs`) names each log's source,
byte count, digest, robot, recorded date and the data's licence STATE;
a fetch that differs from the registry is refused, never read. Ingest
stamps the recording with its provenance so the Studio's telemetry
stage reads "public log": a real robot, and not the project's own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.project.ingest import ingest  # noqa: E402
from rq_pipeline.project.locate import Project  # noqa: E402
from rq_pipeline.robots import public_logs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    sub.add_parser("list", help="every registered log and whether it is fetched")
    fetch = sub.add_parser("fetch", help="download, check and unpack one log")
    fetch.add_argument("name")
    ingest_p = sub.add_parser("ingest", help="fetch if needed, then ingest")
    ingest_p.add_argument("name")
    ingest_p.add_argument("--project", required=True, type=Path)
    ingest_p.add_argument("--as", dest="recording", default=None)
    args = parser.parse_args()
    if args.verb == "list":
        print(json.dumps(public_logs.listing(), indent=1))
        return 0
    entry = public_logs.resolve(args.name)
    source = public_logs.fetch(args.name)
    if args.verb == "fetch":
        print(f"{entry.name} -> {source}")
        return 0
    out = ingest(
        Project(args.project.resolve()),
        source,
        name=args.recording or entry.name,
        adapter=entry.adapter,
        provenance=entry.provenance(),
    )
    print(json.dumps({k: v for k, v in out.items() if k != "channels"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
