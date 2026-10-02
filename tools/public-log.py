"""Public recordings of real robots: list them, fetch one, ingest one.

    python3 tools/public-log.py list
    python3 tools/public-log.py fetch go2-leg-odometry
    python3 tools/public-log.py ingest go2-leg-odometry --project projects/go2-walk

The registry (`trainnr.robots.public_logs`) names each log's source,
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

from _lab import bootstrap, running

bootstrap()

from trainnr.project import index_project, write_index  # noqa: E402
from trainnr.project.ingest import ingest  # noqa: E402
from trainnr.project.locate import Project  # noqa: E402
from trainnr.robots import public_logs  # noqa: E402


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
    try:
        entry = public_logs.resolve(args.name)
        if args.verb == "fetch":
            print(f"{entry.name} -> {public_logs.fetch(args.name)}")
            return 0
        project = Project(args.project.resolve()).use()
        with running(project.root, name=args.recording or entry.name) as run:
            run.stage(f"fetching {entry.name} (size and digest checked)")
            source = public_logs.fetch(args.name)
            run.stage(f"decoding {entry.name} with {entry.adapter}")
            out = ingest(
                project,
                source,
                name=args.recording or entry.name,
                adapter=entry.adapter,
                provenance=entry.provenance(),
                basis=entry.basis,
            )
            run.stage(f"ingested {out.get('stamp')}")
    except KeyError as unknown:  # a refusal by name, not a traceback
        raise SystemExit(
            str(unknown.args[0]) if unknown.args else str(unknown)
        ) from None
    except (OSError, ValueError, FileExistsError) as why:
        raise SystemExit(f"public log {args.name!r}: {why}") from None
    write_index(project, index_project(project))
    print(json.dumps({k: v for k, v in out.items() if k != "channels"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
