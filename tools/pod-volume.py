"""A RunPod network volume over its S3 API - usage, listing, deletion.

    python3 tools/pod-volume.py --bucket <volume-id> du [--depth 2] [--prefix P]
    python3 tools/pod-volume.py --bucket <volume-id> ls trainnr/runs/campaign-1/
    python3 tools/pod-volume.py --bucket <volume-id> rm trainnr/runs/x/ [--yes]

The door that works when the pod is stopped (the log 2026-09-02).
Credentials: a RunPod S3 API key pair as AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY in the environment, or `--env-file .env` to
lift exactly those two names out of a file (nothing else is read).
`rm` is a dry run unless `--yes`: it prints what would go and how much.
"""

import argparse
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from trainnr.cloud.volume import (  # noqa: E402
    VolumeDoor,
    human,
    load_credentials,
    total,
    under,
    usage_by_prefix,
)

DEFAULT_ENDPOINT = "https://s3api-us-nc-2.runpod.io"
DEFAULT_REGION = "us-nc-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bucket", required=True, help="the volume's id")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--env-file", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    du = commands.add_parser("du", help="bytes and objects by prefix")
    du.add_argument("--prefix", default="")
    du.add_argument("--depth", type=int, default=1)
    ls = commands.add_parser("ls", help="every object under a prefix")
    ls.add_argument("prefix")
    rm = commands.add_parser("rm", help="delete everything under a prefix")
    rm.add_argument("prefix")
    rm.add_argument(
        "--yes", action="store_true", help="really delete (default: dry run)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.env_file is not None:
        load_credentials(args.env_file)
    door = VolumeDoor(args.endpoint, args.region, args.bucket)
    if args.command == "du":
        entries = list(door.entries(args.prefix))
        for prefix, usage in usage_by_prefix(entries, args.depth).items():
            print(f"{human(usage.bytes):>10}  {usage.objects:>7}  {prefix}")
        whole = total(entries)
        print(
            f"{human(whole.bytes):>10}  {whole.objects:>7}  total under {args.prefix!r}"
        )
        return 0
    if args.command == "ls":
        for entry in door.entries(args.prefix):
            print(f"{human(entry.size):>10}  {entry.key}")
        return 0
    if not args.prefix.strip("/"):
        print("refusing to delete the whole volume", file=sys.stderr)
        return 2
    doomed = under(door.entries(args.prefix), args.prefix)
    usage = total(doomed)
    print(f"{len(doomed)} objects, {human(usage.bytes)} under {args.prefix!r}")
    if not args.yes:
        print("dry run - pass --yes to delete")
        return 0
    deleted = door.delete([entry.key for entry in doomed])
    print(f"deleted {deleted} objects, {human(usage.bytes)} freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
