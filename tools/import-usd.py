#!/usr/bin/env python3
"""A USD robot asset (Isaac Sim, Omniverse) into a hash-stamped bundle:
Newton reads the stage, its MuJoCo bridge writes the MjSpec, our writer
makes the bundle (docs/e2e-research/77 §6). The same door the Studio's
`onboard_robot` opens, from the shell:

    python ../tools/import-usd.py path/to/Robot.usda my-robot \\
        --variant Physics=Newton_compliant --root fixed --grip-options

`--fetch owner/repo@commit:path/in/repo` first pulls a public asset
folder into the local cache (Git LFS aware) and imports the named file
from it, so the bundle's provenance carries the repository and commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _lab import bootstrap, running

bootstrap()

from rq_pipeline.bundles.locate import robots_dir  # noqa: E402
from rq_pipeline.robot.asset_fetch import AssetFetchError, fetch_tree  # noqa: E402
from rq_pipeline.robot.onboarding import onboard  # noqa: E402
from rq_pipeline.robot.usd_import import (  # noqa: E402
    ROOT_FIXED,
    ROOT_KINDS,
    ImportSettings,
    UsdLayerError,
    missing_line,
)

FETCH_FORM = "owner/repo@commit:path/in/repo"


def parse_variant(text: str) -> tuple[str, str]:
    name, sep, choice = text.partition("=")
    if not sep or not name or not choice:
        raise argparse.ArgumentTypeError(f"a variant is Set=Choice, got {text!r}")
    return name, choice


def parse_fetch(text: str) -> tuple[str, str, str]:
    repository, sep, rest = text.partition("@")
    commit, sep2, path = rest.partition(":")
    if not (sep and sep2 and repository and commit and path):
        raise argparse.ArgumentTypeError(f"--fetch is {FETCH_FORM}, got {text!r}")
    return repository, commit, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "usd",
        help="the root layer (.usd/.usda/.usdc/.usdz); with --fetch, its path "
        "inside the fetched folder",
    )
    parser.add_argument("name", help="the bundle's name (robots/<name>)")
    parser.add_argument(
        "--into",
        type=Path,
        default=None,
        help="the robots directory (default: the library)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        default=[],
        metavar="SET=CHOICE",
    )
    parser.add_argument("--root", choices=ROOT_KINDS, default=ROOT_FIXED)
    parser.add_argument(
        "--maxhullvert",
        type=int,
        default=ImportSettings().mesh_maxhullvert,
        help="convex hull vertices per collision mesh",
    )
    parser.add_argument(
        "--grip-options",
        action="store_true",
        help="declare impratio 10 and an elliptic cone (a gripper's runtime "
        "tuning, not in the USD)",
    )
    parser.add_argument("--fetch", type=parse_fetch, default=None, metavar=FETCH_FORM)
    args = parser.parse_args(argv)

    line = missing_line()
    if line:
        print(f"refused: {line}", file=sys.stderr)
        return 2
    source = Path(args.usd)
    if args.fetch:
        repository, commit, path = args.fetch
        try:
            fetched = fetch_tree(repository, commit, path)
        except AssetFetchError as why:
            print(f"refused: {why}", file=sys.stderr)
            return 2
        source = fetched.root / args.usd
        print(
            f"fetched {repository}@{commit[:7]}:{path} "
            f"({len(fetched.files)} files) into {fetched.root}"
        )
    options = {
        "variants": dict(args.variant),
        "root": args.root,
        "mesh_maxhullvert": args.maxhullvert,
        "grip_options": args.grip_options,
    }
    try:
        with running(None, name=args.name) as run:
            run.stage(f"reading {source.name} with Newton, writing the bundle")
            out = onboard(
                source, args.name, (args.into or robots_dir()) / args.name, options
            )
            run.stage(f"onboarded {out.get('stamp', args.name)}")
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        UsdLayerError,
        ImportError,
    ) as why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
