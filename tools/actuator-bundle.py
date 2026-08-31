"""The certified actuator bundle CLI — wrap BAM fits, verify anyone's.

    cd pipeline && uv run python ../tools/actuator-bundle.py wrap --all
    cd pipeline && uv run python ../tools/actuator-bundle.py wrap \
        feetech_sts3215_7_4V m6 [--out DIR]
    cd pipeline && uv run python ../tools/actuator-bundle.py verify \
        path/to/*.bundle.json

`wrap` turns a vendored (actuator, tier) fit into the interchange
artifact of docs/e2e-research/58 §1: BAM's params verbatim inside, our
provenance/checks/stamp around it. Default output is the committed
store `robots/actuator-bundles/` — deterministic (the wrap date is
deliberately outside the identity), so a re-wrap of unchanged fits is
a no-op diff. `verify` prints each bundle's stamp, its check flags and
its honesty advisories, and exits non-zero on the first violation —
the same function the library runs before every read and write.
"""

import argparse
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()
from rq_pipeline.robot.actuator_bundle import (  # noqa: E402
    read_bundle,
    verify,
    wrap,
    wrap_all,
    write_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "robots" / "actuator-bundles"


def cmd_wrap(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else DEFAULT_OUT
    if args.all:
        paths = wrap_all(out)
        print(f"wrapped {len(paths)} bundle(s) -> {out}")
        return 0
    if not (args.slug and args.tier):
        print("wrap needs SLUG TIER, or --all", file=sys.stderr)
        return 2
    path = write_bundle(wrap(args.slug, args.tier), out)
    print(f"wrapped -> {path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    for name in args.bundles:
        bundle = read_bundle(Path(name))  # raises, loudly, on violation
        checks = bundle["checks"]
        flags = [
            f"{kind}: {', '.join(fields)}"
            for kind, fields in sorted(checks.items())
            if fields
        ]
        print(f"{bundle['stamp']}  ({name})")
        for flag in flags:
            print(f"  check  {flag}")
        for advisory in verify(bundle):
            print(f"  advice {advisory}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    wrap_p = sub.add_parser("wrap", help="vendored fit(s) -> bundle file(s)")
    wrap_p.add_argument("slug", nargs="?", help="actuator directory name")
    wrap_p.add_argument("tier", nargs="?", help="model tier, e.g. m6")
    wrap_p.add_argument("--all", action="store_true", help="every vendored fit")
    wrap_p.add_argument("--out", help=f"output dir (default {DEFAULT_OUT})")
    wrap_p.set_defaults(func=cmd_wrap)

    verify_p = sub.add_parser("verify", help="check bundle files")
    verify_p.add_argument("bundles", nargs="+", help="bundle JSON path(s)")
    verify_p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
