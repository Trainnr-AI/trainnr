#!/usr/bin/env python3
"""Attribution: which parameter would break this deployment first. A
passing plane gate re-run with one dynamics knob turned at a time up its
ladder (latency, friction, payload, gains, encoder noise, tilt, pushes);
the cliff per knob, the knobs ranked, the record beside the manifest,
the fall pictured, the ladders streamed. Spawned by the MCP door
`attribute_deployment`.
"""

from __future__ import annotations

import argparse
import sys

from _lab import bootstrap, deployment_args, resolve_deployment

bootstrap()

from rq_pipeline.deploy.attribution import (  # noqa: E402
    SURVIVED,
    attribute,
    knob,
    still_at_cliff,
    stream,
    write_attribution,
)
from rq_pipeline.deploy.gate import DEFAULT_TOLERANCE  # noqa: E402
from rq_pipeline.deploy.manifest import Key  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.cited import cited_certificate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    deployment_args(parser)
    parser.add_argument(
        "--trials", type=int, default=None, help="the passing gate's own when unset"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="the passing gate's own when unset"
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--workers", type=int, default=None, help="knobs climbed at once (0: here)"
    )
    parser.add_argument(
        "--no-still", action="store_true", help="skip the picture of the fall"
    )
    args = parser.parse_args()
    try:
        project, deployment, manifest, assets = resolve_deployment(args)
        certificate = cited_certificate(project, manifest.raw.get(Key.CERTIFICATE))
        record = attribute(
            deployment,
            assets_dir=assets,
            certificate=certificate,
            runtime=args.runtime,
            trials=args.trials,
            seed=args.seed,
            tolerance=args.tolerance,
            workers=args.workers,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
    base = record["baseline"]
    print(
        f"[attribution] baseline {base['successes']}/{base['trials']} ci95 "
        f"{base['ci95']} against the certificate's lower bound "
        f"{record['certificate']['lower']}",
        flush=True,
    )
    for entry in record["knobs"]:
        rungs = " · ".join(
            f"{r['level']:g}{entry['unit']}: {r['successes']}/{r['trials']} "
            f"[{r['ci95'][0]}, {r['ci95'][1]}]"
            for r in entry["rungs"]
        )
        cliff = entry["cliff"]
        word = (
            SURVIVED if cliff is None else f"cliff at {cliff['level']:g}{entry['unit']}"
        )
        print(f"[attribution] {entry['name']:>16}: {word} — {rungs}", flush=True)
    print(f"[attribution] {record['sensitivity']}", flush=True)
    if not args.no_still:
        still = still_at_cliff(deployment, record, assets_dir=assets)
        record["still"] = still
        write_attribution(deployment, record)
        print(
            "[attribution] still: "
            + (
                still["unrendered"]
                if "unrendered" in still
                else f"{still['file']} ({knob(still['knob']).label(still['level'])}, "
                f"trial {still['trial']}, tick {still['tick']})"
            ),
            flush=True,
        )
    try:
        file = stream(record, deployment)
        print(f"[attribution] stream -> {file}", flush=True)
    except ImportError as missing:  # no viz extra: the record stands alone
        print(f"[attribution] stream not written: {missing}", flush=True)
    write_index(project, index_project(project))
    sys.exit(0)


if __name__ == "__main__":
    main()
