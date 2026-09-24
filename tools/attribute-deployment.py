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
import json
import sys
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.deploy.attribution import (  # noqa: E402
    ATTRIBUTION_FILE,
    SURVIVED,
    attribute,
    knob,
    still_at_cliff,
    stream,
)
from rq_pipeline.deploy.gate import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
)
from rq_pipeline.deploy.manifest import Key, load_manifest  # noqa: E402
from rq_pipeline.deploy.runtime import assets_dir_of  # noqa: E402
from rq_pipeline.deploy.runtimes import DEFAULT_RUNTIME, runtime_names  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.cited import cited_certificate  # noqa: E402
from rq_pipeline.project.locate import DEPLOY_FOLDER, Project  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME, choices=runtime_names())
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--workers", type=int, default=None, help="knobs climbed at once (0: here)"
    )
    parser.add_argument(
        "--no-still", action="store_true", help="skip the picture of the fall"
    )
    args = parser.parse_args()
    project = Project(args.project.resolve()).use()
    deployment = project.folder(DEPLOY_FOLDER) / args.name
    try:
        manifest = load_manifest(deployment)
        certificate = cited_certificate(project, manifest.raw.get(Key.CERTIFICATE))
        assets = assets_dir_of(manifest)
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
        (deployment / ATTRIBUTION_FILE).write_text(
            json.dumps(record, indent=1) + "\n", encoding="utf-8"
        )
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
