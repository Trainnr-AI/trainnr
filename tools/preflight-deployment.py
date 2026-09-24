#!/usr/bin/env python3
"""Pre-flight: every check a deployment must pass before the first tick on
a robot, the ramp-in and the soft stop measured, the record beside the
manifest (docs/77 §10). With `--runtime dds` the robot's state is read
from Unitree's simulator over DDS (a simulation stand-in for the robot)
in their fixed stand, before the handover, and their own stop is measured
beside ours. Spawned by the MCP door `preflight_deployment`.

Ctrl-C while the DDS stand-in walks is the operator's stop: their Passive
chord through the pad, never a killed process with the motors mid-stride.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from _lab import bootstrap

bootstrap()

import numpy as np  # noqa: E402
from rq_pipeline.deploy.gate import DEFAULT_SEED  # noqa: E402
from rq_pipeline.deploy.manifest import load_manifest  # noqa: E402
from rq_pipeline.deploy.preflight import (  # noqa: E402
    PREFLIGHT_FILE,
    STOP_FILE,
    card_line,
    health_of_lowstate,
    measure_dds_stop,
    preflight,
    still_at_handover,
    stream,
    write_record,
)
from rq_pipeline.deploy.runtime import assets_dir_of  # noqa: E402
from rq_pipeline.deploy.runtimes import (  # noqa: E402
    DEFAULT_RUNTIME,
    runtime_names,
    runtime_spec,
)
from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.locate import DEPLOY_FOLDER, Project  # noqa: E402

DDS_STAND_IN = (
    "Unitree's simulator over DDS in their fixed stand (a simulation stand-in)"
)
DDS_COMMAND = np.array([0.5, 0.0, 0.0])


def _dds(manifest: Any, deployment: Path, assets: Path | None, seed: int) -> dict:
    """Their stack up; the robot stood by their controller; the state read
    before handover; the checks; then the handover, a walk and their stop."""
    spec = runtime_spec("dds")
    with spec.stack_for(manifest) as stack:
        runtime = spec.open()(manifest, assets_dir=None, **stack.runtime_options())
        stopped = {"by": None}

        def operator_stop(*_: object) -> None:  # the keyboard's hook
            runtime.stop()
            stopped["by"] = "Ctrl-C"
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGINT, operator_stop)
        try:
            runtime.stand()
            health = health_of_lowstate(runtime.health(), stand_in=True)
            record = preflight(
                deployment,
                assets_dir=assets,
                health=health,
                state_from=DDS_STAND_IN,
                stand_in=True,
                seed=seed,
            )
            if record["passed"]:
                runtime.handover()
                record["their_stop"] = measure_dds_stop(
                    runtime, DDS_COMMAND, stop_file=deployment / STOP_FILE
                )
            else:
                runtime.stop()
                record["their_stop"] = {"not run": "the pre-flight refused"}
        except KeyboardInterrupt:
            raise SystemExit(f"stopped by the operator ({stopped['by']})") from None
        finally:
            signal.signal(signal.SIGINT, previous)
            runtime.close()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the deployment's folder name")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME, choices=runtime_names())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-still", action="store_true")
    args = parser.parse_args()
    project = Project(args.project.resolve()).use()
    deployment = project.folder(DEPLOY_FOLDER) / args.name
    try:
        manifest = load_manifest(deployment)
        assets = assets_dir_of(manifest)
        if args.runtime == "dds":
            record = _dds(manifest, deployment, assets, args.seed)
        else:
            record = preflight(deployment, assets_dir=assets, seed=args.seed)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    for c in record["checks"]:
        mark = {True: "pass", False: "REFUSED", None: "n/a"}[c["passed"]]
        print(f"[preflight] {mark:>7} {c['name']}: {c['measured']}", flush=True)
    for block in ("ramp_in", "soft_stop", "their_stop"):
        if block in record:
            print(f"[preflight] {block}: {json.dumps(record[block])}", flush=True)
    if record["passed"] and not args.no_still:
        record["still"] = still_at_handover(deployment, assets_dir=assets)
    write_record(deployment, record)
    print(f"[preflight] {card_line(record)} -> {deployment / PREFLIGHT_FILE}")
    try:
        print(f"[preflight] stream -> {stream(record, deployment)}", flush=True)
    except ImportError as missing:
        print(f"[preflight] stream not written: {missing}", flush=True)
    write_index(project, index_project(project))
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
