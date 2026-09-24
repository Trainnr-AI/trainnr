#!/usr/bin/env python3
"""Pre-flight: every check a deployment must pass before the first tick on
a robot, the ramp-in and the soft stop measured, the record beside the
manifest (docs/77 §10). A runtime with a stack around it (Unitree's own,
over DDS) has the robot's state read from that stack in its fixed stand,
before the handover, and its own stop measured beside ours; whether that
robot is a stand-in (their simulator) is the stack's to say. Spawned by
the MCP door `preflight_deployment`.

Ctrl-C while the stack's robot walks is the operator's stop: their Passive
chord through the pad, never a killed process with the motors mid-stride;
so is any failure after the handover.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from _lab import bootstrap, deployment_args, resolve_deployment

bootstrap()

from rq_pipeline.bundles.basis import BASIS_OWN, BASIS_SIMULATION  # noqa: E402
from rq_pipeline.deploy.preflight import (  # noqa: E402
    CHECK_MARKS,
    POSES_TRACK,
    PREFLIGHT_FILE,
    STOP_FILE,
    card_line,
    health_of_lowstate,
    measure_dds_stop,
    preflight,
    refuse_stale_stop,
    still_at_handover,
    stream,
    walk_command,
    write_record,
)
from rq_pipeline.deploy.runtimes import runtime_spec  # noqa: E402
from rq_pipeline.project import index_project, write_index  # noqa: E402

CTRL_C = "Ctrl-C"


def _with_stack(
    spec: Any, manifest: Any, deployment: Path, assets: Path, seed: int | None
) -> dict:
    """The stack up; the robot stood by its controller; the state read
    before handover; the checks; then the handover, a walk and its stop.
    The Passive chord goes out whatever happens after the handover."""
    with spec.stack_for(manifest) as stack:
        runtime = spec.open()(manifest, assets_dir=None, **stack.runtime_options())
        basis = BASIS_SIMULATION if stack.stand_in else BASIS_OWN
        stopped = {"by": None}

        def operator_stop(*_: object) -> None:  # the keyboard's hook
            runtime.stop()
            stopped["by"] = CTRL_C
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGINT, operator_stop)
        try:
            runtime.stand()
            health = health_of_lowstate(runtime.health(), stand_in=stack.stand_in)
            record = preflight(
                deployment,
                assets_dir=assets,
                health=health,
                state_from=stack.state_from,
                basis=basis,
                stand_in=stack.stand_in,
                seed=seed,
            )
            if record["passed"]:
                runtime.handover()
                record["their_stop"] = measure_dds_stop(
                    runtime,
                    walk_command(manifest),
                    stand_in=stack.stand_in,
                    stop_file=deployment / STOP_FILE,
                    poses=record.get(POSES_TRACK),
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
    deployment_args(parser)
    parser.add_argument(
        "--seed", type=int, default=None, help="the gate's own when unset"
    )
    parser.add_argument("--no-still", action="store_true")
    args = parser.parse_args()
    try:
        project, deployment, manifest, assets = resolve_deployment(args)
        refuse_stale_stop(deployment)
        spec = runtime_spec(args.runtime)
        if spec.stack is not None:
            record = _with_stack(spec, manifest, deployment, assets, args.seed)
        else:
            record = preflight(deployment, assets_dir=assets, seed=args.seed)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    for c in record["checks"]:
        print(f"[preflight] {CHECK_MARKS[c['passed']]:>7} {c['name']}: {c['measured']}")
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
