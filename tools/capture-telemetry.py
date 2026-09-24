"""Record a robot's telemetry live into a project, as the door does.

    python3 tools/capture-telemetry.py sources
    python3 tools/capture-telemetry.py record walk-1 --project projects/go2-walk \\
        --source dds --network eth0 --seconds 60
    python3 tools/capture-telemetry.py record standin-1 --project projects/go2-walk \\
        --standin projects/go2-walk/deploy/go2-c2-deploy-cited --seconds 10

`record` listens on one registered source (`rq_pipeline.robots.capture.
SOURCES`: `udp` the Pico rig, `dds` Unitree's `rt/lowstate` and `rt/lowcmd`
as one recording) for `--seconds`, then ingests it into the project; the
Studio's Recordings page shows it filling. `--standin DEPLOYMENT` is the
hardware day rehearsed: Unitree's own simulator and controller for that
deployment brought up on loopback (the DDS gate's stack, no virtual pad)
and captured with the basis `simulation`. On the robot: `--source dds
--network <the robot's interface>` and the basis stays `own robot`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.project import index_project, write_index  # noqa: E402
from rq_pipeline.project.ingest import capture, ingest  # noqa: E402
from rq_pipeline.project.locate import INDEX_DIR, Project  # noqa: E402
from rq_pipeline.robots.capture import DEFAULT_SOURCE, sources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    sub.add_parser("sources", help="every source a capture listens on")
    rec = sub.add_parser("record", help="listen, then ingest into the project")
    rec.add_argument("name", help="the recording it becomes")
    rec.add_argument("--project", required=True, type=Path)
    rec.add_argument("--seconds", type=float, required=True)
    rec.add_argument("--source", default=DEFAULT_SOURCE)
    rec.add_argument("--network", default=None, help="dds: the interface (lo, eth0)")
    rec.add_argument("--port", type=int, default=None, help="udp: the port")
    rec.add_argument("--basis", default=None, help="dds: own robot | simulation")
    rec.add_argument(
        "--standin",
        type=Path,
        default=None,
        help="a deployment whose Unitree simulator and controller stand in for "
        "the robot (basis simulation)",
    )
    args = parser.parse_args()
    if args.verb == "sources":
        print(json.dumps(sources(), indent=1))
        return 0
    project = Project(args.project.resolve()).use()
    if args.standin is not None:
        state = _standin(project, args)
    else:
        given = {"network": args.network, "port": args.port, "basis": args.basis}
        options = {k: v for k, v in given.items() if v is not None}
        listener = capture(project, args.name, source=args.source, **options)
        listener.start(window_s=args.seconds + 60.0)
        time.sleep(args.seconds)
        state = listener.stop()
    write_index(project, index_project(project))
    print(json.dumps(asdict(state), indent=1))
    return 0 if state.stamp else 1


def _standin(project: Project, args: argparse.Namespace) -> object:
    from rq_pipeline.robots.dds_capture import standin_capture  # noqa: PLC0415

    def land(_recordings: Path, source: Path, **kw: object) -> dict[str, object]:
        return ingest(project, source, **kw)  # type: ignore[arg-type]

    def reindex(_state: object) -> None:
        write_index(project, index_project(project))

    return standin_capture(
        args.standin.resolve(),
        project.recordings,
        project.root / INDEX_DIR,
        args.name,
        seconds=args.seconds,
        ingest=land,
        on_state=reindex,
    )


if __name__ == "__main__":
    raise SystemExit(main())
