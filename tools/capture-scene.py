#!/usr/bin/env python3
"""Capture a scene from a phone video or a folder of frames: ffmpeg,
COLMAP, Brush, the alignment, the proxy, the gap, the record (docs/78
§3). Spawned by the MCP door `capture_scene`; the log lands beside the
scene as it runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _lab import bootstrap, running

bootstrap()

from trainnr.project import index_project, write_index  # noqa: E402
from trainnr.project.locate import Project  # noqa: E402
from trainnr.scenes.capture import (  # noqa: E402
    FRAMES_PER_SECOND,
    MissingToolError,
    Tools,
    capture_scene,
)
from trainnr.scenes.record import (  # noqa: E402
    CAPTURE_FAILED_WORD,
    CAPTURE_LOG_FILE,
    CAPTURE_STAGE_PREFIX,
    UNRECORDED,
    load_scene_record,
)
from trainnr.scenes.splatters import (  # noqa: E402
    AUTO,
    DEFAULT_SPLATTER,
    DEFAULT_STEPS,
    SPLATTERS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, required=True, help="a video or a folder"
    )
    parser.add_argument("--name", required=True, help="the scene's folder name")
    parser.add_argument("--fps", type=float, default=FRAMES_PER_SECOND)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--scale", type=float, default=None, help="metres per COLMAP unit"
    )
    parser.add_argument("--floor-friction", type=float, nargs=3, default=None)
    parser.add_argument("--brush", type=Path, default=None, help="Brush's binary")
    parser.add_argument(
        "--splatter",
        default=DEFAULT_SPLATTER,
        choices=(AUTO, *SPLATTERS),
        help="the splat trainer; auto takes gsplat where CUDA answers, else Brush",
    )
    parser.add_argument("--device", default=UNRECORDED)
    parser.add_argument("--lighting", default=UNRECORDED)
    args = parser.parse_args()
    project = Project(args.project.resolve()).use()
    try:
        tools = Tools.find(
            brush=args.brush, video=args.source.is_file(), splatter=args.splatter
        )
        with running(project.root, name=args.name) as run:
            run.stage(f"capturing {args.name} from {args.source.name}")
            record_path = capture_scene(
                args.source,
                project.scenes / args.name,
                name=args.name,
                tools=tools,
                fps=args.fps,
                steps=args.steps,
                scale=args.scale,
                floor_friction=args.floor_friction,
                device=args.device,
                lighting=args.lighting,
                on_stage=run.stage,
            )
    except (MissingToolError, FileNotFoundError, ValueError, RuntimeError) as exc:
        # the scene's own log carries the last word, so the index shows a
        # dead capture as failed and not as work in progress
        log = project.scenes / args.name / CAPTURE_LOG_FILE
        if log.parent.is_dir():
            with log.open("a", encoding="utf-8") as out:
                out.write(f"{CAPTURE_STAGE_PREFIX}{CAPTURE_FAILED_WORD}{exc}\n")
        raise SystemExit(str(exc)) from exc
    record = load_scene_record(record_path)
    g = record.gap
    print(
        f"[capture] {record.name}: {record.splat.get('count')} gaussians, "
        f"proxy {record.proxy.get('faces')} faces, gap chamfer {g.chamfer_m} "
        f"p95 {g.p95_m} (m)",
        flush=True,
    )
    write_index(project, index_project(project))
    sys.exit(0)


if __name__ == "__main__":
    main()
