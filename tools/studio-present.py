"""The presenter: the one Python process the Studio asks to show things.

    cd pipeline && uv run --extra sim --extra viz \\
        python ../tools/studio-present.py [--project <dir>] [--once]

Watches `<project>/.index/present.json` — the Studio writes `{"stamp":
...}` when the user clicks Show on an artifact, or `{"stamps": [a, b]}`
for a compare — and streams that into the embedded viewer as itself
(`rq_pipeline.project.present`). The Studio spawns this once per open
project; an agent reaches it through the `show_in_studio` and
`compare_in_studio` MCP tools (docs/76 §10.1), which command the Studio, which
writes the intent here.
"""

import argparse
import os
from pathlib import Path

from _lab import bootstrap

bootstrap()

from rq_pipeline.project import PROJECT_ENV, current_project  # noqa: E402
from rq_pipeline.project.present import serve  # noqa: E402


def watch_parent(pid: int) -> None:
    """A daemon thread: when the Studio's pid is gone, so are we."""
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    def watch() -> None:
        while True:
            time.sleep(1.0)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                os._exit(0)
            except PermissionError:
                continue

    threading.Thread(target=watch, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument(
        "--once", action="store_true", help="present one intent and exit"
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="exit when this process (the Studio) is gone — six presenters "
        "outlived their windows on 2026-09-09",
    )
    args = parser.parse_args()
    if args.parent_pid:
        watch_parent(args.parent_pid)
    if args.project is not None:
        os.environ[PROJECT_ENV] = str(args.project)
    project = current_project()
    print(
        f"presenter watching {project.root} (write .index/present.json to show)",
        flush=True,
    )
    serve(project, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
