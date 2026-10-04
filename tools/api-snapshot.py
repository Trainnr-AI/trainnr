"""The MCP tool API, written down: every built-in tool's name,
description, input schema and annotations, as a client reads them at
session start.

    python3 tools/api-snapshot.py          # check: fails when the API moved
    python3 tools/api-snapshot.py --write  # regenerate, then review the diff

The snapshot lives at trainnr/tests/api/tools.json. A rename, a new
argument or a reworded description is an API change an agent's prompts
and a user's notes depend on, so it shows up as a diff in review rather
than slipping through. The installed tool plugins are left out: the
snapshot is the built-in API. Runs under any python3; without the mcp
extra it re-runs itself through uv in the package's environment.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "trainnr"
SNAPSHOT = PACKAGE / "tests" / "api" / "tools.json"
EXTRAS = ("sim", "mcp", "deploy", "viz")
CHANGED = (
    "the MCP tool API changed: run python3 tools/api-snapshot.py --write "
    "and review the diff"
)
REEXEC_FLAG = "TRAINNR_API_SNAPSHOT_REEXEC"


def render(api: list[dict]) -> str:
    """The snapshot's text: stable key order, one trailing newline."""
    return json.dumps(api, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def current() -> str:
    """The API the code builds now, rendered."""
    sys.path.insert(0, str(PACKAGE))
    from trainnr.mcp_server import build_server, tool_api  # noqa: PLC0415

    return render(tool_api(build_server(plugins=False)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="regenerate it")
    args = parser.parse_args()
    if importlib.util.find_spec("mcp") is None:
        if os.environ.get(REEXEC_FLAG):
            print(
                "the mcp extra did not install; cannot build the API", file=sys.stderr
            )
            return 2
        extras = [flag for extra in EXTRAS for flag in ("--extra", extra)]
        command = ["uv", "run", "--quiet", *extras, "python", __file__, *sys.argv[1:]]
        env = {**os.environ, REEXEC_FLAG: "1"}
        return subprocess.run(command, cwd=PACKAGE, env=env, check=False).returncode
    built = current()
    if args.write:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(built, encoding="utf-8")
        tools = len(json.loads(built))
        print(f"wrote {SNAPSHOT.relative_to(REPO)} ({tools} tools)")
        return 0
    recorded = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.is_file() else ""
    if built != recorded:
        print(CHANGED, file=sys.stderr)
        return 1
    print("MCP tool API matches its snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
