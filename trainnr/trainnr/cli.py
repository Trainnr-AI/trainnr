"""The `trainnr` command: the MCP server, the Studio, the version.

    trainnr mcp        serve the tools over stdio (what an agent's config runs)
    trainnr studio     launch the trainnr Studio on the current project
    trainnr version    the installed version

Each subcommand is a thin door over the module that owns the work; the
pipeline's other operations stay the MCP tools and the scripts under
tools/ until they earn a subcommand.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGE = "trainnr"


def _version() -> str:
    try:
        return version(PACKAGE)
    except PackageNotFoundError:
        return "0.0.0+checkout"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=PACKAGE, description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mcp", help="serve the tools over stdio (needs the `mcp` extra)")
    sub.add_parser("studio", help="launch the trainnr Studio on the current project")
    sub.add_parser("version", help="print the version")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(_version())
        return 0
    if args.command == "mcp":
        from trainnr.mcp_server import main as serve  # noqa: PLC0415

        serve()
        return 0
    if args.command == "studio":
        from trainnr.project import current_project  # noqa: PLC0415
        from trainnr.project.control import launch  # noqa: PLC0415

        answer = launch(current_project())
        print(answer)
        return 0 if answer.get("status") == "done" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
