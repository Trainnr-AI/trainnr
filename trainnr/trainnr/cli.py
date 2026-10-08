"""The `trainnr` command: the MCP server, the Studio, the version.

    trainnr mcp        serve the tools over stdio (what an agent's config runs)
    trainnr studio     launch the trainnr Studio on the current project
                       (its Welcome page before any project exists)
    trainnr sample     list the sample projects, or open one
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
DEFAULT_SAMPLE = "go2-walk"


def _version() -> str:
    try:
        return version(PACKAGE)
    except PackageNotFoundError:
        return "0.0.0+checkout"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=PACKAGE, description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mcp", help="serve the tools over stdio (needs the `mcp` extra)")
    studio = sub.add_parser(
        "studio", help="launch the trainnr Studio on the current project"
    )
    studio.add_argument(
        "--install",
        action="store_true",
        help="download the prebuilt Studio for this platform and exit",
    )
    sample = sub.add_parser("sample", help="list the sample projects, or open one")
    sample.add_argument("action", choices=("list", "open"))
    sample.add_argument("name", nargs="?", default=DEFAULT_SAMPLE)
    sub.add_parser("version", help="print the version")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(_version())
        return 0
    if args.command == "mcp":
        try:
            from mcp.server import MCPServer  # noqa: F401, PLC0415

            from trainnr.mcp_server import main as serve  # noqa: PLC0415
        except ImportError as why:
            print(
                f"trainnr mcp needs the `sim` and `mcp` extras ({why}). From a "
                "checkout: uv run --directory trainnr --extra sim --extra mcp "
                "trainnr mcp",
                file=sys.stderr,
            )
            return 1
        serve()
        return 0
    if args.command == "studio":
        return _studio(install=args.install)
    if args.command == "sample":
        return _sample(args.action, args.name)
    return 2


def _studio(*, install: bool) -> int:
    """Download the prebuilt Studio (`--install`), or launch it on the
    current project, on its Welcome page before any project exists."""
    if install:
        from trainnr.studio_install import main as install_studio  # noqa: PLC0415

        return install_studio([])
    from trainnr.project import current_project  # noqa: PLC0415
    from trainnr.project.control import launch  # noqa: PLC0415

    try:
        project = current_project()
    except FileNotFoundError:
        project = None
    answer = launch(project)
    print(answer)
    return 0 if answer.get("status") == "done" else 1


def _sample(action: str, name: str) -> int:
    """`trainnr sample list` prints the samples as JSON; `trainnr sample
    open <name>` downloads, checks and opens one (the Studio's Open
    button runs this), printing its folder or the reason it could not."""
    import json  # noqa: PLC0415

    from trainnr import samples  # noqa: PLC0415
    from trainnr.project.locate import projects_home  # noqa: PLC0415

    if action == "list":
        print(json.dumps(samples.describe(projects_home()), indent=1))
        return 0
    try:
        root = samples.open_sample(name)
    except (KeyError, samples.SampleError) as why:
        # the Studio shows this line when its Open button fails
        print(f"trainnr sample open: {why.args[0]}", file=sys.stderr)
        return 1
    print(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
