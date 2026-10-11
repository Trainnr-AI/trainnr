"""The `trainnr` command: the MCP server, the Studio, the version.

    trainnr mcp        serve the tools over stdio (what an agent's config runs)
    trainnr studio     launch the trainnr Studio on the current project
                       (its Welcome page before any project exists)
    trainnr sample     list the sample projects, or open one
    trainnr call       run one tool by name, as an agent would
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
REFUSED_EXIT = 3  # `trainnr call`: the tool ran and refused, with its reason


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
    call = sub.add_parser(
        "call", help="run one tool by name with JSON arguments, as an agent would"
    )
    call.add_argument("tool", nargs="?", help="the tool's name (see --list)")
    call.add_argument(
        "--args", default="{}", help="the arguments as a JSON object (default {})"
    )
    call.add_argument("--list", action="store_true", help="list the tools and exit")
    sub.add_parser("version", help="print the version")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(_version())
        return 0
    if args.command == "mcp":
        return _mcp()
    if args.command == "studio":
        return _studio(install=args.install)
    if args.command == "sample":
        return _sample(args.action, args.name)
    if args.command == "call":
        return _call(args.tool, args.args, listing=args.list)
    return 2


def _mcp() -> int:
    """Serve the tools over stdio: what an agent's MCP configuration runs."""
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


def _call(tool: str | None, raw: str, *, listing: bool) -> int:
    """`trainnr call <tool> --args '<json>'`: the tool's result as JSON on
    stdout, exit 0; a refusal is a result too, exit 3 so a caller can tell
    it from success; an unknown tool, bad arguments or a failure is one
    line on stderr, exit 1. The Studio's buttons run this."""
    import json  # noqa: PLC0415

    try:
        from trainnr import mcp_server  # noqa: PLC0415

        server = mcp_server.build_server()
    except ImportError as why:
        print(f"trainnr call needs the `sim` and `mcp` extras ({why})", file=sys.stderr)
        return 1
    if listing:
        for entry in mcp_server.tool_api(server):
            print(entry["name"])
        return 0
    if not tool:
        print("trainnr call: name a tool, or --list", file=sys.stderr)
        return 1
    arguments = _json_object(raw)
    if arguments is None:
        print("trainnr call: --args must be a JSON object", file=sys.stderr)
        return 1
    try:
        result = mcp_server.call_tool(tool, arguments, server=server)
    except mcp_server.ToolCallError as why:
        print(f"trainnr call {tool}: {why}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, default=str))
    refused = isinstance(result, dict) and result.get("status") == "refused"
    return REFUSED_EXIT if refused else 0


def _json_object(raw: str) -> dict | None:
    """`raw` as a JSON object, or None when it is not one."""
    import json  # noqa: PLC0415

    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


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
