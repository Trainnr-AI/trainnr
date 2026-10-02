"""Serve the instrument's MCP surface over stdio (kept for old configs).

    cd trainnr && uv run --extra sim --extra mcp trainnr mcp

is the same server through the package's console script, which is what
`.mcp.json`, the Desktop's agent panel and `uvx trainnr mcp` run; this
file only forwards to it.

The doors an agent works the loop through: describe (bundles with their hash
identity and fit verdicts, the actuator library, the task and engine
registries, run manifests. Any MCP client connects the same way: the
Studio's agent panel hands this command to its ACP session, a Claude
Code terminal session picks it up from `.mcp.json` at the repo root.
The tools themselves live in `trainnr.mcp_server`, where the suite
tests them without the `mcp` extra.
"""

from _lab import bootstrap

bootstrap()

from trainnr.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
