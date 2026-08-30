"""Serve the instrument's MCP surface over stdio.

    cd pipeline && uv run --extra sim --extra mcp python ../tools/mcp-server.py

Read-only tools over the repo's public seams — bundles with their hash
identity and fit verdicts, the actuator library, the task and engine
registries, run manifests. Any MCP client connects the same way: the
Studio's agent panel hands this command to its ACP session, a Claude
Code terminal session picks it up from `.mcp.json` at the repo root.
The tools themselves live in `rq_pipeline.mcp_server`, where the suite
tests them without the `mcp` extra.
"""

from _lab import bootstrap

bootstrap()

from rq_pipeline.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
