#!/usr/bin/env python3
"""An agent drives trainnr: the laptop half of the Quickstart, done by a
real Claude Code session that sees only the trainnr MCP server.

    python3 tools/agent-e2e.py                       # this checkout
    python3 tools/agent-e2e.py --checkout ../other   # another checkout
    python3 tools/agent-e2e.py --model claude-sonnet-5-5

The agent gets a plain-language task (create a project, onboard the Go2,
ingest a public log, identify the dynamics) and nothing else: no tool
names, no hints. The run passes when the fit record exists on disk and
the agent's answer names the pinned count; it reports the tools the
agent called in order, the errors it hit, its turns, time and cost. The
tool names and descriptions are what an agent reads to choose a tool,
so this is the test of them that matters.

Needs `claude` (logged in), `uv`, network for the public log, and the
Go2 model from unitree_rl_mjlab (`--go2`, or `$TRAINNR_GO2_XML`). It runs
in a scratch TRAINNR_HOME, so nothing lands in the user's projects.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The whole laptop half took 20-30 s across runs (2026-10-04); a hung
# session fails at this, rather than holding the terminal forever.
TIMEOUT_S = 900
TASK = (
    "Using the trainnr tools: create a project called demo, onboard the "
    "Unitree Go2 robot from the model file {go2} under the name go2, ingest "
    "the public log iit-go2-chirp, and identify the robot's joint dynamics "
    "from that recording. Then tell me, in one sentence, how many "
    "parameters were pinned out of how many."
)


def _answers_the_record(fit: Path, answer: str) -> bool:
    """The fit is the Go2's, from the chirp log, and the answer states its
    pinned count and its total, in that order, as whole numbers."""
    record = json.loads(fit.read_text(encoding="utf-8"))
    parameters = record.get("parameters", [])
    pinned = sum(1 for p in parameters if p.get("pinned"))
    said = re.search(rf"\b{pinned}\b.*\b{len(parameters)}\b", answer, re.DOTALL)
    return (
        record.get("robot", "").startswith("go2")
        and record.get("recording", "").startswith("iit-go2-chirp")
        and said is not None
    )


def _read_stream(
    stdout: str,
) -> tuple[list[str], list[str], dict[str, object]]:
    """The tools the session called, in order; the results that were errors
    or refusals; and its final result event."""
    calls: list[str] = []
    errors: list[str] = []
    result: dict[str, object] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (event.get("message") or {}).get("content") or []
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "tool_use":
                calls.append(block["name"].removeprefix("mcp__trainnr__"))
            if block.get("type") == "tool_result":
                text = str(block.get("content"))
                # a refusal is a normal result to the client; count it
                if block.get("is_error") or '"status":"refused"' in text.replace(
                    " ", ""
                ):
                    errors.append(text[:160])
        if event.get("type") == "result":
            result = event
    return calls, errors, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkout", type=Path, default=REPO)
    parser.add_argument("--go2", default=os.environ.get("TRAINNR_GO2_XML"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()
    if not args.go2 or not Path(args.go2).is_file():
        print(
            "the Go2 model: git clone --depth 1 "
            "https://github.com/unitreerobotics/unitree_rl_mjlab and pass "
            "--go2 <it>/src/assets/robots/unitree_go2/xmls/go2.xml"
        )
        return 2
    if not shutil.which("claude"):
        print("needs the claude CLI on PATH, logged in: https://claude.com/claude-code")
        return 2
    checkout = args.checkout.resolve()
    with tempfile.TemporaryDirectory(prefix="trainnr-agent-e2e-") as scratch:
        home = Path(scratch) / "home"
        config = Path(scratch) / "mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "trainnr": {
                            "command": "uv",
                            "args": [
                                "run",
                                "--directory",
                                str(checkout / "trainnr"),
                                "--extra",
                                "sim",
                                "--extra",
                                "mcp",
                                "trainnr",
                                "mcp",
                            ],
                            # every location the server reads is pinned
                            # here, so the user's own settings cannot
                            # point the run at their real projects
                            "env": {
                                "TRAINNR_HOME": str(home),
                                "TRAINNR_PROJECTS": str(home / "projects"),
                                "TRAINNR_PROJECT": "",
                                "TRAINNR_RUN_SOURCE": "agent",
                            },
                        }
                    }
                }
            )
        )
        command = [
            "claude",
            "-p",
            TASK.format(go2=args.go2),
            "--mcp-config",
            str(config),
            "--strict-mcp-config",
            "--tools",  # no built-in tools: the trainnr server is all it has
            "",
            "--allowedTools",
            "mcp__trainnr",
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(args.max_turns),
        ]
        if args.model:
            command += ["--model", args.model]
        started = time.monotonic()
        try:
            run = subprocess.run(
                command,
                cwd=scratch,
                capture_output=True,
                text=True,
                check=False,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL (the session ran past {TIMEOUT_S} s)")
            return 1
        elapsed = time.monotonic() - started
        calls, errors, result = _read_stream(run.stdout)
        fits = sorted((home / "projects").rglob("robots/*/fits/*.json"))
        answer = str(result.get("result", ""))
        passed = bool(fits) and _answers_the_record(fits[-1], answer)
        print(f"checkout: {checkout}")
        print(f"tools called ({len(calls)}): {' → '.join(calls)}")
        print(f"tool errors: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
        print(
            f"turns: {result.get('num_turns')}, time: {elapsed:.0f} s, "
            f"cost: ${result.get('total_cost_usd', 0):.3f}"
        )
        print(f"fit records on disk: {len(fits)}")
        print(f"answer: {answer.strip()[:300]}")
        print("PASS" if passed else "FAIL")
        if result.get("is_error") or (run.returncode != 0 and not result):
            print(f"session ended: {result.get('subtype', 'no result event')}")
            print(run.stderr[-2000:])
        return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
