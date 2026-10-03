# Security policy

## Supported versions

trainnr is pre-1.0. Fixes land on `main` and in the next release; there
are no maintained older lines.

| Version | Supported |
|---|---|
| `main` and the latest release | yes |
| anything older | no |

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's
private vulnerability reporting on this repository ("Report a
vulnerability" under the Security tab), which reaches the maintainers
directly. Include the version or commit, the steps to reproduce, and
what an attacker gains. You will get an acknowledgement within a week and
a fix or a decision within thirty days for anything confirmed; we credit
reporters in the release notes unless they prefer otherwise.

## What is in scope

- The `trainnr` and `trainnr-mjlab` Python packages and the `trainnr`
  command-line tool, including the MCP server (`trainnr mcp`) that agents
  connect to over stdio.
- the Studio (`crates/trainnr-studio`), including the files it reads
  from a project directory (`.index/`) and the processes it spawns.
- The Claude Code plugin (`.claude-plugin/`): its skills, agents and the
  MCP server entry.

## Things to know

- The MCP server executes actions on the machine that runs it (training
  runs, simulators, the Studio) in the project directory it is
  given. Treat it like any local developer tool: run it for projects you
  trust, and do not expose its stdio to untrusted agents.
- `tools/cloud-gpu.py` provisions rented machines with an API key read
  from the environment. Keys are never written into the repository; a
  report that one has been is a security report.
