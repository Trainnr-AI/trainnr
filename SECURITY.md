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
- The Studio (`crates/trainnr-studio`), including the files it reads
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

## Threat model

trainnr is a local developer tool that an AI agent drives. What it trusts,
what it does not, and what guards each boundary:

| Boundary | What could go wrong | What guards it |
|---|---|---|
| The agent calling the MCP server | An agent, or a prompt injected into what it reads, asks for an action the user did not intend: a training run, a deployment to a robot, a deletion | Tools act only inside the open project directory; acting tools run as visible background jobs the user can list and cancel (`list_jobs`, `cancel_job`); nothing is deleted by a tool; deploying to hardware is the vendor's runtime, started by the user, after a pre-flight that refuses by name |
| Files the user opens | A crafted robot model, scene, recording or dataset exploits a parser (MuJoCo, USD, URDF, rosbag2, MCAP) | Inputs are the user's own or named public sources with a recorded digest; the parsers are the upstream libraries, kept current by Dependabot and audited weekly (`tools/supply-chain.py`, `cargo-deny`) |
| Downloads | A tampered Studio binary or public log | The Studio comes from this repository's GitHub release over HTTPS and is refused unless its SHA-256 matches; registered public logs are checked against their recorded size and digest |
| Network | Data leaving the machine | The server listens on no network port; the Studio's viewer server binds to the local machine; cloud GPUs are used only when the user asks, with their key read from the environment and never written to the repository |
| The repository | A malicious pull request reaches a release | Only maintainers merge; CI runs a pull request's code with a read-only token and no secrets; every action is pinned to a commit; release builds never restore a build cache; Dependabot and the weekly supply-chain job flag known advisories |

A report that any row's guard can be bypassed is a security report.
