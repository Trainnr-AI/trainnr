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
directly. If you cannot use it, email trainnrai@gmail.com with
"security" in the subject and no details; a maintainer replies with a
private channel. Include the version or commit, the steps to reproduce, and
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

- The MCP server acts on the machine that runs it, as the user who runs
  it: it starts training runs, simulators and the Studio. Run it for
  projects you trust, and do not hand its stdio to an agent you do not.
- A project is data, and some of it is code. A checkpoint (`.pt`) is a
  Python pickle; trainnr reads every checkpoint in PyTorch's weights-only
  mode before the trainer loads it, and refuses one that carries code,
  but open other people's projects with the care you give their code.
- `tools/cloud-gpu.py` provisions rented machines with an API key read
  from the environment. Keys are never written into the repository; a
  report that one has been is a security report.

## Threat model

trainnr is a local developer tool that an AI agent drives. What it trusts,
what it does not, and what guards each boundary:

| Boundary | What could go wrong | What guards it |
|---|---|---|
| The agent calling the MCP server | An agent, or a prompt injected into what it reads, asks for an action the user did not intend | Paths a tool reads or writes (outputs, datasets, checkpoints) must resolve inside the open project, and a call with no project is refused; artifacts are named by plain names (no separators, drive marks or device names); an argument a tool does not take is refused; acting tools run as visible background jobs (`list_jobs`, `cancel_job`); no tool deletes an artifact or overwrites an experiment, deployment, task or project; a robot is never commanded over the network by a tool: the DDS gate and pre-flight run against a simulator on this machine's loopback. Some tools read a file the user names outside the project on purpose: `onboard_robot` (the model's folder, refused when it links outside itself or exceeds 2 GB), `import_experiment`, `capture_scene` (a video), `ingest_recording` |
| A project from someone else | Files in its `.index/`, `mcp-jobs/` or `deploy/` steer the Studio or the server | The Studio writes through no link a project carries and applies no command file older than its own session; a command's id must be a plain name; job status reads the log beside the job's record, never the path the record names; `quit_studio` signals only a process that is a Studio, and the Studio's Stop button refuses pid 1 and a reused pid; a deployment manifest's program names must be plain names; checkpoints are read weights-only first |
| Files the user opens | A crafted robot model, scene, recording or dataset exploits a parser (MuJoCo, USD, URDF, rosbag2, MCAP) | Inputs are the user's own or named public sources with a recorded digest; the parsers are the upstream libraries, kept current by Dependabot and audited (`tools/supply-chain.py`, `cargo-deny`) |
| Downloads | A tampered Studio binary, trainer environment or public log | See *Network* below for every download. The Studio comes from this repository's GitHub release over HTTPS, checked against the SHA-256 published beside it (this catches a corrupted download; a replaced release is guarded by GitHub's immutable releases and the build provenance attestation, `gh attestation verify`); Python packages come from PyPI pinned by hash in the lock files; a robot's fetched files are checked against the git blob ids its `FETCH.json` pins; registered public logs are checked against their recorded size and digest, and one that states no licence needs `accept_unlicensed` |
| The repository | A malicious pull request reaches a release | Only the maintainer merges (rulesets, CODEOWNERS); CI runs a pull request's code with a read-only token and no secrets; every action is pinned to a commit; release builds never restore a build cache; Dependabot and the advisory job flag known vulnerabilities |

### Network

What trainnr sends or receives, and when:

| When | What |
|---|---|
| The Claude Code plugin's session start (`hooks/hooks.json`; off with `TRAINNR_NO_PREFETCH=1`) | `uv sync` of the server's environment from PyPI (about 0.6 GB the first time; uv may download a Python), and the prebuilt Studio from this repository's GitHub release (about 72 MB) |
| The first job of a kind (training, evaluation, export) | `uv` builds the trainer's environment from PyPI (about 6 GB the first time) |
| `launch_studio` or `trainnr studio --install` without a local build | The prebuilt Studio from the GitHub release |
| The first use of a robot whose bundle has a `FETCH.json` (today the microduck; off with `TRAINNR_NO_ROBOT_FETCH=1`) | The files it names from their publisher's GitHub repository at a pinned commit, each checked against its git blob id: the microduck's 38 meshes, about 22 MB, from `pollen-robotics/microduck_rl`. `python -m trainnr.bundles.fetch robots/microduck` fetches them ahead |
| `ingest_public_log` | The named public dataset from its publisher (GitHub, Zenodo) |
| `evaluate_walk(student=…)` naming a Hugging Face repository | That model from the Hugging Face Hub |
| `tools/cloud-gpu.py` and the cloud tools | The RunPod API, with the user's key |

What listens:

| Listener | Address |
|---|---|
| The Studio's viewer server (Rerun gRPC) | `127.0.0.1:9876`; `TRAINNR_VIEWER_BIND` opens it wider, and the Studio warns when it does. Anyone who can reach it can stream into the window, read what it shows, and drive it |
| `play_walk`'s browser viewer (viser) | `127.0.0.1:8080` |
| `start_capture` over UDP | Every interface, because the robot sends from the network; at most an hour per capture; anything that reaches the port is recorded |
| The MCP server | None: it speaks over stdio |

Nothing phones home: there is no telemetry or update check, and Rerun's
analytics are compiled out of the Studio.

### Files outside a project

The projects home (`~/trainnr`, or `TRAINNR_HOME`) holds the projects,
`.current` (the project a new session starts on) and `cache/` (downloads such as public logs); the user's cache
folder (`~/.cache/trainnr/studio` on Linux, `~/Library/Caches` on macOS,
`%LOCALAPPDATA%` on Windows) holds the downloaded Studio; `uv` keeps its cache; the Studio keeps its
window state and Rerun's blueprints in the OS's application-data folder.

A report that any guard above can be bypassed, or that the tables leave
out a call, a listener or a file, is a security report.
