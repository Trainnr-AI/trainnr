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
  from the environment. Keys are never written into the repository, and a
  push to a rented machine leaves out credentials under any common name
  (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `.netrc`, `credentials*.json`);
  its ssh accepts a new machine's host key on first contact
  (`StrictHostKeyChecking=accept-new`). A report that a key has been
  shipped or written is a security report.
- The scripts under `tools/` are developer tools and do more than the
  packages: `tools/install-brush.py` installs the Brush binary into
  `~/.local/bin`, and `tools/udp-wire-bridge.py` listens on every interface
  for the 2025–26 rig's telemetry.

## Threat model

trainnr is a local developer tool that an AI agent drives. What it trusts,
what it does not, and what guards each boundary:

| Boundary | What could go wrong | What guards it |
|---|---|---|
| The agent calling the MCP server | An agent, or a prompt injected into what it reads, asks for an action the user did not intend | Paths a tool reads or writes (outputs, datasets, checkpoints) must resolve inside the open project, and a call with no project is refused; `evaluate_walk(student=…)` takes a folder inside the project or a Hugging Face repo id, and a student whose processor files name a step that is not LeRobot's own is refused before it loads (LeRobot imports the classes they name); artifacts are named by plain names (no separators, drive marks or device names); an argument a tool does not take is refused; acting tools run as visible background jobs (`list_jobs`, `cancel_job`); no tool deletes an artifact or overwrites an experiment, deployment, task or project; a robot is never commanded over the network by a tool: the DDS gate and pre-flight run against a simulator on this machine's loopback. Some tools read a file the user names outside the project on purpose: `onboard_robot` (the model's folder, refused when it links outside itself or exceeds 2 GB), `import_experiment`, `capture_scene` (a video), `ingest_recording` |
| A project from someone else | Files in its `.index/`, `mcp-jobs/` or `deploy/` steer the Studio or the server | Neither the server nor the Studio writes, appends or deletes through a link a project carries (`trainnr.safe_write`: the target and every folder below the project's root are checked, and a staging file is created fresh); the Studio applies a command only when it carries the running Studio's session token (`studio-state.json`), so no command file that came with a project is applied; a command's id and a deployment's name must be plain names; job status and the Studio's log panel read the log beside the job's record, never the path the record names, and the Studio opens a record's viewer file only inside the project; `quit_studio` signals only a process that is a Studio, and the Studio's Stop button signals a pid only when its process started when the record says (both ways; pid 1 refused); a bundle's `FETCH.json` names a repository and a commit by their shapes and writes only inside the bundle; a deployment manifest's program names must be plain names; checkpoints are read weights-only first |
| Files the user opens | A crafted robot model, scene, recording or dataset exploits a parser (MuJoCo, USD, URDF, rosbag2, MCAP) | Inputs are the user's own or named public sources with a recorded digest; the parsers are the upstream libraries, kept current by Dependabot and audited (`tools/supply-chain.py`, `cargo-deny`) |
| Downloads | A tampered Studio binary, trainer environment or public log | See *Network* below for every download. The Studio comes from this repository's GitHub release over HTTPS (a redirect away from HTTPS is refused), checked against the SHA-256 published beside it (this catches a corrupted download; a replaced release is guarded by GitHub's immutable releases and the build provenance attestation, `gh attestation verify`); Python packages come from PyPI pinned by hash in the lock files; a robot's fetched files are checked against the git blob ids its `FETCH.json` pins; registered public logs are checked against their recorded size and digest (a zip member is never inflated past its recorded size), and one that states no licence needs `accept_unlicensed` |
| The repository | A malicious pull request reaches a release | Only the maintainer merges (rulesets, CODEOWNERS); CI runs a pull request's code with a read-only token and no secrets; every action is pinned to a commit; release builds never restore a build cache; Dependabot and the advisory job flag known vulnerabilities |

### Network

What trainnr sends or receives, and when:

| When | What |
|---|---|
| The Claude Code plugin's session start (`hooks/hooks.json`; off with `TRAINNR_NO_PREFETCH=1`) | `uv sync` of the server's environment from PyPI (about 0.6 GB the first time; uv may download a Python), and the prebuilt Studio from this repository's GitHub release (about 72 MB) |
| The first job of a kind (training, evaluation, export) | `uv` builds the trainer's environment from PyPI (about 6 GB the first time) |
| `launch_studio` or `trainnr studio --install` without a local build | The prebuilt Studio from the GitHub release; with `GH_TOKEN` or `GITHUB_TOKEN` set, the release is read through api.github.com with that token (never sent to another host) |
| The Studio's presenter and simulator (`uv run` from the checkout) | uv may sync the server's or the trainer's environment from PyPI the first time, as the jobs above do |
| The first use of a robot whose bundle has a `FETCH.json` (today the microduck; off with `TRAINNR_NO_ROBOT_FETCH=1`) | The files it names from their publisher's GitHub repository at a pinned commit, each checked against its git blob id: the microduck's 38 meshes, about 22 MB, from `pollen-robotics/microduck_rl`. `python -m trainnr.bundles.fetch robots/microduck` fetches them ahead |
| `ingest_public_log` | The named public dataset from its publisher (GitHub, Zenodo) |
| `evaluate_walk(student=…)` naming a Hugging Face repository | That model from the Hugging Face Hub |
| `open_sample` (the Studio's samples) the first time a sample is opened | The sample's archive from the Hugging Face dataset `trainnr/trainnr-artifacts` at a pinned commit (the Go2 walk: 12 MB), its size and SHA-256 checked before it is unpacked; unpacked with the standard library's `data` filter, into the projects home only |
| `tools/cloud-gpu.py` and the cloud tools | The RunPod API, with the user's key |

What listens:

| Listener | Address |
|---|---|
| The Studio's viewer server (Rerun gRPC) | `127.0.0.1:9876`; `TRAINNR_VIEWER_BIND` opens it wider, and the Studio warns when it does. Anyone who can reach it can stream into the window and read what it shows |
| `play_walk`'s browser viewer (viser) | `127.0.0.1:8080` |
| `start_capture` over UDP | Every interface, because the robot sends from the network; at most an hour per capture; anything that reaches the port is recorded |
| `start_capture` over DDS | Joins the DDS domain on the network interface it is given (the robot's), to subscribe to its topics |
| The MCP server | None: it speaks over stdio |

Nothing phones home: there is no telemetry or update check, and Rerun's
analytics are compiled out of the Studio.

### Files outside a project

The projects home (`~/trainnr`, or `TRAINNR_HOME`) holds the projects,
`.current` (the project a new session starts on) and `cache/` (downloads such as public logs); the user's cache
folder (`~/.cache/trainnr/studio` on Linux, `~/Library/Caches` on macOS,
`%LOCALAPPDATA%` on Windows) holds the downloaded Studio; `uv` keeps its cache; the Studio keeps its
window state and Rerun's blueprints in the OS's application-data folder;
the Studio's simulator hands frames to its renderer through a file in the
system's temporary folder, under a random name, readable by the user
alone.

A report that any guard above can be bypassed, or that the tables leave
out a call, a listener or a file, is a security report.
