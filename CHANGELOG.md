# Changelog

All notable changes to trainnr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

First public push, 2026-10-02: `github.com/Trainnr-AI/trainnr`.

### Changed (breaking)

- MCP tools follow one naming rule: `verb_noun`, `list_` for a collection, `describe_` for one item, an existing artifact named by its kind and `name` only for a new one. There are no aliases; an agent prompt or note using an old name must be updated.
  - Renamed: `describe_bundles` → `list_robots`, `describe_bundle(name)` → `describe_robot(robot)`, `describe_runs` → `list_experiments`, `describe_tasks` → `list_tasks` (rows carry `task`), `describe_task_families` → `list_task_families`, `describe_engines` → `list_engines`, `describe_task(task_id)` → `describe_task(task)`, `list_eval_records` → `list_evaluations`, `describe_eval(run)` → `describe_evaluation(evaluation)`, `list_ledger_findings` → `list_findings`, `friction_curve(slug, …)` → `describe_friction(actuator, …)`, `job_status` → `describe_job`, `capture_status` → `describe_capture`, `accept_task(name)` → `check_task(task)`, `describe_viewer_recording` → `describe_viewer_stream`, `simulate_in_studio(task)` → `run_simulation(scene)`, `describe_datasheet(demos_dir)` → `describe_dataset(dataset)`.
  - Merged: `describe_actuators` and `describe_actuator_bundles` → `list_actuators` (one row per actuator, its certified bundles inside); `describe_actuator` and `describe_actuator_bundle` → `describe_actuator(actuator, tier)` (the model and that tier's bundle); `focus_studio_recording(recording)` → `show_in_studio(stream=…)`, which takes exactly one of `artifact` or `stream`.
  - Arguments: `export_deployment(run, …, certificate)` → `(experiment, …, evaluation)`; `gate_deployment(name)` → `(deployment)`; `create_task(task_id, name, overlay)` → `(family, name, settings)`; `play_walk(run)` → `(experiment)`; `ingest_public_log(name, recording_name)` → `(log, name)`; `import_finding(record)` → `(finding)`; `control_simulator(run)` → `(play)`; `multiply_demos(seeds_dir)` → `(dataset)`; `evaluate_walk`'s `judge_in_fit`, `judge_at_scale`, `judge_param` and `delay` → one `conditions` object (`in_fit`, `scale`, `param`, `delay`), which refuses an unknown key.
- Tool input schemas drop pydantic's titles and null wrappers; what a call may pass is unchanged. The surface an agent reads at session start went from 9,161 to 6,175 tokens (cl100k). `tools/api-snapshot.py` checks the API against `trainnr/tests/api/tools.json`.

### Changed

- NOTICE names the copyright holder, Prakhar Aggarwal.

- Projects live in `~/trainnr/projects` (`TRAINNR_PROJECTS`, `TRAINNR_HOME`); `create_project` makes the new project current and the new `use_project` tool chooses one; user caches move to `~/trainnr/cache` (existing checkout caches are kept).
- Tool descriptions are in plain words, every tool is annotated read-only or destructive, and the server reports its version.
- Actuator-bundle readers accept `trainnr-actuator-bundle/1` as well as the frozen `robotiq-actuator-bundle/1`.
- The `remote` and `dds` extras are dependency groups (`uv sync --group remote`), so the published metadata carries no git URL.

- The project is now **trainnr** (packages `trainnr` and `trainnr-mjlab`,
  the Studio `trainnr-studio` (crate, binary, `trainnr studio`, app id
  `ai.trainnr.studio`; the app's persisted window state resets once),
  the `trainnr` command, environment
  variables `TRAINNR_*`, task ids `trainnr/<task>`). The old import names
  and the old task namespace are gone; nothing outside this repository
  ever used them.

### Removed

- The 2025–26 rig (14 Rust crates, the Pico firmware, their tools and
  gates) moved to its own archive repository, `Trainnr-AI/rig`, with its
  history. `recordings/` and the drivetrain bundle stay as evidence.

### Fixed

- A job started after `create_project` is found by `describe_job` and `list_jobs`: the server's job table was fixed at start, before a project existed, so the readers answered "no job …; known: []".
- Tests that render skip, with the reason, where no OpenGL context opens (CI's macOS virtual machines); the capture-tool test no longer assumes the host has apt.

- Every tool's failure reaches the agent with its reason; `trainnr mcp` without the extras names the command that installs them; the Studio download error names the build-from-source line.
- `wsl.env` applies on WSL only and never overrides a variable the user set.
- The Robotiq 2F-85 bundle carries both licences (Robotiq BSD-3, NVIDIA CC-BY) in `LICENSES/`; the USD importer records every licence up to the repository root.
- NOTICE corrected (ALOHA 2 to Trossen Robotics, DFKI CC-BY attribution, the Mip-NeRF 360 licence state, the Studio's linked crates and fonts), one copyright line in both NOTICE files.

- From a fresh clone of the public edition, walked end to end (2026-10-03):
  `evaluate_walk` hands its job the resolved checkpoint path (a bare
  `run/model_N.pt` died in the judge); the artifact drawer's header
  picture (a red triangle after the picture decoder moved off the UI
  thread); the README's tool names and the address of Unitree's `go2.xml`.

### Added

- `tools/agent-e2e.py`: a real agent, given only the trainnr server and a plain-language task, does the laptop half of the Quickstart; a release check.
- `AGENTS.md` (read by Codex, Cursor and other coding agents) and a `CLAUDE.md` that imports it: what the repository is, how to set up, the gates to run before calling a change done, and the rules (sign-off, docs with code, numbers from records, the tool API, paths, words, never a `.env`).

- One version everywhere (`tools/release.py`, checked in CI); a supply-chain
  job (pip-audit on every locked Python set, cargo-deny on the Studio,
  zizmor on the workflows); a pre-commit config; a threat model in
  SECURITY.md and the public API's definition in GOVERNANCE.md.

- The Studio, prebuilt: a release workflow builds it for Linux (x86_64),
  macOS (Apple Silicon) and Windows (x86_64); `launch_studio` and
  `trainnr studio` download the build matching the package version,
  verified by its SHA-256, into the user's cache; the plugin's session hook
  fetches it in the background after install; `trainnr studio --install`
  fetches it ahead.

- A README with a capture of every stage in the Studio, a real quickstart
  session, the robots the loop has run and the headline results with their
  records; the app icon reads "tr".

- Contributing on GitHub: issue forms (bug, feature, robot support) with
  contact links to Discussions and private security reports, a pull request
  template, a sign-off check on every pull request, weekly Dependabot
  updates, line coverage in CI with a README badge, OpenSSF Scorecard,
  every CI action pinned to a commit, and `tools/github-setup.sh` for the
  repository's settings (a ruleset on `main` once public).

- A light theme for the Studio, designed (a white page, warm paper
  panels, soft blue tints): a switch in the title bar (dark, light or the
  system's), kept across launches; the embedded viewer, the pages and the
  card pictures follow; `set_studio_theme` sets it from an agent.
- The project switcher sits at the head of the sidebar; the Simulator's
  empty state is one card with the scene picker.
- The app's heartbeat reports frame time, frame rate and repaint causes.
- The chrome after Zed's: a title bar with the brand and a project › page
  crumb; a status bar along the bottom with what runs, the presenter's
  state, the viewer's panel toggles, the theme switch and the frame time.
- Card pictures are decoded off the UI thread at the card's size; the
  window no longer polls for agent commands (a watcher thread wakes it).
  Under WSLg the app now takes the Wayland window path (`TRAINNR_X11=1`
  for the old one), which presents a frame ten times faster there.
- `trainnr.mcp_tools`: an entry-point group through which an installed
  package registers tools on the same MCP server.
- `trainnr mcp` serves the MCP server over stdio; `trainnr studio`
  launches the app; `trainnr version`.
- The repository installs as a Claude Code plugin and marketplace
  (`.claude-plugin/`), with the MCP server, the skills and the agents.
- `tools/check-layers.py` pins the package layers.
- Governance files: `CONTRIBUTING.md` (DCO), `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`, `CITATION.cff`.

### Security

- `cancel_job` refuses invalid IDs, finished jobs and reused process IDs; a failing tool plugin is skipped instead of stopping the server.
- The Studio installer locks per version, never removes a complete install, and keeps the GitHub token off other hosts.
- Public-log downloads show the licence first and refuse unlicensed data without `--accept-unlicensed` / `accept_unlicensed`.

- The Studio's viewer server listens on this machine only (127.0.0.1:9876);
  it listened on every interface, so anyone on the same network could
  stream into an open Studio. `TRAINNR_VIEWER_BIND` opens it on purpose.
- urllib3 2.8.0, pyjwt 2.15.1, accelerate 1.15.0 and rustls 0.23.45, past
  their published advisories; checkouts no longer keep the token, and the
  release build restores no cache.

