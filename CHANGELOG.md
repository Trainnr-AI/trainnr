# Changelog

All notable changes to trainnr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Not yet released; the repository is being prepared for its first public release. trainnr is licensed under the Functional Source License, Version 1.1, ALv2 Future License (FSL-1.1-ALv2, `LICENSE`): any use other than a competing product or service; each version is also licensed under Apache-2.0 two years after it is made available.

### Changed (breaking)

- `train_walk(recipe=…)` replaces `agent=`: `smoke` checks the stack in minutes and saves nothing, `full` trains and saves the experiment. A count below 1, an unknown recipe, a name with the smoke, and an existing experiment name are refused before a job starts; the reply names the experiment and the next call.
- `list_experiments` returns one short row per experiment (status, iterations, final reward, robot, task, fit, checkpoints), keyed `experiment`; the new `describe_experiment` returns one in full. One call returned 3.7 MB on a project of 13 runs.
- Tools read and write only inside the open project: outputs and datasets (`generate_*`, `multiply_demos`, `describe_dataset`) resolve inside it (a bare name lands in `runs/`), checkpoints for `evaluate_walk`, `export_deployment`, `play_walk` and `generate_walk_demos` must be the project's, and a call with no project is refused. `generate_walk_demos` drops `project`; `run_chain` writes into the project.
- `create_project(name, path=None)`: the path defaults to the name, and a relative path stays in the projects home. `describe_robot` takes a robot's name, never a path. `capture_scene` no longer takes `brush` (Brush is found on PATH or at `$TRAINNR_BRUSH`).
- An evaluation has one name: `list_evaluations` gives each row's `evaluation` (a project evaluation's stamp, `name@hash`; episode records outside a project, their run folder), and that value is what `describe_evaluation(evaluation)` and `export_deployment(evaluation=…)` take. `describe_evaluation` used to take a run folder while export took a stamp. Rows carry `evaluation` instead of `run`; a project's raw `verdict/` files are no longer listed, its evaluations are. `evaluate_walk`'s reply says where the stamp appears.
- MCP tools follow one naming rule: `verb_noun`, `list_` for a collection, `describe_` for one item, an existing artifact named by its kind and `name` only for a new one. There are no aliases; an agent prompt or note using an old name must be updated.
  - Renamed: `describe_bundles` → `list_robots`, `describe_bundle(name)` → `describe_robot(robot)`, `describe_runs` → `list_experiments`, `describe_tasks` → `list_tasks` (rows carry `task`), `describe_task_families` → `list_task_families`, `describe_engines` → `list_engines`, `describe_task(task_id)` → `describe_task(task)`, `list_eval_records` → `list_evaluations`, `describe_eval(run)` → `describe_evaluation(evaluation)`, `list_ledger_findings` → `list_findings`, `friction_curve(slug, …)` → `describe_friction(actuator, …)`, `job_status` → `describe_job`, `capture_status` → `describe_capture`, `accept_task(name)` → `check_task(task)`, `describe_viewer_recording` → `describe_viewer_stream`, `simulate_in_studio(task)` → `run_simulation(scene)`, `describe_datasheet(demos_dir)` → `describe_dataset(dataset)`.
  - Merged: `describe_actuators` and `describe_actuator_bundles` → `list_actuators` (one row per actuator, its certified bundles inside); `describe_actuator` and `describe_actuator_bundle` → `describe_actuator(actuator, tier)` (the model and that tier's bundle); `focus_studio_recording(recording)` → `show_in_studio(stream=…)`, which takes exactly one of `artifact` or `stream`.
  - Arguments: `export_deployment(run, …, certificate)` → `(experiment, …, evaluation)`; `gate_deployment(name)` → `(deployment)`; `create_task(task_id, name, overlay)` → `(family, name, settings)`; `play_walk(run)` → `(experiment)`; `ingest_public_log(name, recording_name)` → `(log, name)`; `import_finding(record)` → `(finding)`; `control_simulator(run)` → `(play)`; `multiply_demos(seeds_dir)` → `(dataset)`; `evaluate_walk`'s `judge_in_fit`, `judge_at_scale`, `judge_param` and `delay` → one `conditions` object (`in_fit`, `scale`, `param`, `delay`), which refuses an unknown key.
- Every MCP tool refuses an argument it does not take, naming it; before, an old name such as `overlay` or `judge_in_fit` was ignored and the call ran with its defaults. Schemas say `additionalProperties: false`.
- Check-task jobs are named `check-task-…` (were `accept-task-…`).
- Tool input schemas drop pydantic's titles and null wrappers; what a call may pass is unchanged. The surface an agent reads at session start went from 9,161 to 6,175 tokens (cl100k). `tools/api-snapshot.py` checks the API against `trainnr/tests/api/tools.json`.

### Added

- `describe_experiment(experiment)`: one experiment's training record, identity, manifests, checkpoints and the evaluations citing it.
- `describe_task` describes a task declared in the project (family, version, settings, the check's verdict, reward previews); a registered task id still builds for real.
- One-line descriptions on the arguments agents misread (`recipe`, `fit`, `task`, `checkpoint`, `evaluation`, `unevaluated`, `student`, `device`, `command`, `follow`, the USD options).
- `describe_job` reports a failed job's last exception line as `error`.
- `tools/third-party-notices.py`, `tools/check-package.py`, a Python licence allow-list (`tools/supply-chain.py --licences`), a Conventional Commits title check on pull requests, the `DCO` file, and a release procedure in GOVERNANCE.md.
- `describe_job(job_id, wait_s)` waits up to 300 s for a running job to end, so an agent waits without a shell loop.
- `describe_identification` gives each fit record's own stamp (`fit`), the value `train_walk` takes; `train_walk` refuses an unknown fit before starting a job.
- A job asked about from another project names the project that holds it.
- `launch_studio` checks the viewer port at the address the Studio binds and, on Linux, past connections still closing; a training job retrying its stream made it refuse an empty port.
- `evaluate_walk(checkpoint)` takes an experiment's name and judges its newest checkpoint; a file inside it still names that file.
- A checkpoint may be named without its `.pt` suffix (`model_149`) in `evaluate_walk` and `export_deployment`; export's refusal lists the run's checkpoints.
- `export_deployment` refuses an `evaluation` the index does not hold and lists the checkpoint's own; before, any text was written into the manifest and the gate judged nothing.
- `train_walk` with a project open saves an unnamed g3 run in the project's `runs/` (`<robot>-walk-<date-time>`); it used to land in the checkout's `runs/`, outside every project.
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

### Changed

- NOTICE names the copyright holder, Trainnr AI.
- The microduck's 38 meshes are no longer in the repository: Pollen Robotics licenses its 3D model files Creative Commons BY-SA-NC, which is not an Apache-2.0 grant. `robots/microduck/FETCH.json` pins each one by its git blob id, and trainnr fetches them from `pollen-robotics/microduck_rl` the first time the robot is used (about 22 MB; `TRAINNR_NO_ROBOT_FETCH=1` refuses, `python -m trainnr.bundles.fetch robots/microduck` fetches ahead). The bundle's stamp is unchanged once they are in place.
- Conduct reports and a security fallback go to trainnrai@gmail.com.
- The plugin, marketplace, registry entry and citation point at this repository until trainnr.ai serves a page; GOVERNANCE.md's network list defers to SECURITY.md's, which is complete.
- The 2025–26 rig is named as a private archive; the documentation no longer links to its repository, which is not public. NOTICE credits Pollen Robotics for the microduck in the README's robots image, under its Creative Commons BY-SA-NC terms.
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

- The draft paper (`docs/paper/`, its figures and its build tools) moved out of this repository; it will be published with its own. The README says the research is ongoing; `tools/check-numbers.py` now traces every success figure the README quotes, and the GPU quickstart's 150-iteration result has its finding record.
- The 2025–26 rig (14 Rust crates, the Pico firmware, their tools and
  gates) moved to a private archive with their history. `recordings/` and
  the drivetrain bundle stay as evidence.

### Fixed

- The microduck's meshes are fetched only by what uses the robot (training, identifying, onboarding, showing it); listing or describing it, a preview or a detail page never downloads. Its stamp is `microduck@ad90736153cc` with or without the meshes (`FETCH.json` records it, with a hash of the files the bundle carries). Two processes fetching at once no longer break each other. Tests that need the meshes fetch them, or skip by name when `TRAINNR_NO_ROBOT_FETCH=1` forbids it.
- `list_experiments` caches a finished run's record read afresh (25 s for 14 runs no tool had refreshed); a job a tool started records its process's start time, so `cancel_job` and the Studio's Stop can stop it.
- `launch_studio` names the process holding the viewer's port, and the port check reads `TRAINNR_VIEWER_BIND`'s port and IPv6 hosts (`[::1]:9876`).
- `tools/github-setup.sh` applies every setting it can and names the ones refused; the read-back decides the exit code.
- NOTICE names BAM's actuator model (ported), Unitree's and DFKI's message layouts, Rerun's example and allocator setup, the mimalloc C library, the Robotiq renders, the gym-aloha constants and the microduck's PPO recipe; both packages ship the full NOTICE; scene records name trainnr's own licence.
- From a fresh clone, walked end to end by an agent (2026-10-05):
  `list_experiments` and `describe_experiment` read a just-finished run's
  record afresh from the trainer's files (it was listed with no status,
  iterations or reward until something refreshed the project); an
  evaluation's trials list the milestones each reached (a failed trial
  read "survived, tracked"); the Studio launches on a WSL boot whose
  runtime directory lacks the Wayland socket, using WSLg's own.
- `tools/github-setup.sh` applies the rulesets even when a public-only
  setting is refused; a release candidate's notes are its version's
  CHANGELOG section; the local gates run the Python suite with CI's
  extras.
- Each MCP server process keeps its own current project; two agent sessions sharing `.current` moved each other's training and exports.
- A fresh plugin install reads its trained runs without TensorBoard (the console log is the record) and builds the trainer's environment with the recorder (`viz`); both failed for a stranger on 2026-10-04.
- `export_deployment` cites by default the newest evaluation in the policy's own world; a newer run under a delay, a scaled gain or another fit is a stress result, never the certificate.
- `evaluate_walk` reads the walk from the experiment's identity, as export and play do.
- `check_drift` refuses the recording a reference fit came from (it can only say "within") and an `against` that names nothing in the project.
- The Studio: a deployment card leads with its evaluation's success rate; the Overview no longer says "All stages complete" as if the policy were good; the checkout's empty sample project is listed only when there is no other.
- Tool descriptions say what is true: `run_simulation`'s scene forms, the Simulator page, what `assay_deployment` stages, that `launch_studio` may download; `stage_deployment` is not marked destructive.
- The versioned pre-commit hook passes on a clean tree; walk scripts `cd` into `trainnr-mjlab`.
- A job started after `create_project` is found by `describe_job` and `list_jobs`: the server's job table was fixed at start, before a project existed, so the readers answered "no job …; known: []".
- Tests that render skip, with the reason, where no OpenGL context opens (CI's macOS virtual machines); the capture-tool test no longer assumes the host has apt.
- Every tool's failure reaches the agent with its reason; `trainnr mcp` without the extras names the command that installs them; the Studio download error names the build-from-source line.
- `wsl.env` applies on WSL only and never overrides a variable the user set.
- The Robotiq 2F-85 bundle carries both licences (Robotiq BSD-3, NVIDIA CC-BY) in `LICENSES/`; the USD importer records every licence up to the repository root.
- NOTICE corrected (ALOHA 2 to Trossen Robotics, DFKI CC-BY attribution, the Mip-NeRF 360 licence state, the Studio's linked crates and fonts), one copyright line in both NOTICE files.
- From a fresh clone, walked end to end (2026-10-03):
  `evaluate_walk` hands its job the resolved checkpoint path (a bare
  `run/model_N.pt` died in the judge); the artifact drawer's header
  picture (a red triangle after the picture decoder moved off the UI
  thread); the README's tool names and the address of Unitree's `go2.xml`.

### Security

- A project from someone else can no longer make trainnr write, append or delete outside it through a link it carries: every write the project layer, the job runner and the bundles make goes through `trainnr.safe_write`, which checks the target and the folders below the project's root and stages each file fresh. One `describe_project` call overwrote a file through a planted `.index/project.json.tmp`, and pruning a linked `.index/commands` deleted files outside the project.
- The Studio applies a command only when it carries the running Studio's session token (`studio-state.json`); a command file that came with a project, whatever its time stamp, is never applied. Its log panel reads the log beside a job's record, never the path the record names; it opens a record's viewer file only inside the project; its Stop button signals a pid only when that process started when the record says (both ways, as `cancel_job` does); and its simulator's frame file has a random name, readable by the user alone.
- `evaluate_walk(student=…)` takes a folder inside the project or a Hugging Face repo id, and refuses a student whose processor files name a step that is not LeRobot's own: LeRobot imports the classes they name, so a crafted checkpoint ran code when it loaded.
- A robot bundle's `FETCH.json` names a repository and a commit by their shapes, and its files land only inside the bundle (a `../` path and a link out of the bundle were written through); an empty raw answer no longer skips the blob check.
- The plugin's session hook runs from the plugin's own folder with `python -P`, so a json.py or a uv.toml in the folder Claude Code opened is never picked up; the Studio installer refuses a redirect away from HTTPS; a public log's zip member never inflates past its recorded size; a deployment's and an actuator's names are plain words; a push to a rented machine leaves out credentials under any common name, not only `.env`.
- A project from someone else cannot steer the Studio into writing outside it: the Studio writes through no link a project carries, applies no command file older than its session, accepts only plain command ids, and its Stop button refuses pid 1 and a reused pid. `quit_studio` signals only a Studio process and trusts no heartbeat from the future; job status reads the log beside the record; a deployment manifest's program names must be plain names.
- Checkpoints are read in PyTorch's weights-only mode before rsl_rl loads them, so a `.pt` that carries code is refused.
- `play_walk`'s browser viewer binds 127.0.0.1; `start_capture` listens at most an hour; a release Studio no longer trusts the checkout path baked in at build time; the Studio warns when `TRAINNR_VIEWER_BIND` opens the viewer beyond this machine.
- Onboarding refuses a model folder that links outside itself or exceeds 2 GB; names refuse drive marks, unprintable characters and Windows device names.
- SECURITY.md lists every network call, listener and file outside a project, and what a shared project can steer.
- Releases: the publish job never replaces a published release's files, archives carry build provenance attestations, the tag must match the version, and archives are root-owned. The Studio's third-party notices reproduce licence texts verbatim and carry the NOTICE files and font and C-library notices they lacked. `tools/github-setup.sh` pins required checks to GitHub Actions, requires pull requests even of the maintainer, turns on immutable releases and reads every setting back.
- `cancel_job` refuses invalid IDs, finished jobs and reused process IDs; a failing tool plugin is skipped instead of stopping the server.
- The Studio installer locks per version, never removes a complete install, and keeps the GitHub token off other hosts.
- Public-log downloads show the licence first and refuse unlicensed data without `--accept-unlicensed` / `accept_unlicensed`.
- The Studio's viewer server listens on this machine only (127.0.0.1:9876);
  it listened on every interface, so anyone on the same network could
  stream into an open Studio. `TRAINNR_VIEWER_BIND` opens it on purpose.
- urllib3 2.8.0, pyjwt 2.15.1, accelerate 1.15.0 and rustls 0.23.45, past
  their published advisories; checkouts no longer keep the token, and the
  release build restores no cache.
