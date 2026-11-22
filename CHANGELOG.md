# Changelog

All notable changes to trainnr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `play_walk` films a walk's checkpoints without a window (`viewer="film"`, #137): one checkpoint, several comma-separated, or `all`, each walking one world from the same seed under one held command for 10 s or until it falls, drawn from the walk's chase camera at the control rate. Each lands beside the run as `play-model_N/play.mp4` with `play.json` (the command, the seconds walked, the fall, the distance covered), and several are joined in iteration order as `film.mp4` with the iteration on the picture. The run's page in the Studio plays them: each checkpoint's film at its iteration on the reward curve's timeline, the whole run on a `video` timeline. The stills now start every checkpoint from the same reset too.
- A robot onboarded again under its own name becomes a new revision instead of being refused: `onboard_robot` keeps the bundle it replaces (under `robots/.revisions/`, out of the lists), lists what changed in model terms (joint ranges with degrees, damping, armature, friction loss, masses, actuator gains and ranges, keyframes), and the same content again returns the same stamp. The robot's page and `describe_robot` show its revisions, and an artifact built on an earlier revision is marked as built on a superseded robot (#119).
- A robot's page shows its keyframes (each joint's value, flagged when outside its range) and its sites, with hinge ranges in degrees beside radians; its preview and its 3D view stand in its `home` or `stand` keyframe, and the 3D view steps through every keyframe on a `keyframe` timeline (#122).
- Limits over several joints, declared with the robot (#121): a bundle's `limits.json` (`trainnr-joint-limits/1`) states poses its parts cannot take although every joint is in range, in three general forms (rules, step tables, grids with the stricter neighbour between grid values; docs/88 §3). The robot's page lists them and the keyframes that break them; a legged walk ends an episode on a broken limit (or pays for it, or ignores them: its `limits` setting) and refuses a stand pose that breaks one; the evaluation counts each trial's steps past a limit; the deployment carries the file (`limits` in `deploy.json`) and the gate fails a trial that breaks one, naming it. Microduck Nano's nine firmware limits, written in these forms, agree with its firmware on 200,000 random poses (the record `nano-limits-match-firmware-2026-11-22`).
- The `servo-bench` identification method (#101): a servo's actuator model from a bench recording of its commanded and measured position on a pendulum arm of known mass (a one-hinge bench model and its `bench.json`). It fits BAM's first-tier law (stall torque, P gain, back-EMF, Coulomb friction, armature, command delay) with intervals and pinned verdicts, the arm's gravity and inertia as the anchor, and writes an actuator bundle in the library's format with the firmware it measured, which a legged walk names as its actuator. On a synthetic SCS0009 bench it pins five of the six within 7 percent of the truth and says the armature cannot be told from this data (the record `servo-bench-synthetic-scs0009-2026-11-22`).
- A walk for any onboarded legged robot (#100): the task family `legged-walk` takes the robot (a bundle onboarded into the project) and its actuator (a library or fitted actuator bundle, or `model` for the model's own actuators, recorded as declared), and reads the rest off the model: the trunk, the feet, the stand keyframe, every joint the policy drives. Settings declare what the model does not say (feet with other names, joints to hold such as a head, the reset height, the command ranges); anything missing is refused by name, as is a stand pose outside its joints' ranges. `check_task`, `train_walk`, `evaluate_walk`, `export_deployment` and `gate_deployment` work on it like the built-in walks; a scene's own floor is left to the simulator's terrain, and a deployment whose policy drives only some joints holds the rest at the stand pose (`action_joints` in `deploy.json`). microduck's walk is now this recipe with its own values.
- `evaluate_walk` on a tracking experiment judges its checkpoint under its tracking task: paired trials through the reference from its first frame, with training's observation noise and start draw, pushes and a per-trial command lag as the task declares, and an exact interval; world 0 streams live into the Studio and the result is an evaluation like a walk's. The court dance's policy succeeded in 18 of 40 trials with pushes and up to 20 ms of lag: 17 of 19 with no lag, 1 of 21 with one 20 ms tick (the record `g1-court-dance-evaluated-2026-11-17`). The Humanoid motion from video template's evaluation step is done by one.
- The tracking task family `g1-tracking`: `create_task` declares the G1 tracking a reference made from video, with its success rule (the clip played to its end with no termination) and its evaluation's conditions (paired trials, pushes, a command lag). `check_task` accepts it when the reference was made from video and two training iterations run on it; `train_tracking` takes the task and its run cites it. The Humanoid motion from video template's "Make the tracking task" step is done by an accepted one.
- `play_walk` on a tracking experiment plays its checkpoint once through the reference in physics, live in the Studio, and films it beside the original video with its falls and tracking errors (`play.json`). The court dance, trained at 16,384 worlds for 2,000 iterations (about 2 h), dances the whole 21.9 s clip without falling (the record `g1-court-dance-tracked-2026-11-17`; one rollout, no interval yet).
- Video to a humanoid, up to training (docs/86 phase 3): `retarget_video` turns footage of a person into a Unitree G1 reference motion as one job: NVIDIA's GEM-X estimates the motion, the SOMA Retargeter maps it onto the G1, its tilted floor is levelled and its jitter smoothed, and the result is a recording holding the G1's joints and root, the motion mjlab's tracking task reads, and the motion replayed beside the video. `train_tracking` trains the G1 to track it in mjlab's tracking task. Both are jobs the Studio shows live: the video job streams its stages, the video, the G1 reference in 3D and every joint raw and smoothed; the training streams like a walk's. `tools/video-setup.sh` installs GEM-X and the retargeter at the measured commits, each in its own environment. A 22 s dance took about 7 minutes on an RTX 3090 Ti (the record `video-to-g1-on-a-24gb-card-2026-11-15`). The Humanoid motion from video template's steps run them.
- Running now has **watch live** for a running job that streams (training, gates, the video job): the viewer turns to its plots and picture as they happen. Each job records its stream.
- Project templates: New project can start from a plan instead of a blank project. Three come first, on NVIDIA's stack: Humanoid motion from video (GEM-X, the SOMA Retargeter and mjlab's tracking task, on a Unitree G1), Manipulation from human video and More from your own demonstrations (GR00T N1.7 and Cosmos Transfer, on an SO-101). A project made from one shows its plan on the Overview, each step marked done, with its button, or planned. `create_project` takes `template`, and `list_templates` lists them. The design is docs/86.
- GR00T N1.7 fine-tunes on one 24 GB card (an RTX 3090 Ti): `tools/gr00t-batch-search.sh` sets up NVIDIA's Isaac-GR00T in its own environment and finds the largest batch each configuration fits, and `tools/gr00t-fit.py` adds three memory savers NVIDIA's launcher does not expose. NVIDIA's settings fit only with the action head frozen, at batch 8; with the savers, the full default fine-tune fits at batch 58 (the record `gr00t-n17-on-a-24gb-card-2026-11-02`).
- Footage: `ingest_video` brings a video into the project as it is, with its duration, frame rate, size, codec and first frame, stamped. It shows on the Recordings page with its first frame, and Show in viewer plays it on its own timeline as frames ffmpeg decodes (15 fps, at most 640 px wide), so it plays whatever the codec and whatever ffmpeg is installed. The first step from video to robot training data (docs/86).
- docs/87 writes down the deployment manifest, `deploy.json` (`trainnr-deploy/1`), for runtimes outside trainnr: one control tick, every key, and each observation source's definition, frame and units, including that a term is scaled before it is clipped and that `last_action` is the clipped, unscaled action (#106).

### Changed

- `train_tracking` draws a command lag of 0 to 40 ms in training by default (two 20 ms control ticks, redrawn each second per world), recorded in the run's identity. mjlab's tracking task draws none, and the court dance's policy trained without one fell in 20 of 21 evaluation trials one tick late (the record `g1-court-dance-evaluated-2026-11-17`); every robot is late, so every user's first run now trains for it. Retrained with it, the same policy succeeded in 30 of 40 trials of the same evaluation instead of 18 of 40, 15 of 21 one tick late instead of 1 of 21; training runs about 1.6 times slower per sample (the record `g1-court-dance-lag-trained-2026-11-17`).
- The gate passes a deployment only when its cited evaluation demonstrated a success (a lower bound above zero): a policy that fell in 0/4 and falls again agreed with its evaluation and read `passed`. The record says `agrees` and `demonstrated` apart, and the card says "failed: nothing demonstrated" (#108).
- A deployment of a robot on position servos (BAM actuators, like the microduck's XL330s) exports no PD gains for their joints: `stiffness`, `damping` and `effort_limit` are null there, and `joints.servos` names the actuator bundle and the supply voltage and position gain its law assumes, which the robot's servos must be set to. Pre-flight reports them instead of comparing a simulated motor's gains (#107).
- A training run started by the agent no longer takes over the Studio's view: its stream (now `trainnr-train-<agent>`) joins Sources, and a note names it (#105).
- An agent session keeps its current project when another session creates or chooses one; the last choice only seeds new sessions. A Studio open on another project is found from any project: `describe_studio` names it, `launch_studio` and `open_in_studio(project=...)` move it, and `open_in_studio` takes a bare project name as `use_project` does (#103, #104).
- The Go2 walk sample is Apache-2.0, as trainnr is: its NOTICE said FSL-1.1-ALv2. The Hugging Face dataset's card and LICENSE say Apache-2.0 too. The sample's files are the same; its prebuilt index uses today's wording, and the deployment's card picture no longer has a black band.

### Fixed

- `identify_system` with `legged-joints` fits a recording that carries a joint the robot's model leaves out (a beak, a gripper, a fan), instead of refusing it whole: it fits the joints the model has a hinge for and names the ones left out in the fit record; `check_drift` reads it the same way. It refuses only a recording none of whose joints is a hinge of the model, listing the model's hinges (#133).
- **watch in simulator** on a legged walk (`legged-walk`, #100) failed with "the legged walk builds from a declared task's spec": the Simulator's walk view now takes the robot, its actuator and its stand pose from the run's identity, as the evaluation and the export do, draws the run's own robot (a scene file's floor left to the view's one ground), and so do the experiment card's stills.
- `onboard_robot` copies only the files an MJCF model reaches (its includes and the meshes and textures they name) and a licence or readme beside it, not the whole folder; the reply lists what was copied and how much was left beside, and the stamp covers exactly the copy, so an unrelated file in the folder no longer changes a robot's stamp (#120). A robot onboarded again from the same folder can get a new stamp for this reason (the README's Go2 is now `go2@9fd00d4b4ee1`); bundles already in a project keep theirs.
- `launch_studio` no longer refuses when the viewer's port is held by a Studio on another trainnr home (another session's window): the new Studio takes a free port, recorded in the project (`.index/viewer.json`), and the feeds and jobs of a session on that project stream there. A port held by anything other than a Studio is still refused, a `TRAINNR_VIEWER_BIND` the operator set is never replaced, and within one home the open Studio is still moved to the project (#124).
- `describe_robot` returns what the robot's page in the Studio shows (the census, joints, keyframes, actuators, bodies, sensors, sites, the importer's audit, the files) and says when there are no fit records, instead of only the stamp and the file list (#123).
- An agent connected from the Studio's Connect panel no longer fails to start when its client runs in the user's home folder (where a Windows terminal opens): the entry ran `python -m trainnr.cli`, and the home's `trainnr` folder, the default trainnr home, was imported in place of the package. On Windows the entry now runs the engine's `trainnr.exe`, and the panel offers Update for an entry made before; the macOS and Linux launcher passes `-P` and is rewritten when the panel opens.
- The Studio's Running now panel no longer cuts off a run's buttons under its progress bar, and its log is a framed panel that shows a line repeated in the tail once with its count. A tracking run no longer offers a Simulator scene it does not have.
- A template step is done when its own artifact exists, not any result of its stage: "Turn a video into motion" needs a recording made from video, and "Train it to track the motion" a finished tracking run. A step with a run still training reads running with its progress, and a done step names what proved it.
- `train_tracking` stops MuJoCo-Warp printing its solver and line-search limit warnings from the GPU: with 4,096 G1 worlds they wrote 9.3 GB of job log in 400 iterations. The solver's limits are mjlab's as before.
- The ROS 2 readers (MCAP and rosbag2) leave out an `Imu` field its publisher marks as not provided (covariance[0] = -1, REP-145), and say so in the recording's notes: a robot without an accelerometer got a channel of zeros in m/s^2 (#102).
- An agent connected from the Studio's Connect panel opens the installed app with `launch_studio` instead of downloading a second copy of it: the entry names the app that wrote it (`TRAINNR_STUDIO`; on Linux the AppImage). On macOS and Linux the launcher carries it and is rewritten when the panel opens, so an entry made before gets it too; on Windows the panel's Add or Update writes it.

### Security

- The Studio's lock takes rustix 1.1.6 (RUSTSEC-2026-0486) and async-executor 1.14.1 (1.14.0 was yanked); both Python locks take opentelemetry-api 1.45.2 (CVE-2026-126360). pillow 12.3.0 (CVE-2026-128066, CVE-2026-128067) stays a written exception: LeRobot 0.6.1 requires Pillow<13, and trainnr writes no TIFF and opens only the user's own images.
- The Studio's lock takes pulldown-cmark 0.13.6 (RUSTSEC-2026-0433). thrift 0.17 (RUSTSEC-2026-0435, fixed in 0.23) stays where Rerun 0.36's parquet 58 pins it, a written exception in the Studio's `deny.toml`: parquet reads its metadata with its own reader, not the affected one.

## [0.2.0] - 2026-10-30

### Added

- `describe_studio` says what a frame costs to draw: `frame_cost_ms` and `frame_cost_max_ms`, the mean and the worst over the last second, counting the pages and the rendering but not the wait for vsync (eframe's `cpu_usage`). `frame_ms` stays the time from one frame to the next.
- `trainnr call <tool> --args '<json>'` runs one tool by name, as an agent would: the same registry, argument checks, refusals and plugin tools. It prints the result as JSON and exits 3 on a refusal. The Studio's buttons run it.
- After a design review: secondary text reads at 4.5:1 or better in both themes (the light theme's muted grey was never applied); evaluation cards are titled by their condition and no longer show a check mark; the view switch is no longer covered by the page's buttons; an empty project's Start here has a button for each step; actions stay off until their input exists and say what is missing; deployment steps read in the workflow's order.
- After the design review's second round: a project's Overview leads with the best policy per condition; form problems show under their field and a click outside no longer discards a form; downloads show progress and can be cancelled; the command palette has a Search button; painted chips and step numbers are labelled for screen readers; plurals, sentence case and the activity feed are cleaned up; a deployment's card no longer has a black band; the Simulator offers the project's deployments to play.
- The README opens with the app: download it, open the sample, make a project, connect your coding agent; the plugin's path follows.
- The release carries the app to double-click: a macOS disk image (drag trainnr onto Applications) and a Linux AppImage (`tools/package-studio.py`), unsigned for now. A click on the Studio's version in the status bar looks for a newer release and links its page; it never looks on its own.
- Windows gets an installer: `trainnr-*-windows-x64-setup.exe` installs for the user alone, with no administrator prompt, a Start-menu entry and an uninstaller. The Windows exe no longer needs the Visual C++ Redistributable: its C runtime is linked in.
- The macOS app opens from a download again: v0.2.0-rc.1's bundle was unsealed (the binary carried only the linker's signature), and macOS called it damaged. Without a Developer ID the app is now sealed ad hoc, so macOS asks to open it from Privacy & Security instead.
- `trainnr version` and the light commands work on a plain `pip install trainnr` again: 0.2.0-rc.1's command line imported the job layer, and through it MuJoCo, when it started.
- Connect your agent: a panel from the Welcome page and the status bar adds trainnr to Claude Code, Codex or Cursor with one button each (nothing is written at launch), shows which already have it, and gives the line for any other client. The entry runs a launcher at a fixed path (`<trainnr home>/engine/bin/trainnr`), so an update does not break it.
- The Studio runs without a checkout: the first action that runs a tool sets up its engine (uv when none is on PATH, at a pinned version and SHA-256; the release's runtime archive, `trainnr-runtime-v*.tar.gz`, 9 MB; the core packages), showing each step on the button. The sample opens with no engine at all: the Studio downloads, checks and unpacks it itself. The release builds the runtime archive (`tools/runtime-archive.py`); jobs run the engine's uv (`TRAINNR_UV`).
- The Studio's pages carry their actions as buttons: Add robot, Identify dynamics, Add recording, Check environment, Train, Evaluate, Export and Check for drift. Each opens a form drawn from the tool's own input schema, with pickers over the project's artifacts and native file dialogs, and runs the tool on the window's project through `trainnr call`. `trainnr forms` prints the forms (`trainnr/forms.py`).
- The Studio makes a project without an agent: New project on the Welcome page and the Projects page asks for a name and runs `create_project`. The Welcome page now reads "Work here, or let your coding agent drive it" (docs/85, the desktop app plan).
- The Studio says whether an agent is connected. The status bar carries a dot on every page, at every stage of a project, with the call count and the last tool on hover; the Welcome page and an empty project's Start here card carry the fuller line: connected, with the last tool it called, or not connected, with the README's plugin install and a chip that copies each command. `trainnr mcp` records each session in `<user home>/agents/<pid>.json` while it serves and removes its own on the way out, so sessions are counted rather than overwriting each other (the plural says sessions, because one agent window can serve twice); the Studio needs a heartbeat that is fresh and not in the future, and a live pid, so a session killed hard reads as gone. The Studio still never speaks MCP. In the app, the status bar's Connect agent button carries it and opens the Connect panel; no agent is said quietly, not as a warning.
- The Studio opens on a Welcome page before any project exists: start your own project through the agent, open the Go2 sample with one click, and the loop's nine stages; it follows to the first project the agent makes or opens. `launch_studio`, `trainnr studio` and the Studio's doors work before the first project; `trainnr sample list` and `trainnr sample open` do from a shell what the sample tools do. The Projects page lists the samples.
- Sample projects: `list_samples` and `open_sample` open a finished project to explore before your own. The Go2 walk (12 MB) is the whole loop, from identification to a gated deployment and a drift check, downloaded once from the Hugging Face dataset at a pinned commit, size and SHA-256 checked, and unpacked into the projects home. `tools/make-sample.py` builds a sample from a real project, deterministically, refusing any file with a private marker.
- The Python packages are uploaded to PyPI by `.github/workflows/pypi.yml` when a release is published (the release's own attested files, trusted publishing, a maintainer's approval per package); a release candidate goes to TestPyPI. `uvx --from "trainnr[sim,mcp]" trainnr mcp` runs the server with no checkout.

### Changed

- trainnr is licensed under the Apache License 2.0, the whole repository: the Python packages, the Studio, the CLI, the agent plugin, the robot bundles and the docs. Releases up to and including 0.2.0-rc.2 were made available under FSL-1.1-ALv2. Each package's and release's `LICENSE` is the Apache text; the separate `LICENSES/Apache-2.0.txt` copy is gone.
- The simulator's overlays (contacts, forces, joints, actuators and the rest) draw without the floor's reflection unless the Rendering panel asks for it: the mirror pass drew every shape twice. On the kitting scene under WSL, contact forces render in 29 ms a frame instead of 81 and all overlays at 5 frames a second instead of 2.6.
- A running scene's controls under the simulator's picture are whole again: the viewer's hidden top bar was drawn over them, covering the buttons and taking their clicks.
- The Studio's simulator picture is sharp: it renders at the panel's full size, widened to rows of a whole 256 bytes, which WSLg's graphics layer uploads in about a tenth of the time. It used to drop to half size when frames ran slow, measured by the time between repaints, which tracks the stream's own frame rate, and on WSLg it never came back.
- The Studio's card and drawer pictures are sharp: each is decoded to the exact pixel size it is drawn at with a sharp filter, instead of shrunk to fit with a fast one and stretched to fill the card; the drawer no longer enlarges the card's smaller copy.
- The Studio's empty pages say what each page holds and what to ask the agent, in prompts that copy on a click; an empty project's Overview shows a Start here card with the first three steps instead of a row of zeros; the Overview's next step for an empty project is one short line.
- The README says what a Mac trains: the smoke recipe and `check_task` run on the CPU; a full run needs a GPU. Its disk figures are the ones measured on a fresh install (the server's environment about 1 GB, the Studio about 200 MB unpacked, the trainer's environment 1.8 GB on a Mac and 6 GB on Linux with CUDA, built by the first `check_task`), and a new section says how to uninstall or reset and what stays. The Studio's Compute card no longer names `TRAINNR_ENDPOINT`, which no tool reads.
- The first session after the install says what to say first (the plugin's session hook, once); `describe_studio` before any project answers with the next step instead of failing; `create_project`'s reply names `launch_studio` as the next step; `onboard_robot` refusing a missing file inside Unitree's repository gives the `git clone` that fetches it.
- The Studio opens dark on a first launch (`set_studio_theme` and the title-bar switch still choose, and the choice is kept); the README's screenshots follow the reader's GitHub theme, dark and light.
- The README's Studio section describes the prebuilt download (the plugin's first session, `launch_studio` on demand, `trainnr studio --install`) now that v0.1.0 is published; building from source is CONTRIBUTING's.

### Fixed

- The downloaded Studio keeps its licences: `trainnr studio --install` and the plugin kept only the binary from the release archive and dropped its LICENSE, NOTICE and THIRD_PARTY_LICENSES.md, the notices of the third-party crates compiled into it. They now stay beside it; a Studio installed before keeps its folder until the next version.
- The Simulator's walk scene runs its policy again when the trainer's environment was synced without the `viz` extra: the Studio's walk view now asks for it, as the trainer's own runs do, instead of dying on `import rerun`.
- A walk's stamp hashes its robot with its knobs: `go1-walk` and `go2-walk` at their defaults shared one hash. A walk already declared keeps the stamp its `task.json` recorded.
- A plugin install lists no projects from its own copy of the repository: `list_projects` on a fresh install showed `sample` from the plugin's cache folder.
- A job no longer inherits the server's `VIRTUAL_ENV`; every job log began with uv's warning about it.
- The `describe_job` hints wait 60 s, not 300: Claude Code moves a call past 120 s to the background.
- The environment stage is proved by an accepted task, not a declared one: after `create_task` the Studio and the index say "declared, not yet accepted" and name the `check_task` call, until it accepts. `list_robots` says each robot's `scope` (`project` or `library`), and `list_projects` counts the stages the project's loop passes through, as the Studio does.

### Security

- The Studio no longer builds rmcp 1.8 (GHSA-33f5-2c5q-wgwj, GHSA-9g45-5xwm-f3wc, GHSA-9pj6-vhgr-3mwh, GHSA-c9xm-49cp-xcr9): it depends on Rerun's viewer and crash handler directly instead of through Rerun's `native_viewer` feature, which also brought the `rerun` command's MCP server, a part the Studio never ran. Thirteen crates leave the build and none is added.
- The locks take the fixes for every advisory published against them: aiohttp 3.14.5, certifi 2026.10.23, filelock 4.2.4, GitPython 3.2.1, msgpack 1.2.4 and opentelemetry-api 1.45.1 in trainnr's; certifi, filelock, GitPython, opentelemetry-api, Pillow 13.0.0 and setuptools 84.2.1 in trainnr-mjlab's; pulldown-cmark 0.13.5 in the Studio's (RUSTSEC-2026-0414). trainnr's Pillow (12.3.0) and setuptools (81) stay where LeRobot 0.6.1 caps them, below the fixed versions, each advisory a written exception in `tools/supply-chain.py`.
- The Studio's lock takes rustls 0.23.46, rustls-webpki 0.103.16, webpki-roots 1.0.11, tokio 1.53.3 (now also its floor in Cargo.toml), zstd-sys 2.1.2 and clipboard-win 5.4.3 (RUSTSEC-2026-0346, -0347, -0348, -0349, -0381, -0382, -0386, -0410; the yanked clipboard-win 5.4.1). No other crate or dependency edge moves. Dated exceptions in deny.toml: zvariant, zvariant_derive and zbus_names, unmaintained since they merged into zbus 6, until AccessKit moves to zbus 6; and zstd 0.13 and zstd-safe 7 (RUSTSEC-2026-0391, -0398, -0403), pinned through Rerun 0.36's arrow, parquet and mcap, until the Rerun update that takes zstd 0.14.

## [0.1.0] - 2026-10-06

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

- A release ships everything a version is, built by CI from its tag: the Studio for three platforms, the `trainnr` and `trainnr-mjlab` packages (sdist and wheel), a CycloneDX SBOM for each (`tools/sbom.py`) and `SHA256SUMS`; every file has a build-provenance attestation and each SBOM an SBOM attestation. Publishing a release runs `release-verify` on Linux, macOS and Windows: checksums, attestations, the wheel installed clean naming the version, the Studio installed through trainnr's own installer. The workflow is `release.yml` (was `studio-release.yml`).
- The research artifacts (the walk-c1 checkpoints and the study logs) are in the Hugging Face dataset `trainnr/trainnr-artifacts`, pinned by commit in `docs/artifacts/README.md`; they are no longer a GitHub release.
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

- NOTICE and LICENSE name the copyright holder, Prakhar Aggarwal; trainnr is published under the Trainnr AI name.
- The microduck's 38 meshes are no longer in the repository: Pollen Robotics licenses its 3D model files Creative Commons BY-SA-NC, which is not an Apache-2.0 grant. `robots/microduck/FETCH.json` pins each one by its git blob id, and trainnr fetches them from `pollen-robotics/microduck_rl` the first time the robot is used (about 22 MB; `TRAINNR_NO_ROBOT_FETCH=1` refuses, `python -m trainnr.bundles.fetch robots/microduck` fetches ahead). The bundle's stamp is unchanged once they are in place.
- Conduct reports and a security fallback go to trainnrai@gmail.com.
- The plugin, marketplace, registry entry and citation point at this repository until trainnr.ai serves a page; GOVERNANCE.md's network list defers to SECURITY.md's, which is complete.
- The 2025–26 rig is published as [rigrs](https://github.com/Trainnr-AI/rigrs) (MIT OR Apache-2.0), and the documentation links it. NOTICE credits Pollen Robotics for the microduck in the README's robots image, under its Creative Commons BY-SA-NC terms.
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
  gates) moved out of this repository; they are published as [rigrs](https://github.com/Trainnr-AI/rigrs).
  `recordings/` and
  the drivetrain bundle stay as evidence.

### Fixed

- The release's SBOM step writes its files: `tools/sbom.py` resolves `--out` before running the writers inside each package's folder (the first release candidate, 2026-10-06).
- `list_projects` marks the current project as current when the projects home is reached through a link, as macOS's `/var` and `/tmp` are; it compared one resolved path with one that was not.
- `identify_system`, `check_drift` and the other tools that take an artifact by name refuse one that a running job of the project is still writing by naming the job and the wait (`describe_job`), not "no artifact"; `ingest_public_log`'s handle says `next`, and the README's CPU quickstart shows the wait. Asked right after the handle, as the quickstart read, the first identification of a fresh install was refused (macOS, 2026-10-06).
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

- The locks take werkzeug 3.1.9 and multidict 6.9.1 (CVE-2026-102598, CVE-2026-104874), and trainnr-mjlab's takes fsspec 2026.9.0. trainnr's keeps fsspec 2026.2.0 (CVE-2026-104851), because datasets 4.8.5 requires fsspec<=2026.2.0 and LeRobot 0.6.1 requires datasets<5; the advisory gate lists it as a dated exception until LeRobot moves.
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
