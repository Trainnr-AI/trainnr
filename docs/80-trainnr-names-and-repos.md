# trainnr: the names and the repositories

*Decided 2026-10-01 to 2026-10-03. What every surface is called and
where every module lives. It follows the 2026-09-07 decision (the name
is trainnr, the organisation is Trainnr AI) and two structural studies
against primary sources: Rerun (rerun-io/rerun) and goose
(aaif-goose/goose). The audit of the former names is in §5 and §6.*

## 0. The two models we studied, in one paragraph each

**Rerun** keeps everything that ships in lockstep in one repository: every
Rust crate, the Python and C++ SDKs, the web viewer, examples, docs
content, tests. One version number for all of it, bumped in lockstep every
two weeks. Internal crates carry a short prefix (`re_viewer`, `re_sdk`);
only the two or three things a user types are unprefixed (`pip install
rerun-sdk`, `cargo install rerun-cli`, the `rerun` binary that is both the
viewer and the server, `rerun viewer-mcp` as a subcommand). The crates are
laid out in folders that are layers, and a CI script fails any import that
points upward. Satellite repositories exist only for a different reason to
exist: heavy examples with their own dependencies, loader plugins as
executables on PATH, generic libraries spun out, a private website
renderer that reads the public docs folder. Product surfaces are named for
what they do: Viewer, SDK, Hub. MIT or Apache, no CLA.

**goose** (Block's agent, now under the Linux Foundation) is likewise one
monorepo with one version: a Rust workspace (`goose`, `goose-cli`,
`goose-mcp`, `goose-sdk`), an Electron desktop app, Docusaurus docs,
recipes. Satellites exist only where the release channel differs: the VS
Code extension, the mobile apps. The brand is lowercase in prose, the CLI
is `goose`, every crate is `goose-*`, the app is "goose Desktop". Their
hard-won lesson: the desktop app *spawns and connects to* one server over
a standard protocol and owns no state, so the app and the core version
independently; they paid a year of a custom daemon (`goosed`) before
landing there. Extensions are external MCP servers in a registry, never
code in core. Apache-2.0, governance files from day one.

## 1. The brand

- **trainnr**, lowercase in prose, like goose and rerun. "Trainnr AI" is the
  company and the GitHub organisation (`github.com/Trainnr-AI`); the site
  is trainnr.ai. Capitalised only where an operating system demands it
  (`Trainnr.app`).
- The line and the loop's eight turns are the root README's. Its honesty
  rule holds here too: no real deployment is claimed until one is on the record.
- No "robotiq" anywhere a person reads (§5 lists the one identifier that
  must stay, and why). "Robotiq" the gripper vendor keeps its name in the
  2F-85 asset, because it is theirs.

## 2. Surfaces, and what each is called

Names say what the thing does, the Rerun rule. Internal words that a user
never needed ("the doors", "the press", "the referee") become the industry's.

| Former word | The name | Why |
|---|---|---|
| the Studio | **trainnr Studio** (decided 2026-10-03): the app you install; binary `trainnr-studio`, command `trainnr studio`, app id `ai.trainnr.studio` | The tools were already `launch_studio`, `describe_studio`, `set_studio_theme`, and `$TRAINNR_STUDIO` names the binary, so the user-facing name and the API agree. The app is not a viewer only: it lists the project, runs the simulator, mirrors jobs. Inside it the words are exact: **the viewer** (the embedded Rerun viewer), **the simulator** (the MuJoCo window), **the pages** (Robots … Monitoring). |
| the doors (MCP tools) | **tools** of the **trainnr MCP server** | MCP's own word. Tool prefix `mcp__trainnr__*` from a checkout's `.mcp.json`, `mcp__plugin_trainnr_trainnr__*` through the plugin. |
| the press | **data generation** (`generate_walk_demos`, `generate_kitting_demos`, `generate_planned_demos`) | W&B / LeRobot vocabulary; "press" stays as the module name only. |
| the referee | **success criterion** / **judge** | Gymnasium vocabulary. |
| a certificate | **evaluation** (the page already says so) | The word "certificate" stays inside the JSON file it names and in the paper's method name; users see evaluations. |
| the gate | **gate** | A CI word; keep. |
| the twin | **simulation stream** (`trainnr-sim-<scene>`) | "twin" is house. |
| the presenter | (internal) | Never user-facing. |
| the instrument | **trainnr** | Drop the metaphor from tool descriptions. |
| the rail | (internal) | The sidebar. |
| "the loop" | **the loop** | The README's word; the Overview's strip is labelled the sim-to-real pipeline. |

"Desktop" (goose's word) was the alternative; the decision of 2026-10-03
chose Studio.

## 3. The repositories (organisation `Trainnr-AI`)

One product repository and satellites only where the lifecycle differs,
the rule both Rerun and goose follow.

*Decided 2026-10-03 (one repository for now; a split later if needed): everything
below the first three rows is deferred. The product repository holds
mjsim (docs/84), the telemetry readers, the exact statistics, the scene
chain as modules and directories with their seams named,
so a split later is a move, not a rewrite. The rows stay as the shape a
split would take.*

| Repository | Holds | Why separate |
|---|---|---|
| **`trainnr`** (public) | the pipeline (`trainnr`), the mjlab plugin (`trainnr-mjlab`), the Studio (`crates/trainnr-studio`), the CLI and tools, the actuator library and the five robot bundles, docs, findings and artifacts, the agent definitions and skills | It ships in lockstep: one version, one tag builds every artifact. |
| a hosted service (not in this repository) | a control plane: registry, scheduler, organisations | Different owners and cadence; it extends the tool through its plugin seams (`trainnr.mcp_tools`). |
| the website | trainnr.ai; reads `docs/` from the product repo at a pinned commit, Rerun's `landing` pattern | Its own build and release cadence. |
| **`trainnr-robots`** (public, later) | large third-party robot bundles (microduck 20 MB, ALOHA 2 15 MB, the 2F-85) as release tarballs the pipeline fetches on demand through `asset_fetch` | Asset licences differ from code licences and the files are big; the small actuator library and nominal bundles stay in the product repo. First release: everything stays in `trainnr`; the split is a later, mechanical move. |
| **`mjsim`** (public, planned; docs/84) | the MuJoCo simulator window: the egui widget (Rust) and the simulation stream (Python), one version; the Studio and `trainnr` depend on it | its own cadence and its own users (any egui app, any MuJoCo user); the name is free on PyPI and crates.io |
| a paper repository | planned: the paper moved out of this repository on 2026-10-05 and will be published in its own | A paper freezes at submission and is cited by DOI; the product repository says only that the research is ongoing. |
| **`rig`** (published as [rigrs](https://github.com/Trainnr-AI/rigrs) on 2026-10-06) | the 2025–26 rig: the 14 crates and the firmware (29 kLOC), extracted with history | A different product era; out of the product repo before the first public push. |
| **`.github`** (public) | the organisation profile README | GitHub's convention. |

Not repositories: a docs repo (docs live beside the code and a checker ties
them, as Rerun does), a separate MCP repo (the server is a subcommand of
the package, as `rerun viewer-mcp` is), a separate Studio repo (it
versions with the file contract it reads; goose's lesson).

## 4. "Developed separately, imported seamlessly": the layout inside `trainnr`

The monorepo gets what makes Rerun's modules independent: folders that
are layers, a check that fails an upward import, and packages that install
on their own.

```
trainnr/
  trainnr/              PyPI `trainnr`, import `trainnr`      (L0: the pipeline; no GUI, no trainer)
  trainnr-mjlab/        PyPI `trainnr-mjlab`, import `trainnr_mjlab`   (L1: the trainer plugin; imports L0)
  crates/
    trainnr-studio/     the app; its own Cargo workspace (rerun 0.36) (L2: reads L0's files, spawns L0/L1)
  tools/                the gates and the lab scripts; `trainnr` CLI subcommands over time (L2)
  robots/               the actuator library + the robot bundles (data)
  docs/  docs/findings/  docs/artifacts/
  .claude/agents  .claude/skills  .claude-plugin/  .mcp.json  server.json
  CHANGELOG.md  CITATION.cff  CONTRIBUTING.md  GOVERNANCE.md  LICENSE  NOTICE  SECURITY.md
```

Rules:

- **Layers.** L0 never imports L1 or L2; L1 imports L0; L2 talks to L0 by
  files and processes only. A script (a layer check under tools/, after
  Rerun's crate-layer check) greps imports and fails CI on a
  violation. *Target:* L0 still shells out to L1 by path
  (`mcp_actions.py`); the plan is an entry point L1 registers
  (`trainnr.trainers`), so L0 needs no path to L1.
- **Own installs.** Each Python package builds and installs alone:
  `pip install trainnr` (core), `pip install trainnr[mcp,deploy,viz]`,
  `pip install trainnr-mjlab` (pins `trainnr==x.y`). *Target:* a `uv`
  workspace at the root; today there is no root pyproject, and
  `trainnr-mjlab` reaches `trainnr` through a path dependency.
- **One protocol between the app and the core.** The Studio reads
  `.index/*.json` and drives the pipeline through the MCP server's tools;
  it owns no state. Every path it assumes today (`spawn.rs` markers) is
  replaced by "the `trainnr` package on PATH" plus `$TRAINNR_REPO` for
  development.
- **One version, one tag.** *Target.* Today 0.1.0 is written by hand in
  three files (both pyprojects and the Studio's Cargo.toml). The plan:
  `version` lives in one place and is written
  into both pyprojects, the Cargo workspace and the Studio's about box;
  a tag `vX.Y.Z` builds wheels, the Studio for three platforms, and the
  source bundle; a floating `stable` tag, goose's way. Alphas pinned exact.
- **Extension points.** Robot adapters, identification methods, trainers,
  engines, model sources are entry-point groups under `trainnr.*`; a
  third party's package registers and is found, no fork needed.

## 5. Identifiers: the rename table

| Identifier | Now | Becomes | Kind of change |
|---|---|---|---|
| PyPI / import | `rq-pipeline` / `rq_pipeline` | `trainnr` / `trainnr` | no shim: nothing outside this repository ever imported it (decided 2026-10-02, after the rename landed) |
| PyPI / import | `rq-mjlab` / `rq_mjlab` | `trainnr-mjlab` / `trainnr_mjlab` | same |
| entry-point groups | `rq_pipeline.tasks` … | `trainnr.tasks` … | both sides in-tree |
| MCP server | `robotiq` (`.mcp.json`, `name=`) | `trainnr` | agents' tool prefix changes |
| Studio crate / binary | `studio-shell` | `trainnr-studio` | launch path, `$TRAINNR_STUDIO` |
| app id / eframe name | `robotiq_studio` / "robotiq studio" | `ai.trainnr.studio` / "trainnr Studio" | resets the app's persisted window state once |
| gym / task namespace | `robotiq/<task>` | `trainnr/<task>` | no alias: the only `robotiq/` task ids were twelve files in local, untracked projects, rewritten; no stamp moves (the stamp hashes the bare name) |
| LeRobot env type | `--env.type=robotiq` | `--env.type=trainnr` | |
| the module named after the old brand (envs/robotiq.py, since removed) | | a module named for what it holds (the gym environments) | |
| Rerun app ids | `robotiq-sim-`, `rq-gate`, `rq-press`, `rq-walk-*`, `rq-reward-preview`, `robotiq-walk-worlds`, `robotiq-instrument` | `trainnr-sim-`, `trainnr-gate`, `trainnr-data`, `trainnr-walk-*`, `trainnr-reward-preview`, `trainnr-walk-worlds`, `trainnr-actuators` | replays of old streams keep their old ids; the cloud feed's uuid5 seed changes (new recording ids) |
| env vars | `RQ_*` (19), `ROBOTIQ_*` (2) | `TRAINNR_*` | documented in one page |
| dataset ids | `rq-pipeline/rig`, `rq-pipeline/aloha2-kitting` | frozen for existing datasets; new defaults `trainnr/<name>` | |
| user agents | `rq-pipeline/cloud-gpu` | `trainnr/<version>` | |
| actor word | `by: studio` | `by: studio` | unchanged |
| the old rig's ids | `robotiq_hil`, `robotiq_rig`, … | untouched; they leave with the rig | |

**The one identifier that keeps the old word**: the bundle schema string
`robotiq-actuator-bundle/1`. It is hashed into every actuator model's stamp
(`actuator_bundle.py:125`); renaming it re-stamps every actuator bundle
and breaks the provenance of every finding, `xl330-m6@e57c2563`
among them. It is an internal constant inside a JSON file, never shown.
The path out is `trainnr-actuator-bundle/2` at the next change that
already breaks bundle compatibility, with a migration that re-wraps and
records old→new stamps. Until then it stays, and the brand doc says so.

## 6. What stays frozen, and what a user never sees

Frozen (decided 2026-09-09): kind ids (`certificate`, `deploy`, `batch`…),
the `trainnr-*/N` schema strings (several are hashed into stamps), the
existing dataset ids, the bundle schema string above. Internal words that
may stay in code: press, referee, presenter, rail, twin (as
`TWIN_APP_PREFIX`), door (in comments only).

## 7. Licence and governance

*Changed 2026-10-05, before the first public release: the licence is
FSL-1.1-ALv2 (the Functional Source License with an Apache-2.0 future
licence), not Apache-2.0. Any use but a competing product or service, and
each version Apache-2.0 two years after its release; contributions come in
under Apache-2.0 (CONTRIBUTING.md). Still one licence for the whole
repository, one NOTICE. The paragraph below is the 2026-10-02 decision.*

Apache-2.0 for everything (the 2026-09-07 AGPL-for-the-Studio idea is
dropped: one licence, one NOTICE; Rerun and goose both ship one licence).
DCO sign-off on every commit (goose is a foundation project without one
and calls that an accident). NOTICE names every third party the audit
listed as missing (microduck, Robotiq 2F-85, BAM, mjlab, rerun, Newton,
Unitree Go2, Mip-NeRF 360, Neverwhere). CONTRIBUTING (issues first,
conventional commit titles, one changelog line per PR), SECURITY (GitHub
private reporting), CODE_OF_CONDUCT (Contributor Covenant), CITATION.cff
(the software), GOVERNANCE (maintainers, decisions), CODEOWNERS.

## 8. The public edition

The public repository is produced from the maintainers' repository by a
filter that keeps the tool, its documents and its records; the rig lives
in its own archive. A push happens only after the rename has landed,
`tools/verify.sh` is green on a cold clone, a new user's agent has walked
the Go2 loop from that clone, and the README is the Go2 path.

Since 2026-10-04 the public repository is the source of truth: changes
land there by pull request, and the filter is retired.

## 9. Status

Everything this document decided has landed: the rename, the governance
files and NOTICE, the rig extracted to its archive with history, and the
public edition built, gated on a cold clone and walked by a new user's
agent (2026-10-02 and 2026-10-03).
