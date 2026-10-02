# trainnr: the names and the repositories

*A proposal, 2026-10-01, for the operator's decision. It settles what every
surface is called and where every module lives before a single rename is
made. It follows the 2026-09-07 decision (docs/70 §1: the name is trainnr,
the org is Trainnr AI) and two structural studies made today against
primary sources: Rerun (rerun-io/rerun) and goose (aaif-goose/goose).
The full audit of our current names is in §6.*

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
- One line: *measure your robot, train on the measurement, certify, deploy.*
- No "trainnr" anywhere a person reads (§5 lists the one identifier that
  must stay, and why). "Robotiq" the gripper vendor keeps its name in the
  2F-85 asset, because it is theirs.

## 2. Surfaces, and what each is called

Names say what the thing does, the Rerun rule. Internal words that a user
never needed ("the doors", "the press", "the referee") become the industry's.

| Today | Proposed | Why |
|---|---|---|
| the Studio (the desktop window) | **trainnr Desktop** — the app you install; binary `trainnr-desktop` | "Studio" is what Foxglove dropped and what LM Studio and Android Studio own; "Desktop" is goose's word and is neutral. The app is not a viewer only: it lists the project, runs the simulator, mirrors jobs. Inside it, the words are exact: **the viewer** (the embedded Rerun viewer), **the simulator** (the MuJoCo viewport), **the pages** (Robots … Monitoring). |
| the doors (MCP tools) | **tools** of the **trainnr MCP server** | MCP's own word. Tool prefix `mcp__trainnr__*`. |
| the press | **data generation** (`generate_demos`) | W&B / LeRobot vocabulary; "press" stays as the module name only. |
| the referee | **success criterion** / **judge** | Gymnasium vocabulary. |
| a certificate | **evaluation** (the page already says so) | The word "certificate" stays inside the JSON file it names and in the paper's method name; users see evaluations. |
| the gate | **gate** | A CI word; keep. |
| the twin | **simulation stream** (`trainnr-sim-<scene>`) | "twin" is house. |
| the presenter | (internal) | Never user-facing. |
| the instrument | **trainnr** | Drop the metaphor from tool descriptions. |
| the rail | (internal) | The sidebar. |
| "the loop" | **the pipeline** (Sim-to-real pipeline, as the Overview says) | |

The alternative is to keep **Studio**: 450 mentions in the docs already,
and "Studio" is honest for a window that is more than a viewer. The cost
is the collision with three well-known products and the current muddle
where "Studio", "viewer" and "viewport" mean three things. The decision is
the operator's; everything else in this document is unchanged by it.

## 3. The repositories (organisation `Trainnr-AI`)

One product repository and satellites only where the lifecycle differs,
the rule both Rerun and goose follow.

| Repository | Holds | Why separate |
|---|---|---|
| **`trainnr`** (public) | the pipeline (`trainnr`), the mjlab plugin (`trainnr-mjlab`), the desktop app (`crates/trainnr-desktop`), the CLI and tools, the actuator library and the four nominal robot bundles, docs, the paper, findings and artifacts, the agent definitions and skills | It ships in lockstep: one version, one tag builds every artifact. |
| **`trainnr-cloud`** (private) | the control plane of docs/64: registry, scheduler, orgs, billing | Closed by decision; different owners and cadence. |
| **`trainnr-www`** (private or public) | trainnr.ai; reads `docs/` from the product repo at a pinned commit, Rerun's `landing` pattern | Vercel deploys; a designer's repo. |
| **`trainnr-robots`** (public, later) | large third-party robot bundles (microduck 20 MB, ALOHA 2 15 MB, the 2F-85) as release tarballs the pipeline fetches on demand through `asset_fetch` | Asset licences differ from code licences and the files are big; the small actuator library and nominal bundles stay in the product repo. First release: everything stays in `trainnr`; the split is a later, mechanical move. |
| **`rig`** (public archive) | the 2025–26 rig: the 14 crates and the firmware (29 kLOC), extracted with history | A different product era; out of the product repo before the first public push. |
| **`.github`** (public) | the organisation profile README | GitHub's convention. |

Not repositories: a docs repo (docs live beside the code and a checker ties
them, as Rerun does), a separate MCP repo (the server is a subcommand of
the package, as `rerun viewer-mcp` is), a separate Desktop repo (it
versions with the file contract it reads; goose's lesson).

## 4. "Developed separately, imported seamlessly": the layout inside `trainnr`

The monorepo gets what makes Rerun's modules independent: folders that
are layers, a check that fails an upward import, and packages that install
on their own.

```
trainnr/
  python/
    trainnr/            PyPI `trainnr`, import `trainnr`      (L0: the pipeline; no GUI, no trainer)
    trainnr-mjlab/      PyPI `trainnr-mjlab`, import `trainnr_mjlab`   (L1: the trainer plugin; imports L0)
  crates/
    trainnr-desktop/    the app; its own Cargo workspace (rerun 0.36) (L2: reads L0's files, spawns L0/L1)
  tools/                → `trainnr` CLI subcommands over time (L2)
  robots/               the actuator library + nominal bundles (data)
  docs/  docs/paper/  docs/findings/  docs/artifacts/
  .claude/agents  .claude/skills  .mcp.json
  Cargo.toml  pyproject.toml (uv workspace: python/*)  RELEASES.md
```

Rules:

- **Layers.** L0 never imports L1 or L2; L1 imports L0; L2 talks to L0 by
  files and processes only. A script (a layer check under tools/, after
  Rerun's crate-layer check) greps imports and fails CI on a
  violation. Today the one violation is that L0 shells out to L1 by path
  (`mcp_actions.py`); it becomes an entry point L1 registers
  (`trainnr.trainers`), so L0 needs no path to L1.
- **Own installs.** Each Python package builds and installs alone:
  `pip install trainnr` (core), `pip install trainnr[mcp,deploy,viz]`,
  `pip install trainnr-mjlab` (pins `trainnr==x.y`). In the repo they are a
  `uv` workspace, so development uses the path; a release uses the pin.
- **One protocol between the app and the core.** The Desktop reads
  `.index/*.json` and drives the pipeline through the MCP server's tools;
  it owns no state. Every path it assumes today (`spawn.rs` markers) is
  replaced by "the `trainnr` package on PATH" plus `$TRAINNR_REPO` for
  development.
- **One version, one tag.** `version` lives in one place and is written
  into both pyprojects, the Cargo workspace and the Desktop's about box;
  a tag `vX.Y.Z` builds wheels, the Desktop for three platforms, and the
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
| Desktop crate / binary | `studio-shell` | `trainnr-desktop` | launch path, `$TRAINNR_DESKTOP` |
| app id / eframe name | `robotiq_studio` / "robotiq studio" | `ai.trainnr.desktop` / "trainnr Desktop" | resets the app's persisted window state once |
| gym / task namespace | `robotiq/<task>` | `trainnr/<task>` | no alias: the only `robotiq/` task ids were twelve files in local, untracked projects, rewritten; no stamp moves (the stamp hashes the bare name) |
| LeRobot env type | `--env.type=robotiq` | `--env.type=trainnr` | |
| the module named after the old brand (envs/robotiq.py, since removed) | | a module named for what it holds (the gym environments) | |
| Rerun app ids | `robotiq-sim-`, `rq-gate`, `rq-press`, `rq-walk-*`, `rq-reward-preview`, `robotiq-walk-worlds`, `robotiq-instrument` | `trainnr-sim-`, `trainnr-gate`, `trainnr-data`, `trainnr-walk-*`, `trainnr-reward-preview`, `trainnr-walk-worlds`, `trainnr-actuators` | replays of old streams keep their old ids; the cloud feed's uuid5 seed changes (new recording ids) |
| env vars | `RQ_*` (19), `ROBOTIQ_*` (2) | `TRAINNR_*` | documented in one page |
| dataset ids | `rq-pipeline/rig`, `rq-pipeline/aloha2-kitting` | frozen for existing datasets; new defaults `trainnr/<name>` | |
| user agents | `rq-pipeline/cloud-gpu` | `trainnr/<version>` | |
| actor word | `by: studio` | `by: desktop` | the event log's word |
| the old rig's ids | `robotiq_hil`, `robotiq_rig`, … | untouched; they leave with the rig | |

**The one identifier that keeps the old word**: the bundle schema string
`robotiq-actuator-bundle/1`. It is hashed into every actuator model's stamp
(`actuator_bundle.py:125`); renaming it re-stamps every actuator bundle
and breaks the provenance of every finding, the paper's `xl330-m6@e57c2563`
among them. It is an internal constant inside a JSON file, never shown.
The path out is `trainnr-actuator-bundle/2` at the next change that
already breaks bundle compatibility, with a migration that re-wraps and
records old→new stamps. Until then it stays, and the brand doc says so.

## 6. What stays frozen, and what a user never sees

Frozen (docs/07 2026-09-09): kind ids (`certificate`, `deploy`, `batch`…),
the `trainnr-*/N` schema strings (several are hashed into stamps), the
existing dataset ids, the bundle schema string above. Internal words that
may stay in code: press, referee, presenter, rail, twin (as
`TWIN_APP_PREFIX`), door (in comments only).

## 7. Licence and governance

Apache-2.0 for everything (the 2026-09-07 AGPL-for-the-Studio idea is
dropped: one licence, one NOTICE; Rerun and goose both ship one licence).
DCO sign-off on every commit (goose is a foundation project without one
and calls that an accident). NOTICE names every third party the audit
listed as missing (microduck, Robotiq 2F-85, BAM, mjlab, rerun, Newton,
Unitree Go2, Mip-NeRF 360, Neverwhere). CONTRIBUTING (issues first,
conventional commit titles, one changelog line per PR), SECURITY (GitHub
private reporting), CODE_OF_CONDUCT (Contributor Covenant), CITATION.cff
(the paper), GOVERNANCE (maintainers, decisions), CODEOWNERS.

## 8. The public export

History is filtered, never wiped (docs/64): `git filter-repo` on a fresh
clone removes `claude-sync/` and `.env` from every commit and moves the rig
crates out; the 267 MB of walk-c1 training logs stay (they are the paper's
per-trial evidence; the alternative, a release tarball the records point
to, is the operator's call). The first push to `Trainnr-AI/trainnr` happens
only after the rename lands, `tools/verify.sh` is green on a cold clone,
the stranger test passes from that clone, and the README is the Go2 path.

## 9. Order of work

*Status 2026-10-02 (night): 1 written into docs/64 and docs/70; 2 landed (the
rename commit); 3 landed (NOTICE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT,
GOVERNANCE, CITATION.cff, CODEOWNERS, CHANGELOG, the README, the org
profile under `tools/export/`), with the plugin manifest and the MCP
Registry `server.json` beside them; 4 landed (the extraction commit, and
the archive built with history by `tools/export/rig/build-rig-archive.sh`:
177 commits, 195 files, the crates' tests and a firmware build green);
5 run (`tools/export/public-export.sh`: 892 commits, no private path in
any of them; the cold-clone gate in docs/07). The push awaits the operator.*

1. This document decided (the Desktop name, the robots split timing, the
   logs), then written into docs/64 and docs/70 as the standing decision.
2. The rename, one commit: directories, packages, entry points, the MCP
   server, the Desktop, the ids, the env vars, the docs; the layer check
   added. (The compat shims for the two import names and the task
   namespace were added and then removed the same day: nothing outside
   this repository ever used the old names.)
3. Governance files, NOTICE, README, CITATION; the org profile.
4. The rig extracted to its own repository with history.
5. The export script and the cold-clone run; then the first push.
