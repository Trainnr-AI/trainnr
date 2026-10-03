# tools/

The lab bench: the gates, the scripts the MCP tools spawn, the studies
behind the paper, and the paper build. Shell tools run from the repository
root; Python tools run from `trainnr/` so uv picks up its environment:

```sh
cd trainnr && uv run --extra sim --extra viz python ../tools/<name>.py
# on WSL2 the GPU routing lives in one file:
cd trainnr && uv run --env-file wsl.env --extra sim --extra viz python ../tools/<name>.py
# tools that need the train environment (LeRobot):
cd trainnr && ../tools/wsl-run.sh .venv-train/bin/python ../tools/<name>.py
```

Every tool's own docstring or header is the authority on its flags; this
page says what each one is for.

## Gates

| Tool | What it proves |
|---|---|
| `verify.sh` | Everything, one command: the Studio crate's fmt, clippy and tests; the doc, layer, number and unsafe gates; the Python packages (ruff, mypy, both unit suites); the USD import suite. Nothing here needs hardware. |
| `check-docs.py` | No document names a path that no longer exists; every link resolves. |
| `check-layers.py` | A lower layer never imports a higher one (`trainnr` ← `trainnr-mjlab` ← the Studio), lazy imports included. |
| `check-numbers.py` | Every k/n success figure in the manuscript and the ledgers exists in a finding record under `docs/findings/`. |
| `check-unsafe-gates.py` | `unsafe` stays forbidden in the crate's manifest. |
| `findings.py` | Renders `docs/findings/*.json` into `docs/68-findings.md`; `--check` fails when the page is stale. The ledger is generated, never hand-edited. |
| `coverage.sh` | Rust line coverage for the Studio crate. |
| `coverage-badge.py` | Turns coverage.py's JSON report into the README's coverage badge (CI, after the tests). |
| `github-setup.sh` | Applies the repository's GitHub settings: merges, Actions, labels and, once public, the ruleset on `main` and the security features; lists who can merge. Idempotent. |
| `loc-report.py` | Line counts by area. |

## The loop's doors

The scripts the MCP server spawns, in the loop's order. Each runs as a job
with its log beside the artifact it writes.

| Tool | What it does |
|---|---|
| `capture-telemetry.py` | A robot's telemetry live into a project: `sources` lists the adapters; `record <name> --source dds --network <if> --seconds 60` on a Unitree robot records `rt/lowstate` and `rt/lowcmd` as one recording; `--standin <deployment>` rehearses on Unitree's simulator. |
| `public-log.py` | Public recordings of real robots: `list` the registry with each log's licence state, `fetch` one with every byte count and digest checked, `ingest` one into a project with its provenance. |
| `capture-scene.py` | A phone video or a folder of frames into a scene: ffmpeg, COLMAP, Brush, alignment, the collision proxy, the see-versus-touch gap, the record. |
| `install-brush.py`, `install-gsplat.py` | The two splat trainers the capture chain uses: Brush's release binary for this machine (checksum verified), or gsplat into the train environment with CUDA wheels pinned to torch's build. |
| `import-usd.py` | A USD robot asset (Isaac Sim, Omniverse) into a hash-stamped bundle through Newton's importer; `--fetch owner/repo@commit:path` for a public asset at a pinned commit. |
| `audit-bundle.py` | What the importer changed: a bundle's compiled model against the description it came from, each change explained or UNEXPLAINED; exit 1 on an unexplained change. |
| `actuator-bundle.py` | Wrap vendored BAM fits into certified actuator bundles; verify anyone's (stamps, rail and floor checks, honesty advisories). |
| `sync-bam-actuators.py` | Vendor new or changed servos from a BAM checkout into `robots/actuators/`; refuses a changed file with no `--version`. |
| `fit-report.py` | A bundle's fit records: intervals, cross-run spread, EXCEEDS verdicts. |
| `show-legged-fit.py` | A legged-joints fit in both viewers: measured, rigid and modelled torque per joint, residuals and bootstrap histograms; MuJoCo stills under the fitted terms. |
| `accept-task.py` | The acceptance critic: a manipulation task's scripted expert must pass its referee on every paired trial and the do-nothing floor on none; a walk task's robot bundle is checked for the walk's shape, then the learnability smoke runs. Exit 1 on rejection. |
| `kitting-demos.py` | Referee-filtered scripted kitting episodes over the task's declared band with randomization, each manifest carrying the expert's stamp; shards across parallel generators. |
| `planner-demos.py` | The planner expert presses an SO-101 task (`lift`, `block_stack`, `tool_insert`): beats per seated scene, chained IK, kept by the referee, every declared camera captured; `--shards N --parallel M` presses in disjoint ranges and merges. |
| `press-multiply.py` | Multiply kept kitting seeds into new episodes at device scale on MJX-Warp. |
| `study.py` | A study with its arms as data: press, convert, train, judge, paired. |
| `paired-study.py` | The guessed-versus-identified randomization study's two datasets. |
| `e2e-smoke.py` | The imitation chain in one command: demos, LeRobot v3, `lerobot-train` with in-loop evaluation through our env, `lerobot-eval` with records, the fold; `--scale smoke` or `--scale cloud`; any contiguous slice of the stages. |
| `train-watch.py` | A training run as a live Rerun dashboard, from its files: the run manifest, losses, in-loop and final evaluation funnels, GPU samples; `--rrd` saves the stream. |
| `rl-watch.py` | Reinforcement learning watched live: thousands of worlds on the GPU, sixteen on screen. |
| `gate-deployment.py` | The sim-to-sim gate on a project's deployment: the exported ONNX policy driven through its manifest alone under `--runtime mujoco` or `dds` (Unitree's simulator), judged the evaluation's way. |
| `preflight-deployment.py` | Before the first tick on a robot: seven checks refused by name with their numbers (policy widths, joint order, gains, a dry rollout's targets and torques, compute per tick, the robot's state against the SDK's watchdogs), then the ramp-in and the stops measured; `preflight.json` beside the manifest. |
| `attribute-deployment.py` | Which parameter breaks a deployment first: a passing gate re-run with one dynamics knob at a time up its ladder, the cliff per knob against the certificate's lower bound. |
| `assay-deployment.py` | The perturbation assay: a deployment on a captured scene, nominal and moved, each stage gated with one seed. |
| `mcp-server.py` | A forwarder to `trainnr mcp`: the MCP server over stdio, 76 tools, for a client configured before the console script existed. |
| `studio-present.py` | The presenter: the one Python process the Studio asks to show things; watches the project's index and streams the chosen artifact into the embedded viewer. |
| `studio-render-stream.py` | MuJoCo's own render as a subprocess service for the Studio's simulator: frames out on stdout, camera commands in on stdin; a scene is a preview task, a walk, or a deployment. |
| `studio-instrument-view.py` | The instrument's records into the Studio's viewer with a blueprint: evaluation funnels, fit parameters as estimate and interval series, the friction budget. |
| `studio-cloud-feed.py` | A rented card's training live in the Studio: tails the remote log over ssh and streams the trainer's metrics to the Studio's ingest port. |
| `gen-app-icon.py` | Generates the Studio's icon asset; re-run and commit together with the asset. |

## Scenes in the viewers

| Tool | What it does |
|---|---|
| `show-aloha2.py`, `show-cargo-chaos.py` | One scene each in the MuJoCo viewer and Rerun. |
| `show-many.py` | Batched domain-randomized worlds side by side. |
| `show-2f85-isaac.py` | The USD-born 2F-85 bundle closing on a cube: MuJoCo stills and the Rerun stream; exits 0 when the cube is held. |
| `show-gripper-pick.py` | The `gripper-pick` task's acceptance rungs (pick, no-close, limp) through the expert, replayed from the judged rows; exits 0 when only the expert rung succeeds. |
| `camera-match.py` | Simulated renders beside released real frames: camera placement is calibration. |
| `solver-study.py` | A constraint-solver sweep on the kitting scene, judged by the referee, with penetration, iterations, slip and grip force per row. |
| `determinism-probe.py` | Is MJX-Warp bit-repeatable, at what cost? One subprocess per mode; the verdict needs a CUDA device. |

## Studies and campaigns behind the paper

| Tool | What it does |
|---|---|
| `lift-envelope-cell.py` | One cell of the lift's expert envelope: the scripted pick at a damping and gain scale, N trials, one process per cell. |
| `walk-c1-arm.sh` | One arm of the walk study on one pod: train the teacher under a declared randomization span, judge it at the bundle's point fit on 40 matched trials. |
| `walk-c1-refit-pods.sh` | The identified-interval arms of the walk study on six pods in parallel, on the refit bundle. |
| `walk-c1-fold.py` | Fold the walk arms' certificates into one finding and its figure. |
| `walk-mismatch-matrix.sh`, `walk-matrix-fold.py` | Every walk policy judged in each mismatched world (the actuator pinned at the fit times a scale, or drawn from a wide span), and the fold into a finding. |
| `walk-axes-on-box.sh` | The per-axis mismatch matrices where a GPU is free: the nine tracked walk checkpoints judged on each axis at each scale, then one fold per axis. |
| `walk-refit-rejudge-on-box.sh` | Re-judges the refit arms' certificates with the fixed randomization seam; both folds rewrite their records with a `revised` note so the manuscript's citations keep resolving. |
| `walk-envelope.sh` | One policy certified at a grid of actuator-parameter scales around the fit: where it fails. |
| `walk-latency-fold.py` | The latency-budget certificates folded into one finding: the same student under an inference budget of n control ticks. |
| `walk-dagger-round.sh`, `dagger-fold.py` | One DAgger round on the walk on a pod, and its fold: base student versus round student on the same 40 trials. |
| `campaign-distill.sh` | The distillation campaign on a rented card: press teacher demonstrations, export, train a vision student, certify it. |
| `bam-bootstrap.py` | A fitted interval for a BAM actuator from Rhoban's public bench logs: refit, bootstrap, the record. |
| `sts-study.py`, `sts-figure.py` | The STS3215 benchmark ingest and its parameter fits; the study's figure. |

## The paper build

| Tool | What it does |
|---|---|
| `paper.py` | Derives `docs/paper/paper-1.md` (the reading copy with figure list and provenance appendix) from `docs/paper/manuscript.md`; `--check` fails when stale. |
| `paper-tex.py` | The manuscript as LaTeX into `docs/paper/latex/main.tex`, figures copied beside it; `--pdf` compiles with latexmk. |
| `finding-figure.py` | Draws the figure of one or more finding records into `docs/figures/` and writes the paths onto the record. |
| `paper-figures.py` | The designed composite figures into `docs/figures/paper/`. |
| `paper-stills.py`, `paper-lift-stages.py`, `paper-walk-stages.py` | The paper's robot stills from the simulator, never a screenshot: the lift at its stages, the microduck walking and falling. |
| `paper-figure-sheet.py` | A review sheet: each figure with its title, caption and the paragraphs that cite its record, one HTML page. |

## Cloud

| Tool | What it does |
|---|---|
| `cloud-gpu.py` | A rented GPU as a runbook: `offers`, `launch`, `push` (rsync, never `.env`), `bootstrap`, `run`, `pull`, `terminate`; `follow <id> NAME --watch` mirrors a running pod's light files and opens the dashboard. RunPod first, behind `trainnr/cloud`'s provider seam (docs/34). |
| `_pod-bootstrap.sh`, `_pod-resume.sh` | Sent over ssh by `cloud-gpu.py`: the train environment on a fresh machine, and a fresh pod over a network volume made ready in one line. |
| `pod-volume.py` | A RunPod network volume over its S3 API: `du`, `ls`, `rm` (dry run unless `--yes`); works when the pod is stopped. |

## The archived rig

These tools read and replay the recordings of the 2025–26 rig under
`recordings/`. The rig's firmware and crates live in
[Trainnr-AI/rig](https://github.com/Trainnr-AI/rig); the live paths need
that checkout's `hil-host`, the replays run from here.

| Tool | What it does |
|---|---|
| `rig-rerun.py` | A rig session's `.wire` file as a live Rerun dashboard with the rig drawn in 3D from its odometry. |
| `replay-errand.py` | A recorded fetch errand replayed whole: MuJoCo window and Rerun stream side by side. |
| `sim-errand.py` | The errand in pure simulation under the firmware's own state machine, as the prediction the real run was judged against. |
| `udp-wire-bridge.py` | The rig's WiFi telemetry (UDP 9870) into a growing `.wire` file. |
| `show-rig.py`, `show-yellow.py` | The car-and-arm rig and its yellow arm in the MuJoCo viewer. |
| `debug-inference.sh` | Runs any command with ONNX Runtime's and Rust's logging gates open, so execution-provider diagnostics show. |

## Plumbing (not tools, but they live here)

| File | What it is |
|---|---|
| `_lab.py` | The shared bench the Python tools import: path bootstrap, Rerun session plumbing, `running(...)` to put a tool's run in the project's job table so the Studio shows its progress. The MuJoCo-to-Rerun mirror lives in `trainnr/trainnr/viz.py`. |
| `wsl-run.sh` | Runs a command under `trainnr/wsl.env`, the WSL2 GPU routing in one file. |
| `setup-hooks.sh`, `hooks/pre-commit` | Installs and is the pre-commit gate: the crate's fmt, clippy and tests, docs, layers, numbers, unsafe, ruff over `trainnr/` and `tools/`, the unit suite. |
| `.ruff.toml` | Extends the package's lint contract to `tools/`, with the per-file exceptions and their reasons. |
