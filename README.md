<p align="center">
  <img src="docs/figures/readme/logo.png" width="96" alt="trainnr">
</p>

<h1 align="center">trainnr</h1>

<p align="center">
  <b>The physical AI platform for robot learning, run from your coding agent.</b><br>
  Simulation, reinforcement learning, deployment, and a data flywheel from<br>
  the robot's telemetry back into training.
</p>

<p align="center">
  <a href="https://github.com/Trainnr-AI/trainnr/actions/workflows/gates.yml"><img src="https://github.com/Trainnr-AI/trainnr/actions/workflows/gates.yml/badge.svg?branch=main" alt="gates"></a>
  <a href="https://github.com/Trainnr-AI/trainnr/actions/workflows/gates.yml"><img src="https://github.com/Trainnr-AI/trainnr/raw/badges/coverage.svg" alt="coverage"></a>
  <a href="#licence"><img src="https://img.shields.io/badge/licence-FSL--1.1--ALv2-blue" alt="licence: FSL-1.1-ALv2"></a>
  <a href="trainnr/pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-blue" alt="python ≥ 3.10"></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/MCP-76%20tools-6f42c1" alt="MCP server: 76 tools"></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/Claude%20Code-plugin-d97757" alt="Claude Code plugin"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/DCO-signed--off-green" alt="DCO"></a>
  <!-- After the repository is public and the scorecard workflow has run once:
  <a href="https://scorecard.dev/viewer/?uri=github.com/Trainnr-AI/trainnr"><img src="https://api.scorecard.dev/projects/github.com/Trainnr-AI/trainnr/badge" alt="OpenSSF Scorecard"></a>
  -->
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#what-it-does">What it does</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="docs/paper/README.md">Paper</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<img alt="The Studio's Overview of a Go2 project: every stage of the loop proved, and the best policy under each test condition with its exact interval" src="docs/figures/readme/studio-overview.png">

trainnr is an open-source platform for **physical AI**: robot learning
from the robot's own data, end to end. It identifies your robot's dynamics
from its telemetry into a **simulation** model (real-to-sim), creates
training data on that model, trains policies with **reinforcement learning
for robotics** (MuJoCo, mjlab) or imitation learning (LeRobot), evaluates
them with exact confidence intervals, and gates the **deployment** on the
robot's runtime (sim-to-real). The deployed robot's **telemetry** comes
back as new data: it is checked for drift, re-identified, and fed **back
into training**. That cycle is the **data flywheel**, and every turn of it
is on the record.

Every step is a tool of one **MCP server**, so your coding agent runs the
loop by conversation. Every result is a **record** the next step cites,
stamped `name@hash`. The **Studio**, a desktop app, shows each stage as it
lands. For developers with one robot and companies with a fleet.

## Quickstart

In Claude Code, install the plugin (the MCP server, seven agents and two
skills):

```sh
claude plugin marketplace add Trainnr-AI/trainnr
claude plugin install trainnr@trainnr
```

Then ask for the loop in plain words:

```text
you    Create a project, onboard Unitree's Go2 and identify it from the IIT chirp log.
agent  create_project("go2")                        → a project with its index
       onboard_robot(go2.xml, "go2")                → go2@e5aa641994fc, shape checked
       ingest_public_log("iit-go2-chirp")           → 24 s · 200 Hz · 6 channels · public log
       identify_system("go2", "iit-go2-chirp")      → 31 of 36 parameters pinned at 95 %

you    Train a walk on that fit, evaluate it, and gate the export.
agent  create_task("trainnr/go2-walk"), accept_task → accepted: learnability smoke
       train_walk(agent="g3", fit="fit@…")          → an mjlab experiment on the fit
       evaluate_walk(trials=8)                      → survived 8/8, tracked 0/8, CP95 [0.00, 0.37]
       export_deployment, gate_deployment           → ONNX + manifest, sim-to-sim gate passed
       preflight_deployment, check_drift            → 7/7 checks; drift within interval
```

That is a real session from a fresh clone (2026-10-03), with a short
150-iteration training run, which survives but does not yet track its
commands. The full recipe certifies at 38 of 40 (below). The robot model
is Unitree's own `go2.xml` from
[`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab)
(`src/assets/robots/unitree_go2/xmls/`).

<details>
<summary><b>Any other MCP client</b> (Cursor, Codex, a checkout)</summary>

**Any MCP client**, from a checkout (the server is `trainnr mcp`, served over
stdio; `uvx trainnr mcp` once the package is on PyPI):

```sh
git clone https://github.com/Trainnr-AI/trainnr && cd trainnr
claude mcp add trainnr -- uv run --directory "$PWD/trainnr" --extra sim --extra mcp trainnr mcp
```

Cursor, in `.cursor/mcp.json`; Codex, in `~/.codex/config.toml`:

```json
{ "mcpServers": { "trainnr": { "command": "uv",
  "args": ["run", "--directory", "/path/to/trainnr/trainnr", "--extra", "sim", "--extra", "mcp", "trainnr", "mcp"] } } }
```

```toml
[mcp_servers.trainnr]
command = "uv"
args = ["run", "--directory", "/path/to/trainnr/trainnr", "--extra", "sim", "--extra", "mcp", "trainnr", "mcp"]
```

</details>

<details>
<summary><b>The Studio</b> (desktop app) and <b>training</b> (CUDA GPU)</summary>

**The Studio** (Rust; the embedded Rerun viewer and the MuJoCo simulator):

```sh
cd crates/trainnr-studio && cargo build --release && cd ../..
uv run --directory trainnr trainnr studio        # or the launch_studio tool
```

**Training** needs the mjlab trainer and a CUDA GPU:
`cd trainnr-mjlab && uv sync --extra viz`. Everything else, including every
test, runs on a laptop without a GPU.

</details>

## What it does

### Real-to-sim: telemetry into a simulation model

The robot's own telemetry, fitted into the model it trains on. Every
parameter carries a 95 % interval and a verdict: pinned when the data
constrains it, NOT PINNED when it does not. A public log works too, with
its provenance shown as such.

<img alt="A robot's page in the Studio: the asset, and its system identification table with estimates, intervals and verdicts" src="docs/figures/readme/studio-identification.png">

`onboard_robot` · `ingest_recording` · `ingest_public_log` · `start_capture` · `identify_system`

### Data creation and RL for robotics

Reinforcement learning on [mjlab](https://github.com/mujocolab/mjlab), or
imitation learning through [LeRobot](https://github.com/huggingface/lerobot),
on the fitted model with a randomization span that comes from the fit's
intervals instead of a guess. Demonstrations are generated under recorded
randomization and kept by the task's success criterion, with a datasheet.

<img alt="An experiment's page: the reward curve, provenance, the trainer's settings and the training table" src="docs/figures/readme/studio-experiment.png">

`create_task` · `accept_task` · `train_walk` · `generate_walk_demos` · `multiply_demos` · `run_chain`

### Evaluation with an interval, not a clip

A clip cannot tell 10 % from 99 %. An evaluation here is paired,
seed-matched trials with an exact (Clopper-Pearson) confidence interval and
a funnel of how far each episode got; gates read the lower bound.

<img alt="An evaluation's page: 38 of 40 with its 95 % interval, survived and tracked bars, provenance and the protocol" src="docs/figures/readme/studio-evaluation.png">

`evaluate_walk` · `describe_eval` · `list_eval_records` · `friction_curve`

### Sim-to-real deployment through a gate

The export is ONNX with normalization folded in, plus a manifest read from
the built environment. The gate drives it through the manifest alone on two
runtimes, plain MuJoCo and Unitree's own simulator over DDS, and passes it
within a declared tolerance of its evaluation. A failed gate is attributed
to its cause, latency first. Pre-flight checks joint order, gains, ranges,
torques and compute before the first tick, and measures the stops.

<img alt="A deployment's page: provenance, the manifest, and the sim-to-sim gate's verdict" src="docs/figures/readme/studio-deployment.png">

`export_deployment` · `gate_deployment` · `attribute_deployment` · `preflight_deployment` · `stage_deployment`

### Simulation in a captured scene

MuJoCo inside the Studio, with the embedded [Rerun](https://rerun.io)
viewer beside it. A phone video becomes a scene: Gaussian splats for what
the camera sees, a collision proxy for what the feet touch, and the gap
between the two measured. Here the deployed Go2 walks a captured garden.

<img alt="The Studio's simulator: the deployed Go2 walking in a captured garden, with joint, command, force and contact plots" src="docs/figures/readme/studio-simulator-garden.png">

`capture_scene` · `import_scene` · `simulate_in_studio` · `control_simulator` · `screenshot_studio`

### Deployment telemetry back to training: the data flywheel

The deployed robot's telemetry is identified again and judged against
its fitted intervals. What left its interval is named, re-identification
is recommended, and the new fit becomes the model the next policy trains
on. Each turn adds recordings, fits, datasets, policies and evaluations,
all stamped and queryable, so the next turn starts from more data than
the last.

<img alt="A drift check's page: parameters out of interval, within interval and undetermined, with the recommendation" src="docs/figures/readme/studio-drift.png">

`check_drift` · `describe_project` · `describe_runs` · `list_ledger_findings`

## Robots

<img alt="Robots the loop has run in simulation: a Unitree Go2, a microduck biped, an SO-101 arm, a Robotiq 2F-85 gripper" src="docs/figures/readme/robots.png">

| Robot | What ran | From |
|---|---|---|
| Unitree Go2 | the whole loop: identify from public logs, train, evaluate, two gates, pre-flight, drift, a captured scene | Unitree's `unitree_rl_mjlab` model |
| microduck (biped) | walk training across fifteen randomization spans, the paper's walk study | Pollen Robotics' model, BAM actuator fits |
| SO-101 arm | the lift study and generated demonstrations | TheRobotStudio's model |
| Robotiq 2F-85 | imported from Isaac Sim USD through Newton, audited equal to the source | NVIDIA's Isaac Sim asset |

Your own robot enters by its MJCF or USD file (`onboard_robot`); the reply
says whether it is in the shape a task needs before anything trains.

## Measured, not claimed

| Result | Number | Record |
|---|---|---|
| Go2 walker certified on its declared world | 38 of 40, 95 % CI [0.83, 0.99] | [`dds-gate-go2-c2`](docs/68-findings.md#dds-gate-go2-c2-2026-09-13) |
| The same policy gated on Unitree's simulator over DDS | 20 of 20 | [`dds-gate-go2-c2`](docs/68-findings.md#dds-gate-go2-c2-2026-09-13) |
| A Go2 identified from a public chirp log (IIT) | 31 of 36 parameters pinned | [`go2-legged-fit-public-logs`](docs/68-findings.md#go2-legged-fit-public-logs-2026-09-24) |
| The Go2 walking a captured garden under full physics | 6 of 6, [0.54, 1.00] | [`go2-walks-the-captured-garden`](docs/68-findings.md#go2-walks-the-captured-garden-2026-09-23) |

Every number in this repository resolves to a record under
[`docs/findings/`](docs/findings/README.md), and a commit gate refuses a ratio
in the paper that no record carries. All results are in simulation: the
hardware step is the vendor's own runtime, and nothing here claims a real
deployment yet.

The method and its measurements are the paper,
[*How Wide a Span, and What Does Identification Buy?*](docs/paper/README.md)
(draft v3).

<details>
<summary><b>Why the verdicts can be trusted</b></summary>

The instrument was tested on itself and the misses are on the record
([`docs/68-findings.md`](docs/68-findings.md)): a fit that converged to twice the
truth with a tight interval until its anchor was audited, which is why a fit
record without an anchor statement is refused; a timestamp convention that
biased a fit, caught three more times afterwards, which is why truth
recovery on synthetic data gates every new excitation harness; a
randomization study whose arms were identical because the event never
fired, which is why the mjlab linter exists. Evaluations gate on the lower
confidence bound, never on the point estimate.

</details>

<details>
<summary><b>The loop, step by step</b>, with every tool</summary>

| Step | What happens | The tools | What it leaves |
|---|---|---|---|
| 1. Capture | telemetry from the robot or a public log, a phone video of the space | `start_capture`, `ingest_recording`, `ingest_public_log`, `capture_scene` | recordings with their rate, dropouts and licence state; scenes |
| 2. Real-to-sim | hardware proximity, measured: the robot's own telemetry fitted into the model it trains on, every parameter with an interval and a pinned / NOT PINNED verdict; the captured space with its see-versus-touch gap measured | `onboard_robot`, `identify_system`, `import_scene` | the robot bundle `name@hash`, the fit record, the scene record |
| 3. Build data | demonstrations generated under recorded randomization, kept by the task's success criterion, exported as LeRobot v3 with a datasheet | `generate_kitting_demos`, `generate_walk_demos`, `multiply_demos` | datasets with provenance |
| 4. Train | reinforcement learning on mjlab or imitation through LeRobot, on the identified model with a measured randomization span | `train_walk`, `run_chain` | experiments and policies |
| 5. Evaluate | a clip cannot tell 10 % from 99 %; an evaluation here is paired, seed-matched trials with an exact confidence interval and a milestone funnel, and the gate reads the lower bound | `evaluate_walk`, `list_eval_records`, `describe_eval` | evaluations that gate on the lower bound |
| 6. Sim-to-real | the deployment is the last evaluation: the export, a gate on two runtimes (ours and the vendor's) against the same rows at a declared tolerance, a failed gate attributed to its cause (latency first), pre-flight before the first tick | `export_deployment`, `gate_deployment`, `attribute_deployment`, `preflight_deployment` | deployments with their gate and pre-flight |
| 7. Deploy and watch | the policy on the vendor's runtime; its telemetry back into turn 1; the robot re-identified against its fitted intervals | `stage_deployment`, `check_drift` | drift records |
| 8. Query and learn | every record above, from the agent or the Studio: which policy, on which model, under which conditions, with what interval | `describe_project`, `describe_runs`, `friction_curve`, `list_ledger_findings` | the next turn's question |

</details>

<details>
<summary><b>What is in the repository</b></summary>

| Directory | Layer | What it is |
|---|---|---|
| [`trainnr/`](trainnr/README.md) | L0, Python package `trainnr` | bundles and their hash identity, system identification over `mujoco.sysid`, exact small-n statistics, the evaluation protocol, the deployment gate, drift, scenes, the project index, the MCP server and the `trainnr` command |
| [`trainnr-mjlab/`](trainnr-mjlab/README.md) | L1, Python package `trainnr_mjlab` | the mjlab trainer: identified actuator physics as an mjlab actuator, the randomization events, the linter that turns silent no-ops into errors, the walk tasks |
| [`crates/trainnr-studio/`](crates/trainnr-studio/) | L2, Rust | the Studio; talks to L0 through the project's files and processes only |
| [`robots/`](robots/) | data | the actuator library (BAM's fits, each with provenance) and the nominal robot bundles |
| [`tools/`](tools/) | scripts | the gates (`verify.sh`, `check-docs.py`, `check-numbers.py`, `check-layers.py`), the paper build, the cloud runbook |
| [`docs/`](docs/) | the record | dated research against primary sources, numbered decisions, the findings and the paper |
| [`.claude/`](.claude/), [`.claude-plugin/`](.claude-plugin/) | the plugin | the agents, the skills and the manifest that make this repository a Claude Code marketplace |

The layers never import upward (`tools/check-layers.py`). Robot and dataset
identity is `name@hash` everywhere; nothing is nameable without its hash.

</details>

## Community

- **Questions and ideas:** [Discussions](https://github.com/Trainnr-AI/trainnr/discussions).
- **Bugs, features, robot support:** [open an issue](https://github.com/Trainnr-AI/trainnr/issues/new/choose); the forms ask for what a maintainer needs.
- **Pull requests:** read [`CONTRIBUTING.md`](CONTRIBUTING.md); every commit is signed off (DCO).
- **Security:** report privately, never in an issue ([`SECURITY.md`](SECURITY.md)).
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) · [`GOVERNANCE.md`](GOVERNANCE.md) · [`CITATION.cff`](CITATION.cff) · [`CHANGELOG.md`](CHANGELOG.md)

## Licence

[FSL-1.1-ALv2](LICENSE): any use other than a competing product or service, and each version under Apache-2.0 two years after it is made available. Third-party material is listed in [`NOTICE`](NOTICE). The 2025–26
rig this toolchain grew up on lives in its own archive,
[`Trainnr-AI/rig`](https://github.com/Trainnr-AI/rig). "Robotiq" in
`robots/robotiq-2f85-isaac` is Robotiq Inc.'s gripper; trainnr is not
affiliated with Robotiq.
