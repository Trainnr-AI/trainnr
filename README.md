# trainnr

**Identify, train, evaluate, gate and monitor robot policies, from your coding
agent.** trainnr is the real-to-sim-to-real pipeline as one MCP server, a
desktop app and a Claude Code plugin: the robot's actuators are identified
from its own telemetry into a content-hashed bundle, policies are trained on
the identified model with a measured randomization span, evaluated with
paired trials and exact confidence intervals, gated against the deployment
stack before any hardware is touched, and watched for drift afterwards.
Every stage leaves a record that cites the records it was made from, by
version.

The one-paragraph argument, with the measurements behind it, is the paper:
[*How Wide a Span, and What Does Identification Buy?*](docs/paper/manuscript.md)
(draft v3; every number in it resolves to a record under
[`docs/findings/`](docs/findings/)). The positioning, with evidence per
claim, is [`docs/33-what-we-say.md`](docs/33-what-we-say.md).

## The pipeline, on a Unitree Go2, in one conversation

With the plugin installed, this is the whole loop an agent walks through the
MCP tools, as a new user's agent did on 2026-09-28
([the record](docs/07-progress-log.md)):

| Stage | Tool | What it leaves behind |
|---|---|---|
| Project | `create_project_dir` | a project directory with its index |
| Asset | `onboard_robot` (Unitree's `go2.xml`) | the robot bundle `go2@<hash>` |
| Telemetry | `ingest_public_log` or `ingest_recording` | a recording with its rate, dropouts and licence state |
| System identification | `identify_system` | the fit: each parameter with an interval and a pinned / NOT PINNED verdict |
| Environment | `create_task` (`trainnr/go2-walk`) | the declared task, accepted or refused |
| Policy | `train_walk(fit=…)` | an mjlab experiment on the identified model, with its randomization span |
| Evaluation | `evaluate_walk` | paired trials, exact intervals, the funnel |
| Deployment | `export_deployment`, `gate_deployment` | the ONNX export and a sim-to-sim gate at the trained command envelope |
| Pre-flight | `preflight_deployment` | the checks before the first tick on a robot, soft stop measured |
| Monitoring | `check_drift` | the robot re-identified from a new recording against its fitted intervals |

The desktop app shows each stage as it lands: the Overview's pipeline strip,
the robot and its fit, the experiment's curves, the evaluation's intervals,
the deployment gate, and the simulator with the policy driving the robot.

![The Overview after the loop closed](docs/figures/go2/overview-after-loop.png)

## Install

**Claude Code** (the plugin: the MCP server, seven agents, two skills):

```sh
claude plugin marketplace add Trainnr-AI/trainnr
claude plugin install trainnr@trainnr
```

The tools then appear as `mcp__trainnr_trainnr__<tool>`; ask the agent to
"create a project and onboard the Go2" and it starts at the top of the table.

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

**The desktop app** (Rust; the embedded Rerun viewer and the MuJoCo simulator):

```sh
cd crates/trainnr-desktop && cargo build --release && cd ../..
uv run --directory trainnr trainnr desktop        # or the launch_studio tool
```

**Training** needs the mjlab trainer and a CUDA GPU:
`cd trainnr-mjlab && uv sync --extra viz`. Everything else, including every
test, runs on a laptop without a GPU.

## What is in the box

| Directory | Layer | What it is |
|---|---|---|
| [`trainnr/`](trainnr/README.md) | L0, Python package `trainnr` | bundles and their hash identity, system identification over `mujoco.sysid`, exact small-n statistics, the evaluation protocol, the deployment gate, drift, scenes, the project index, the MCP server and the `trainnr` command |
| [`trainnr-mjlab/`](trainnr-mjlab/README.md) | L1, Python package `trainnr_mjlab` | the mjlab trainer: identified actuator physics as an mjlab actuator, the randomization events, the linter that turns silent no-ops into errors, the walk tasks |
| [`crates/trainnr-desktop/`](crates/trainnr-desktop/) | L2, Rust | the desktop app; talks to L0 through the project's files and processes only |
| [`robots/`](robots/) | data | the actuator library (BAM's fits, each with provenance) and the nominal robot bundles |
| [`tools/`](tools/) | scripts | the gates (`verify.sh`, `check-docs.py`, `check-numbers.py`, `check-layers.py`), the paper build, the cloud runbook |
| [`docs/`](docs/) | the record | dated research against primary sources, numbered decisions, the progress log, the findings and the paper |
| [`.claude/`](.claude/), [`.claude-plugin/`](.claude-plugin/) | the plugin | the agents, the skills and the manifest that make this repository a Claude Code marketplace |

The layers never import upward (`tools/check-layers.py`). Robot and dataset
identity is `name@hash` everywhere; nothing is nameable without its hash.

## Why the verdicts can be trusted

The instrument was tested on itself and the misses are on the record
([`docs/07`](docs/07-progress-log.md)): a fit that converged to twice the
truth with a tight interval until its anchor was audited, which is why a fit
record without an anchor statement is refused; a timestamp convention that
biased a fit, caught three more times afterwards, which is why truth
recovery on synthetic data gates every new excitation harness; a
randomization study whose arms were identical because the event never
fired, which is why the mjlab linter exists. Evaluations gate on the lower
confidence bound, never on the point estimate.

## Contributing, security, citing

[`CONTRIBUTING.md`](CONTRIBUTING.md) (Developer Certificate of Origin
sign-off, the gates, where decisions are written),
[`SECURITY.md`](SECURITY.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md),
[`GOVERNANCE.md`](GOVERNANCE.md), [`CITATION.cff`](CITATION.cff). The 2025–26
rig that this toolchain grew up on lives in its own archive,
[`Trainnr-AI/rig`](https://github.com/Trainnr-AI/rig); its tour is
[`docs/27-rig-tour.md`](docs/27-rig-tour.md).

[FSL-1.1-ALv2](LICENSE): any use other than a competing product or service, and each version under Apache-2.0 two years after it is made available. Third-party material is listed in [`NOTICE`](NOTICE).
"Robotiq" in `robots/robotiq-2f85-isaac` is Robotiq Inc.'s gripper; trainnr
is not affiliated with Robotiq.
