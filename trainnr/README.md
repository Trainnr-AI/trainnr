# trainnr

<!-- mcp-name: io.github.trainnr-ai/trainnr -->

The Python package behind the end-to-end robotics platform: identify a
robot's dynamics from its telemetry, generate datasets, train policies,
evaluate them with exact confidence intervals, gate and deploy them, and
watch the deployment's telemetry for drift. Every step is an MCP tool
and every result is a hash-stamped record the next step cites.

The premise: **the robot is an artifact, not an import.** A robot enters
as a bundle `name@hash` (canonical MJCF or a USD asset read by Newton,
measured dynamics with intervals, provenance), and every stage codes
against that bundle, never against an embodiment.

The architecture and its decisions:
[`docs/22-pipeline-architecture.md`](../docs/22-pipeline-architecture.md);
the loop this package serves: [`docs/76-the-loop.md`](../docs/76-the-loop.md).

## Layout

Subpackages under `trainnr/trainnr/`, lowest layer first.

| Package | What it holds |
|---|---|
| `stats` | The honesty layer: exact binomial intervals, rank-correlation intervals, top-pick probability, per-factor main effects by Fisher's exact test. Standard library only, so a signed report is recomputable anywhere. |
| `bundles` | Hash-stamped artifact identity (`name@hash`, `require_stamp`) and where bundles live (`TRAINNR_ROBOTS_DIR`). Nothing is nameable without its hash. |
| `protocol.py` | The vocabulary every layer shares: `EpisodeProtocol`, milestones, `Placement`, `CameraSpec`. Standard library only, so a third-party task package imports this and nothing heavier. |
| `physics` | Engines by name (`trainnr.engines` entry points): CPU MuJoCo as the metrology reference and MJX-Warp as the batched throughput instrument; start-state validation on the model's own geometry; the rollout contract and `instrument_stamp`. |
| `robot` | The robot as a measured artifact: onboarding from MJCF or USD with an import audit, fail-loudly model gates, system identification over `mujoco.sysid` with fit records, intervals and pinned / NOT PINNED verdicts, identification methods by name. |
| `robots` | How a real robot's telemetry enters: recording adapters (`wire`, `lerobot`, `mcap`, `rosbag2`, Unitree DDS capture), joint-order maps, the public-log registry. |
| `collect` | Recordings become datasets with provenance: the demonstration press, shards, LeRobot v3 export, datasheets. |
| `tasks` | Scenes built as programs over the robot bundle, with their episode protocols and scripted experts, self-registering through `trainnr.tasks` entry points; task declaration as data (`Task.stamp`), overlays, the acceptance critic. |
| `evaluate` | The judge: paired trials with milestone chains, per-trial records and their fold, variations drawn by trial index, camera rigs as data, certificates, finding records, figures. |
| `envs` | The ecosystem's door: every registered task as a gymnasium env, the LeRobot `EnvConfig` (`--env.type=trainnr`), LeRobot's policy loader, an openpi chunk policy over its own client. |
| `deploy` | A policy exported as ONNX with a manifest a runtime drives it from; the sim-to-sim gate on plain MuJoCo and on Unitree's simulator over DDS; attribution of a failed gate to its cause; pre-flight; the deployment mirror into the viewer. |
| `fleet` | After deployment: drift judged against the robot's identified intervals from fresh telemetry. |
| `scenes` | Captured scenes: the Gaussian splat the cameras see, the collision proxy the solver touches, and the record of the gap between them. |
| `rl` | Learning on top of certified policies: SmoothRL's objective transcribed, the residual and replay pieces. |
| `project` | One directory per effort: the manifest, the artifact index the Studio reads, the presenter, the Studio's control files, previews. |
| `cloud` | Rented GPUs behind one seam: the provider contract and registry, RunPod first, the transfer rules (never `.env`, never private files). |
| `mcp_server.py`, `mcp_actions.py`, `mcp_jobs.py` | The MCP surface: the describe tools, the acting tools (each spawns the CLI that owns the work), and job handles for the long-running ones. |
| `cli.py` | The `trainnr` command. |
| `plugins.py`, `paths.py`, `viz.py` | The plugin door both registries share; where the checkout is; the mesh-true Rerun mirror of any MuJoCo model. |

The layer rule, pinned by `tests/test_layers.py` inside the package and by
`tools/check-layers.py` across packages: `stats` ← `bundles`, `protocol` ←
`evaluate` ← `physics`, `tasks`, `collect`, `robot` ← `envs` ← tools. This
package never imports `trainnr_mjlab` or the Studio.

## Tasks

Registered task families (`describe_task_families` lists them):

| Family | Robot | What it is |
|---|---|---|
| `trainnr/lift-study` | SO-101 | the lift at a declared randomization span, the study's unit |
| `trainnr/kitting` | ALOHA 2 | parts into tray slots, with the scripted expert that presses demonstrations |
| `trainnr/gripper-pick` | Robotiq 2F-85 (USD-born) | a pick with the acceptance ladder's rungs |
| `trainnr/go2-walk` | Unitree Go2 | mjlab's velocity task on the onboarded Go2, trained through `trainnr-mjlab` |
| `trainnr/microduck-walk` | microduck | the biped's walk through the certified actuator stack |
| `trainnr/go1-walk` | Unitree Go1 | mjlab's own Go1 task with the study's span knob |

The SO-101 reach, lift, block-stack and tool-insert scenes and the ALOHA 2
transfer-cube scene are task builders in the same registry.

## The command and the plugin doors

```sh
trainnr mcp        # serve the tools over stdio (what an agent's config runs)
trainnr studio     # launch the Studio on the current project
trainnr version    # the installed version
```

Six entry-point groups let a third party extend the platform from its own
package, without a fork:

| Group | What registers | Built-in example |
|---|---|---|
| `trainnr.tasks` | a task module | `so101`, `aloha2` |
| `trainnr.engines` | a physics engine over the same MJCF | `mujoco`, `mjx-warp` |
| `trainnr.gpu_providers` | a rented-GPU vendor | `runpod` |
| `trainnr.robot_adapters` | a telemetry format | `wire`, `lerobot`, `mcap`, `rosbag2` |
| `trainnr.model_sources` | a robot model format for onboarding | `usd` |
| `trainnr.identification_methods` | a system-identification method | `drivetrain-ratio` |

A seventh, `trainnr.mcp_tools`, lets another package add tools to the MCP
server (`register_plugin_tools`).

## Develop

Everything runs through [uv](https://docs.astral.sh/uv/), which provisions
the interpreter too.

```sh
cd trainnr
uv run python -m unittest discover -s tests   # the suite; heavy parts skip without their extra
uvx ruff format --check . && uvx ruff check . # the same gate pre-commit runs
```

The base install is dependency-light on purpose: the statistics and bundle
layers must be recomputable anywhere a report is audited. The stages that
need more are extras:

| Extra | What it brings |
|---|---|
| `sim` | MuJoCo (compatible-release pinned: the engine version is part of the identified artifact) and gymnasium |
| `train` | LeRobot with dataset, SmolVLA and training support; Python 3.12 or newer |
| `mjx` | MJX with the Warp implementation, the batched second engine; CPU anywhere, fast on NVIDIA |
| `gpu` | MuJoCo Warp itself; NVIDIA only |
| `viz` | the Rerun SDK and TensorBoard's event reader (the presenter reads the file the trainer writes) |
| `viz-query` | reading a saved viewer recording back as columns (Rerun with DataFusion) |
| `scene` | Open3D and CoACD for captured scenes: the surface-to-proxy audit and collision proxies |
| `deploy` | ONNX Runtime and onnx for the manifest-driven runtime and the gate |
| `mcp` | the MCP SDK and psutil (the Studio's process liveness) |
| `remote` | openpi's websocket client, pinned to a commit; not on PyPI |
| `dds` | cyclonedds, evdev and unitree_sdk2py for the DDS gate against Unitree's simulator |
| `usd` | Newton's importers for USD assets |

`mjx` and `usd` conflict (Newton's release and our engine pin disagree on
mujoco-warp); a USD import happens once, at onboarding, and the bundle it
writes runs on every extra. LeRobot wants Python 3.12 while `sim` is
measured on 3.11, so training commonly lives in a second environment
(`.venv-train`) beside the sim one.

On WSL2 the GPU routing and headless rendering variables live in one file,
`wsl.env` (`uv run --env-file wsl.env …` or `../tools/wsl-run.sh`); its
header says what each line does. Harmless on native Linux, unnecessary on
macOS and Windows.
