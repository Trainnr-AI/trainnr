# tools/

The lab bench. Shell tools run from the repo root; Python tools run
from `pipeline/` so uv picks up its environment:

```sh
cd pipeline && uv run --extra sim --extra viz python ../tools/<name>.py
# on the WSL box, the GPU routing lives in ONE file:
cd pipeline && uv run --env-file wsl.env --extra sim --extra viz python ../tools/<name>.py
# tools that need the train venv (LeRobot):
cd pipeline && ../tools/wsl-run.sh .venv-train/bin/python ../tools/<name>.py
```

`_lab.py` is the shared bench (path bootstrap, Rerun session plumbing)
and `_rig3d.py` the shared MuJoCo→Rerun mirror; neither is a tool.

## Gates — run these before pushing

| Tool | What it proves |
|---|---|
| `verify.sh` | Everything, one command: Rust fmt/clippy/tests, doc and unsafe gates, the Python pipeline (ruff + unit suite), every firmware variant, wire replays, the emulator HIL. `--serial <port>` adds real silicon |
| `check-docs.py` | No doc names code that no longer exists |
| `check-unsafe-gates.py` | `unsafe` stays forbidden in every crate |
| `coverage.sh` | Rust line coverage |
| `loc-report.py` | Line counts by area |

## Rig sessions (hardware on the desk)

| Tool | What it does |
|---|---|
| `rig-rerun.py` | THE session viewer: live wire → Rerun dashboard + MuJoCo twin |
| `udp-wire-bridge.py` | WiFi telemetry (UDP 9870) → the same `.wire` stream the USB path writes |
| `replay-errand.py` | A recorded `.wire` session through the same viewers, offline |
| `build-robot.sh`, `build-pico2.sh` | Firmware images (RP2040 / RP2350 + feature variants) |
| `sim-*.sh`, `setup-emulator.sh` | The rp2040js emulator harness and per-peripheral smoke runs |
| `sim-errand.py`, `sim-hil.sh` | The errand mission in pure sim / HIL against the emulator |

## Sim sessions (no hardware)

| Tool | What it does |
|---|---|
| `show-aloha2.py`, `show-cargo-chaos.py` | One scene each, MuJoCo viewer + Rerun (the house rule: every session gets both) |
| `show-rig.py`, `show-yellow.py` | The car+arm rig and the yellow arm in the MuJoCo passive viewer only — their Rerun mirror is queued (docs/32 §10.1) |
| `show-many.py` | Batched domain-randomized worlds side by side |
| `kitting-demos.py` | T5's data source: referee-filtered scripted kitting episodes drawn over the task's whole declared band with ±30% DR — trajectories + frames + manifests, each manifest carrying the expert's own stamp (`expert_stamp`); `--first-episode K` shards one batch across parallel generators |
| `camera-match.py` | Sim renders beside released real frames — camera placement is calibration, not decoration |
| `studio-render-stream.py` | MuJoCo's own render as a subprocess service for `crates/studio-shell`: frames out on stdout, orbit/zoom/size commands in on stdin — the app's 3D viewport without linking MuJoCo's C API into Rust |
| `mcp-server.py` | The instrument's MCP surface over stdio: bundles (name@hash, fit records, SPREAD), the actuator library, task/engine registries, run manifests — read-only tools for any MCP client; the Studio's agent panel and `.mcp.json` both point here |
| `studio-instrument-view.py` | The instrument's records into the Studio's embedded Rerun viewer, one shot with a blueprint: eval funnels as bar charts with readings, fit parameters as estimate+interval series per sweep with SPREAD verdicts, the STS3215 M1-vs-M6 friction budget on a real velocity axis |
| `gen-app-icon.py` | Generates the Studio's dock icon (`crates/studio-shell/assets/icon-256.rgba` + PNG preview) — the committed asset's provenance; re-run and commit both together if the mark changes |

## Fitting and studies

| Tool | What it does |
|---|---|
| `cloud-gpu.py` | A rented GPU as a runbook: `offers` (priced, in stock, CUDA ≥ 13 hosts) → `launch` → `push` (rsync, never `.env`) → `bootstrap` (the train venv, the WSL recipe) → `run` → `pull` → `terminate`; every vendor behind `rq_pipeline/cloud`'s provider seam, Runpod first (docs/34); `follow <id> NAME --watch` mirrors a running pod run's light files every 20 s and opens the dashboard on them |
| `train-watch.py --follow <runs/NAME-act>` | The run as a live Rerun dashboard, from its files (no torch needed): the chain's run manifest as a text panel — every parameter and the dataset's stamps — `train/*` (loss, grad norm, lr, samples, update/dataloading seconds), `eval/inloop/*` and `eval/final/*` (success rate and the milestone funnel per checkpoint, the eval videos), `machine/gpu_*` (nvidia-smi samples), the trainer's resolved config at each checkpoint; `--rrd FILE` saves the stream; `--play-checkpoints` adds both viewers (train venv). Works on a mirrored pod run (`cloud-gpu follow`) |
| `e2e-smoke.py` | The T5 chain in one command: demos → LeRobot v3 → `lerobot-train` with in-loop eval through our env → `lerobot-eval` with records → the fold. `--scale smoke` (default: minutes on the WSL card) or `--scale cloud` (the ACT sim recipe: 50 demos, 100k steps, checkpoints every 20k, 20 paired starts); any knob overrides its preset; `--from <stage>`/`--until <stage>` run any contiguous slice of the four stages — the CPU-bound demos and convert on this box, the GPU-bound train and eval on a rented card (docs/34); it writes `runs/NAME-watch/` from its first second (the run manifest, every stage's output teed into `chain.log`, nvidia-smi samples) for the dashboard |
| `determinism-probe.py` | Is MJX-Warp bit-repeatable, at what cost? Subprocess-per-mode (compile option); the verdict needs the WSL CUDA device |
| `accept-task.py` | The critic loop as a command: the scripted expert must pass a task's referee on every paired trial and the do-nothing floor must pass none; prints the verdict, the funnel and every reason, exits 1 on rejection (`--tray-y`, `--in-slot-xy` compose kitting variants) |
| `solver-study.py` | Constraint-solver sweep on the kitting scene: solver × cone × integrator plus an impratio sweep, judged by the referee, with penetration, solver iterations, peak pad slip and peak grip force per row |
| `fit-report.py` | A bundle's fit records: intervals, cross-run spread, EXCEEDS verdicts |
| `sts-study.py` | The STS3215 benchmark ingest (YouTube-sourced) → parameter fits |
| `sts-figure.py` | The study's figure, from `sts-study.py`'s JSON |
| `sync-bam-actuators.py` | Vendor new/changed servos from a BAM checkout or repomix pack into `robots/actuators/` — refuses a changed file with no `--version` |
| `train-watch.py`, `rl-watch.py` | Watch a training run / RL policy roll out live in Rerun |
| `debug-inference.sh` | Runs any command with ONNX Runtime's and Rust's logging gates open (`ORT_LOG`, `RUST_LOG`) so execution-provider diagnostics show — e.g. `cargo run -p vision --bin bench` |

## Repo plumbing (not tools, but they live here)

| File | What it is |
|---|---|
| `_lab.py`, `_rig3d.py` | The shared bench and the MuJoCo→Rerun mirror the Python tools import |
| `_firmware.sh` | `build_uf2 <crate>`: asks cargo where a firmware binary landed, sourced by the `sim-*.sh` scripts |
| `wsl-run.sh` | Runs a command under `pipeline/wsl.env`, the WSL box's GPU routing in one file |
| `setup-hooks.sh`, `hooks/pre-commit` | Installs and is the pre-commit gate (fmt, clippy, docs, unsafe, ruff over `pipeline/` and `tools/`, the unit suite) |
| `harness/*.ts`, `patches/` | The rp2040js emulator harnesses and the patch `setup-emulator.sh` applies |
| `.ruff.toml` | Extends the pipeline's lint contract to `tools/`, with the per-file exceptions and their reasons |

