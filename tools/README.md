# tools/

The lab bench. Shell tools run from the repo root; Python tools run
from `pipeline/` so uv picks up its environment:

```sh
cd pipeline && uv run --extra sim python ../tools/<name>.py
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
| `show-rig.py`, `show-yellow.py`, `show-aloha2.py`, `show-cargo-chaos.py` | One scene each, MuJoCo viewer + Rerun (the house rule: every session gets both) |
| `show-many.py` | Batched domain-randomized worlds side by side |
| `kitting-demos.py` | T5's data source: referee-filtered scripted kitting episodes with DR, trajectories + frames + manifests |
| `camera-match.py` | Sim renders beside released real frames — camera placement is calibration, not decoration |

## Fitting and studies

| Tool | What it does |
|---|---|
| `solver-study.py` | Constraint-solver sweep on the kitting scene: solver x cone x integrator, judged by the referee |
| `fit-report.py` | A bundle's fit records: intervals, cross-run spread, EXCEEDS verdicts |
| `sts-study.py` | The STS3215 benchmark ingest (YouTube-sourced) → parameter fits |
| `sts-figure.py` | The study's figure, from `sts-study.py`'s JSON |
| `train-watch.py`, `rl-watch.py` | Watch a training run / RL policy roll out live in Rerun |
| `debug-inference.sh` | Checkpoint inference smoke on the train venv |
