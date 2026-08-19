# rq-pipeline

Train, validate and serve robot policies **with numbers you can sign**:
identified dynamics, gated simulation, confidence intervals on everything.

The platform premise: **the robot is an artifact, not an import.** Any robot
enters as a hash-stamped bundle — canonical MJCF, measured dynamics with
intervals, calibration state, an interface adapter — and every pipeline stage
codes against that bundle, never against an embodiment. That is what makes
"anyone can come and train their robot" a property of the architecture instead
of a roadmap item.

Architecture, stage roadmap and the design decisions behind them:
[`docs/22-pipeline-architecture.md`](../docs/22-pipeline-architecture.md).

## Layout

| Package | What it is |
|---|---|
| `rq_pipeline/stats` | The honesty layer: exact binomial intervals, Fisher-z rank-correlation intervals, top-pick probability. Dependency-free — a signed report must be recomputable anywhere. |
| `rq_pipeline/bundles` | Hash-stamped artifact identity (`name@hash`). Calibration is device state; nothing is nameable without its hash. |
| `rq_pipeline/physics` | The backend protocol. MuJoCo is the default (CPU, first-party sysid); Newton slots in behind the same protocol for GPU rollouts. |
| `rq_pipeline/robot` | Fail-loudly model gates: a USD import that silently drops actuators is refused, not discovered in week three. |

## Develop

Everything runs through [uv](https://docs.astral.sh/uv/) — it provisions the
interpreter too, so the only prerequisite is uv itself.

```sh
cd pipeline
uv run python -m unittest discover -s tests   # fast suite, < 5 s; heavy roundtrips run under --extra sim/train
uvx ruff format --check . && uvx ruff check . # the same gate pre-commit runs
```

Optional extras pull the heavy stages when a machine can carry them:
`uv sync --extra sim` (MuJoCo ≥ 3.5), `--extra train` (LeRobot, needs
Python ≥ 3.12), `--extra gpu` (MuJoCo Warp, needs an NVIDIA GPU).
