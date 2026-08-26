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
| `rq_pipeline/physics` | CPU MuJoCo: load, census, keyframes, batched sysid rollouts, and `Stepper` — the one stepping loop every episode runs on. A GPU engine would arrive as another env over the same tasks. |
| `rq_pipeline/robot` | Fail-loudly model gates, fit records with spread verdicts, damped-least-squares arm IK, `mujoco.sysid` identification. A USD import that silently drops actuators is refused, not discovered in week three. |
| `rq_pipeline/collect` | The wire in: status-line parsing, frame assembly, CSV/excitation ingest, LeRobot dataset export. |
| `rq_pipeline/tasks` | Scene builders + episode protocols: SO-101 (reach/lift/stack/insert), ALOHA 2 (transfer, kitting + its scripted demo generator), the mobile-manipulator rig. |
| `rq_pipeline/evaluate` | The harnesses: paired-trial scoring over sensors or pixels, ArmnetBench camera rig, sim↔real certificates. |
| `rq_pipeline/envs` | The ecosystem's door: every task as a gymnasium env (`gym.make("robotiq/kitting-v0")`) with paired starts through the seed, plus the LeRobot `EnvConfig` that lets `lerobot-eval` and `lerobot-train` run our rollouts (`--env.type=robotiq --env.discover_packages_path=rq_pipeline.envs`). The judge stays in `evaluate`. |

## Develop

Everything runs through [uv](https://docs.astral.sh/uv/) — it provisions the
interpreter too, so the only prerequisite is uv itself.

```sh
cd pipeline
uv run python -m unittest discover -s tests   # fast suite, < 5 s; heavy roundtrips run under --extra sim/train
uvx ruff format --check . && uvx ruff check . # the same gate pre-commit runs
```

Optional extras pull the heavy stages when a machine can carry them:
`uv sync --extra sim` (MuJoCo ≥ 3.5 and gymnasium), `--extra train` (LeRobot, needs
Python ≥ 3.12), `--extra gpu` (MuJoCo Warp, needs an NVIDIA GPU),
`--extra viz` (the Rerun viewer the tools log to).

Two environment facts that cost an afternoon each, recorded so they
cost nothing again:

- **Training lives in a second venv.** LeRobot 0.6.1 needs Python 3.12,
  and `uv sync` prunes anything it didn't install — so the WSL box keeps
  `.venv-train` (3.12 + torch/CUDA + these packages) beside the untouched
  3.11 sim venv, selected with `--python .venv-train/bin/python`.
- **Headless rendering on WSL needs `MUJOCO_GL=egl`.** Without it the
  Renderer wants a display and vision rollouts die in the harness, not
  in your code.
