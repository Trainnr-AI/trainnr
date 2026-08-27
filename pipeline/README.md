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
| `rq_pipeline/stats` | The honesty layer: exact binomial intervals, Fisher-z rank-correlation intervals, top-pick probability, and per-factor main effects by Fisher's exact test (`effects`). Dependency-free — a signed report must be recomputable anywhere. |
| `rq_pipeline/bundles` | Hash-stamped artifact identity (`name@hash`, `require_stamp` the one rule) and where bundles live (`RQ_ROBOTS_DIR`). Calibration is device state; nothing is nameable without its hash. |
| `rq_pipeline/protocol.py` | The vocabulary every layer shares — `EpisodeProtocol`, milestones, `Placement` (where a body starts), `CameraSpec` — standard library only, so a third-party task package imports this and nothing heavier. |
| `rq_pipeline/physics` | Engines by name (`registry.py`, the `rq_pipeline.engines` entry-point group): `mujoco` — CPU MuJoCo, the metrology reference (load, census, keyframes, batched sysid rollouts, `Stepper`) — and `mjx-warp` — batched rollouts and a batched stepper on MJX-Warp, float32, admitted through the gauntlet in `tests/test_mjx_backend.py`. Start validation on the model's own geometry (`placement.py`: a declared start is admissible or refused by name before a trial is spent). The rollout contract, the FULLPHYSICS row layout and `instrument_stamp` (engine, version, CPU architecture) shared in `backend.py`. |
| `rq_pipeline/robot` | Fail-loudly model gates, fit records with spread verdicts, damped-least-squares arm IK, `mujoco.sysid` identification. A USD import that silently drops actuators is refused, not discovered in week three. |
| `rq_pipeline/collect` | The wire in: status-line parsing, frame assembly, CSV/excitation ingest, LeRobot dataset export. |
| `rq_pipeline/tasks` | Scene builders + episode protocols, self-registering (`@register`, the `rq_pipeline.tasks` entry-point group for plugins): SO-101 (reach/lift/stack/insert), ALOHA 2 (transfer, kitting + its scripted demo generator), the mobile-manipulator rig. `KittingSpec` is the kitting task as data (`Task.stamp` names it); `acceptance.py` accepts a task only when its scripted expert passes every paired trial and the do-nothing floor passes none. |
| `rq_pipeline/evaluate` | The judge: paired-trial protocols with milestone chains, the per-trial record and its fold, variations drawn by trial index, camera rigs as data, sim↔real certificates. `scheduler.py`: a chunk-predicting policy executed on the protocol's own horizon, every fetch checked. |
| `rq_pipeline/envs` | The ecosystem's door: every registered task as a gymnasium env (`gym.make("robotiq/kitting-v0")`) with paired starts through the seed and its contract strings in one place (`contract.py`), plus the LeRobot `EnvConfig` that lets `lerobot-eval` and `lerobot-train` run our rollouts (`--env.type=robotiq --env.task=<id> --env.discover_packages_path=rq_pipeline.envs`) and LeRobot's policy loader for tools. The judge stays in `evaluate`. `openpi_policy.py`: a policy served by openpi as a chunk policy, over their own client. |

The layer chain — `stats ← bundles, protocol, robot.model_checks ←
evaluate ← physics, tasks, collect, robot (sysid) ← envs ← tools` — is
pinned by `tests/test_layers.py` over every import, lazy ones included.

## Develop

Everything runs through [uv](https://docs.astral.sh/uv/) — it provisions the
interpreter too, so the only prerequisite is uv itself.

```sh
cd pipeline
uv run python -m unittest discover -s tests   # fast suite, < 5 s; heavy roundtrips run under --extra sim/train
uvx ruff format --check . && uvx ruff check . # the same gate pre-commit runs
```

Optional extras pull the heavy stages when a machine can carry them:
`uv sync --extra sim` (MuJoCo 3.11.x — pinned, the engine version is part of
the identified artifact, docs/e2e-research/49 — and gymnasium), `--extra train`
(LeRobot, needs Python ≥ 3.12), `--extra mjx` (MJX-Warp, the batched second
engine; runs on Warp's CPU backend anywhere, fast on an NVIDIA GPU),
`--extra gpu` (MuJoCo Warp itself, NVIDIA only), `--extra remote` (openpi's
websocket client, for pi0 / pi0.5 served from their process), `--extra viz`
(the Rerun viewer the tools log to).

Two environment facts that cost an afternoon each, recorded so they
cost nothing again:

- **Training lives in a second venv.** LeRobot 0.6.1 needs Python 3.12,
  and `uv sync` prunes anything it didn't install — so the WSL box keeps
  `.venv-train` (3.12 + torch/CUDA + these packages) beside the untouched
  3.11 sim venv, selected with `--python .venv-train/bin/python`.
- **The WSL box's GPU routing is one file, `wsl.env`** (`uv run --env-file wsl.env …`
  or `../tools/wsl-run.sh`). Headless rendering there needs `MUJOCO_GL=egl`. Without it the
  Renderer wants a display and vision rollouts die in the harness, not
  in your code.
