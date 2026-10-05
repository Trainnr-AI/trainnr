# trainnr_mjlab

The platform's reinforcement-learning trainer: the walk tasks on
[mjlab](https://github.com/mujocolab/mjlab), trained on the robot the
project onboarded and identified, with the measured actuator physics and
its intervals as the randomization basis. It is the layer `trainnr`'s
`train_walk`, `evaluate_walk`, `export_deployment` and related tools spawn;
it imports `trainnr` and nothing imports it back.

Its own package and its own environment on purpose: mjlab pins torch, CUDA
and the MuJoCo release it was built against, a third instrument beside the
sim and train environments. The pin is exact (`mjlab==1.6.0`) and an
upgrade is one deliberate change. Since 2026-09-22 `pyproject.toml`
overrides mjlab's `mujoco~=3.11` to `mujoco==3.13.0` and
`mujoco-warp==3.13.0` (the Gaussian-splat ray tracer the scene loop renders
with); every walk certificate names the instrument it was judged on.

## Modules

| Module | What it does |
|---|---|
| `actuator.py`, `kernel.py` | BAM's servo law (Rhoban, ICRA 2025, Apache-2.0) as an mjlab actuator, the batched torch kernel ported once, built only from a verified bundle |
| `bundle.py` | the refusal at the door: actuator bundles are read from the certified store `robots/actuator-bundles/` and verified; a tampered stamp refuses by name |
| `firmware.py` | the servo firmwares' own constants, one copy |
| `entity.py` | `entity_from_bundle`: a hash-stamped robot bundle under `robots/` as an mjlab `EntityCfg` |
| `dr.py` | `dr_from_bundle`: domain randomization with its basis on the record, ranges from the bundle's identified intervals or a declared span |
| `events.py` | the per-world field-expansion event, made unforgettable: forgetting it fails loudly at the actuator's `initialize` |
| `linter.py` | the DR no-op linter: a term that randomizes a field the actuator overwrites every step is a refusal, not a silent no-op |
| `walks.py` | the walk specs by robot, so the tools take `--robot` instead of a copy each |
| `go2_walk.py`, `microduck_walk.py`, `go1_walk.py` | the three walk tasks: mjlab's velocity task on the onboarded Go2, microduck's walk through the certified stack, mjlab's own Go1 task with the span knob |
| `fit_walk.py` | a walk trained under a measured fit: joints set at the fit's estimates and randomized over its intervals |
| `lag_dr.py` | command lag as a training randomization, on any walk |
| `scene_stage.py` | a captured scene's heightfield as the walk's terrain |
| `envelope.py` | the command envelope a checkpoint trained under, read from its curriculum stage |
| `sim_options.py` | mujoco_warp's overflow warnings as the walks want them |
| `walk_train.py` | training: smoke or the full recipe, checkpoints and the TensorBoard events the presenter reads |
| `walk_verdict.py` | the locomotion certificate: seeded paired trials, tracking and fall counts, exact intervals, the instrument on every row; a vision student can be judged through the policy bridge |
| `walk_export.py` | the actor as ONNX with normalization folded in, plus the manifest a runtime drives it from |
| `walk_press.py` | an RL teacher presses demonstrations into a dataset |
| `walk_play.py`, `walk_view.py`, `walk_stills.py` | a checkpoint rolled out in the viewers, in the Studio's simulator, or as one still per checkpoint |
| `recorder.py` | a Rerun `RecorderTerm`: mjlab rollouts streaming to the viewer during training |
| `reward_preview.py` | every reward term streamed per step under a controller that has learned nothing, before a run is paid for |
| `demo_recorder.py` | mjlab's own cartpole narrating itself into the Studio, the recorder's smoke |

## Usage

The MCP tools build these command lines (`trainnr/trainnr/mcp_actions.py`);
they run from a checkout the same way. On WSL2 prefix with
`--env-file ../trainnr/wsl.env`.

```sh
cd trainnr-mjlab && uv sync --extra viz

# train: agent `smoke` is minutes and saves nothing, `g3` is the full recipe
# (the train_walk tool's recipe="smoke" and recipe="full")
uv run python -m trainnr_mjlab.walk_train --agent g3 --robot go2 \
    --project ../projects/<name> --iterations 1500 [--fit <fit@hash>] [--dr-span 0.10]

# evaluate a checkpoint: paired trials, exact intervals
uv run python -m trainnr_mjlab.walk_verdict ../projects/<name>/runs/<run>/model_1499.pt \
    --trials 40 --seed 1000 --robot go2 --project ../projects/<name>

# export for deployment: ONNX plus manifest
uv run python -m trainnr_mjlab.walk_export ../projects/<name>/runs/<run>/model_1499.pt \
    --project ../projects/<name> --robot go2 --name <deployment>

# the tests that run without a GPU
uv run python -m unittest discover -s tests -t .
```

Extras: `bam` installs Rhoban's own package as the parity reference for the
kernel's tests (never a runtime dependency); `viz` installs the Rerun SDK
the recorder streams to, pinned to the Studio's own release line.

Training needs a CUDA GPU, and so does `reward_preview.py` today (its
device is `cuda:0`). Evaluation of a trained checkpoint runs on the CPU,
slowly. The environment is about 6 GB; the first tool job that needs it
(training, evaluation, export) prepares it.
