# mjlab read at the source: Isaac Lab's shape on MuJoCo Warp, and where the measurement layer isn't

*2026-08-31. Source: a repomix pack of `mujocolab/mjlab` v1.6.0 (394
files, docs + src, released 2026-08-08), read by five field agents —
actuators, engine, environment architecture, training/eval/ops,
viewers/positioning — each against the primary source only. File paths
below are theirs. This is the framework-layer companion to
[52](52-warp-determinism-mjwarp.md) (mujoco_warp itself) and
[49](49-gpu-path-mjxwarp.md) (our GPU path); everything here is from
their source unless marked INFERENCE.*

## 1. What mjlab is

Berkeley (Zakka, Yi, Liao, Le Lay, Sreenath, Abbeel — BAIR/Hybrid
Robotics, `CITATION.cff`), Apache-2.0, paper arXiv:2601.22074. Their own
sentence: Isaac Lab "provides a comprehensive manager-based API … but
requires the Omniverse runtime"; MuJoCo Playground is "minimal
abstractions and monolithic environment definitions"; "mjlab fills this
gap. It adopts Isaac Lab's manager-based design … and pairs it with
MuJoCo Warp" (`docs/source/motivation.rst`). Velocity: 11 releases in
6.5 months (1.0.0 on 2026-01-28 → 1.6.0 on 2026-08-08), minor releases
roughly monthly. Ecosystem: 13 downstream repos including an *official*
Unitree port and PAL Robotics — and the pattern is that vendors **fork**
mjlab rather than plug in (`docs/source/faq.rst`: the plugin guide is "a
future release"). Three robots ship (Unitree G1, Go1, i2rt YAM) and they
say they will not expand the library. Claude Code is wired into their CI
review loop (`.github/workflows/claude*.yml`, `.claude/commands/`).

Stated non-goals, verbatim, worth keeping on file: "Cross-simulator
portability is a non-goal; mjlab favors precise control and
interpretability over backend generality." "High-fidelity RGB rendering
is out of scope." "MuJoCo Warp does not yet guarantee determinism …
mjlab training runs will not be perfectly reproducible even when
setting a seed" (`motivation.rst`, `faq.rst`).

## 2. The engine seam (theirs vs ours)

Their whole wrap is four files (`src/mjlab/sim/`): one `mjwarp.Data`
with a leading world dimension, model fields shared across worlds until
a domain-randomization function's `@requires_model_fields` decorator
expands them, zero-copy torch views (`WarpBridge.__setattr__` *raises*
— pointer stability for CUDA graphs is a hard invariant), and four
captured CUDA graphs (step/forward/reset/sense) gated on driver ≥ 12.4.
CPU MuJoCo compiles the spec, seeds the data, feeds the native viewer
and the NaN dumps — it never steps physics.

What their changelog proves about single-backend life: a CUDA-700
illegal memory access from mass DR at ≥128 envs on consumer GPUs,
`qfrc_constraint` populated wrongly across vectorized worlds, stride-0
broadcast views silently becoming size-1 arrays across a mjwarp minor
bump — each fixed by chasing an upstream release
(`docs/source/changelog.rst`). They track that churn with a Claude
command (`mjlab:.claude/commands/update-mjwarp.md`: one upstream commit → one
minimal-diff PR → `uv lock`) and a CI matrix that resolves both `locked`
and `unlocked` "so a new upstream release breaks a PR instead of a
release". Our two-instrument seam (CPU MuJoCo + MJX-Warp, docs/49)
would have caught the `qfrc_constraint` class in-house as an A/B
divergence; theirs cannot, by declared design.

**Steal — the NaN guard** (`mjlab:src/mjlab/utils/nan_guard.py`): a `watch()`
context manager around the step keeping a 100-deep rolling deque of
cloned `qpos/qvel/act/mocap`, checking NaN *and* Inf across
`qpos, qvel, qacc, qacc_warmstart, sensordata` per world; on first hit
it dumps `.npz` + `.mjb` (via `mj_getState`/`mj_saveModel`, so
`mj_setState` replays it) with `latest` symlinks and a scrubbing viewer
(`uv run viz-nan`). Off = a branch. On our seam this upgrades from
crash log to **divergence oracle**: replay the dumped state through CPU
MuJoCo and diff. They also split diagnose vs survive honestly —
`nan_detection` as a termination is documented as "a band-aid, not a
cure".

**Steal — the nightly harness's structure, not its standards**
(`scripts/benchmarks/`): systemd timer, fresh clone, `clear_gpu`
between phases; measures `physics_sps` and `env_sps` separately and
reports `overhead_pct` so a manager-layer regression is visible when
physics is flat; every row stamped with a 7-char commit, appended to a
JSON series, rendered to a public gh-pages chart. Two blind spots we
must not copy: no hardware fingerprint on the series (one box,
`/home/kevin`), and **no thresholds — it asserts nothing**. Ours keys
on `(gpu, driver, cuda, engine version)` — we already measured the same
engine version giving different verdicts on two architectures
(docs/33) — and gates.

Scaling facts for our WSL/3090 Ti context: Linux + CUDA 12.4+ for
training (below 12.4 you silently lose graph capture); WSL "tested less
frequently" but not blocked; `device="cpu"` still claims VRAM on every
visible GPU unless `CUDA_VISIBLE_DEVICES=""` (their issue #949); memory
dials are per-world `nconmax`/`njmax`, `maxhullvert`, broadphase
choice; a hand-rolled Cholesky avoids cuSOLVER's ~4 GB resident
overhead.

## 3. The actuator layer: a clean interface with no epistemics

The interface is well-factored — `ActuatorCfg` lifecycle
(`edit_spec` pre-compile → `initialize` post-compile → `compute` per
step), built-ins that lower onto native MJCF elements for implicit
integration, explicit PyTorch actuators, and a fusion contract where
keeping the inherited `compute` *is* the opt-in to batched execution
(`mjlab:fused_group.py`). The models run from ideal PD through a linear
torque-speed DC clamp up to MuJoCo's native `<dcmotor>` with back-EMF,
thermal, cogging and LuGre stick-slip slots (`mjlab:builtin_actuator.py`).

Where the values come from is the finding. Their entire hardware record
type is `ElectricActuator(reflected_inertia, velocity_limit,
effort_limit)` — three bare floats (`mjlab:utils/actuator.py`). Gains are
derived, not measured: `STIFFNESS = armature·ω_n²`,
`DAMPING = 2ζ·armature·ω_n` with ζ = 2.0 justified in their own docs as
overdamping "when the true system inertia is underestimated"
(`docs/source/actuators.rst`). The G1's linkage ratio is "we assume a
nominal 1:1"; the YAM's effective inertias are a hand-listed table with
no source; its gripper effort is derated ×0.1 "for sim stability"
(`asset_zoo/robots/*/\*_constants.py`). Grep across 394 files for
sysid/fit/interval/provenance: nothing. The only richer friction model
(LuGre, five parameters) ships **used by no robot**, excluded from the
supported DR surface, with no story for where its numbers would come
from. All three shipped robots use plain position actuators and set no
`frictionloss` at all.

`LearnedMlpActuator` completes the picture: it *consumes* a TorchScript
actuator-net (`torch.jit.load(network_file)`, history-stacked pos-error
+ vel, output clipped by the DC curve) but the repo contains no
training pipeline, no data collection, and no weights — the only
networks are three-line test stubs. The framework will happily ingest
identified artifacts; nothing in it can produce one.

**Implication for us** (INFERENCE, flagged): our BAM-derived M1–M6
records with intervals and recording hashes drop into their `ActuatorCfg`
extension point without fighting the architecture — a robotiq-identified
actuator would be a differentiated *upstream contribution*, not a
competing framework. One technical caution from their own docs: implicit
integration favors models expressed through MuJoCo's native slots; an
explicit PyTorch friction law loses both stiffness handling and fusion.
Whether M1–M6 can be expressed through `<dcmotor>`'s native parameters
is an open question worth one experiment.

## 4. Tasks as configs vs tasks as programs

Nine managers (action, observation, reward, termination, event,
command, curriculum, metrics, recorder), assembled from one flat
dataclass of `dict[str, TermCfg]`s; a fixed step order; a module-level
registry filled by import side effects with paired train/play configs
(`managers/`, `mjlab:envs/manager_based_rl_env.py`, `tasks/registry.py`). The
migration page's real argument for dicts over Isaac's `@configclass`:
misspelled fields raise at construction instead of silently creating
attributes.

**Identity is the structural gap.** No config hash, fingerprint or
stamp exists anywhere in the dump (grep: only asset-download SHA256 and
viewer mesh dedup). It cannot be retrofitted cheaply: curriculum terms
work by **mutating the live config objects** — `commands_vel` rewrites
`cfg.ranges.lin_vel_x` in place on schedule, `reward_curriculum`
rewrites term weights (`mjlab:tasks/velocity/mdp/curriculums.py`) — so the
thing you would hash changes under you mid-run, and their FAQ concedes
a fixed seed does not reproduce a run anyway. Our content-stamped
TaskSpecs and per-record instrument stamps are not a style choice
against this; they are a capability their idiom forecloses.

**No scripted experts.** Nothing in mjlab produces reference actions;
staged Gaussian reward shaping (`reaching·(1+bringing)` with hand-picked
σs in the lift-cube task) stands in for a demonstrator. Where we can
generate an expert trajectory, that shaping layer — and much of the
config surface managing it — dissolves.

**Steal, all four cheap:**
- The **event lifecycle vocabulary** — `startup` / `reset` / `interval`
  / `step` with per-env timers and `min_step_count_between_reset`
  throttling (`docs/source/events.rst`). Sharper than our current
  perturb-at-reset-only framing.
- **`@requires_model_fields`** — every DR function *declares* which
  simulator fields it touches and what recomputation that forces
  (`envs/mdp/dr/`). They use it for CUDA-graph correctness; for us it
  is a perturbation's identity — the declaration a paired-trial
  protocol needs to prove trial A and B differ in one dimension only.
- **Paired train/play registration** — eval config ≠ train config as a
  registered object, not a runtime flag.
- **The latched `episode_success`** — `torch.maximum(success, at_goal)`
  surviving to episode end (`mjlab:manipulation/mdp/commands.py`) is a clean
  vectorized one-rung funnel; our milestone vector is its ordered
  generalization.

Their DR surface is broad and typed (~50 functions over geom, body —
including a log-Cholesky pseudo-inertia guaranteeing physical
consistency — joint, tendon, camera, light, material, pair friction;
per-world mesh *variants* with per-variant inertials proven byte-equal
to independent compiles). Worth knowing exists before we build any of
it ourselves.

## 5. Training, eval, ops: a dashboard, not an experiment

One algorithm (PPO via rsl_rl), asymmetric actor-critic as observation
groups, tyro CLI where the whole config tree is flags (with a repo-wide
AST test enforcing sweep-safe explicit booleans), single-node multi-GPU
via torchrunx, cloud = SkyPilot on Lambda with W&B sweeps whose
objective is **training mean reward** — the search optimizes the
training curve, not a held-out metric (`scripts/cloud/sweep.yaml`).

Evaluation, precisely: the only real harness is tracking-only
(`mjlab:tasks/tracking/scripts/evaluate.py`), scores 1024 episodes **on the
same motion artifact the run trained on**, defines success as "survived
to timeout without tripping a termination", and emits a JSON of six
floats with no seed, no config hash, no commit. No held-out anything,
no paired trials, no baseline comparison, no intervals, no eval-record
schema. Velocity/manipulation/cartpole tasks have no evaluation at all
beyond reward curves and `uv run play`. The nightly trains one task and
asserts nothing. Recorders are a hook API with zero shipped
implementations — no dataset export, no BC corpus. Deployment ends at
`policy.onnx` in a W&B artifact; everything past the sim boundary lives
in the 13 forks.

**Steal:**
- **The ONNX metadata envelope** (`mjlab:rl/exporter_utils.py`): joint names,
  gains, default pose, obs term names/scales/clips/history, action
  scale baked into the deployed file — the policy carries its own I/O
  contract. Our eval records and certificates should be this
  self-describing (they largely are; the delta is embedding the obs
  pipeline spec in the exported artifact itself).
- **Artifact lineage**: `wandb.run.use_artifact(motion)` makes
  motion→run→checkpoint queryable from either end; their eval and play
  scripts *recover* the training motion by scanning `used_artifacts()`.
- **Eval-per-commit with a caching report** (`mjlab:generate_report.py`):
  re-evaluates only new runs, preserves history, publishes a
  commit-tagged series. Cheap and durable; add the gates they lack.

## 6. Viewers and sensors: 3D-first, telemetry-thin

Two viewers off one config: the MuJoCo passive viewer (per-frame
GPU→CPU copy of one world, contacts recomputed by C MuJoCo) and a viser
web viewer (per-sensor RGB/depth panes, checkpoint hot-swap from W&B —
their closest thing to a Studio). Telemetry is a reward-term strip:
native capped at 12 `MjvFigure` plots, viser a 150-sample deque per
term — cleared on reset, never persisted, nothing scrubbable, no
cross-run comparison; "we are exploring training-time visualization …
but this is not yet available" (`faq.rst`). Everything else goes to
W&B.

Sensors worth knowing: a 675-line contact sensor with regex matching,
reduction modes and an air-time tracker whose `history_length` exists
because decimation can swallow a brief contact between reads; a
BVH-accelerated raycast sensor (grid/pinhole/ring patterns) sharing its
BVH with cameras; RGB-D rendered by **mujoco_warp's ray-traced GPU
renderer** (packed ABGR unpacked by a Warp kernel, zero-copy into
torch, 160×120 default, no published throughput). Terrain: a
difficulty-interpolated grid of box/heightfield sub-terrains with a
promote/demote curriculum and morphological flat-patch spawn sampling.

## 7. What we adopt, in order — and what we don't

Adopt (cheap, high-leverage, in priority order):
1. **NaN guard → divergence oracle** on the engine seam (§2). One file.
2. **Milestone latch + event lifecycle + declared-fields DR** patterns
   into the task layer as we grow it (§4) — vocabulary, not code.
3. **Nightly eval-per-commit harness structure** with our gates and a
   hardware fingerprint (§2, §5) — pairs with the acceptance runs we
   already have.
4. **Self-describing exported artifacts** (ONNX-style metadata envelope)
   for anything we hand a deployment (§5).
5. **Their dependency-churn ritual** for our own mjwarp exposure: one
   upstream commit per PR + a locked/unlocked CI matrix (§2).

Rejected, with reasons:
- **Adopting mjlab as our training substrate.** It is Apache-2.0 and
  good; but its two structural commitments — single backend by
  principle, task identity foreclosed by mutation-based curricula — are
  the negations of our two load-bearing properties (two instruments,
  content-stamped tasks). We would spend our differentiation to save
  plumbing we mostly already have. Interop, not adoption: an
  mjlab-compatible actuator artifact (§3) is the right bridge.
- **Their evaluation layer.** Nothing to take; §5 is the evidence.
- **Viser-style viewer work.** The Studio's embedded Rerun already
  exceeds their telemetry ceiling; their 3D-viewer polish (ghost
  meshes, checkpoint hot-swap) is noted for ideas only.

## 8. Open questions carried forward

- Can M1–M6 friction express through MuJoCo's native `<dcmotor>`
  slots (implicit integration + their fusion path) without losing what
  BAM measured? One experiment decides (§3).
- Their claim "on par or faster than Isaac Lab" is experiential, no
  numbers published; their nightly publishes env-steps/s for one box.
  If we ever quote comparative throughput, measure it ourselves.
- The actuator-net path (`LearnedMlpActuator`) as a robotiq *output*
  format: an identified-artifact exporter targeting their loader is a
  small, legible upstream contribution — worth it only when a customer
  runs mjlab.

---

*Postscript 2026-08-31: §8's `<dcmotor>` question is resolved by the
BAM source read ([57](57-bam-source-and-microduck.md) §2) — M6's
load-dependent friction cannot be expressed as a native MuJoCo
actuator (BAM's own deprecated exporter says so); the working
mechanism is Rhoban's per-step `dof_frictionloss`/`dof_damping`
writes, already production-tested by Pollen. §7's "rejected as
substrate" stands, but the bridge plan it sketched is superseded by
[58](58-cross-framework-setup.md): the bridge half-exists upstream;
the gap is the artifact layer around it.*
