# The pipeline as software: architecture of `pipeline/`

Started **2026-08-20** on branch `h9/pipeline`. This turns the research design
in [docs/e2e-research/30-the-pipeline.md](e2e-research/30-the-pipeline.md)
into a codebase. Scope here: what exists, what each module owes the others,
and the decisions that shape everything downstream. The research docs carry
the evidence; this doc carries the contracts.

## 1. The one principle everything hangs off

> **The robot is an artifact, not an import.** Any robot enters the platform
> as a hash-stamped bundle — canonical MJCF, measured dynamics with confidence
> intervals, calibration state, an interface adapter — and no stage may code
> against an embodiment. A new robot is new *data*, never new *code paths*.

This is what "generic — anyone can come and train their robot" means as an
architectural property. The same rule applies to scenes (`scene-bundle@hash`)
and policies (`policy@hash`): stages exchange artifacts, artifacts carry the
hashes they were produced under, and an evaluation that cannot name the exact
robot-bundle, scene-bundle and calibration it ran against is not a result.

## 2. Module map, against the research design

The stages of [30-the-pipeline.md](e2e-research/30-the-pipeline.md) map onto
packages. Built means: typed, tested, gated, on `main`'s quality bar.

| Stage | Package | Status |
|---|---|---|
| statistics under everything | `pipeline/rq_pipeline/stats/` | **built** — intervals + ranking, including the **exact permutation test** for n ≤ 8 policies (all n! pairings enumerated — R2's fix) and **cross-task pooling** (`pooling.py`, R9's fix): Fisher-z inverse-variance combine, Cochran's Q heterogeneity policing, Fisher's-method combination of exact p's; chi-squared machinery stdlib-only in `intervals.py`, verified against closed forms |
| artifact identity | `pipeline/rq_pipeline/bundles/` | **built** — content hashing, `name@hash` stamps, and the **typed `RobotProfile`**: every robot constant (tick scale, camera fps, servo band) lives once in the bundle's `profile.json` with per-value provenance strings, loaded through a validating frozen dataclass — code and tests read named fields, never literals |
| physics abstraction | `pipeline/rq_pipeline/physics/` | **built** — protocol + **MuJoCo adapter** (`mujoco.rollout` batched, census wired to the fail-loudly gate, gravity-verified; MuJoCo 3.11 with the sysid toolbox importable in-venv) |
| ② onboard: model gates | `pipeline/rq_pipeline/robot/` | **built** — fail-loudly import census |
| ② onboard: identification | `pipeline/rq_pipeline/robot/` | **built, and rehearsed end-to-end** — staged excitation, `mujoco.sysid` fit wrapper, identifiability report (NOT PINNED verdicts for unconstrainable parameters), plus the full Paper 0 rehearsal: true drivetrain → synthetic sweep → integer-tick wire degradation → `identify()` recovers gear ~2%/damping ~5% single-run. The first robot bundle skeleton lives at `robots/rig-drivetrain/`, and **fit records** (`robot/fit_record.py`) write each run's fit into the bundle's `fits/` — recording stamp and anchor statement required, cross-run spread reported beside per-run intervals |
| ① scene ingestion | planned `rq_pipeline/scene` | scan QA gate, USD scene bundle, cousin substitution |
| Paper 2 sim substrate | `robots/so101-nominal/`, `robots/aloha2-nominal/` | **bundled** — Menagerie's `trs_so_arm100` byte-identical (Apache-2.0, upstream docs preserved) + a sensor-adding wrapper (`so101.xml`), since harness policies observe sensors and upstream ships none. Census, wrapper-purity and closed-loop-hold pinned by tests. Every number nominal by design — that IS Paper 2's first condition |
| Paper 2 task suite | `pipeline/rq_pipeline/tasks/` | **started** — scenes composed as programs (`MjSpec` attach; the arm's contact options restored explicitly because attach drops them), referee sensors appended after the robot's, targets computed from the bundle's own keyframes. Three tasks run end-to-end through the harness with graded scripted policies: `reach`, `lift` (squeeze grip in the measured pad pocket, 15/15 across ±4 mm jitter), and **`block_stack`** — the first ArmnetBench-named task: pick, staged base swing, a measured-landing-point drop onto cube B, high retract (a low retract demolishes the fresh stack — measured), 9/9 across pick jitter. **`tool_insert`** joins them (same pick-swing-drop delivered into a walled pocket — one behaviour, two tasks, like a real policy across a suite; low walls on purpose, 24 mm walls let the cube cock and perch). **The pooled Gate A dry run passes**: 3 tasks × 4 named policies → harness → per-task certificates (each with an exact p, none passing at n=4) → cross-task pooling (homogeneous, combined exact p) — and the POWER lesson is now an assertion: 3×n=4 cannot clear a 0.5 pooled gate, which is why Paper 2 needs 6-7 retrained policies. The ALOHA 2 rig joins with `transfer_cube` (gym-aloha's protocol on the bundle, cosmetic `act_sim` look for released checkpoints) and **`kitting`** — whose scripted demonstrator (chained grip-centre IK, closed-loop clamped correction, verify-and-retry against the task's own referee) is T5's data source via `tools/kitting-demos.py`. **`gripper-pick`** (`tasks/gripper_pick.py`, 2026-09-25) is the first task on a robot imported from USD: the Isaac 2F-85 hung from a task-declared Cartesian carriage, a spec'd spawn band and lift-and-hold referee, a registered expert and a ladder (`LADDER`: pick, no-close, limp) that the acceptance critic accepts |
| assembly menu | `pipeline/rq_pipeline/tasks/components.py` | **built** — `compose(car=..., arm=...)`: the differential-drive base (RobotSpec geometry), the arm, or the **mobile manipulator** (arm mounted on the chassis, facing forward). Design walked through four measured failures: tip-forward (one caster), beach-on-roll (two casters), and a parking-brake bug — **MuJoCo combines contact friction by element-wise max**, so low-friction casters inherit the floor's grip unless `priority` says otherwise. Drives 2.3 m/5 s, spins, arm fully extended does not tip it |
| ④ collect | `pipeline/rq_pipeline/collect/` | **built end to end** — `.wire` reader (Rust-census conformance) → frame alignment (27 frames, 50 Hz-counter clock) → **LeRobot export** behind the `train` extra, roundtrip-tested: export, reload, shapes and the real wire clock all verified. The rig's own data can now feed a training run |
| ⑤ curate | planned `rq_pipeline/curate` | PSD ranker first (needs nothing), rollout scoring later |
| ⑥ expand | planned `rq_pipeline/expand` | green-screen style augmentation + the renderer-agnostic sensor-degradation stage |
| ⑦ train | WSL box, `.venv-train` | **running** (2026-08-26, docs/31) — LeRobot 0.6.1 trains against this repo's exports: T0 ACT smoke (300 steps, 2.1 GB), T6 PPO on MuJoCo Warp (36k env-steps/s, 2048 worlds). Thin wrappers stay a non-goal until a second trainer forces an abstraction — today the dataset contract IS the interface |
| ③/⑧ evaluate | `pipeline/rq_pipeline/evaluate/` | **built end to end** — the certificate artifact (bundle-stamped, gated on the Fisher lower bound, per-policy intervals, exact permutation p whenever n ≤ 8) **and the harness that feeds it** (`harness.py`): census-gated closed-loop episodes, paired trials across policies, sensors-only observations, exact-join to real outcomes by name. The whole Gate A chain runs in one test: four policies → sim ranking → join → honest FAIL at n=4 with exact p = 1/24. **The real side has real data**: `evaluate/armnetbench.py` loads the committed third-party aggregate (`data/armnetbench-v01-so101-counts.json` — 2,099 SO-101 rollouts, 7 policies × 8 tasks, Apache-2.0, provenance in-file) into `join_with_real`'s shape, strict-by-default on the `suboptimal` label. **Pixels too** (`evaluate/vision.py`, 2026-08-25): the same paired-trial skeleton over rendered cameras (ArmnetBench's three-camera rig as data), camera-count census-gated, with a LeRobot checkpoint adapter that runs released policies through their own pre/post-processors |
| ⑨ envelope | stays in `firmware/` | the Tier 0 boundary is hardware's job; the pipeline only *verifies* it exists |

## 2.1 The two entry maps: where the ML enters, where the physics enters

The two questions every newcomer asks, answered against the stage map above.

**ML enters at exactly three points.** ⑦ TRAIN is the VLA (LeRobot wrappers,
ACT → π0.5/MolmoAct2 per doc 20's ladder), consuming a `demo-set@hash` and
emitting a `policy@hash`. ⑥ carries the world-action model in its only
sanctioned role — an auxiliary training signal, never a data generator. ⑩
OPERATE is the RL: on-site refinement of a **frozen** generalist by a small
chunk-level learner fed sparse human success labels — the RLT reference
design ([30 §⑩](e2e-research/30-the-pipeline.md)). Evaluation (③/⑧) contains
no ML at all: there the models are the *subject*, and the certificate that
judges them must not share their failure modes.

**Physics enters at five points, in a strict promotion order** — dynamics is
measured before the simulator runs, the simulator is validated before it is
trusted, trusted before used:

| Stage | Physics' role |
|---|---|
| ② onboard | **Dynamics becomes data**: excitation + `mujoco.sysid` fit → parameters with intervals. CPU MuJoCo as fitting substrate |
| ① scan | Scene physics, minimally: collision geometry + cousin mass/friction. The splat is never the physics |
| ③ validate | **First end-to-end rollouts, on probation** — their purpose is to test the simulator itself (Gate A) |
| ⑥/⑦ expand + train | Sim as data factory, only after the gate — synthetic episodes and DR **centred on identified values** (randomising around a guess trains robustness to the wrong distribution) |
| ⑧ evaluate | Sim as certified instrument: batch rollouts rank policies under ③'s certificate |

And one loop: ⑪ telemetry watches parameter drift back into ② —
re-identification is scheduled maintenance, because a drifting fit is a
wearing gearbox. Dynamics is a subscription, not a measurement.

In build order both precede the ML: the MuJoCo adapter (§5 item 2) brings
physics into the codebase, and Paper 0 (docs/23-research-agenda.md) brings
dynamics-as-measurement onto owned hardware, before anything trains.

## 3. Decisions, with the reasoning attached

**MuJoCo is the simulation loop; Newton is a backend, not a foundation.**
The identification stage is `mujoco.sysid`, which is CPU MuJoCo — that alone
anchors the canonical path. Newton (Apache-2.0, NVIDIA + DeepMind + Disney)
would slot in for GPU rollouts as another gymnasium env over the same tasks
(`rq_pipeline/envs`, the backend-agnostic seam since 2026-08-26 — the
`PhysicsBackend` Protocol with its single implementation was retired in
docs/32 step 5); MuJoCo itself is becoming a solver inside Newton, so this
is one converging stack, not a fork risk. MJCF stays the robot's source of truth and
USD carries scenes — the split, and the silent-import failure that mandates
the census gate in `pipeline/rq_pipeline/robot/model_checks.py`, are settled
in [23-simulation-and-real2sim.md](e2e-research/23-simulation-and-real2sim.md) §3/§6.

**The statistics module is dependency-free and came first, deliberately.**
No leaderboard in this field publishes confidence intervals and no shipping
framework computes them; the product's differentiation is honest numbers, so
the honesty layer is the foundation everything else imports. Clopper-Pearson
is implemented from the continued fraction (~40 lines) rather than pulling
scipy: a signed report must be recomputable on an auditor's laptop.

**The gate statistic is Fisher-z, and our own first test run is the reason.**
The percentile bootstrap looked like the obvious interval — and the suite's
first execution caught it asserting certainty ([1.0, 1.0]) on perfectly
monotone rankings at n = 5, exactly the overconfidence defect C1 in
[31-defects.md](e2e-research/31-defects.md) exists to prevent. The bootstrap
stays as a diagnostic; the certificate uses `fisher_rank_ci`, gates on the
*lower* bound, and reports `top_pick_probability` — the number a deployment
decision actually turns on. The lesson is now a permanent regression test in
`pipeline/tests/test_ranking.py`.

**uv and ruff run the Python world, and the gate enforces both.** uv
provisions interpreters (the host's Python 3.9 stops mattering), pins the
environment, and runs the suite; ruff plays the role clippy + rustfmt play
for the crates. `tools/hooks/pre-commit` runs format-check, lint and the unit
suite on every commit — the same discipline the Rust side has, mechanical
rather than conventional. Floor is Python ≥ 3.10; heavy stages are extras
with their own floors (LeRobot ≥ 3.12, Warp needs NVIDIA).

**Apache-2.0, commercialisable.** Core dependency-free; every planned
dependency (MuJoCo, LeRobot, Newton, gsplat) is Apache/BSD. The research
corpus's licence traps — INRIA mesh extraction, CC-BY-NC asset tiers,
proprietary ovrtx — stay outside the dependency tree by policy.

## 4. What the platform owes a stranger's robot

The test for "generic" is a robot we have never seen. Its onboarding path,
per stage ② of the research design: MJCF triage → interface capability census
(what can it report: position? velocity? current?) → staged safe excitation →
`mujoco.sysid` fit → **a robot-bundle whose parameters carry intervals and
whose validity scope is written down**. Every later stage consumes that bundle
blind to what the robot is. The rig in this repo — camera, two motors, three
servos on a Pico — is deliberately the first stranger: if the pipeline's
abstractions cannot swallow the hardware twenty centimetres away, they cannot
swallow a customer's arm either.

## 5. Next, in order

The original three next-steps (LeRobot adapter, MuJoCo adapter, first
robot-bundle) all landed — see the stage map. As of **2026-08-26**:

1. **T5 at scale**: batch `kitting-demos` generation → LeRobot conversion →
   an ACT policy trained on our own scripted demonstrations (WSL card).
2. **Camera matching against released real videos** (`tools/camera-match.py`
   discipline) before any sim↔real correlation is signed — a vision policy
   failing on cosmetics must not be read as a dynamics gap.
3. **Identification on a feedback-capable arm** stays gated on hardware the
   venture has deliberately deferred buying; until then the STS3215 study
   (docs/28, YouTube benchmark ingest) carries the dynamics side.
