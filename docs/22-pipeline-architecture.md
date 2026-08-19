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
| statistics under everything | `pipeline/rq_pipeline/stats/` | **built** — intervals + ranking, 20 tests |
| artifact identity | `pipeline/rq_pipeline/bundles/` | **built** — content hashing, `name@hash` stamps |
| physics abstraction | `pipeline/rq_pipeline/physics/` | **built** — `PhysicsBackend` protocol; MuJoCo adapter next |
| ② onboard: model gates | `pipeline/rq_pipeline/robot/` | **built** — fail-loudly import census |
| ② onboard: identification | `rq_pipeline/robot` (grows) | next — excitation + `mujoco.sysid` wrapper + identifiability report |
| ① scene ingestion | planned `rq_pipeline/scene` | scan QA gate, USD scene bundle, cousin substitution |
| ④ collect | planned `rq_pipeline/collect` | LeRobot dataset adapters; the rig's `.wire` → LeRobot converter is the first one |
| ⑤ curate | planned `rq_pipeline/curate` | PSD ranker first (needs nothing), rollout scoring later |
| ⑥ expand | planned `rq_pipeline/expand` | green-screen style augmentation + the renderer-agnostic sensor-degradation stage |
| ⑦ train | planned `rq_pipeline/train` | thin LeRobot wrappers, ACT first |
| ③/⑧ evaluate | planned `rq_pipeline/evaluate` | `mujoco.rollout`-based harness emitting certificates via `stats` |
| ⑨ envelope | stays in `firmware/` | the Tier 0 boundary is hardware's job; the pipeline only *verifies* it exists |

## 3. Decisions, with the reasoning attached

**MuJoCo is the simulation loop; Newton is a backend, not a foundation.**
The identification stage is `mujoco.sysid`, which is CPU MuJoCo — that alone
anchors the canonical path. Newton (Apache-2.0, NVIDIA + DeepMind + Disney)
slots in behind `PhysicsBackend` for GPU rollouts when a machine has the
hardware; MuJoCo itself is becoming a solver inside Newton, so this is one
converging stack, not a fork risk. MJCF stays the robot's source of truth and
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

1. **`.wire` → LeRobot dataset adapter** in `collect` — the rig's recordings
   become training data, exercising bundles + calibration stamping end to end.
2. **MuJoCo adapter** behind `PhysicsBackend` + the census gate wired to real
   imports (also settles the "USD splat silently skipped" inference from
   [30-the-pipeline.md](e2e-research/30-the-pipeline.md) stage ①).
3. **Identification walk-through on the SO-101 class** once the arm kit is
   assembled: excitation, fit, intervals — the first robot-bundle, and the
   experiment nobody has published.
