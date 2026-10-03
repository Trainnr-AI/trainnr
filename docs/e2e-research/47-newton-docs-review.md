# Newton, read docs-first: the solver stable and our parity ledger

*2026-08-27, branch `physics-newton`. Source: a repomix pack of
[newton-physics/newton](https://github.com/newton-physics/newton) at
`main`, version `1.6.0.dev0` (post-v1.5.0 development), reviewed
locally — every claim below is from the repo's own docs/source unless
marked MEASURED (run here today). This extends
[36-newton-status.md](36-newton-status.md) (2026-08-25); the verdict
refresh is at the bottom. Field-agent reports:
[48-solver-landscape.md](48-solver-landscape.md) (solver theory vs our
scenes) and [49-gpu-path-mjxwarp.md](49-gpu-path-mjxwarp.md) (the
MJX-Warp probe), [50-newton-delta-probe.md](50-newton-delta-probe.md)
(the Newton install probe — which verified §3's sensor-drop row by
execution and corrected §5's Isaac Lab read).*

**One naming collision, pinned first:** MuJoCo's default constraint
solver is literally named "Newton" (`mjSOL_NEWTON` — Newton's method on
the convex contact problem). The NVIDIA/DeepMind/Disney engine is also
named Newton. In these docs, "the Newton solver" always means MuJoCo's
`mjSOL_NEWTON`; the engine is always "Newton (engine)". MEASURED
2026-08-27: our kitting and lift scenes both resolve to
`solver=Newton, integrator=Euler, cone=elliptic, impratio=10,
timestep=0.002, iterations=100, ls_iterations=50, tolerance=1e-8`.

## 1. The solver stable (docs/solvers/index.rst, main)

| Solver | Coordinates | Formulation | Fit for our arms | Status |
|---|---|---|---|---|
| `SolverMuJoCo` | generalized | wraps **mujoco_warp**; MuJoCo semantics | **the only candidate** — see §2 | stable |
| `SolverFeatherstone` | generalized | semi-implicit articulated | no joint friction, no effort limits, no equality/mimic | stable |
| `SolverXPBD` | maximal | position-based (XPBD) | "does not use armature, joint friction, effort limits, or velocity limits" (their words) | stable |
| `SolverVBD` | maximal | vertex block descent, compliant-ALM migration in flight | limited joint support; cloth/soft territory | **experimental** |
| `SolverSemiImplicit` | maximal | penalty-style, no iteration parameter | soft bodies/particles | stable |
| `SolverStyle3D` | — | implicit cloth | no joints at all | stable |
| `SolverKamino` | maximal | PADMM, hard frictional contact, kinematic loops | experimental; D6 unsupported | **experimental** |
| `SolverImplicitMPM` | — | implicit MPM (granular/fluids) | no rigid bodies | stable |
| Coupled (ADMM/proxy) | mixed | partitions one Model across solvers | multiphysics coupling | **experimental** |

The matrix settles the manipulation question by itself: `SolverMuJoCo`
is the **only** Newton (engine) solver with generalized coordinates AND
`joint_target_mode` AND effort limits AND equality/mimic constraints —
the only one that can run our arms as authored. Differentiability:
MuJoCo/XPBD/VBD/Kamino all ❌; only Featherstone/SemiImplicit are
"basic" diffsim. The engine's breadth is cloth/MPM/cables/coupling —
none of which our current tasks need.

## 2. SolverMuJoCo is mujoco_warp with a conversion layer

`SolverMuJoCo` wraps mujoco_warp behind Newton's solver interface, with
compatible-release pins on both `mujoco` and `mujoco-warp`. The step
cycle is push Newton→MuJoCo, integrate with mujoco_warp, pull back.
Knobs mirror MuJoCo (`solver`, `integrator`, `cone`, `impratio`,
`iterations`, ...) with three-level resolution: constructor > per-world
`model.mujoco.*` custom attributes (populated by MJCF import) > MuJoCo
default — **with one silent exception: `integrator` defaults to
`implicitfast` where MuJoCo defaults to Euler** ("Newton-opinionated,
for stability on stiff systems"). Anyone comparing trajectories across
the seam must pin the integrator explicitly or they are comparing
integrators, not engines.

Two mode switches matter:

- `use_mujoco_cpu=True` — classic CPU MuJoCo instead of mujoco_warp.
  This plus "macOS (CPU only)" in the tested-configurations table means
  Newton (engine) runs on this Mac, GPU-free.
- `use_mujoco_contacts` — `True` (default) uses MuJoCo's own collision
  pipeline and preserves imported `contype/conaffinity` masks verbatim
  (single-import case); `False` feeds Newton's pipeline (non-convex
  trimesh, SDF, hydroelastic — things MuJoCo lacks) but re-derives
  contact `solref` through Newton's force-space conversion.

## 3. The parity ledger — what our stack loses crossing the seam

The 08-25 report named "parity" as Newton's real cost but couldn't
enumerate it. The docs enumerate it themselves
(docs/solvers/mujoco.rst, "Unsupported MuJoCo features"):

| MuJoCo feature | Fate in Newton (engine) | What breaks for us |
|---|---|---|
| **`<keyframe>`** | **not imported** | our entire episode protocol starts from `neutral_pose` keyframes (`EpisodeProtocol.home`); the ALOHA reset-state jam (docs/22) is exactly what keyframes protect us from |
| **`<sensor>`** | **not imported** — Newton has its own sensor pipeline | the whole harness contract ("policies observe sensors"), the referee framepos sensors, the bundle's jointpos/jointvel census, `LEFT_GRIPPER_POS_SLICE` — all of it would need re-plumbing through Newton sensors |
| Cameras/lights in MJCF | ignored (own viewer pipeline; `SensorTiledCamera` instead) | our vision harness renders MJCF-declared cameras per step |
| `<plugin>`, `<composite>`, `<flex>`, `<skin>` | not supported/imported | unused by us today |
| Version pin | `mujoco~=3.11.0` on main (1.6.0.dev0) | CORRECTED 2026-08-27: the repo's lock also runs 3.11.0 (upstream latest is 3.12.0) — so Newton's pin currently *matches* ours. The deeper point survives intact: the engine version is part of the identified artifact, and an accidental 3.11→3.12 bump broke nine of our tests the same day (see docs/e2e-research/49 postscript) |
| Contact `solref` | force-space re-conversion unless MJCF-authored (`SOLREF_MODE_RAW` preserved verbatim) | our gym-aloha-copied solimp/solref are MJCF-authored → preserved; but `save_to_mjcf` round-trip is lossy for force-space joints |
| Multi-world | model built from the **first world only**, replicated; worlds must be structurally identical | per-trial DYNAMICS variation (our `variations.py` DR) has no first-class door here; per-world values exist only via custom attributes |
| Non-convex meshes | convex-hulled at conversion (MuJoCo semantics — same as ours) | no change |
| Collision masks | reproduced exactly up to 32 biclique bits, then a "may allow extra contacts" fallback | our scenes are small; exact path applies |

The first two rows are decisive. Keyframes and sensors are the two
mechanisms our protocol and instrument rules are built on, and neither
survives import. Adopting Newton (engine) for evaluation would mean
rebuilding the reset mechanism and the entire observation layer in
Newton idioms — not an adapter, a port. MEASURED 2026-08-27
([51-solvermujoco-roundtrip.md](51-solvermujoco-roundtrip.md)): the
round trip also drops all 13 visual meshes, rewrites the root body's
mass, flips the integrator in the emitted file — and the importer
zeroes the position servos' kv, so Newton simulates this robot with
UNDAMPED servos: the very parameter sysid identifies. (Mechanism found
2026-08-27 in their source: the importer reads an explicit `kv` but
not MuJoCo's compiled `dampratio`, which our bundles author; docs/e2e-research/51.)

## 4. What is worth taking regardless: the tuning corpus

The docs' tuning chapters are the best compact treatment of
MuJoCo-style contact tuning we have seen, and they apply to OUR CPU
MuJoCo scenes verbatim (the semantics are MuJoCo's own):

- **The constraint mental model** (docs/concepts/simulation_tuning_mujoco.rst):
  contacts are not world-space spring-dampers but a *soft servo in
  constraint space* — `a + d(bv + kr) = (1-d)a₀`. `solref` = how error
  is corrected (timeconst/dampratio); `solimp` = how much authority the
  constraint has (impedance curve d(r): d0, dmax, width, midpoint,
  power). Direct format `solref=(-stiffness,-damping)` is called out as
  "clearer for system identification" — relevant to Paper 0/sysid work.
- **"Make harder vs make stable" are different actions**: harder =
  raise ke / cut timeconst / raise dmax / cut width, costs stability
  margin; stable = raise kd toward critical, cut dt, fix geometry.
  Chasing hardness to fix jitter (or vice versa) is the named
  anti-pattern.
- **refsafe floor**: positive-format solref is clamped to
  `timeconst >= 2*dt` — past that floor, raising gains hardens nothing;
  only a smaller dt does. At our dt=0.002 the floor is 4 ms.
- **The grasping template**, in priority order: commanded/clamped grip
  force first, then friction mu, then contact stiffness, then elliptic
  cone + impratio for stick-slip — "never use stiffness as a substitute
  for friction capacity." This is the exact debugging ladder for the
  kitting close-and-lift failures, and it validates the bundle's
  existing elliptic+impratio=10 choice.
- **Drive/contact sanity checks**: ωₙ≈√(k/m_eff), ζ≈d/(2√(km_eff)),
  with the honest caveat that there is no universal safe ωₙ·Δt — sweep
  dt/stiffness bounded and watch, don't tune to a magic number.
- **Buffer discipline for batched worlds**: `nconmax`/`njmax` are
  per-world, tuned to the busiest world; a marginally-stable parameter
  diverges in *some* world at 2048 worlds even if most are fine.

## 5. Governance and stability, updated

- Versioning is now formally documented (docs/guide/compatibility.rst):
  major.minor.micro; deprecations live at least one full minor cycle;
  breaking changes only in minors; every deprecation in CHANGELOG +
  runtime `DeprecationWarning`. The 08-25 finding "no formal
  API-stability guarantee anywhere" is now stale — a real deprecation
  policy exists, though minors still ship breaking changes by policy.
- Only the latest minor line is maintained; no backports.
- Tested configs: Ubuntu 22.04/24.04, Windows, macOS (CPU only);
  NVIDIA Ada/Blackwell; Python 3.10+; CUDA 12/13 (12.3+ required for
  reliable CUDA graph capture).
- Newton's FAQ claims Isaac Lab's integration includes "basic
  manipulation" — but Isaac Lab's OWN docs (updated 2026-08-26, see
  [50](50-newton-delta-probe.md)) still say locomotion-only with no
  manipulation environments. The primary source on Isaac Lab wins:
  the revisit trigger has not moved.
- `warp.sim` was deprecated in Warp 1.8, removed in 1.10 — Newton is
  its successor, so Warp-side sim code has exactly one home now.

## 6. Verdict refresh vs [36-newton-status.md](36-newton-status.md)

**The recommendation stands, now with the cost itemized.** "Stay
MJX/mujoco_warp for GPU rollouts" survives this review strengthened:
the docs themselves show that going through Newton (engine) buys us a
conversion layer that drops keyframes and sensors (the two things our
evaluation stack is built on), pins the same MuJoCo as our
lock (`~=3.11.0`; our train venv runs 3.12.0 on purpose, as a second
instrument), and adds a solref/actuator translation whose
fidelity we would have to re-verify — all to reach the same
mujoco_warp we can call through MJX with the identified `mjModel`
intact. What Newton (engine) uniquely offers — cloth/MPM/cables,
solver coupling, `SensorTiledCamera` RTX rendering, USD-native scenes,
ONNX policy controllers — is real and none of it is on our critical
path.

**Two refinements to the standing position:**

1. The revisit trigger gains a second leading indicator: Isaac Lab
   listing "basic manipulation" (2026-08-27, FAQ) means the trigger
   (stable, non-beta manipulation RL) is plausibly nearer than
   "late-2026". Watch it, don't act on it.
2. The engine's tuning documentation (§4) is adoptable **today** at
   zero integration cost, because it documents MuJoCo semantics. The
   solver-study harness this branch builds should measure exactly the
   axes those docs name: solver × cone × integrator × dt, judged by
   referee outcome + penetration + solver iterations on our scenes.

## 7. MEASURED (2026-08-27): the sweep §6.2 called for

`tools/solver-study.py`, one scripted kitting episode (trial 0, proven
band) per config on this Mac, judged by the task's own referee.
pen = worst contact interpenetration; niter = mean constraint-solver
iterations per 50 Hz tick:

| solver | cone | integrator | verdict | wall s | pen mm | niter |
|---|---|---|---|---|---|---|
| **newton** | **elliptic** | **euler** | **PASS** (baseline) | 1.3 | 1.5 | 2.0 |
| newton | elliptic | implicitfast | PASS | 1.5 | 1.8 | 2.1 |
| newton | pyramidal | euler | fail | 1.5 | 4.7 | 1.8 |
| cg | elliptic | euler | fail | 2.4 | 11.4 | 12.0 |
| cg | pyramidal | euler | fail | 3.8 | 7.4 | 9.0 |
| pgs | pyramidal | euler | fail | 7.9 | 3.1 | 56.3 |

Three conclusions, each load-bearing for the GPU-path decision:

1. **Our scenes REQUIRE mjSOL_NEWTON + elliptic.** Every cheaper solver
   fails the referee outright, and CG's 11.4 mm penetration means the
   pads sink a quarter of the way into a 40 mm part — not a tuning gap,
   a physics collapse. Any GPU backend that lacks the Newton solver or
   elliptic cones is disqualified for our manipulation scenes; this is
   now the first parity question to ask of mujoco_warp, not the last.
2. **The pyramidal failure validates the bundle's authored choice** by
   measurement (4.7 mm penetration and a lost grasp), agreeing with the
   engine docs' own grasping guidance ("prefer elliptic + impratio for
   stick-slip").
3. **implicitfast passes the referee at ~equal cost — but it is NOT
   behaviorally equivalent** (revised 2026-08-27 when the sweep gained
   the docs/e2e-research/48 §3 diagnostics): peak pad normal force is **318 N vs
   Euler's 22.6 N** — a 14x spike at the fingertips the verdict column
   cannot see. Referee-equivalent, contact-dynamics different; the
   integrator decision needs the force trace, not just the outcome.

Caveat, stated honestly: one trial per config, and the scripted
choreography's closed-loop corrections partially adapt to whatever
physics they get — so "fail" measures the pipeline outcome (the thing
we ship), not solver accuracy in isolation. The penetration column is
the solver-only signal.

**Second pass, same day — the sweep grew the docs/e2e-research/48 §3 diagnostics**
(peak tangential pad-part slip from the elliptic efc_vel rows; peak
pad normal force via mj_contactForce) plus an impratio sweep:

| solver | cone | integ | impratio | verdict | pen mm | slip mm/s | grip N |
|---|---|---|---|---|---|---|---|
| newton | elliptic | euler | 10 | PASS | 1.5 | 130 | 22.6 |
| newton | elliptic | implicitfast | 10 | PASS | 1.8 | 136 | **318.5** |
| cg | elliptic | euler | 10 | fail | 11.4 | **5166** | 169.0 |
| newton | elliptic | euler | 1 | PASS | **5.1** | 140 | 23.6 |
| newton | elliptic | euler | 3 | PASS | 1.4 | 118 | 21.1 |
| newton | elliptic | euler | 30 | PASS | 2.6 | 164 | 23.3 |

Three refinements: (a) CG's failure is now legible — 5.2 m/s of pad
scrubbing across the part, the grasp never holds; (b) **impratio=1
still passes this trial** — the anti-slip knob's effect here shows up
as penetration (5.1 mm at 1 vs 1.4-1.5 mm at 3-10), not as slip, so
the recipe's value on THIS task is contact hardness at the pads
rather than slip suppression (one trial; the DR band may differ);
(c) the implicitfast force spike above. Peak — not mean — forces;
a mean-force trace is the follow-up before any integrator decision.

### 7.1 Re-measured on the second machine (WSL, x86-64, 2026-08-27 — the merge review)

The same tool, the same MuJoCo 3.11.0 (the lock), the same scene and
trial, on the RTX 3090 Ti box's x86-64 CPU:

| solver | cone | integrator | verdict | wall s | pen mm | niter |
|---|---|---|---|---|---|---|
| **newton** | **elliptic** | **euler** | **PASS** (baseline) | 1.2 | 2.39 | 2.0 |
| newton | elliptic | implicitfast | PASS | 1.3 | 2.30 | 2.1 |
| newton | pyramidal | euler | fail | 1.7 | 2.95 | 1.9 |
| cg | elliptic | euler | fail | 2.2 | 11.02 | 19.0 |
| cg | pyramidal | euler | fail | 3.3 | 11.79 | 8.2 |
| pgs | pyramidal | euler | **PASS** | 3.9 | 1.58 | 57.6 |

Two rows moved. **PGS passes here and fails on the Mac**, and every
penetration figure differs (baseline 1.5 mm there, 2.39 mm here; the
pyramidal Newton row 4.7 vs 2.95). Same engine version, same inputs,
different CPU architecture — arm64's and x86-64's floating point do
not agree to the last bit, and a pinch grasp amplifies the last bit
through contact (the mechanism docs/07 measured the night before
between MuJoCo builds: bit-identical for 84 steps, then a box-box
contact, then centimetres).

So conclusion 1 above is corrected: **mjSOL_NEWTON + elliptic is the
only configuration that passes on both machines** — CG and the
pyramidal cone fail on both, and PGS is a marginal pass on one
architecture at 28x the solver iterations. The bundle's authored
choice still stands, on stronger grounds; "every cheaper solver fails
outright" does not. And the instrument stamp learned from it: every
`instrument` now ends in the CPU architecture
(`mujoco-3.11.0+x86_64`, `physics/backend.py::instrument_stamp`),
because a version alone did not name the thing the verdict depends
on.

### 7.3 The second pass, re-measured on x86-64 (WSL, 2026-08-27, the same sweep with the diagnostics)

| solver | cone | integ | impratio | verdict | pen mm (arm64 → x86) | slip mm/s (arm64 → x86) | grip N (arm64 → x86) |
|---|---|---|---|---|---|---|---|
| newton | elliptic | euler | 10 | PASS · PASS | 1.5 → 2.39 | 130 → 151 | 22.6 → 26.6 |
| newton | elliptic | implicitfast | 10 | PASS · PASS | 1.8 → 2.30 | 136 → 149 | **318.5 → 21.5** |
| cg | elliptic | euler | 10 | fail · fail | 11.4 → 11.02 | **5166 → 2135** | 169 → 275 |
| newton | elliptic | euler | 1 | PASS · PASS | **5.1 → 5.22** | 140 → 161 | 23.6 → 21.1 |
| newton | elliptic | euler | 3 | PASS · PASS | 1.4 → 2.05 | 118 → 106 | 21.1 → 21.1 |
| newton | elliptic | euler | 30 | PASS · PASS | 2.6 → 2.97 | 164 → 165 | 23.3 → 24.7 |

The reading, machine by machine: **the 318 N implicitfast spike is an
arm64 transient** — on x86-64 the same configuration peaks at 21.5 N,
below Euler's 26.6 — so the second pass's first reversal ("implicitfast
is NOT behaviorally equivalent") is an instrument-specific measurement,
not a property of the integrator on this scene; the integrator decision
still needs the force trace, on the instrument the decision is for. What
survives both architectures: impratio=1 hardens nothing and lets the pads
sink 5 mm (5.1 / 5.22 — the second reversal holds), CG fails with
metre-per-second scrubbing on both (5.2 / 2.1 m/s — the failure is
robust, its number is not), and Newton + elliptic passes at 21–27 N of
grip everywhere. Every row's verdict agrees across the two machines
except PGS (§7.1); every row's *number* differs.
