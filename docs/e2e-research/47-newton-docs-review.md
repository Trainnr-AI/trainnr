# Newton, read docs-first: the solver stable and our parity ledger

*2026-08-27, branch `physics-newton`. Source: a repomix pack of
[newton-physics/newton](https://github.com/newton-physics/newton) at
`main`, version `1.6.0.dev0` (post-v1.5.0 development), reviewed
locally — every claim below is from the repo's own docs/source unless
marked MEASURED (run here today). This extends
[36-newton-status.md](36-newton-status.md) (2026-08-25); the verdict
refresh is at the bottom. Field-agent reports on the install probe,
solver theory, and the GPU path land separately.*

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
| Version pin | `mujoco~=3.11.0` on main (1.6.0.dev0) | **cannot coexist with the MuJoCo 3.12 we run** in one env; the identified-model contract is version-locked and Newton lags the lock |
| Contact `solref` | force-space re-conversion unless MJCF-authored (`SOLREF_MODE_RAW` preserved verbatim) | our gym-aloha-copied solimp/solref are MJCF-authored → preserved; but `save_to_mjcf` round-trip is lossy for force-space joints |
| Multi-world | model built from the **first world only**, replicated; worlds must be structurally identical | per-trial DYNAMICS variation (our `variations.py` DR) has no first-class door here; per-world values exist only via custom attributes |
| Non-convex meshes | convex-hulled at conversion (MuJoCo semantics — same as ours) | no change |
| Collision masks | reproduced exactly up to 32 biclique bits, then a "may allow extra contacts" fallback | our scenes are small; exact path applies |

The first two rows are decisive. Keyframes and sensors are the two
mechanisms our protocol and instrument rules are built on, and neither
survives import. Adopting Newton (engine) for evaluation would mean
rebuilding the reset mechanism and the entire observation layer in
Newton idioms — not an adapter, a port.

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
- FAQ now says Isaac Lab's Newton integration includes "basic
  manipulation" among initial environments (the 08-25 report: "not
  there yet") — still experimental, but the trigger condition in
  36's revisit rule is inching closer.
- `warp.sim` was deprecated in Warp 1.8, removed in 1.10 — Newton is
  its successor, so Warp-side sim code has exactly one home now.

## 6. Verdict refresh vs [36-newton-status.md](36-newton-status.md)

**The recommendation stands, now with the cost itemized.** "Stay
MJX/mujoco_warp for GPU rollouts" survives this review strengthened:
the docs themselves show that going through Newton (engine) buys us a
conversion layer that drops keyframes and sensors (the two things our
evaluation stack is built on), lags the MuJoCo version lock
(`~=3.11.0` vs our 3.12), and adds a solref/actuator translation whose
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
3. **implicitfast is behaviorally equivalent here** (PASS, ~equal cost)
   — so Newton (engine)'s silent integrator flip would not by itself
   break these scenes, and implicitfast is available as a stiffness
   headroom lever if identified gains ever need it.

Caveat, stated honestly: one trial per config, and the scripted
choreography's closed-loop corrections partially adapt to whatever
physics they get — so "fail" measures the pipeline outcome (the thing
we ship), not solver accuracy in isolation. The penetration column is
the solver-only signal.
