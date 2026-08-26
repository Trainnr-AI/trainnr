# Constraint solvers, mapped to our scenes: the field report

*2026-08-27, one agent against primary sources (MuJoCo computation/XML
docs, mjdata.h, mujoco_warp README, Newton engine docs, the sysid
toolbox source — all fetched/read that day), per the
one-agent-per-field discipline. Companion to
[47-newton-docs-review.md](47-newton-docs-review.md); the report
follows, lightly trimmed. Naming collision applies throughout:
"MuJoCo's Newton solver" (`mjSOL_NEWTON`) is unrelated to the NVIDIA
Newton engine.*

## 0. What our scenes actually run (read from code, not guessed)

Both rigs: `solver=Newton, iterations=100, tolerance=1e-8,
ls_iterations=50, integrator=Euler, timestep=0.002 (500 Hz),
cone=elliptic, impratio=10, noslip off`, control at 50 Hz. For the
SO-101 scenes everything but cone/impratio arrives as an unchosen
MuJoCo global default (`pin_nominal_options` — added the same day —
now declares the whole block); the ALOHA bundle authors cone/impratio
in XML and inherited the rest.

**Correction to prior shorthand:** the gripper pads are **condim=6**
(sliding + torsional + rolling) and the parts condim=4; MuJoCo's
priority rule takes the max, so **pad–part contacts run condim 6**.
Part–table contacts run condim 4.

**A folk trick worth knowing:** the inherited `solimp="2 1 0.01"`
(gym-aloha, Menagerie SO-ARM100 both) sets d_min=2 on a parameter
defined on [0,1]; the engine clamps to `mjMAXIMP=0.9999`
(mjmodel.h, verified) — i.e. "impedance pinned near 1, near-rigid
contact". Inherited folklore, not documented guidance.

## 1. The solver family (MuJoCo computation chapter, 2026-08-27)

All three solve the SAME convex soft-constraint problem; they differ
in how they get to the optimum:

- **Newton** (default): exact Newton's method, analytic second-order
  derivatives, Cholesky with rank-1 updates. "Quadratic convergence
  near the global minimum… usually around 5 [iterations], and rarely
  more than 20." The docs call it "better in every way."
- **CG**: nonlinear conjugate gradient (Hager-Zhang), first-order.
  Niche: very large models where Newton's per-iteration cost bites.
- **PGS**: projected Gauss-Seidel on the dual, sub-linear; niche is
  more-DOFs-than-constraints; needs special handling for elliptic
  cones (coordinate descent sticks in a continuum of local minima).
- **noslip**: not a solver — a post-pass PGS sweep with hard friction;
  the docs themselves call it "an ad-hoc correction that can
  occasionally cause instabilities." We do not use it and should not.

**impratio** (XML reference, verbatim): settings larger than 1 make
friction forces "harder" than normal forces, "having the general
effect of preventing slip, without increasing the actual friction
coefficient" — elliptic cones only; pyramidal behavior is
undocumented (open flag). And the modeling chapter's grip recipe,
verbatim: **"When contact slip is a problem, the best way to suppress
it is to use elliptic cones, large impratio, and the Newton algorithm
with very small tolerance."** Our configuration IS the documented
recipe — and `tools/solver-study.py` measured the same conclusion
independently (every other solver/cone fails the kitting referee).

## 2. Integrators, and the one place we sit against documented advice

- Euler: semi-implicit, damping implicit. Docs: "use for
  compatibility with older models."
- **implicitfast: "similar computational cost to Euler, yet provides
  increased stability, and is therefore a strict improvement. It is
  the recommended integrator for most models."**
- RK4: energy-conserving systems — not our regime.

We run Euler at 500 Hz in both rigs, and nobody chose it — it is the
inherited default (now pinned deliberately). Flipping to implicitfast
is measured-equivalent on kitting (docs/47 §7) but **changes the
nominal condition Paper 2 defines** — a measured decision to schedule,
not a hygiene commit.

**mujoco_warp gaps** (README, 2026-08-27): "IMPLICITFAST midpoint
integrator feature is not supported" (ambiguous phrasing — reads as
the midpoint sub-feature, not implicitfast wholesale; verify before
relying), no PGS/noslip, no PLUGIN actuators/sensors, Flex
experimental, no autodiff. **Intersection with our scenes: none** —
we need Newton solver + elliptic + Euler, all supported.

## 3. Solver diagnostics a harness should record (mjdata.h, verbatim)

| Field | Meaning | Why record it |
|---|---|---|
| `solver_niter[]` | iterations, **per island** (sum/max over islands) | converged or hit the cap |
| `solver[].improvement/.gradient/.nactive/.nchange` | per-iteration stats | convergence trace; contact-set churn = chattering grasp |
| `solver_fwdinv[2]` | forward-inverse consistency (needs `fwdinv` flag) | independent error check |
| `contact.dist` | "neg: penetration" | THE physical degradation metric |
| `efc_vel` | J·qvel in constraint space | **tangential slip at the pads — the direct measure of what impratio buys** |
| `efc_force`, `efc_state`, `efc_pos`, `efc_aref` | delivered force, cone regime, violation, reference accel | grip force actually applied; which cone regime each contact ended in |
| `ncon`, `nefc`, `warning[]` | scale + instability tripwires | normalize everything above |

Judgement recipe: compare max pad penetration, integrated tangential
`efc_vel` during the grip window, `solver_niter` distribution, and
task verdicts. (`tools/solver-study.py` records the first, third and
fourth today; `efc_vel` slip is the natural next column.)

## 4. The Newton ENGINE's solver stable

Eight solvers; the coordinate split carries the argument. Only
`SolverMuJoCo` (mujoco_warp: same physics, GPU) and
`SolverFeatherstone` use generalized coordinates; XPBD/VBD/Kamino/
SemiImplicit enforce joints as pairwise maximal-coordinate
constraints — the docs note they "do not use the articulation
kinematic-tree structure," which is exactly where compliance/drift
shows on a 6-DOF arm pinching a 20 g cube through condim-6 contact.
VBD and Kamino carry explicit experimental warnings. **The docs
contain no quantitative accuracy statement for XPBD/VBD on stiff
articulated contact** — flagged unverifiable from primary source;
stronger claims belong to the XPBD literature, not these docs.

## 5. Solver settings and sysid fidelity

The `mujoco.sysid` toolbox **does not pin or recommend solver
settings** — Gauss-Newton over batched `mujoco.rollout` steps the
user's compiled model with whatever `opt` it carries (verified in the
toolbox's residual module: `opt.timestep` used only for resampling,
no overrides). Consequence (synthesis, flagged as ours): **fits are fits
OF the discretization** — evaluate identified parameters under a
different solver/integrator and you silently change the model being
validated. Hence `pin_nominal_options` + `test_nominal_options.py`,
added 2026-08-27: the option block is part of the identified artifact.
One documented sysid note: the negative "direct" solref format is
"the recommended format for system identification" — relevant if we
ever fit contact stiffness/damping.

## Open flags (not verified)

1. impratio under pyramidal cones — undocumented.
2. mujoco_warp's "IMPLICITFAST midpoint" phrasing — test before
   relying on implicitfast-on-Warp.
3. No quantitative XPBD/VBD accuracy statements exist in Newton
   engine docs.
4. The sysid Colab notebook was not read cell-by-cell.
5. `solimp` d_min=2 clamp trick — verified from headers, blessed by
   no doc.
