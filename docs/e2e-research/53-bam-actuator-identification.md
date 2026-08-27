# BAM: a published, working answer to our own open question

*2026-08-28, read from two operator-supplied repomix packs:
[Rhoban/bam](https://github.com/Rhoban/bam) (ICRA 2025, Duclusaud &
Passault, Apache-2.0) and a downstream user, `mjlab_microduck` (an
RL locomotion project consuming BAM through
[mjlab](https://github.com/mujocolab/mjlab)). Every claim below is
read directly from the packed source unless marked INFERENCE.*

**The one-line finding:** docs/27 posed "does parametric identification
converge on a machine with backlash, no torque feedback and a plastic
gearbox?" as an open question. BAM is a published answer: yes, on
exactly that hardware class, including our own servo — Feetech
STS3215 is one of BAM's seven shipped actuators, with real
oscilloscope-measured firmware constants and downloadable raw
trajectories. It is not an identification of *our* arm (different
rig, different mounting, different load) — but it retires the
"has anyone even tried" half of the question, and gives us numbers
to compare against.

## 1. What BAM is

A friction-model identification pipeline plus a library of
pre-identified servos, built around one insight: MuJoCo's and
IsaacGym's native Coulomb-Viscous friction is "too simplistic to
accurately represent complex friction phenomena like Stribeck,
load-dependence or quadratic effects" (their words, citing their own
ICRA 2025 paper). BAM provides:

- Six friction models, **M1→M6**, in increasing expressiveness — M1
  is exactly MuJoCo's native model (`τ_fm = Kv|θ̇| + Kc`); M6 adds
  Stribeck (presliding), load-dependence (friction scales with
  transmitted torque), directionality (motor-side vs external-side
  contributions differ — captures backdrivability asymmetry), and a
  piecewise quadratic term for "harmonic-drive-like" nonlinear
  coupling.
- An identification pipeline: a **pendulum test bench** (rigid arms
  of varying length, weights, a mounting bracket, a bus interface),
  four excitation trajectories (`sin_time_square`, `lift_and_drop`,
  `up_and_down`, `sin_sin`, 6 s each) played at multiple P-gains and
  mass/length combinations, fit by **CMA-ES with BIPOP restart**
  (Optuna, 100k trials default) minimizing mean absolute position
  error across all logs simultaneously, with a held-out P-gain as a
  validation split to catch overfitting.
- Two rig-level nuisance parameters fit **alongside** the motor
  model, explicitly not treated as motor properties: `q_offset`
  (mounting misalignment) and `command_delay` (bus + firmware
  transport latency) — both flagged as needing re-estimation if the
  actuator is remounted or moved to different electronics. This is
  the same discipline our own `pin_nominal_options` doctrine follows
  (the rig's discretization is part of the identified artifact, not
  incidental).
- Seven shipped, pre-identified actuators: Dynamixel MX-64, MX-106,
  XL-320, XL-330, eRob80:50, eRob80:100, and **Feetech STS3215
  (7.4V)**. Raw recorded trajectories are downloadable from
  HuggingFace for STS3215 and three others (`feetech_sts3215_raw.zip`
  — a bucket-style link, not a stable dataset-repo URL; treat as
  best-effort hosting, not archival-grade, until re-verified).

## 2. The STS3215 identification, specifically

**Measured firmware constants** (BAM's Feetech actuator module, comment:
"determined using an oscilloscope and STS3215 actuators"):

| Constant | Value | What it is |
|---|---|---|
| `vin` | 7.4 V | the bench's supply voltage |
| `kp` (firmware) | 32 | the servo's own P-gain register, firmware units |
| `error_gain` | 0.166 | position-error × firmware-kp → duty-cycle conversion, oscilloscope-measured |
| `max_pwm` | 0.97 | measured duty-cycle ceiling (their comment: "TODO, but can we assume 1.0?" — flagged uncertain even by them) |
| `default_max_velocity` | `(3400·2π)/4096` rad/s | a firmware register-derived speed limit |

**A previously-undocumented-to-us firmware behavior**: the class
docstring states "the firmware rate-limits the target position it
feeds to its P controller, which makes `compute_control` stateful."
BAM's STS3215Actuator class carries an internal `q_target_smooth` state
that low-pass-filters the commanded target before the P-loop ever
sees it. **Our SO-101/ALOHA bundles do not model this** — our
position actuators apply the commanded target directly. If real, this
smoothing would shape every fast setpoint change our choreography
issues, and would be indistinguishable from "the arm is a bit
sluggish" without knowing to look for it specifically.

**Identified parameters** (`bam/params/feetech_sts3215_7_4V/`), the
two ends of the model hierarchy:

```
m1 (Coulomb-Viscous, MuJoCo-native equivalent):
  kt=1.178 N·m/A   R=2.479 Ω   armature=0.0261 kg·m²
  friction_base=0.0515   friction_viscous=0.0599
  q_offset=-0.0506 rad   command_delay=1.05 ms

m6 (full directional/quadratic model):
  kt=1.275   R=2.753   armature=0.0216
  friction_base=0.0533   friction_viscous=0.0282
  friction_stribeck=4.3e-5   dtheta_stribeck=0.371   alpha=9.93
  load_friction_motor=0.0449   load_friction_external=0.244
  load_friction_motor_stribeck=1.2e-4  load_friction_external_stribeck=0.142
  load_friction_motor_quad=0.0076   load_friction_external_quad=0.0034
  q_offset=-0.0683 rad   command_delay=4.98 ms
```

Two things worth flagging honestly: (a) these are **not** our arm —
BAM's own pendulum bench, their own mounting, their own mass/length
sweep, at 7.4V (our bundles' nominal is unstated/inherited — worth
checking against our own supply). Reusing them as our nominal
condition would be borrowing someone else's rig's `q_offset` and
`command_delay`, which their own docs say is rig-specific and not
guaranteed to transfer. (b) No MAE/accuracy number for the STS3215 fit
specifically was found in the packed docs — the ICRA paper likely has
it; not verified here.

## 3. The integration paths, and a real version wall

**MuJoCo CPU** (`bam.mujoco.MujocoController`, PyPI
`better-actuator-models[mujoco]`, unbounded `mujoco` dependency —
compatible with our `~=3.11.0` pin): actuators must be declared
`<motor>` in the MJCF, **not** `<position>` — our current SO-101/ALOHA
actuators are `<position>` with fixed gains, so adopting BAM's CPU
path means an actuator-type migration, not a drop-in overlay. At
runtime BAM **overwrites** `dof_frictionloss`, `dof_damping`, and
`armature` — any value we author in XML for those fields becomes dead
once a `MujocoController` drives that joint. Also models an optional
battery-sag term (`V_eff = V_in − R_drop·I`, current drawn as a
signed sum across all joints on one controller, clamped at zero so
regeneration can't raise the modeled bus voltage) — something our
rig's own SG90/STS setup has never modeled.

**MJX-Warp via mjlab** (`bam.mjlab.BamActuatorCfg`, PyPI
`better-actuator-models[mjlab]`): fully vectorized over parallel
environments in PyTorch, with per-env domain randomization built in —
supply voltage, drop resistance, **friction-budget scale**, and
command delay in simulation steps, all sampled once at init and held
for the episode. **This is a real version wall for us**: the `mjlab`
extra pins `mujoco-warp>=3.7,<3.8` and `warp-lang>=1.12,<1.13`
("mjlab 1.3 targets the MuJoCo Warp 3.7 era... otherwise `Simulation()`
raises `ls_parallel was removed`"). Our pin is `mujoco~=3.11.0` /
measured `warp-lang 1.14.0` (docs/49, docs/52). Trying `bam[mjlab]`
against our stack today would hit exactly the kind of breakage
Tuesday night's accidental 3.11→3.12 bump caused on our own suite —
this is not speculative, it's the same failure shape, pinned in their
own comment. A comparison would need an isolated old-stack venv (the
newton-probe pattern), not a same-env install.

## 4. What the downstream user (`mjlab_microduck`) teaches, beyond BAM's own docs

Two lessons from a project actually running BAM in a training loop:

**A "who owns this field" trap, generalized.** `FrictionDRBamActuator`
exists because "the canonical `bam.mjlab.BamActuator` exposes per-env
gain scaling (kp/kd) but no friction hook, and under BAM MuJoCo's
`dof_frictionloss` is zeroed in `edit_spec` (BAM computes friction
itself in `compute()`). So the stock `dr.dof_frictionloss` is a no-op
here." mjlab's own generic domain-randomization helper silently does
nothing once BAM has claimed that MJCF field — exactly our own
repo's recurring bug shape (two systems, one fact, one of them wins
silently) but discovered and fixed by someone else, for free.

**Backlash as a genuinely separate physical effect, with a concrete
MJCF pattern.** `BacklashEncoderBamActuator` adds an unactuated
`passive_<joint>_backlash` hinge in series with each servo joint in
the MJCF (motor output → backlash play → link). The real firmware's
magnetic encoder sits on the **output** side of that gap, so the
position-loop error is computed on `qpos[main] + qpos[backlash]`
while velocity feedback (which drives back-EMF and friction — rotor
physics) stays motor-side only. This is a direct, concrete recipe for
the backlash our own docs/27 already measured on a real STS3215 from
third-party video (**0.0151 rad ≈ 0.87°, ~10 encoder counts, ~2× the
<0.5° datasheet spec**) — a phenomenon neither our SO-101 nor ALOHA
bundle currently models at all.

## 5. Corrections to our own docs

- **docs/27's open question is answered, in part.** "Does parametric
  identification converge on a machine with backlash, no torque
  feedback and a plastic gearbox?" — yes, BAM does exactly this,
  published, for STS3215 among others. What remains genuinely open:
  whether it converges on *our* arm, in *our* rig configuration, with
  *our* electronics — a different, narrower question than "has anyone
  ever tried."
- **docs/23's "no published `mujoco.sysid` application to any hobby
  servo" stands, narrowly** — BAM does not use `mujoco.sysid`; it has
  its own CMA-ES rollout-matching optimizer against a from-scratch
  friction-budget simulator. The claim about `mujoco.sysid`
  specifically is unaffected; the broader "nobody has identified a
  hobby servo" reading it invited is now wrong and should not be
  repeated that way.

## 6. What this is worth doing, if anything (scoped options, not a recommendation)

1. **Cheapest, zero-hardware**: download BAM's raw STS3215 logs and
   the `m6.json`, replay them through our own scene at the docstring's
   reported conditions, and check whether adopting BAM's friction
   *model shape* (M1→M6, as a MuJoCo `<motor>` overlay) changes
   anything about the kitting/reach tasks' behavior versus our current
   `<position>` actuators — a physics comparison, not a claim about
   our specific hardware.
2. **CPU integration only** — `bam[mujoco]` has no version conflict
   with our stack. The GPU path (`bam[mjlab]`) is blocked by the
   version wall in §3 until BAM updates its mjlab pin or we accept a
   pinned side-venv comparison.
3. **Backlash modeling** — the `BacklashEncoderBamActuator` pattern is
   directly reusable MJCF-and-signal-routing knowledge independent of
   BAM itself: an extra passive hinge, position read through it,
   velocity not. Worth remembering whenever the deferred feedback-arm
   purchase happens and identification on our own hardware begins.
4. **Do nothing yet** — this is research, not a blocker; nothing here
   changes what's shippable today.

## 7. Postscript (2026-08-28, same day): the library is landed, all 8 actuators

Following the operator's direct question — "how can we use all the
BAM servos and keep adding more?" — the answer is built, not just
designed:

- **`robots/actuators/`**: all 8 BAM actuators vendored (Dynamixel
  MX-64/MX-106/XL-320/XL-330, eRob80:50/eRob80:100, Feetech STS3215
  7.4V, Waveshare ST3025), all 6 model tiers each (48 JSON files), a
  `PROVENANCE.json` per actuator naming source/citation/license — a
  directory with none is refused at load time, not read as trustworthy.
- **`rq_pipeline/robot/friction_budget.py`**: the M1→M6 equations as
  ONE function, not six — the models are strictly nested (confirmed
  by inspecting all 48 files' field sets: M3/M4 use an undirected
  `load_friction_base`, M5/M6 replace it with the directional
  `load_friction_motor`/`load_friction_external` pair, never both),
  so which optional `FrictionParams` fields are populated selects the
  tier. Pure numpy, lazily imported (not a core dependency) —
  identical under `jax.numpy`, so it runs unchanged on CPU MuJoCo or
  MJX-Warp with no engine import at all. 7 tests check it against
  values hand-computed from the equations, independent of the code.
- **`rq_pipeline/robot/actuator_library.py`**: `list_actuators`,
  `list_models`, `load_actuator(slug, tier)` — the one door the data
  enters through, provenance-checked, fail-loud on an unknown
  actuator/tier/source. 10 tests, run against the real vendored
  files, not fixtures — including a pin-both-ends check of STS3215's
  published `kt` against the value cited in §2 above.
- **`tools/sync-bam-actuators.py`**: reads either a BAM git checkout
  or a repomix XML pack (the format used all session) and vendors
  new/changed `params/*`, refusing a changed file with no `--version`
  — verified against the actual pack: dry-run reports 0 changes
  (idempotent), and a deliberately corrupted value is correctly
  refused without `--version` and correctly accepted with it.

**Deliberately NOT built this pass**: the per-step controller that
would let M2-M6 (velocity- and torque-dependent) actually drive a
running simulation — those terms need live `tau_m`/`tau_e` recomputed
every physics step, the way BAM's own `MujocoController.update()`
does, which is real design work (where does it hook into `Stepper`?)
kept separate on purpose. M1 alone needs no such hook — it's exactly
`dof_frictionloss`/`dof_damping` MuJoCo already applies natively, so a
scene can adopt an M1 fit today as a static override with zero new
runtime code. 320 tests green.
