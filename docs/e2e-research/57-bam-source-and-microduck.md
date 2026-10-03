# BAM's code and its first production consumer: the bridge already exists, and where provenance and randomization are left to the reader

*2026-08-31 (versions as of that day: BAM v1.0.2, mjlab 1.6). Sources:
repomix packs of `Rhoban/bam` (v1.0.2, 125 files)
and `pollen-robotics/microduck_rl` (163 files), one field agent each,
primary source only. Companion to [53](53-bam-actuator-identification.md)
(BAM's paper and published fits) and [56](56-mjlab.md) (mjlab whole).
Foreign paths carry a repo prefix in backticks. This read feeds the
cross-framework design in [58](58-cross-framework-setup.md).*

## 1. The headline: Rhoban ships a BAM→mjlab actuator, and Pollen runs it in production

`bam:bam/mjlab.py` — `BamActuatorCfg`/`BamActuator`, torch-vectorized
over `(num_envs, num_joints)`, evaluating the identified law each step
and writing `dof_frictionloss`/`dof_damping` per world. Pollen's
microduck (14 Dynamixel XL330s, ~800 g biped, 13 registered mjlab
tasks) subclasses it (`microduck:src/mjlab_microduck/actuator/
friction_dr_bam.py`) and trains everything through it. The bridge
docs/56 §7 proposed is not hypothetical; it is deployed. What follows
is what its current shape gets wrong — each item measured, not
inferred.

## 2. BAM's models and fitter, from code

All six models are ONE class toggled by four booleans
(`bam:bam/model.py`). The friction budget (θ̇ = joint velocity, τ_m =
motor torque, τ_e = external torque; every K is a fitted constant):

- M1: viscous + Coulomb — `Kv·|θ̇| + Kc`
- M2: + Stribeck boost at low speed — `exp(−|θ̇/θ̇_s|^α)·K_cs`
- M3: + load-dependent term `Kl·|τ_m − τ_e|`
- M5: the load term splits into motor and external coefficients
  (`|K_m·τ_m − K_e·τ_e|`) with Stribeck twins
- M6: + a quadratic load term, gated to opposing-sign torques only
  (a code detail their docs omit)

Friction is applied as a *clip on the stopping torque*, not an added
force — which is why the MuJoCo mechanism is per-step writes into
`dof_frictionloss`/`dof_damping`, and why BAM's own MJCF exporter
(`bam:bam/to_mujoco.py`) is deprecated with the warning that
load-dependent friction "can't be exported exactly to MuJoCo".
**This resolves docs/56 §8's open question: the `<dcmotor>`/native-slot
path is dead; Rhoban's write-the-dof-fields mechanism is the way.**

The fitter (`bam:bam/fit.py`): Optuna CMA-ES (bipop restarts), 100k
trials default, box bounds, objective = **position MAE in radians,
nothing else** (current/voltage/temperature are logged and never enter
the loss). Rollouts during fitting are BAM's own Euler simulator over a
vectorized batch of all logs. A `--validation_kp` holdout exists — its
MAE is printed to W&B **and never written into the params file**.

Data: a pendulum bench, six-second trajectories (a chirp, a
lift-and-torque-disable drop that separates viscous friction from
back-EMF, slow cubics), swept over P-gains and masses — ~240 recordings
for the XL330. Drivers ship for Dynamixel, Feetech, eRob. Eight motors
× six models = 48 published fits; Unitree Go1 has an actuator class and
**no params**.

Not modeled anywhere: backlash, encoder quantization, temperature
(logged, unused), elasticity, thermal derating. `q_offset` is fitted
globally ("if the actuator is remounted … that assumption no longer
holds" — their docs); `command_delay` is a property of the *rig*, and
they say so.

## 3. The artifact has no envelope

A published fit is a flat `{param: float}` JSON plus exactly two keys:
`"model"`, `"actuator"`. Lost: schema version, units, date, git SHA,
dataset hash, bench conditions, optimizer config, train MAE, the
held-out MAE they computed, author. **No uncertainty of any kind.**
Rail-hitting fits ship silently (`bam:bam/params/xl330/m4.json`:
`alpha ≈ 9.999999997`; several load terms at ~1e-13). Key sets differ
per motor (eRob has no `R`; MX lacks `q_offset`; Feetech adds
`error_gain_ratio`) and the loader silently ignores unknown keys.
Directory names disagree with the `actuator` key
(`params/feetech_sts3215_7_4V/` contains `"actuator": "sts3215"`).
Their docs claim `vin`/`kp` are read from the JSON; the code reads them
from hard-coded class constructor defaults. Zero tests in the repo; CI
runs `uv sync` only.

## 4. The version wall, confirmed at the pin

`bam:pyproject.toml` `[mjlab]` extra: `mjlab>=1.3,<1.4`,
`mujoco-warp>=3.7,<3.8`, `warp-lang>=1.12,<1.13`, Python `>=3.12,<3.13`
— against today's mjlab 1.6 / mujoco-warp 3.11 / warp 1.16+. The inline
comment names the exact removed APIs (`ls_parallel`, `wp.context`).
Microduck escapes by depending on an **unpinned git branch**
(`mjlab_frictionloss`) with a commented-out path to a developer's home
directory as the alternative. The wall is in the *mjlab-facing kernel
only* — the model math is numpy with no pins, and the whole M6 kernel
mirrored in `microduck:scripts/validate_bam_testbench.py` is ~30 lines.

## 5. What production use looks like (microduck), catalogued

- **Provenance:** fitted params resolve via a private API
  (`bam.model._resolve_json_path`) in one script and from
  `~/Rhoban/bam/params/xl330/m6_new.json` on a laptop in another. No
  committed JSON, no hash, no date. Nothing can answer "which fit was
  this ONNX trained against." Four rival actuator XML classes are named
  after versions and individual developers.
- **Silent no-op DR, five instances:** `dr.dof_frictionloss` and
  `dr.joint_damping` randomize fields the BAM actuator overwrites every
  step; an IMU randomizer wrote a field nothing reads; a mass
  randomizer was a no-op under mjlab 1.3; config-dict writes after
  manager init are ignored. None raised. A no-op startup event exists
  *only* to carry the `@requires_model_fields` decorator, and every env
  must remember to register it or per-env friction silently collapses.
- **Fitted = exact, guessed = randomized:** the only actuator DR is a
  hand-typed ±10% scalar on the friction budget; `kt`, `R`, Stribeck
  shape are treated as exact, while voltage sag `(0.0, 0.2)` and delay
  `3–6 ticks` are undocumented guesses (their own testbench uses 0–3
  for the same servo). Meanwhile genuinely *measured* DR exists too —
  CoM ranges capped by the foot support polygon after an audit, a
  1-tick joint-velocity lag traced to Dynamixel firmware — proof the
  team knows the difference and had no machinery to enforce it.
- **The deployment contract is folklore:** the on-robot runtime is a
  Rust binary in another repo; the flag `--action-scale 0.8` contradicts
  the trained `scale = 1.0`; a kp-ratio default would silently detune
  kp 200→120; the obs-comparison script is hardcoded to a stale 51-D
  layout against the current 61-D contract. All documented only in
  prose plan documents.
- **Verification is one servo on a bench** with thresholds "chosen by
  feel" (`0.01` rad, `1.5×`), validating a *replica* kernel rather than
  the class they train with, reading data from `~/Rhoban/bam`. For the
  whole robot: train → deploy → watch the video ("rolls but
  face-plants 1 in 3" — their AGENTS.md's own reporting standard).
- Real engineering worth respecting alongside: the obs normalizer baked
  into the ONNX graph (hand-conversion bug class killed), a 61-D obs
  contract shared across 13 tasks for runtime hot-swap, NaN
  terminations + contact-buffer sizing lessons, an aarch64 CUDA torch
  pin locked by a test.

## 6. What this proves for the design in 58

1. Demand is real and current: two independent teams (Rhoban, Pollen)
   built the identified-actuator→mjlab path without us.
2. The path's weakest links are exactly trainnr's strengths: artifact
   provenance, uncertainty, refusal semantics, declared perturbations,
   machine-readable contracts, statistics with thresholds that are
   declared rather than felt.
3. The maintenance vacuum is concrete: the kernel is small, the wall is
   real, and nobody owns keeping it current with mjlab's monthly churn.
4. The mechanism question is settled (per-step dof writes), so the
   engineering risk of the bridge is low and measured.
