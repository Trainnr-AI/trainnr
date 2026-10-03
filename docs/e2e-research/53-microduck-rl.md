# microduck_rl, read code-first: the sim2real recipe of an 800 g servo biped, and what it says about our servo model

> **Archived (2026-09-07).** Written on branch `platform-2026-08-29`, which was retired on 2026-08-31 (docs/07, "a gate dissolved"): the design it describes - the pipeline's own MJX reinforcement-learning package and the actuator law as a runtime module - was superseded by `trainnr_mjlab` (docs/e2e-research/58) and never merged, so code paths named below may not exist on `main`. Restored to `main` unchanged because `trainnr/trainnr/robot/friction_budget.py` and docs/e2e-research/53's successors cite it by number.

*2026-08-28, branch `physics-newton-mac-2026-08-27`. Source: a repomix
pack of [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
at `develop` (163 files, 37,007 lines, Apache-2.0; the newest dated
design doc is 2026-08-04), reviewed locally — every claim below is from
the pack unless marked FETCHED (PyPI / GitHub, today) or MEASURED (run
here today). Five field agents read one field each against the pack —
actuator physics, backlash, RL task design, deployment, process — and
their reports are §1–§5; §0 is what the operator measured from the
BAM package itself before they returned.*

**Why this repo.** It is the complete sim2real recipe for a robot in
our price class — Dynamixel XL330 servos, a Raspberry Pi, ~800 g — with
the actuator modelled *"down to its voltage control law instead of an
ideal PD"* because *"at this scale … actuator fidelity is most of the
sim2real gap"* (`README.md`, "Actuator model"). That is our wedge stated
by someone else, with a different answer to it: they take Rhoban's
BAM identification of the servo type; we identify the joint in the
arm. The repo pins `mjlab==1.3.0`, `warp-lang==1.12.0`, `torch==2.9.1`,
`better-actuator-models`, Python `>=3.12,<3.13` (`pyproject.toml`, each
pin with a dated reason).

## 0. MEASURED first: the BAM package, and where `kp=17.8` came from

`better-actuator-models` 1.0.2 (PyPI, released 2026-07-16; FETCHED)
installs on Python 3.12 only — our train venv, not the 3.11 sim venv —
and its core needs nothing but numpy and colorama. Extras: `mujoco`,
`mjlab` (pins mjlab 1.3, mujoco-warp 3.7, warp 1.12) and
`identification` (rustypot ≥ 1.5, dynamixel_sdk, optuna, cmaes, wandb,
PyQt5 — the bench toolchain ships in the wheel). Installed into a
scratch venv here (MEASURED):

| What ships | Detail |
|---|---|
| Identified parameters | In the 1.0.2 wheel: `xl330`, `xl320`, `mx64`, `mx106`, `erob80_50`, `erob80_100` — models M1–M6 each; **`feetech_sts3215_7_4V` — M1 only**; a `unitree_go1` actuator class without parameters. Upstream `main` (FETCHED by field A at b8207484, 2026-08-27) is ahead of the wheel: the STS3215 at **M1–M6** (M6: `kt 1.275, R 2.75, armature 0.0216, friction_base 0.053, K_v 0.028, max_velocity 5.10 rad/s, error_gain_ratio 1.16, command_delay 5.0 ms`) and a `waveshare_st3025`. **Nothing for the XM430/XM540** (ALOHA-2) on either. |
| Model ladder (*bam/model.py*) | M1 Coulomb; M2 Stribeck; M3 load-dependent; M4–M6 combinations, M6 = Stribeck + load-dependent (motor/external, linear and quadratic) + viscous. Parameters: `kt`, `R`, `armature`, `q_offset`, `friction_base`, `friction_viscous` (M1), plus the Stribeck/load terms. |
| The STS3215 law (*bam/feetech/actuator.py*) | `STS3215Actuator(VoltageControlledActuator)`: `vin = 7.4`, firmware `kp = 32`, `error_gain = 0.166`, `max_pwm = 0.97` (*"TODO, but can we assume 1.0?"*); duty = clamp(kp · error_gain · (q_target − q), ±max_pwm); applied voltage = vin · duty; torque from the voltage law with back-EMF and the friction model. |
| The bench (*bam/testbench.py*, *bam/feetech/record.py*) | a `Pendulum` — a mass on a lever (`--mass`, `--length`), trajectories such as `lift_and_drop`, the servo driven through its own bus (`--port`, `--id`, `--kp`, `--vin`); *fit.py* fits with optuna/cmaes. |
| *bam/to_mujoco.py* | **deprecated by BAM itself**: *"the identified model is collapsed into the constant `kp`, `damping`, `frictionloss`, `armature` and `forcerange` of a MuJoCo position actuator, which cannot reproduce the load-dependent friction or the torque-dependent damping that BAM identifies. Prefer `bam.mujoco.MujocoController` (CPU) or `bam.mjlab.BamActuatorCfg` (GPU), which evaluate the actual model at each simulation step."* The mapping: `forcerange = vin·kt/R`, `kp = error_gain·kp_fw·vin·max_pwm·kt/R`, `damping = friction_viscous + kt²/R`, `frictionloss = friction_base`, `armature` as identified. |

**The provenance of the industry's constants.** Running BAM's own
`voltage_controlled_to_mujoco` on the shipped STS3215 M1 parameters
(`kt = 1.2116`, `R = 2.6762`, `armature = 0.0284`, `friction_base =
0.0524`, `friction_viscous = 0.0591`; MEASURED):

| Setting | `kp` | `damping` | `frictionloss` | `armature` | `forcerange` |
|---|---|---|---|---|---|
| 7.4 V, `max_pwm = 0.97` (BAM default) | 17.263 | 0.608 | 0.052 | 0.028 | 3.35 N·m |
| 7.4 V, `max_pwm = 1.0` | **17.797** | **0.608** | 0.052 | 0.028 | 3.35 N·m |
| the same 7.4 V winding driven at 12 V, `max_pwm = 1.0` — *not* the STS3215-12V unit, whose winding differs | 28.0 | 0.608 | 0.052 | 0.028 | 5.43 N·m |

`kp = 17.8, damping = 0.60` is the pair Lightwheel's `leisaac` and
Positronic ship for every SO-101 joint and its gripper
([29 §5](29-the-company.md), [30 §3.3](30-the-pipeline.md)). It is not
a guess: it is one STS3215 identified on a pendulum bench at 7.4 V,
collapsed to first order by an export BAM now deprecates, then copied
to six loaded joints, a gripper and a 12 V arm. Our docs said "one
guess, copied"; corrected today to "one measurement, copied" in
docs/23, docs/25, 29, 30 and the `so101-nominal` README — the wedge is
unchanged (one servo on a bench is not six joints in an arm at the
arm's voltage — and BAM's servo is the 7.4 V winding, while the SO-101
evidence we hold is a 12 V unit, docs/26), the caption is now true. Beside it, `so101-nominal`
itself is Menagerie's `kp = 50, dampratio = 1, frictionloss = 0.1,
armature = 0.1` on every joint, and the real STS3215's measured
nonlinearities — 0.87° backlash, a 10-count firmware dead zone, an
overload governor ([27 §5](27-open-questions.md)) — are in none of the
three parameterisations. microduck's backlash twin is ±1° per joint
(§2): the same number, modelled.

**Two version facts for our GPU path** (FETCHED): mjlab 1.6.0
(2026-08-09) pins `mujoco ~= 3.11.0`, `mujoco-warp ~= 3.11.0`,
`warp-lang >= 1.14`, `rsl-rl-lib == 5.4.2`; mujoco-warp 3.12.0
(2026-08-20) needs `mujoco >= 3.11`, `warp >= 1.15`. microduck stays on
mjlab 1.3.0 (mujoco 3.10.x) on purpose (`pyproject.toml`). Our locked
sim venv is mujoco 3.11.0, the train venv 3.12.0 ([52](52-warp-determinism-mjwarp.md)).

## 1. Actuator physics and identification (field A)

**The torque, per physics step** (5 ms). The pack's
*actuator/friction_dr_bam.py* is a 114-line shim over upstream
`bam.mjlab.BamActuator` (FETCHED at Rhoban/bam `main` b8207484 and the
pinned branch `mjlab_frictionloss` 57d13ead); it overrides one method,
multiplying the friction budget by a per-env `friction_scale`. Upstream
`compute`: (1) the previous step's `qfrc_actuator` is the motor-side
load; (2) sag — `I = Σ duty·τ_prev/kt`, `vin = vin_env − R_drop·I`,
floored at `vin_min`; (3) the firmware P law — `duty = (q_target −
q)·kp·error_gain`, clipped to ±`max_pwm`, `V = vin·duty` (XL330
`error_gain = (4096/2π)/(256·885)`; microduck sets firmware kp 200);
(4) the DC motor — `τ_m = kt·V/R − kt²·q̇/R`, the second term the
back-EMF; (5) the external load `τ_e = −qfrc_bias + qfrc_constraint −
qfrc_friction`, the friction term scattered from the previous solve so
load-dependent friction does not feed on itself; (6) the M6 budget —
`friction_base + |K_e·τ_e − K_m·τ_m| + s·(K_cs + |K_es·τ_e − K_ms·τ_m|
+ Q)`, `s = exp(−(|q̇|/dθ_s)^α)`, `Q` quadratic in whichever of τ_e,
τ_m is larger; viscous `K_v` unscaled; (7) **friction is not added to
the torque** — `dof_frictionloss` and `dof_damping` are written into
the per-world model every step and MuJoCo's solver does the
stick/slip; (8) `τ_m` drives a `<motor>` — `edit_spec` converts every
targeted `<position>` to a motor with `forcerange = ±max(vin)·kt/R`,
sets the joint `armature` to BAM's, zeroes damping and frictionloss,
and stiffens `solref_friction` because MuJoCo-Warp has no noslip and a
held joint creeps. Consequence: *robot/microduck/joints_properties.xml*'s
`chosen_actuator` (`kp=0.55`, `damping 0.053`, `frictionloss 0.0048`) is
dead text under BAM — the robot's MJCF no longer states its own
actuator physics.

**DR** (*robot/microduck_constants.py* L121-134): `motor_name="xl330",
model="m6", kp_fw=200`, per-env `vin_range=(6.5, 8.2)` sampled at
startup and held across resets, `vin_drop_gain_range=(0.0, 0.2)`,
`vin_min=6.0`, a delay of 3–6 physics steps = 15–30 ms through mjlab's
`DelayBuffer` on the position command (lag resampled per step); the
friction scale 0.9–1.1 per episode, reset to 1.0 first; gain scales
averaged to one scalar per env. The identified `command_delay` in the
JSON (10.2 ms for the XL330) is **not used** by the mjlab actuator —
only by BAM's own simulator for fits. A drift: the pack's kwarg
`vin_drop_gain_range` (V = gain·Σ|τ|) exists in neither upstream ref,
which expose `vin_drop_resistance_range` in ohms; the pack pins BAM by
git branch and ships no lockfile, so the revision that resolves is not
verifiable.

**The bench.** One XL330 on a printed bracket, a single hinge, a
printed arm with a weight, ±80° (*robot/xl330_test_bench/xl330_test_bench.xml*;
the payload is 0.1 kg in the XML comment and 0.12 kg in
*robot/testbench_constants.py* — they disagree); rustypot at 1 Mbaud.
Protocol A, kernel validation (*scripts/validate_bam_testbench.py*):
BAM's processed logs replayed through BAM's Python simulator and
through a hand-written CPU-MuJoCo M6; pass = mean |bam − mj| ≤ 0.01
rad and MAE_mj ≤ 1.5 × MAE_bam — kernel against kernel, **no absolute
sim-vs-real threshold**; its reference `m6_new.json` is a local file
absent upstream. Protocol B, closed loop (*scripts/testbench_sim2real.py*):
a PPO policy trained on the bench env (no DR) drives 4 s holds at seeded
random targets at 50 Hz; on the real servo P = 200 is written and read
back, **I = D = 0**; compared: MAE(q_sim, q_real), each side's tracking
MAE, velocity RMS, action MAE, a four-panel plot — **no pass
criterion**, judged by eye. BAM's own bench for the shipped parameters
is a pendulum on a U2D2 with several arm lengths and masses (FETCHED
docs); the fit is CMA-ES over a scalar position MAE, ~100k trials.

**Ours against theirs.** A MuJoCo `<position kp dampratio>` — SO-101
`kp=50 dampratio=1 forcerange ±3.5`, joint `frictionloss 0.1 armature
0.1` (Menagerie), ALOHA-2 per-joint `kp 10.4–265` with Menagerie
damping, friction and armature; identification = `identify()` over a
`ParameterSpec` writing kp, damping, frictionloss and armature into the
`MjSpec` (`trainnr/trainnr/robot/sts_synth.py`; real fits only for
the N20 drivetrain, `trainnr/trainnr/robot/drivetrain_fit.py`); DR
= appliers scaling the identified model
(`trainnr/trainnr/physics/variations.py`); no delay, deliberately
("an explicit fitted parameter, when we model it",
`trainnr/trainnr/physics/mujoco_backend.py`).

| Effect | BAM M6 | Ours | An arm at 50 Hz, pick-and-place |
|---|---|---|---|
| Voltage saturation + back-EMF (`τ ≤ kt(vin − kt·q̇)/R`) | yes | a `forcerange` clamp, not speed-dependent | minor, except the gripper close |
| Coulomb + Stribeck | yes | Coulomb (`frictionloss`) only | **major**: settle hysteresis, the hold band under gravity (the droop `tasks/so101.py` documents at kp 50) |
| Load-dependent, directional friction | yes | none | **major** for a geared STS3215 holding a load |
| Viscous | yes | `damping` (fit) / `dampratio` (nominal) | same |
| Supply sag | per env | none | skip on a bench supply; the Pico car's battery only |
| Command delay | 3–6 steps, a hand range | none | the identified 5–10 ms is under one 20 ms tick — small for ACT |
| Firmware rate-limited target (STS3215 `max_velocity` ≈ 5.1 rad/s, stateful) | yes | none | **major** for ACT chunks with large per-tick goal jumps |
| Dead zone, overload governor ([27 §5](27-open-questions.md)) | none | none (synthetic only) | open on both sides |

**Could the parameters drop in?** STS3215: upstream ships M1–M6 for
the **7.4 V** winding; our STS3215 evidence is a 12 V unit — a
different winding — and which unit we buy decides whether these apply.
XM430/XM540: no parameters; the nearest Dynamixels are the MX-64/106
at 15 V; fitting one needs their pendulum bench,
`bam.dynamixel.all_record` and `bam.fit`. For every Dynamixel BAM
models a P-only law and the bench script forces I = D = 0 — shipped
firmware defaults are not what is modelled. What replacing our position
actuator would change: the actuator becomes Python before every
`mj_step` — ten calls per control tick reading `qfrc_*` and `efc` and
writing `dof_frictionloss`, `dof_damping`, `ctrl` — against a
`Stepper.advance` that is `ctrl[:] = u` then `mj_step × 10`; on
MJX-Warp there is no host code per step today, and BAM's GPU path
exists only inside mjlab's torch actuator framework; a stamp would have
to pin the BAM revision as well as the JSON, because the law lives in
the package (the `vin_drop_*` drift is exactly that failure); and our
own Newton row in docs/33 — "a conversion layer erases what
identification measured" — would apply to us: `edit_spec` discards the
MJCF's kp/kv. The identified position law and the BAM law are
alternatives, not layers.

## 2. Backlash and the model variants (field B)

**The mechanism** (*robot/microduck/add_backlash.py*;
*robot/microduck/robot_walk_backlash.xml* lines 83–93). After every
self-closing `<joint … class="chosen_actuator"/>` the script emits a
second hinge on the same body and axis — `passive_<name>_backlash`,
`class="backlash"` — with `damping 0.01, frictionloss 0, armature
0.001, limited, range ±1°` (from `--backlash-deg 2.0`, the TOTAL play,
"what you measure wiggling the horn with the servo held"),
`solreflimit 0.01 1` ("the default 0.02 lets the joint overshoot its
limits ~2× under load; 0.01 = 2·sim_dt, the stiffest stable setting"),
`solimplimit 0.95 0.999 0.0001 0.5 2`. Two hinges on one body compose
in series — link angle = `qpos[servo] + qpos[backlash]` — and the tiny
armature is what makes the singular 2×2 mass block on a shared axis
solvable: the motor winds through the dead zone before the link moves.
No equality, no intermediate body. Fourteen joints get it (the roller
wheels are `passive_wheel` and skipped; the head servos are not
excluded). Generation is a line-by-line regex rewrite of the
onshape-to-robot export, the last `post_import_commands` entry of
*config_mjcf_walk_backlash.json*; the six export configs differ only by
output name and that line, six near-identical `sed -i` lists are
hand-duplicated, and the backlash scenes' keyframes are hand-widened
qpos strings with interleaved zeros. **The 2° is a CLI default; no
measurement of the XL330's play appears anywhere in the pack.**

**The encoder reads through the play** (*tasks/mdp.py*,
*actuator/friction_dr_bam.py*, *tasks/backlash.py*):
`joint_pos_rel_backlash = joint_pos[servo] + joint_pos[backlash]·mask −
default`, with the encoder-bias DR on the servo term only ("one
encoder per servo → one bias per joint; the backlash summand stays
raw"); `joint_vel_rel_backlash` sums the velocities ("the firmware
derives present_velocity from encoder positions, so it also sees the
backlash motion"); the mask is 0 where no passive hinge exists, so the
same functions run on plain models. The servo's own loop closes on the
output side too — `BacklashEncoderBamActuator.get_command` adds the
backlash angle to the measured position before the BAM PD, while
`cmd.vel` stays motor-side ("drives back-EMF and friction, which are
rotor physics"); torque is applied to the servo joint only. Rewards
follow the sensor view — head-pose tracking measures the summed angle
("measuring the servo alone would let the head droop the backlash play
reward-free AND penalize the policy for compensating it");
`dof_pos_limits` is scoped to the servo joints because the backlash
joints "spend their life pinned against their ±1° limits"; AGENTS.md's
invariant: an observation remapped to a sensor view must have its
tracking reward measure the same view. Action and observation dims are
unchanged, so "runtime/export need no changes" — a claim with **no test
behind it**: `tests/` mentions backlash once, in a comment.

**Ours.** No trainnr bundle models gear play;
`trainnr/trainnr/robot/sts_synth.py` models the firmware dead zone
in the *measurement* path (`DEAD_ZONE_RAD = 10 · QUANTUM_RAD`), and
docs/26 names "backlash as a mechanical element" as the open risk.
Where a twin goes: one function `add_backlash(spec, play_by_joint)`
beside `scale_dynamics` in `trainnr/trainnr/tasks/aloha2/rig.py` —
an uncompiled-`MjSpec` mutation adding the passive hinge per servo
joint — applied to the arm spec before it is attached
(`trainnr/trainnr/tasks/so101.py`, `attach_arm` in
`trainnr/trainnr/tasks/components.py`); no XML rewriting, no
duplicated configs. Two design facts first: our arm state is
`<jointpos>/<jointvel>` on the servo joints, so a twin must keep
`state_width` and report servo + backlash per joint (a sensor on the
passive hinge, summed in the state reader); and a MuJoCo `<position>`
actuator feeds back on its own joint — motor side — so output-side
feedback needs the servo PD in code, a torque actuator plus a
firmware-emulation step (microduck does it inside BAM). A twin needs
its own stamp — `bundle_hash` is over bundle bytes and `Task.stamp`
over the spec, so the play values must live in a stamped field or the
nominal model and the twin share an identity, and `certify()` could
not say which one a policy was evaluated on. Tests to pin: `njnt` grows
by exactly the servo count with `nu` and `state_width` unchanged; every
passive range = ±play/2, at 0 in the home keyframe; `state[j] ==
qpos[servo] + qpos[backlash]` after a random step; the stamps differ;
and the scripted expert clears `tasks/acceptance.py` on the twin at the
measured play — microduck's "no runtime change" claim, made into the
test they never wrote.

| Robot | Servo | Play we hold | Verdict |
|---|---|---|---|
| Yellow arm | SG90 / MG90S | none — "every dynamics number beyond geometry is NOMINAL hobby-servo guesswork" (`trainnr/trainnr/tasks/yellow.py`) | needs it most; no encoder to read through |
| SO-101 | STS3215 | 0.0151 rad ≈ 0.87°, ~2× the datasheet, plus a 10-count firmware dead zone, stacking to a ~1.8° blind band ([27 §5](27-open-questions.md); a third-party video, not archived) | first candidate — the number exists |
| ALOHA-2 | XM430 / XM540 | none | measure before modelling; Menagerie's armature 0.38–0.40 and damping 18–20 likely swallow it |

## 3. RL task design on mjlab (field C)

**One env cfg, anatomised** (*tasks/microduck_velocity_env_cfg.py*, the
shared base; *tasks/microduck_standup_env_cfg.py*, the episodic
template). Each is mjlab's `make_velocity_env_cfg()` mutated (L264) —
sim dt, decimation, base terminations and un-overridden base rewards
are inherited and not in the pack. Physics dt 0.005 × decimation 4 →
50 Hz control (*robot/microduck/add_backlash.py*, *scripts/infer_policy.py*);
a velocity episode is 1,000 steps = 20 s, standup 6 s; PPO rolls 24
steps per iteration, so every curriculum "iteration" is ×24 env steps.
Scene: one entity, three sensors (feet net-force contact with air
time, trunk self-collision, a per-foot terrain ray); `num_envs` is a
CLI flag (4096). Actuator: the BAM M6 XL330 voltage model with firmware
kp 200, per-env supply 6.5–8.2 V, a load-sag gain 0–0.2 with a 6.0 V
floor, a command delay of 3–6 ticks (*robot/microduck_constants.py*
L121-134) and a per-env friction scale on the Coulomb + Stribeck + load
budget (*actuator/friction_dr_bam.py* L29-54); under BAM the MJCF
`dof_frictionloss` is zeroed, so mjlab's stock friction DR is a silent
no-op (AGENTS.md, invariants). Action: joint positions, scale 1.0, the
14 servo joints selected by `^(?!passive_).*`. Observations are
asymmetric: the actor's 61-D (§4), the critic privileged — base linear
velocity, unbiased joint positions, the true IMU, foot forces, heights
and air time; actor-only realism — uniform noise (angular velocity
±0.03, gravity ±0.01, joint position ±0.001, joint velocity ±0.25), an
IMU lag of 0–1 ticks redrawn every 64 steps, a fixed 1-tick
joint-velocity lag (the Dynamixel moving average), encoder bias ±0.015
rad, IMU misalignment ≤ 6° (L565-634). Commands: twist ranges fixed (a
widening curriculum "outpaced the robot"), the standing-env share 2 % →
25 % by curriculum, a 15 % turn-in-place bucket; 4-D head and 6-D body
pose commands resampled every 2–5 s with a `zero_command_prob`.

Rewards (velocity; weight): pose +1, upright +2, linear/angular
velocity tracking +2/+2, air time +3 inside [0.125, 0.300] s, head-pose
tracking +2, head-pose bias 0 → +3 by iteration 1,500, action rate −0.1
→ −1.0 by 1,500, foot slip −0.1, self-collision −1, body angular
velocity −0.05, angular momentum −0.02, foot clearance and swing height
inherited. "Task mass" ≈ 11 is the number every other env is rebalanced
against (standup divides its task weights by 4 to keep the task ↔
regulariser ratio). Terminations: a custom `nan_state` over joint and
root state *and* contact forces, plus the inherited `fell_over` and
`time_out`; standup deletes `fell_over` because it starts fallen.
Curricula are step functions of the global step counter that mutate
the live manager cfgs (*tasks/mdp.py* `reward_weight` L3443,
`event_param_curriculum` L4454): the action-rate weight (6 stages), the
standing share (6), the head-pose range 5 % → 100 % of reachable (5),
the CoM range capped at ±15 mm because the heel sits 20 mm behind the
ankle (4). DR by mode: mass and inertia ±5 % at startup — fixed per env
for the run; CoM ±3 → 15 mm and the friction scale 0.9–1.1 at reset; a
±0.3 m/s push every 3–6 s (was ±0.5: "trains a permanently nervous
fall-recovery gait"); supply voltage inside the actuator cfg.
Non-accumulation is the invariant: "An accumulating CoM randomizer once
degraded every long run for months." Standup's reverse curriculum is a
spawn mix — standing / sitting / face-down / face-up 0.40 / 0.40 / 0.20
/ 0 at stage 0, ramped to 0.15 / 0.20 / 0.30 / 0.35 by iteration 2,500 —
with the anti-violence penalties switched on at 3,000. Runner: actor and
critic (512, 256, 128) ELU, observation-normalised, PPO lr 1e-3
adaptive on a 0.01 KL target, 5 epochs × 4 minibatches, 50k iterations
for the gait, 15k for standup.

**The reward lessons, as written** (AGENTS.md;
*docs/roller_standup_policy_summary.md*). General: "RL optimizes the
letter of the reward" — encode what counts as the manoeuvre in hard
state gates, not penalty nudges; no jackpots — a "reach X" reward is
rate-limited or slewed, "slow IS the argmax"; never gate a positive
reward on a bad state, use potential-based shaping; compare reward
*mass*, not weights, "PPO sees relative advantage"; a tracking
Gaussian's std ≈ the error you still care about; multiplicative
composites beat sums at goal states; introduce smoothness *after* skill
discovery — "any attempt-tax active while a hard skill is being explored
makes 'do nothing' win"; the infallible check, every
`Episode_Reward/<penalty>` in wandb ≤ 0 (a penalty weight on a function
that was already negative *paid* for violence, logged +0.0118, found
months later); a command input that is never non-zero has dead weights
forever, and zero-command behaviour is trained explicitly; a metric
that steps down exactly at a curriculum boundary means the pacing is
wrong — stretch, never move earlier; "measure before theorizing — run a
headless eval of the actual checkpoint before changing rewards"; one
correction at a time (three at once could not be attributed).
Biped-specific: one target from t = 0 rather than waypoints; a 25 cm
robot tumbles at 3.5–5.5 rad/s naturally; the 38 %-of-mass head must
oscillate while walking and is the pivot for back recovery (penalising
head impact froze the policy).

**Symmetry** (*tasks/symmetry.py*): a permutation + sign table over
the 61-D observation and the 14-D action (left ↔ right legs, yaw and
roll negated) wired as rsl_rl's mirror *loss* (coefficient 0.5), not
augmentation — and `ENABLE_SYMMETRY = False` in every cfg, dead until a
2026-08-13 key fix; "never for asymmetric tasks".

**Testing an RL env.** Cfg tests build the cfg on the CPU with no
simulator and pin: term presence, weights with their intended sign,
parameter values, command ranges, terminations, curriculum
monotonicity with stage 0 equal to the event default, observation-term
ORDER parity between envs (the hot-swap contract, *tests/test_spin_cfg.py*),
joint indices resolved by compiling the real `MjSpec` and reading the
names at the used indices (*tests/test_roller_standup_cfg.py* L123-145),
and the sign lock `test_already_negative_penalties_use_positive_weights`
(L457-477). The mdp functions are tested with duck-typed fakes, no mjlab
runtime (*tests/test_nan_guard.py*). The NaN guard is four layers
(*tasks/mdp.py*): `nan_to_num` monkeypatched into the reward manager and
into PPO's returns (a curriculum weight jump once collapsed the
advantage std), a termination on any non-finite joint, root or contact
value (L964-1016), `_finite` wrappers on the critic's sensor terms —
plus `nconmax`, solver iterations and a softened terrain `solref` on
the sim side.

**mjlab facts.** CUDA-only (README L25; `device="cuda:0"`); per-world
model fields must be declared with `@requires_model_fields` or per-env
writes "collapse to a single shared value"; managers deep-copy cfgs at
init, so a write to `env.cfg` afterwards is a silent no-op. Cost: "1–2 h
for a usable gait at 4096 envs"; a trick ≈ 1,000 iterations, a gait
4,000–6,000; MEASURED by them, 2.32 s/iteration at 4096 envs
(*docs/superpowers/specs/2026-08-04-spin-env-design.md*) ≈ 42k
env-steps/s, ≈ 3.9 h for 6,000 iterations; the smoke test is 64 envs ×
5 iterations "for cents". Determinism is addressed nowhere: the only
"seed" in the pack is the bench schedule and "back-recovery was
seed-lucky (1 success / 3 failures with equivalent rewards)" — on warp
1.12.0, before the deterministic mode we priced in
[52 §3](52-warp-determinism-mjwarp.md) existed; float32 is not
discussed.

**What transfers.** (a) T6: mjlab would give us the manager stack,
tested non-accumulating DR, step curricula, rsl_rl PPO and an ONNX
export, at the price of a third pinned instrument (torch 2.9.1 / warp
1.12 / mujoco 3.10 / Python 3.12, CUDA-only, no Mac) that conflicts with
the 3.12 + warp ≥ 1.16 determinism path — worth it only if T6 becomes a
main line; ours is brax PPO at 36k env-steps/s on 2048 worlds
(`tools/rl-watch.py`). (b) Task spec: their terminations, curricula,
commands and the spawn-mix table are already data — the pattern for
`KittingSpec` start bands; they have no acceptance gate (acceptance =
"builds, steps NaN-free, obs is 61D"), ours is expert-passes-all /
floor-passes-none (`trainnr/trainnr/tasks/acceptance.py`). (c) DR:
the startup/reset split and "cap the range from geometry" fit
`physics/variations.py` and the batched GPU path; their centre is CAD +
BAM, ours is identified — the philosophy is ours (docs/31 §3). (d) For
ACT: a 1-tick joint-velocity lag and a 3–6-tick action delay are cheap
realism knobs `envs/` does not have.

## 4. Deployment and the sim2real contract (field D)

**The contract.** 61 floats, no history, scale 1.0, no clip: `[0:3]
base_ang_vel · [3:6] projected_gravity · [6:20] joint_pos_rel · [20:34]
joint_vel · [34:48] last_action · [48:51] twist · [51:55] head cmd ·
[55:61] body cmd` (*tasks/symmetry.py* docstring; README "Conventions");
the only scaling is the `EmpiricalNormalization` baked into the ONNX
(*scripts/export.py* L227-232). It is spelled in at least six hand
copies: the env cfgs' term order with `zero_command_padding` up to the
unified 13-D command; the permutation tables in *symmetry.py*
("Migrated 2026-08-13 from the old 51-D layout"); *scripts/infer_policy.py*'s
`DEFAULT_POSE` ("matches HOME_FRAME in microduck_constants.py" — a
regex dict there, not an ordered vector) and its `get_observations()`,
whose docstring still says "Total: 51D"; *scripts/plot_observations_comparison_plotly.py*
with hard-coded 51-D offsets; *scripts/testbench_sim2real.py*'s own
one-joint layout; and the runtime's *obs.rs* (FETCHED,
pollen-robotics/microduck): "the highest-risk code in the crate — a
flat array of 61 floats where every index must match the policy's
training. Incorrect offsets produce plausible-looking failures rather
than loud errors" — which also hard-codes body x, y and yaw to zero, so
the deployed command space is 10 of 13 slots. The one machine-readable
copy — ONNX metadata `joint_names / default_joint_pos /
observation_names / command_names / action_scale`, written by *mdp.py*'s
export patch — is read by nobody: *infer_policy.py* reads only
`gait_period` (which the patch does not write), the runtime "never
reads model metadata" (*policy.rs*). Versioning is a CLI flag,
`--new-cmd-obs` ("Old policies (51D obs, head_offset added to ctrl) need
this flag OFF"); nothing in the artefact says which; the rehearsal
script warns on a width mismatch, the runtime refuses.

**Export** (*scripts/export.py*): the actor only, normaliser folded in
("always deploy ONNX produced by scripts/export.py, never a
hand-converted checkpoint"), opset 18, names from rsl_rl's `as_onnx()`;
the metadata carries joint names, stiffness and damping, the default
pose, command and observation names, the action scale — and no task
id, run id, git sha, layout hash or the `new_cmd_obs` flag; `run_path`
is `None` for every wandb-sourced export. **No numerical check** of the
ONNX against the torch policy exists anywhere in the pack.

**The rehearsal** (*scripts/infer_policy.py*, 1,386 lines): CPU MuJoCo
at dt 0.005 × 4 = 50 Hz with real-time sleep; the walk ↔ stand swap on
|v_cmd| ≤ 0.05 (the runtime's threshold too); timed sessions for
ground-pick, sit, kick, roulade; a `--delay` action ring buffer; the
XL330 current limit via BAM's kt, but "not the BAM voltage model";
keyboard commands, random pushes, pause. `--save-csv` writes one row
per tick (`step, time, obs_0..60, action_0..13`); `--record` a pickle
for the plotly overlay against a real recording — 48 proprioceptive
dims, no alignment, no metric, visual only. The one quantitative
sim2real tool is *scripts/testbench_sim2real.py*: the same ONNX driving
one XL330 through rustypot against MuJoCo + BAM, npz traces, a
comparison plot.

**The robot** (FETCHED): a Rust workspace (`robotd`, `duck-control`) on
a Radxa RK3566 (four A55 cores); 15 XL330 on one 1 Mbps UART, one
`sync_read` and one `sync_write` per tick; an LSM6DSV16X IMU on the same
bus; a 50 Hz tokio loop with `MissedTickBehavior::Skip`; ONNX through
`ort` 2.0-rc with one intra-op thread; every net checked 61-in / 14-out
at load; policies are paths in */etc/robot/robotd.toml*, read at start
and "not watched" — a swap is a config change and a restart, and each
release pins its policies. Nothing is imported from microduck_rl:
`JOINT_NAMES` lives in the IPC protocol crate, *obs.rs* restates the
layout by hand.

**Mapped onto ours.** `EpisodeProtocol` (`trainnr/trainnr/protocol.py`)
is the hashed contract — trials, steps, control interval, home,
milestones, observables, placements, executed horizon — and a mismatch
is refused before a trial is spent; the names are spelled once
(`ObservationKeys` in `trainnr/trainnr/envs/contract.py`); records
carry `source name@hash` and the instrument `mujoco-3.11.0+x86_64`
(`trainnr/trainnr/evaluate/records.py`); the firmware-mirror test
and the wire replays fail the build on drift
(`trainnr/tests/test_firmware_mirror.py`, docs/24); `train-watch
--play` rehearses on the evaluation env itself, so there is no second
observation assembly. What we lack: a deployable artefact — our
policies are torch LeRobot checkpoints or an openpi websocket, no ONNX,
no embedded metadata (0 hits for `onnx` in `trainnr/` and `tools/`); a
runtime-side observation assembly to pin (moot until a Rust runtime
exists — then it wants the mirror-test pattern, not a third copy); a
per-tick observation/action stream with a metric for sim-vs-real; the
single-servo bench.

## 5. Process and infrastructure (field E)

**AGENTS.md** (248 lines; *CLAUDE.md* is one line, `@AGENTS.md` — one
file for every agent vendor): a preamble ("Sim2real transfer is the
whole point: every convention below exists because breaking it produced
a policy that worked in the viewer and failed on hardware"), then
Commands · Repo map · Invariants · Building a new env · Reward design ·
Commands, observations, dead weights · Curricula · Training ops &
reading a run · Sim2real footguns. About eight lines are agent
behaviour — "A 5-iteration smoke test at 64 envs catches ~95% of config
errors for cents. Never launch a long run without one"; "Verify physics
assumptions in sim BEFORE training"; "Write cfg tests … lock in the
invariants"; "Measure before theorizing"; "Report what rollouts actually
show ('rolls but face-plants 1 in 3'), not 'it works!'. The user decides
when it's good enough"; "A fresh `uv sync` is the ground truth … Keep
`pyproject.toml` honest" — the other ~200 are domain, and every rule
names its scar ("A 5 mm-wrong STAND_Z once turned the goal into an
impossible target for days"). A gotcha log compiled into rules.

**Spec → plan → code → handoff.** A spec
(*docs/superpowers/specs/2026-08-04-roller-standup-design.md*, 375
lines) has a purpose; a `Décisions actées` table (decision | choice |
alternatives rejected); the architecture (file, factory signature, task
id, parent); measured constants with their method ("by exact kinematics
on scene_rollers.xml"); joint indices; rewards removed / kept / added
with weights and a one-line rationale each; observation and command;
reset; curriculum; network and PPO; the numbered tests the plan must
write; the exact train command and the metric to watch; risks; out of
scope. A plan (*docs/superpowers/plans/2026-08-04-roller-standup.md*,
1,236 lines; the same skeleton in all seven) lists tasks with Files
(create / modify / test) and Interfaces (consumes / produces, exact
signatures), each in TDD order with the *expected* output written before
it runs: the failing tests in full → run them, "Attendu:
ModuleNotFoundError" → implement → "Attendu: 10 passed" → "no other test
regresses: 4 failed, 56 passed" (the four are pre-existing, named, "do
not fix them") → commit with the literal message. The last task is
always the check static tests cannot make: 64 envs × 3 iterations on the
GPU with a table of plausible errors → causes, then `play` at 16 envs to
*look*. Plans end with a self-review (spec coverage, "Placeholders:
none"). There is no dated log; the record is the handoff doc written as
the last task (*docs/roller_standup_policy_summary.md*), which keeps
growing after training — "Correction anti-violence (après premier test
robot)": root cause, run ids, a before/after table, a frozen-policy
experiment, and the method lesson "une correction à la fois" — plus
results appended to the spec, lessons promoted into AGENTS.md, and cfg
docstrings that "encode a 5-run lesson arc". They have, we lack: the
decisions-with-rejected-alternatives table; the expected failure and
the expected pass count stated before code; a per-policy handoff page
that outlives its plan; "out of scope" in every spec. We have, they
lack: a dated newest-first log, a research corpus with an index,
positioning with evidence and date, doc/code gates
(`tools/check-docs.py`, `tools/verify.sh`).

**HF Jobs as the cloud** (*src/mjlab_microduck/hf_jobs.py*,
*train_cli.py*): `uv run train … --hf-jobs` strips the flag and submits
— image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, flavour `l4x1`
(also `a10g-large`, `a100-large`; no prices anywhere), a 12 h timeout;
the working tree (`git ls-files -co --exclude-standard`) is tarred into
a private HF *dataset* repo mounted read-only at `/src`; a bootstrap
shell installs a pinned `uv 0.11.30` with `UV_LINK_MODE=copy`, runs `uv
sync` with a self-healing cache clean, starts a 60 s checkpoint watcher
(*scripts/hf/uploader.py*) that commits `model_*.pt` and params to a
private *model* repo, trains, then auto-exports ONNX "while the env is
still warm"; secrets `HF_TOKEN` / `WANDB_API_KEY` from the environment
or `~/.netrc`; scheduling polled up to 1,200 s ("the pytorch image pull
alone is ~5 min"), resubmitted up to 3× on transient mount errors, the
log stream re-attached in a loop, Ctrl-C detaches. Dated gotchas: a
stale wheel cache from an older uv (2026-07-21); the FUSE bucket "does
not support hardlinks, so uv sync falls back to full-copying ~6 GB"
(the cache is off by default); GPU-node mounts failing ~7 min into
SCHEDULING; the HF README still documents the flipped cache flag. The
shape is container + command + env + secrets + volumes + timeout + log
stream + cancel — no SSH, no rsync. Our `GpuProvider` Protocol fits
(`MachineSpec.ssh=False` exists) but our push → bootstrap → run → pull
runbook does not; a `@provider("hf-jobs")` is "submit a job", with push
and pull re-expressed as a dataset-repo upload and a model-repo
download.

**Pinning** (*pyproject.toml*): `requires-python >=3.12,<3.13` because
"a floating upper bound let HF jobs pick 3.13.14 (2026-07-21)"; `scipy`
declared because mjlab 1.3.0 forgets to; `torch==2.9.1` as a *direct*
dependency because uv applies `[tool.uv.sources]` to direct deps only,
and `==` because the CUDA index would drag it to 2.13; overrides for
BAM's protobuf and zmq stubs ("uv sync fails on a FRESH install even
though a warm local cache tolerated it"); "do NOT override mujoco — an
override here previously forced mujoco down to 3.4.0".
*tests/test_aarch64_cuda_torch.py* tests the manifest and the lockfile:
torch direct, the aarch64 source a CUDA index, one version across
platforms. Ours (`trainnr/pyproject.toml`) pins the engine by
compatible release (`mujoco~=3.11.0`, with the dated reason "NINE tests
broke") and runs two venvs as two instruments; we have no test that
reads `uv.lock`.

**wandb**: per-task project / experiment / run names in each runner
cfg; `Episode_Reward/*`, `Episode_Termination/*`, the learning rate,
checkpoints as run files; resume by run name + checkpoint;
*scripts/play_latest.py* resolves the newest run by an email substring
and a task-id table and prints a run card (duration, progress, top-8
rewards, last checkpoint, host and GPU); throwaway runs use the
tensorboard logger "to avoid polluting wandb".

## 6. Verdicts

| Verdict | Item | Field | Why |
|---|---|---|---|
| ADOPT | The pendulum bench protocol, and BAM's released raw STS3215 logs as a validation corpus for `identify()` | A | real STS3215 data exists today; our fit can be scored on it with zero hardware |
| ADOPT | Command delay as an explicit integer-lag parameter in `Stepper` | A | already promised (R7); BAM shows the identified 5–10 ms is tick-scale |
| ADOPT | A backlash twin as a series passive hinge with `passive_` naming, the encoder AND the servo loop reading through the play, rewards on the sensor view | B | physically right; the one sim2real convention in the pack worth copying whole |
| ADOPT | Cfg-style tests: weights with their sign, indices resolved on the compiled model, observation-order parity | C | the same shape as our 303; "index resolved on the real model" is new |
| ADOPT | The DR startup/reset split, ranges capped from geometry, non-accumulation as an invariant | C | fits `physics/variations.py` and the batched GPU path |
| ADOPT | The reward-design lessons, into docs — no code | C | general; the referee is ours, but T6 has rewards |
| ADOPT | ONNX export with the normaliser folded in, plus the ORT-vs-torch numerical check they lack — when we ship a policy | D | a checkpoint is not an artefact; `ort` is already our named runtime |
| ADOPT | The single-servo bench pattern: the same policy driving one servo, sim vs real | D | the cheapest true sim2real number this repo can produce |
| ADOPT | In specs: a decisions-with-rejected-alternatives table and an out-of-scope list; in plans: the expected failure and the expected pass count written before the code; a per-policy handoff page that keeps growing after training | E | cheaper to audit than prose; docs/07 has no per-policy living page |
| ADOPT | In the cloud chain: a 60 s checkpoint watcher, resubmission on transient errors, auto-export in the warm environment, a pinned `uv` in the bootstrap, `--dry-run`, a `requires-python` ceiling | E | each has a dated incident behind it in the pack |
| ADAPT | BAM's STS3215 M6 as the *prior centre* for our own fit, our intervals on top, the 7.4 V vs 12 V winding declared | A | they publish a point; we publish the interval |
| ADAPT | The firmware rate-limited target and `error_gain_ratio` into `sts_synth.py`'s corruption model | A | stateful, cheap, likely load-bearing for ACT chunks; absent from docs/26 |
| ADAPT | The M6 friction budget as a CPU `Stepper` plug-in behind the engine seam, opt-in per bundle | A | keeps the MJX path pure; lets §8.3 measure whether Coulomb + Stribeck + load changes a verdict |
| ADAPT | Per-joint measured play with DR around it, in a stamped field — not one CLI default for every joint | B | ours is centred on a measurement (0.87° STS3215) and must be distinguishable in a certificate |
| ADAPT | Sum servo + backlash in the state reader, not in an RL observation function | B | `state_width` and the LeRobot plugin stay untouched |
| ADAPT | Manager-style cfg-as-data — terminations, curricula, the spawn-mix table with probabilities ramped by iteration — for `KittingSpec` start bands | C | maps onto `KittingSpec` + `EpisodeProtocol` without a framework |
| ADAPT | The four-layer NaN guard → one non-finite assertion at the Warp `Stepper` seam, counted per 10⁶ steps | C | our contact regime on Warp has no such counter |
| ADAPT | Observation delays and encoder bias as actor-only DR in `envs/` | C | for state-based training; two realism knobs we lack |
| ADAPT | Artefact metadata (protocol hash, source stamp, key order, instrument) that the runtime *reads* and refuses on | D | theirs is written and read by nobody |
| ADAPT | A per-tick observation/action stream with the instrument stamp and a metric, not a 4,600-pixel figure | D | into records + Rerun |
| ADAPT | `@provider("hf-jobs")` as a submit-a-job shape — tarball to a dataset repo, checkpoints back through a model repo | E | fits the `GpuProvider` Protocol (`ssh=False`), not the push/run/pull runbook; only when a run needs a no-SSH vendor |
| ADAPT | A lockfile test on the model of *tests/test_aarch64_cuda_torch.py*: every `mujoco*` spec `~=3.11.0`, one resolved version per venv | E | we have dated pin comments and no test that reads `uv.lock` |
| SKIP | `bam.mjlab.BamActuator` as-is; supply-sag DR for bench-powered arms; the XM430/XM540 via BAM | A | mjlab-shaped and branch-pinned; no battery; no parameters |
| SKIP | The regex/sed/six-config variant pipeline; their backlash tests | B | not our toolchain; there are none to adopt |
| SKIP | mjlab as the T6 framework (now); the symmetry mirror loss; evaluation by wandb means | C | a third pinned instrument against the 3.12 + warp ≥ 1.16 path; off in their own repo; no counts, no intervals |
| SKIP | Contract versioning by CLI flag; six hand copies of the layout | D | the version belongs in the hash; our recurring-bug shape (docs/24) |
| SKIP | No dated log, no research index | E | docs/07 and this corpus are strictly more |

## 7. Positioning sentences (each with its evidence; dated 2026-08-28)

1. "The industry's SO-101 constants are one bench servo at 7.4 V, collapsed to first order by an export its own authors deprecate, then copied to six joints, a gripper and a 12 V arm; we identify the joint in the arm at the arm's own supply, with an interval." — §0 (MEASURED with BAM's `to_mujoco`); *leisaac* and Positronic per [30 §3.3](30-the-pipeline.md).
2. "BAM ships one fitted point per servo model — 17 scalars, no interval, no unit identity, no date; we ship the customer's unit with intervals and NOT-PINNED verdicts." — *bam/params/&lt;motor&gt;/m6.json*; *bam/fit.py* optimises a scalar MAE; ours `trainnr/trainnr/robot/identify.py`, `trainnr/trainnr/robot/fit_record.py`.
3. "Under BAM the robot's MJCF no longer states its actuator physics — the XML's `kp=0.55` is overwritten into a `<motor>` at load; our bundle's MJCF *is* the identified model on every engine." — *bam/mjlab.py* `edit_spec`; *robot/microduck/joints_properties.xml*; ours `trainnr/trainnr/physics/mjx_backend.py`, the docs/33 Newton row.
4. "Their sim2real gate is a plot and a printed MAE with no threshold; the only numeric pass is kernel-against-kernel at 0.01 rad; we gate on counts, Clopper–Pearson intervals and declared thresholds." — *scripts/testbench_sim2real.py*, *scripts/validate_bam_testbench.py*; ours docs/32.
5. "microduck injects the same ±1° of play into all 14 servos from a script default, with no measurement in the repo and no test on the mechanism; we put the measured play per joint into a stamped bundle and randomise around it." — *robot/microduck/add_backlash.py* `--backlash-deg 2.0`, `tests/` (one comment); ours *planned*, §2.
6. "They accept an environment when a 64-env × 5-iteration smoke test builds, steps NaN-free and has 61 observations; we accept a task only when its scripted expert passes every paired start and a hold-still floor passes none." — AGENTS.md step 5; ours `trainnr/trainnr/tasks/acceptance.py`.
7. "They read a run off wandb means and a video — zero hits for confidence, interval, Clopper or Wilson in 37,007 lines; we count paired trials and gate on a lower bound." — pack-wide grep; AGENTS.md "rolls but face-plants 1 in 3"; ours `trainnr/trainnr/evaluate/certificate.py`.
8. "Their observation layout lives in six hand copies with no test across them — their own runtime calls it 'the highest-risk code in the crate'; ours is spelled once and hashed, and a mismatch is refused before a trial is spent." — §4; ours `trainnr/trainnr/envs/contract.py`, `trainnr/trainnr/protocol.py`.
9. "They call a 1-in-4 back recovery 'seed-lucky' on warp 1.12.0, where a deterministic mode does not exist; we measured that RUN_TO_RUN determinism arrives with warp ≥ 1.16 and priced the upgrade." — *tasks/microduck_standup_env_cfg.py*, *pyproject.toml*; ours [52 §3](52-warp-determinism-mjwarp.md).
10. Credit, and a gap of ours: "They export an artefact the robot loads, with the normaliser inside; we still hand the robot a torch checkpoint." — *scripts/export.py*; ours `trainnr/trainnr/envs/lerobot_policy.py`.

## 8. What to run here (short local runs; the cloud for anything longer)

1. **Their data, our fit.** Download BAM's raw STS3215 logs (FETCHED: `feetech_sts3215_raw.zip` in BAM's docs), convert BAM's log JSON to our CSV, run `identify()` with our four-parameter servo model on the same logs and report our MAE beside M1 … M6's (replayed through `bam.simulate.Simulator`). Expected: ours ≈ M1; the M1 → M6 gap on real STS3215 data is the number that decides the CPU plug-in. A *tools/bam-replay.py*, ~2 min CPU, zero hardware.
2. **The structured-residual test docs/26 §4 wants.** Synthetic STS3215 data from BAM's M6, fitted with our four parameters: the intervals will report "pinned" on a wrong model — frictionloss absorbing `K_c + K_cs`, damping absorbing `K_v` plus the load terms. A real target for the residual-whiteness diagnostic.
3. *(Built and measured for M1, 2026-08-29 — [36 §5](../36-actuator-law.md), `tools/servo-ab.py`, retired with the actuator-law branch: 4/4 → 2/4; M6 awaits its parameter fetch.)* **Does M6 change a verdict?** The SO-101 lift/reach scripted expert, nominal `<position>` against `bam.mujoco.MujocoController(feetech_sts3215_7_4V, m6)` on the same bundle and paired starts; hold droop and settle hysteresis in both viewers. Expected: a ±`friction_base`-wide hold band (0.053 N·m ≈ 1–2° at the shoulder) invisible in the linear model. And the cost: `Stepper.advance` with and without the per-step controller, steps/s and `lerobot-eval` s/episode.
4. **The backlash twin, A/B.** `add_backlash` on `so101` at 0.87° and at 2°; first the wind-through check in the viewer (actuator on the servo joint vs a tendon: the link must lag by exactly the play in the first case and not the second), then `tasks/acceptance.py`, 20 starts, nominal vs twin — the number that says whether play changes a verdict.
5. **Observation realism.** A 1-tick joint-velocity lag and a 3–6-tick action delay in `envs/`, the T5 policy re-run through the harness; the funnel shift is their named sim2real footgun measured on our task.
6. **Their DR widths through our sensitivity test.** CoM ±15 mm, friction 0.9–1.1, mass ±5 % on kitting via `trainnr/trainnr/stats/effects.py` — SENSITIVE or not at their ranges.
7. **The artefact.** Export the T5 ACT checkpoint to ONNX (opset 18), compare ORT against torch on 1,000 recorded observations (max abs action error — the check their pipeline lacks), and time one tick under `ort` with one intra-op thread.
8. **Process items, each an hour.** *trainnr/tests/test_pins.py* reading `pyproject.toml` and `uv.lock`; a `requires-python` ceiling matching the tested venvs, both suites re-run; a pinned `uv` + `UV_LINK_MODE=copy` in `tools/_pod-bootstrap.sh`, bootstrap timed before and after; a 60 s checkpoint watcher tested against `run --deadline-min` killing the job mid-run.
9. **When arms are in hand.** The STS3215 hysteresis loop — a slow ±3° triangle at three amplitudes, `present_position` against an external reference — separating mechanical play from the firmware dead zone that docs/27 lumps; then a BAM-style pendulum at 7.4 V and at 12 V, `bam.fit` and `identify()` on identical logs: the interval BAM does not publish, and the winding delta.

