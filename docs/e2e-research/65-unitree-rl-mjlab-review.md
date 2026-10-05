# Unitree's own mjlab stack, read code-first

Research date: **2026-09-02**. Source: a full dump of
`unitreerobotics/unitree_rl_mjlab` at commit `8a5edab` (~106k lines,
read by two agents; line references below are into that dump). This is
the closest thing that exists to a *vendor's official RL stack on our
own foundation* — Unitree ships it, it pins `mjlab==1.2.0` from PyPI
unforked, and it trains the policies their humanoids actually run. We
read it for three questions: what are its hand-coded tasks, what is the
general way of defining a task, and is there a community/plugin schema
we should adopt. Everything below is a reported-from-code observation,
not a benchmark claim.

## 1. What it is

Two task families behind 22 registered task IDs — nothing else:

- **velocity**: locomotion under a joystick twist command. Eight robots
  (go2, a2, as2, g1 29-dof, g1 23-dof, h1_2, h2, r1), each in a Rough
  and a Flat variant, all produced by ONE factory function
  (`make_velocity_env_cfg`, dump L103008). Flat is Rough with the
  terrain generator swapped for a plane and the height-scan pieces
  popped; `play=True` is a third mutation of the same cfg.
- **tracking**: whole-body motion imitation — the G1 dance demos. A
  re-implementation of BeyondMimic (`whole_body_tracking`), G1 only,
  driven by mocap CSVs converted to npz.

No manipulation, no navigation, no loco-manipulation. A C++ `deploy/`
tree (FSM + ONNX inference at 50 Hz over a 1 kHz command loop) carries
policies to hardware, and a vendored `unitree_mujoco` bridge lets the
same deploy binary run sim2sim before touching the robot.

## 2. What a task costs to define

A task is one Python function returning `ManagerBasedRlEnvCfg` — dicts
of typed terms (observations, rewards, terminations, commands, DR
events, curriculum, metrics), each term a cfg pointing `func=` at a
function from an **mdp vocabulary** with `params=`. The base cfg leaves
per-robot "holes" (`body_names=()`, per-joint posture stds, gait
offsets) marked `# Set per-robot`; each robot's file fills them by
imperative mutation. The measured economics (2026-09-02, from the dump):

- **New robot on an existing family ≈ ~180 lines** of pure config
  mutation + one registration block + one PPO cfg. Essentially data.
- **New task family ≈ ~1200 lines** of custom Python: command logic,
  ~15 reward terms, terminations, a runner subclass.

The identical PPO recipe serves all eight robots (512/256/128 elu,
adaptive lr, 50 Hz policy over 200 Hz physics) — the per-robot deltas
live entirely in the cfg holes. That one factory serving eight bodies
is the strongest evidence in the repo that the mdp-vocabulary pattern
generalizes.

## 3. There is no community schema

Greps for `entry_points`, `gym.register`, `gymnasium`, YAML task files:
zero hits in `src/`. Discovery is "import the package; every
`config/<robot>/__init__.py` calls `register_mjlab_task` at import
time" — instantiated cfgs, not factories, so play/train variants are
eager duplicate calls. There is no external-task convention, no plugin
mechanism, and nothing for a third party to target.

**The opening this leaves**: the de-facto standard is just "the mjlab
manager API plus a self-registering package" — exactly the surface we
already build on, minus everything we add (hash-stamped robot bundles,
declared DR bases, refusal linting, certificates, an MCP surface). A
community schema does not exist to adopt; ours can be the one that
does.

## 4. The steal list (ranked)

1. **The locomotion reward vocabulary** (velocity/mdp, L105836+): gait
   phase scheduling (`feet_gait` — per-leg phase offsets against a
   sin/cos clock that zeroes when standing), `variable_posture`
   (per-joint-regex stds switched by commanded speed:
   standing/walking/running), `soft_landing` (first-contact impulse
   penalty), `angular_momentum_penalty` ("natural arm swing"), feet
   air-time/clearance/slip. Mature, eight-robot-proven, directly
   transplantable to our walk and successors.
2. **Adaptive failure-bin sampling** (tracking, L102203+): the
   reference motion is binned per second; bins where episodes
   terminated get higher resample probability (geometric kernel
   λ=0.8, 10% uniform mix), with sampling entropy logged. This is our
   "press where the policy fails" loop already working in RL form.
3. **The deploy contract**: `deploy.yaml` travels WITH the policy and
   carries the joint-order map, kp/kd, action scale/offset, and the
   ordered observation list (order in the file IS the concat order);
   normalization is baked into the ONNX so all runtime scales are 1.0.
   Plus the **sim2sim gate**: the same C++ binary runs against a
   MuJoCo bridge (`./g1_ctrl --network=lo`) before hardware.
4. **Motion ingest with a sim honesty check** (their csv-to-npz script,
   L48803+): mocap CSV → 30→50 fps resample (slerp for quaternions,
   central-difference SO(3) velocities) → **replayed through the
   actual MuJoCo model** to harvest per-body states, asserting the sim
   reproduces the commanded motion. A whole data lane (human motion →
   imitation training) we don't have, verified the way we would verify
   it.
5. **Asymmetric critic + staged curricula**: the critic sees noiseless
   privileged observations (base lin-vel, clean height scan, contact
   forces log1p-compressed) while the actor eats heavy noise
   (joint_vel ±1.5); command ranges widen on a step schedule and
   reward weights anneal.
6. Small gems: a **No-State-Estimation task variant** that simply
   drops the observations the real robot cannot estimate — and is the
   one they deploy; multi-GPU via torchrunx with per-rank
   `MUJOCO_EGL_DEVICE_ID` pinning; a NaN guard that dumps offending
   states; a browser viewer for headless boxes.

## 5. The convergence: they measure physics instead of randomizing over ignorance

Their entire dynamics DR table (velocity, L103186+): foot friction
0.3–1.6 (shared draw), encoder bias ±0.015 rad, torso CoM offset
±0.05 m, velocity pushes every 5–6 s. **No mass scaling, no gain
randomization, no latency, no motor friction.** Instead, kp/kd are
*derived from identified motor physics*: per-motor rotor inertias
reflected through the two-stage planetary gearbox
(`reflected_inertia_from_two_stage_planetary`, L96080+), then
stiffness = armature·ω² at a chosen 10 Hz natural frequency with
damping ratio 2.0; ankles modeled as two motors in parallel linkage;
per-joint action scale = 0.25·effort_limit/stiffness.

That is Unitree independently landing on our thesis — get the physics
right and keep DR narrow — from the vendor side, where they can read
the motor datasheet. We arrive from the customer side, where nobody
can: fitted parameters with confidence intervals, command delay
measured, the whole thing stamped into a certified bundle. Same
doctrine, and ours works on robots Unitree didn't build. It also names
a benchmark worth running on a robot: **derived-gains + narrow DR
vs identified-intervals DR**, same task, same trials.

## 6. Where our discipline is ahead (each with the evidence)

- **Deploy fall-detection is stubbed**: `bad_orientation` in the C++
  deploy has its tilt test commented out and returns `false`
  (L2974–2996) — the fall transition registered by every RL state can
  never fire; only the comms timeout and the operator's button remain.
  There is also no action clip, rate limit, or joint-limit clamp
  between policy output and motor command. Our loud-refusal and
  certificate culture is a real differentiator, not decoration.
- **Two different G1 definitions** across the two families (their own
  `src/assets` for velocity, mjlab's asset zoo for tracking) — the
  same robot, two truths. Our one-bundle-one-stamp rule exists to make
  this impossible.
- **Shadowed dead code**: the local `MotionCommandCfg` carrying the
  adaptive-sampling knobs appears unused by the registered tasks,
  which import mjlab's — the knobs set at L101697 configure a class
  that never runs.
- **Zero provenance**: no policy, dataset, or deploy bundle records
  what physics, what code, or what DR produced it.

## 7. What this changes for us

Folded into the data design: the mimic lane as a fourth expert
source, failure-bin sampling as prior art for the funnel→press loop,
the deploy.yaml/sim2sim contract as the shape of the sim-to-real seam
(built since: `trainnr/deploy`), and the reward vocabulary + curricula
as the adoption list for trainnr_mjlab when the walk grows terrain.
The draft paper's §8 (published separately) re-reads this repository at commit 1425b15
(in preparation).
