# 77 — The Unitree loop: a Go2 from asset to deployment, by the doors alone

*Started 2026-09-10 on branch `unitree-e2e-2026-09-10`. Prakhar: "start
with a new branch where we do end to end with a new unitree robot and in
the process fix all the bugs and issues and ui ux to make the tool super
helpful efficient", with `unitreerobotics/unitree_rl_mjlab` as the
reference. This document is the plan and the running log of what the
loop taught us; every friction found on the way is fixed in the same
stage and recorded here.*

## 1. What the reference is, read on 2026-09-10

`unitree_rl_mjlab` (Apache 2.0; shallow clone at
`~/.cache/trainnr/unitree_rl_mjlab`, commit `1425b15`) is Unitree's
reinforcement-learning stack on mjlab — the same simulator stack our
walk study runs on (docs/e2e-research/58; rq_mjlab pins mjlab 1.6.0,
the reference pins 1.2.0 and mujoco-warp 3.5.0). What it holds:

- Seven robot packages under `src/assets/robots` (go2, a2, as2, g1 at 29
  and 23 dof, h1_2, h2, r1): an MJCF per robot **without actuators or
  keyframes** (mjlab injects position actuators and the home pose from
  Python constants), collision geoms tagged `*_collision`, named IMU and
  foot sites; plus a `scene_*.xml` twin **with** actuators for the
  sim-to-sim viewer. Go2's constants: hip and thigh kp 20 kd 1 effort
  23.5 armature 0.01; calf kp 40 kd 2 effort 45 armature 0.02; home z
  0.32, thigh 0.9, calf −1.8; foot friction 0.6.
- Velocity tasks (`src/tasks/velocity`): mjlab manager-based configs,
  observations in dict order (base angular velocity, projected gravity,
  command, gait phase, joint positions, joint velocities, last actions,
  height scan on rough terrain), one joint-position action scaled 0.25
  around the default offset, seventeen rewards, push and friction and
  encoder-bias and centre-of-mass randomization, 50 Hz control
  (timestep 0.005, decimation 4), 20 s episodes; flat ids `Unitree-Go2-
  Flat` and `-Rough`, registered by import side effect.
- Training and play scripts (tyro CLIs over rsl_rl, unpinned), ONNX
  export bolted onto checkpoint saving with metadata attached.
- `deploy/`: a C++ runtime over onnxruntime with an FSM (passive → fixed
  stand → RL) and, per robot, a hand-maintained `deploy.yaml` carrying
  the joint-order map (Go2's MJCF FL,FR,RL,RR against the SDK's
  FR,FL,RR,RL: a real reordering), kp/kd, default joint positions, the
  action scale and offset and clip, and the ordered observation list.
- `simulate/`: a vendored MuJoCo 3.3.6 viewer that republishes the
  simulated state as Unitree DDS LowState and takes LowCmd — so the same
  deploy binary runs against the simulator and the robot, gamepad in
  hand. Linux only.

What we reuse and what we do not. Reuse: the robot assets (with
attribution), the Go2 gains and home pose as a **declared** basis, the
velocity task's observation order, action scale rule, randomization
ranges and command curriculum as the environment spec, and the
`deploy.yaml` field set as the shape of our deploy manifest (A6). Not
reused: their registration (flat ids by import side effect; ours are
`namespace/name` with a content-stamped spec), their configs' mutability
(the ordered observation list must be read out of the built
environment, not the source), their hand-copied deploy numbers (ours are
minted at export from the identity record and embedded in the ONNX
metadata), and their C++ deploy path for now — our sim-to-sim gate runs
in MuJoCo from Python, hardware-free, and their DDS simulator is a
second gate we can add on a Linux box.

## 2. The loop for a Go2, stage by stage

The states are docs/76's. For a reinforcement-learning loop two of them
read differently, and the window must say so rather than show a gap.

| state | for the Go2 | door | status 2026-09-10 |
|---|---|---|---|
| asset onboarded | Menagerie `go2.xml` into `projects/go2-walk/robots/go2` | `onboard_robot` | done by the door; two frictions (§3) |
| telemetry recorded | none — no Go2 in the room; the SDK2 adapter (A2, gated) is the door when one arrives | `ingest_recording` | honest gap, shown as such |
| system identified | **declared, not identified**: the reference's gains and armatures as the actuator basis, with a domain-randomization span around them (the way `rq_mjlab/go1_walk.py` already does for the Go1) | `identify_system` when a recording exists | declared |
| environment defined | the Go2 walk (`rq_mjlab/src/rq_mjlab/go2_walk.py`): mjlab 1.6's velocity task wired to the Go2's names over the project's bundle, the reference's constants declared; a registry family with a stamped spec is the next step | `create_task`, `accept_task` (a walk's critic is the learnability smoke — §3) | walk built 2026-09-10; family to build |
| data generated | **not a stage for RL**: the policy learns from its own rollouts; the strip must say "not needed" | — | friction (§3) |
| policy trained | rq_mjlab's walk trainer, `--robot go2 --project …`, 4096 envs, 8000 iterations on the pod | `train_walk(robot="go2", name=…)` → generic `train_policy` (A5) | door built and smoked 2026-09-10; the real run needs the pod |
| policy evaluated | the walk verdict: paired trials, exact interval, the funnel; under the fit and under pushes | `certify_walk` → generic `evaluate_policy`, `certify` (A5) | to build |
| deployment exported | the deploy manifest (A6): joint-order map, gains, scale, offset, ordered observations, control rate, limits; ONNX; the sim-to-sim gate re-certifying through the manifest in MuJoCo | `export_deployment`, `gate_deployment` | to build |
| drift monitored | synthetic drift into the declared basis until a Go2 exists; then the SDK2 adapter | `monitor_drift` (A7) | to build |

## 3. Frictions found, in the order the loop found them

1. **Onboarding ignores the open project** (found and fixed
   2026-09-10). `onboard_robot` wrote to the checkout's robot library
   (`$RQ_ROBOTS_DIR` or `robots/`), while the loop says the project holds
   its robots and the Studio lists the project's. Fixed: the bundle
   locator (`rq_pipeline/bundles/locate.py`) searches registered roots
   before the library — a project registers its `robots/` when it is
   made current (`Project.use`) — so a robot onboarded into a project
   builds tasks like a library rig; the door lands the bundle in the
   open project and reindexes; the describe doors list both.
2. **Onboarding picks the wrong model file** (found and fixed
   2026-09-10). The Go2 directory holds `go2.xml`, `go2_mjx.xml`,
   `scene.xml` and `scene_mjx.xml`; the door was told `go2.xml`, but
   nothing recorded it, so the reader's "largest XML" rule chose the MJX
   twin and the drawer showed its counts (30 sensors the base model
   does not have). Fixed: onboarding writes `bundle.json` (schema
   `trainnr-robot/1`: the model file compiled, the source path, the
   census; no timestamps, so two onboardings of one directory are one
   version) and every reader asks it first
   (`rq_pipeline/bundles/bundle.py`). The rig's `profile.json` stays what
   it is, a measurement, not provenance. Also found: the actuator store
   was listed as a robot bundle.
3. **The pipeline strip assumed every loop has a dataset stage** (fixed
   2026-09-10). A reinforcement-learning loop has none; "Dataset" read
   as a gap forever. Now a project says how its policy learns
   (`project.json` `loop`: imitation or reinforcement, set at
   `create_project_dir` or read off the first run, whose summary says
   what it learned by), the index marks the stage `needed: false` with
   the reason, the next move never names it, and the strip draws it dim
   with the reason on hover and counts "1 of 8 stages".
4. **A walk has no scripted expert** (settled 2026-09-10). Acceptance
   for a locomotion task cannot be "the expert passes every trial". Its
   honest gate is learnability: the walk's smoke — two environments, two
   iterations, the environment built from the project's bundle, the
   actor's twelve outputs and the critic's seventy-two inputs shaped by
   the declared constants — which the train door runs in ten seconds on
   a laptop CPU. `accept_task` on a walk family will run exactly that and
   record it as the verdict; a `floor` half (hold home, must not track)
   comes with the verdict tool.
5. **The Menagerie Go2 is not in the simulator's shape** (2026-09-10).
   Its collision geoms are unnamed, it has no foot sites, and its trunk
   is `base`; mjlab's velocity task wants `*_collision` geoms, foot
   sites and a trunk body by name. The reference's `go2.xml` has all
   three and no actuators or keyframe (mjlab injects them from the
   constants), so it is the project's robot; Menagerie's stays the
   viewer-friendly twin. Onboarding should say which shape a model is
   in — a census of named collision geoms and sites — so the agent learns
   this before training, not from a traceback.
6. **A second venv, a second world** (2026-09-10). The walk runs in
   rq_mjlab's own environment (mjlab pins), so the train door spawns it
   with `--project` and the walk resolves the robot through the same
   project-first locator the pipeline uses; nothing about the project's
   layout is known to the walk package.

## 4. Decisions to take before building (asked 2026-09-10)

- **Robot: Go2.** The Go1 walk already exists in rq_mjlab on mjlab's own
  asset; the Go2 is the current robot, has the reference's gains, deploy
  config and sim-to-sim twin, and mjlab 1.6's asset zoo does not ship
  it — so this exercises onboarding for real.
- **Environment source: our registry, the reference's numbers.** A
  `robotiq/go2-walk` family in `rq_pipeline.tasks` with a spec dataclass
  holding the velocity task's knobs, stamped by content; rq_mjlab builds
  the mjlab environment from it (as `go1_walk.py` does). Not a
  dependency on the reference package (its mjlab pin conflicts with
  ours, and its configs cannot be hashed).
- **Compute: the RunPod pod for training and evaluation, the Mac for
  everything else**, per the standing rule (short local runs).
- **Hardware: none in this pass.** The rig connects at A6/A7 per the
  earlier call; a Go2's SDK2 adapter is gated research.
