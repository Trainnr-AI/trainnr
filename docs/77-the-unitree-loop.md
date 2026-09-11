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
| environment defined | the walk families (`rq_pipeline/tasks/walks.py`): `robotiq/go2-walk` with a `WalkSpec` (span, terrain, episode length, trials) stamped by content, built over the project's bundle; rq_mjlab builds the simulator environment from it (`rq_mjlab/src/rq_mjlab/go2_walk.py`) | `create_task`, `accept_task` (the learnability smoke, §3) | done 2026-09-10: `go2-flat` declared, accepted in 10 s, cited by a smoke run |
| data generated | **not a stage for RL**: the policy learns from its own rollouts; the strip must say "not needed" | — | friction (§3) |
| policy trained | rq_mjlab's walk trainer, `--robot go2 --project …`, 4096 envs, 8000 iterations on the pod | `train_walk(robot="go2", name=…)` → generic `train_policy` (A5) | door built and smoked 2026-09-10; the real run needs the pod |
| policy evaluated | the walk verdict: paired trials, exact interval, the funnel; under the fit and under pushes | `certify_walk` → generic `evaluate_policy`, `certify` (A5) | to build |
| deployment exported | the deploy manifest (A6, `rq_mjlab/src/rq_mjlab/walk_export.py`): joint and actuator orders, gains, home pose, action scale and offset, the ordered observations, the control rate, the SDK joint map, every number read from the BUILT environment; ONNX with normalization folded in and checked against the actor; the trained scene as MJCF; the sim-to-sim gate (`rq_pipeline/deploy/`) driving the ONNX through the manifest alone in plain MuJoCo | `export_deployment`, `gate_deployment` | built 2026-09-11 on a laptop checkpoint; the certified Go2 policy's export waits for its checkpoint here |
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

7. **A bundle's stamp depended on the machine it was onboarded on**
   (found and fixed 2026-09-10, the WSL box). `bundle.json` recorded the
   source as an absolute path, and the record lives inside the directory
   the stamp hashes, so the Go2 onboarded from the same clone at the same
   commit was `go2@b6170cf88b09` on the Mac and `go2@a185f9103878` on the
   box. Fixed: the record keeps the source's last three path components
   (`unitree_go2/xmls/go2.xml`), the same on every machine; the box's Go2
   is `go2@5003bf617b5f`, and the Mac's stamp changes when it re-onboards.
   The task stamp (`go2-walk@0e7e123a7de7`) was identical on both machines
   from the start, as a content hash should be.

8. **A run training in the project was invisible in the Studio** (found
   and fixed 2026-09-10, the box). The Studio polls the index file every
   second, but nothing rewrote the index while a run trained, and the
   experiment card's curve came only from a console log an imported
   study arm carries - a door-launched run has neither. Fixed three
   ways: `walk_train` tees its console into the run folder as
   `train.log`; `rq_pipeline.project.live` turns that log into
   `training.json` whenever the log is newer (status `running` until
   the done line, iterations so far, the curve so far); and the
   presenter the Studio spawns does that and re-indexes every 15 s. The
   Go2 run's card showed its reward curve at iteration 1278 of 8000.
   Open: a run launched by hand, not by the door, is absent from the
   Compute card (no job record) - the next run goes through
   `train_walk`.

9. **A certificate judged inside the project never became an
   evaluation** (found and fixed 2026-09-10, the box). `certify_walk`
   on `model_1400` of the training run wrote its 40/40 verdict where
   rq_mjlab writes one - `runs/go2-c1/verdict/walk-verdict-cuda.json`
   beside the checkpoint - and the Evaluations page stayed at zero: the
   only path from a verdict to an evaluation artifact was the importer,
   which refuses a run already in the project. Fixed: the importer's
   policy and certificate writers are shared helpers, and the live loop
   (`live.refresh_verdicts`, every presenter tick) turns each verdict
   under a run into a policy (`policies/go2-c1-model_1400`, the judged
   checkpoint copied with a manifest citing the run) and an evaluation
   (`certificates/go2-c1-model_1400-cuda`) citing that policy; the
   Studio spawns the presenter when a project opens, not at the first
   Show. Seen on the Evaluations page within a tick: 40 / 40, exact
   interval [0.91, 1.00], funnel survived 40, tracked 40; the overview
   at 4 of 8 stages (asset, environment, policy, evaluation).

10. **A run's version moved every time a checkpoint landed** (found and
    fixed 2026-09-10, the box). A run was stamped like a bundle, by
    hashing its folder, so the card read `go2-c1@b7630dc54e54` one tick
    and `go2-c1@a8300622d290` the next, and the policy written a minute
    earlier cited a run version that no longer existed - the lineage
    linked nothing. Fixed: `kinds.stamp_run` hashes the identity the
    run was launched with (`identity.json`, or a LeRobot run's
    `run.json`), the rule tasks already followed; the index, the
    importer and the live loop share it. The policy's and the
    certificate's `run` now resolve to the run card. Open: the
    certificate cites the task by rq_mjlab's walk-spec hash
    (`go2-walk@83d8a180bfda`) while the project's task is
    `go2-walk@0e7e123a7de7` (the declared spec's content hash), so the
    environment card is not yet cited by its evaluation.

11. **What the policy looked like every hundred iterations** (asked
    2026-09-10: "one image of each 100 iterations, like 80 images",
    for after the run). `rq_mjlab.walk_stills <run> --project <root>
    --robot go2 --every 100 --tick 100`: the env built once at the
    nominal point with one world, every hundredth checkpoint's actor
    loaded into it in turn, a rollout capped at 100 control ticks (2 s)
    and the pose at the last tick rendered through the project's own
    scene file from the chase camera; `stills/stills.json` keeps what
    the world did in those ticks, `stills/sheet.png` lays every still
    out in iteration order framed in the outcome's colour (tracking
    green, fallen red, up but off-command blue). The run's presenter
    puts the stills on the iteration timeline next to the curves, so
    scrubbing the reward curve in the Studio shows the robot at that
    checkpoint. Smoke on the box with three checkpoints: 17 s including
    the env build, so the eighty-one stills of a finished run are a
    three-minute pass. Seen: the sheet, and the Live view with the
    still beside the curves at iteration 5669. Caveat: the rollout is
    not bit-reproducible across tool invocations (the same seed gave
    err ratio 0.37 then 0.49 at model_4000) - a still is a picture,
    not a judgment; the certificate remains the judgment.

12. **Two finished certificates showed as running for an hour** (found
    and fixed 2026-09-10, the box). The job manager's exit-code watcher
    is a thread in the process that opened the door; called from a
    script that returned, no `.exit` file ever landed, and the Studio
    counted a job running while no exit was recorded. Fixed twice: the
    job now runs under a runner (`python -m rq_pipeline.mcp_jobs
    --exit-file … -- <tool>`) that writes the exit file itself, whoever
    launched it; and the Studio counts a job as running only while its
    process exists (Linux, through `/proc`; elsewhere the exit file
    alone decides). The Compute card reads "idle · no jobs running"
    with both certificates ended.

13. **A rotated verdict would have become a second evaluation** (found
    before it happened, 2026-09-10). rq_mjlab never overwrites a
    verdict: certifying `model_7999` moved `model_1400`'s file to
    `walk-verdict-cuda.seed1000.n40.json`, which the live loop would
    have imported as a new evaluation of `model_1400`. An evaluation is
    now named by policy, instrument, seed and trial count
    (`go2-c1-model_1400-cuda-seed1000-n40`), so the rotated file maps
    to the card that exists and a re-judge under another seed gets its
    own.

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
  Amended 2026-09-10: the pod launch was refused (account balance too
  low), and Prakhar: "we will do the training run tomorrow on wsl gpu"
  — the RTX 3090 Ti box. §5 is the runbook for it.
- **Hardware: none in this pass.** The rig connects at A6/A7 per the
  earlier call; a Go2's SDK2 adapter is gated research.

## 5. Tomorrow on the WSL box: the run, by the doors

**Done on the box, 2026-09-10.** The reference cloned at commit
`1425b15` into `~/.cache/trainnr/`, the venvs synced with their extras
(the recorder tests need `rerun`, the dataset tests `pyarrow`: sync
with `--extra viz` and `--extra sim --extra viz --extra mcp`). The
project made by hand (`projects/go2-walk/project.json`, loop
`reinforcement`), then by the doors: `onboard_robot` (friction 7 above,
then `go2@5003bf617b5f`), `create_task("go2-walk", "go2-flat", {span
0.10, flat, 20 s, 40 trials})` → `go2-walk@0e7e123a7de7`, `accept_task`
→ ACCEPTED by the learnability smoke (2 environments, 2 PPO iterations).
The run: `walk_train --agent g3 --robot go2 --project <root> --dr-span
0.1 --task-stamp go2-walk@0e7e123a7de7 --log-dir <root>/runs/go2-c1
--seed 42 --no-recorder`; identity `go2@5003bf617b5f`,
`unitree-go2-declared-pd@3b68245f8d2a`, 4096 environments, 8000
iterations at ~1.5 s each on the 3090 Ti (about 3.5 h); console log
`<root>/runs/go2-c1.log`. Then `certify_walk`.

**The loop closed, 2026-09-10 20:25 IST.** The run took 2 h 24 min
(8000 iterations, 1.03-1.13 s each, ~102k steps/s, final reward 74.0,
episode length 1000 of 1000). `certify_walk` on `model_1400` mid-run
and on `model_7999` at the end: both 40/40 survived and tracked, exact
95 % interval [0.912, 1.000], median error ratio 0.25 and 0.25, under
the trained ±0.10 law span. The project holds, all seen in the Studio:
the robot, the environment, the run with its curves, two policies, two
evaluations citing them and the run, and 81 stills (one per hundred
iterations, item 11) on the run's iteration timeline; the overview
reads 4 of 8 stages (asset, environment, policy, evaluation) with
"next: run system identification on a recording". What the loop does
not have on this robot: telemetry from a real Go2, hence no sys-id and
no identified interval - the span was declared, not measured, which is
the Go1 finding's shape (docs/74) and the reason the overview's next
move points at a recording.


What exists by the end of 2026-09-10, all pushed on the branch: the Go2
onboarded into `projects/go2-walk` (the reference's `go2.xml`,
`go2@b6170cf88b09`); the walk declared as `go2-flat`
(`go2-walk@0e7e123a7de7`, span 0.10, flat, 20 s episodes, 40 trials)
and accepted by the learnability smoke; the train door taking the
declared task. The box needs what a pod needed: the rq_mjlab venv
(mjlab 1.6, mujoco-warp, torch with CUDA — `cd rq_mjlab && uv sync`)
and the project directory (gitignored; copy `projects/go2-walk`, the
bundle is 24 MB of meshes). Then, with `TRAINNR_PROJECT` set to the
project, by the door:

    train_walk(agent="g3", task="go2-flat", name="go2-c1", seed=42)

which the trainer receives as `--robot go2 --project <root> --dr-span
0.1 --task-stamp go2-walk@0e7e… --log-dir <root>/runs/go2-c1 --seed 42`;
4096 environments, 8000 iterations (the walk study took 1.9 h on an
RTX PRO 6000; expect longer on the 3090 Ti). The run's `identity.json`
cites the task; the index shows the experiment with its reward curve as
it trains (the tensorboard log is not read yet — the console log is,
once the run ends). Then `certify_walk(<checkpoint>, robot="go2")` for
the evaluation, and the loop reaches "policy evaluated".

## 6. The deployment stage, built 2026-09-11 on the Mac

`export_deployment(run, checkpoint, name)` spawns `rq_mjlab.walk_export`
in the walk package's venv: the actor as ONNX (rsl_rl's exporter,
observation normalization inside the graph, a fixed batch of one, as a
runtime uses it), checked against the torch actor on sixty-four random
inputs (max |Δ| recorded; 9.5e-7 on the laptop checkpoint); the manifest
(`deploy.json`, schema `trainnr-deploy/1`) read from the built
environment — the policy's joint order, the simulator's actuator order
and the map between them, kp/kd/effort per joint, the home pose, the
action rule and scale, the seven observation terms in order with their
widths and sources, 50 Hz from timestep 0.005 and decimation 4, the
twist ranges, the fell-over limit, the reference's SDK joint map for
the Go2 as a declared fact; and `scene.xml`, the entity compiled with
its injected actuators and keyframe plus a floor and the training
timestep, so a plain MuJoCo loads the model the policy trained in with
the bundle's meshes. The door finds the checkpoint's policy and newest
evaluation in the index and cites them.

`gate_deployment(name, trials, seed, tolerance)` spawns
`tools/gate-deployment.py` in the pipeline's venv (`--extra deploy`:
onnxruntime): `rq_pipeline.deploy.runtime` computes each observation
term from MuJoCo state the way mjlab does (the IMU's velocimeter and
gyro from the sensors, gravity rotated into the base frame, joint
positions relative to home, joint velocities, the last action, the held
command), runs the ONNX, maps actions to controls, steps at the
manifest's rate; `rq_pipeline.deploy.gate` judges each seeded held
command the certificate's way (survived and error ratio below 0.5,
floored at 0.1 m/s), writes the exact interval, and passes when the
rate is within the stated tolerance (0.10) of the cited certificate's.
Cross-checked on the laptop checkpoint: the gate says survived 4,
tracked 0; mjlab's own verdict on the same checkpoint says survived 4,
tracked 0 (error ratios 0.76–1.0 against 1.01–1.12 — the protocols
differ, the judgment agrees). Twenty-second episodes run in half a
second each in plain MuJoCo. The Studio shows the deployment's card (the
robot), its drawer (the manifest's facts, the gate's verdict and trials,
the joints and observations as tables) and the loop at 5 of 8 stages;
"Show in viewer" on a deployment presents the trained scene from
`scene.xml` with the bundle's meshes, the gate's error ratio per trial
as bars against the 0.5 bound, and a reading with the lineage
(`rq_pipeline/project/present.py::_present_deploy`, checked by capture).

Frictions found (docs/77 §3 continued): 14, the project's folder is
`deploy/`, not `deployments/`; 15, the entity's spec is attached to the
scene and cannot be serialized — a fresh entity is; 16, rsl_rl's ONNX
takes one observation at a time; 17, the scene namespaces the entity's
names (`robot/FL_hip_joint`) — the manifest keeps the robot's own.

What the real Go2 deployment needs next: the certified `model_7999.pt`
from the box (with its verdicts) so the export cites `go2-c1`'s
evaluation and the gate judges against 40/40; then the reference's DDS
simulator on a Linux box as a second, independent gate; then A7.

**Done on the box, 2026-09-11.** The Mac's two commits pulled, both
venvs synced (onnx, onnxruntime 1.30), 634 pipeline tests green. Then
by the doors: `export_deployment(run="go2-c1", checkpoint="model_7999.pt",
name="go2-c1-deploy")` - `policy.onnx`, `deploy.json` citing the
policy `go2-c1-model_7999@62a6474496a7`, the run, the robot, the
declared task `go2-walk@0e7e123a7de7` and the evaluation
`go2-c1-model_7999-cuda-seed1000-n40@cc63d83c85c4`, and `scene.xml`;
`gate_deployment("go2-c1-deploy", trials=40, seed=1000)` - **38 / 40,
exact 95 % interval [0.831, 0.994], passed** (rate 0.95 against the
certificate's 1.00, tolerance 0.10). No trial fell; the two misses
are the two smallest held commands (0.12 and 0.14 m/s, just above the
0.1 m/s floor) at error ratios 0.82 and 1.00 - the regime where the
ratio criterion is harshest and the two protocols (mjlab's batched
env under law DR against plain MuJoCo at the nominal point) differ
most. The Studio: the Deployments page with the card and drawer, the
overview at 5 of 8 stages, the deployment in the viewer (the trained
scene with the bundle's meshes, the gate's error ratio per trial with
the two misses standing out, the reading with the lineage), both jobs
marked done in the activity feed.

Friction 18 (fixed): a run trained by the door cites its task by the
project's stamp (`go2-walk@0e7e…`), and the export door handed that
stamp to the task registry, which knows families by id
(`robotiq/go2-walk`) - refused. The door now resolves the stamp
through the environment card the index holds (its spec records the
family id); a run naming the family id directly is taken as is.

Friction 19 (fixed, 2026-09-11, the box): **the Studio window never
appeared on the operator's screens under WSLg**, through a day of
work seen only by the agent's in-app screenshots. WSLg's compositor
announced the window to Windows every time (`robotiq_studio` in the
RAIL app list), but Windows never showed a native Wayland window from
this app, while an X11 test window (`xmessage`, through Xwayland)
showed at once, and the Studio relaunched with `WAYLAND_DISPLAY`
hidden showed at once - the operator's screenshot. winit takes
Wayland whenever that variable is set, so the binary now asks the
event loop for X11 when it runs on WSL (`main.rs::prefer_x11_under_wslg`,
eframe's event-loop hook and winit's X11 extension trait - no
environment edits, the crate forbids unsafe); elsewhere winit's own
choice stands. Not the cause, ruled out on the way: the Vulkan adapter
(Mesa's dzn over D3D12 draws the app fine), the OpenGL backend (falls
to llvmpipe and refuses an R32Float target), the driver path.

Friction 19, continued (2026-09-11, evening): with the window visible,
two more things on the X11 path. (a) The process holds one full core:
the hot thread is "WSI", the presentation thread inside Mesa's dzn
Vulkan driver, spin-waiting to present to an Xwayland surface that
has no vertical blank. Presenting without vsync changed nothing; the
OpenGL path (Mesa's D3D12 driver, WSLg's usual route) refuses the
viewer's R32Float render target on this box even under the driver
environment. Left as is: the load is inside the driver, the window
stays live. (b) The operator's pointer landed a row below where it
hovered, while a pointer moved through X directly (XTest) hit the
right row - so the offset was added between Windows and the X server.
Dropping the window-manager frame (client-drawn chrome) did not
remove it and lost the window controls; reverted. A plain relaunch
with the frame, not maximized, put the pointer right ("ok fixed
now"). The offset is therefore tied to the window's state on the
Windows side (it had been resized to 1908 x 999 and driven from the
frame), not to the app; the measurement to take if it returns is the
X server's pointer position against the operator's, in one
"park the mouse" round trip.

Friction 20 (fixed, 2026-09-11, the box): **the certificate and the
manifest described the curriculum's first stage, whatever the
checkpoint had trained on.** The reward curve's step at iteration
5000 (86.5 to 73.2; entropy 1.76 to 4.45; episode length unchanged at
1000) is mjlab's velocity task widening the commanded twist - the
`command_vel` curriculum, stages at 0, 5000 and 10000 iterations of
24 steps: forward -1.0..1.0 and turn ±0.5 rad/s, then forward
-1.5..2.0 and turn ±0.7, then forward -2.0..3.0 (never reached in
8000). A fresh environment restarts the curriculum, so every
certificate had judged stage one and the deployment manifest's ranges
(what the gate draws from) were stage one too; the two checkpoints'
40/40 were the same question asked twice. Fixed: `rq_mjlab.envelope`
pins the config's command ranges to the stage the checkpoint's
iteration had reached and removes the curriculum; the certificate
records `protocol.commands` and `protocol.command_basis`, the
manifest `commands` and `command_basis`; an evaluation's name carries
a hash of its protocol, so a re-judge under another envelope is its
own card, and the card's subtitle leads with the envelope. Verdict
backups are named by the policy they judged too (two checkpoints of
one run under one suffix shared a backup name and the second
rotation would have overwritten the first).

Re-judged at the trained envelopes, 40 trials, seed 1000, on the
3090 Ti (both a minute):

| checkpoint | envelope judged | survived / tracked | interval | median error ratio |
|---|---|---|---|---|
| model_1400 | stage 1: forward -1..1, turn ±0.5 | 40 / 40 | [0.912, 1.000] | 0.250 |
| model_7999 | stage 2: forward -1.5..2.0, turn ±0.7 | 40 / 40 | [0.912, 1.000] | 0.199 |

So the second half of training did buy something the first
certificate could not see: the final policy tracks commands up to
2 m/s with a lower median error than the early one shows at half the
speed. The deployment re-exported citing the stage-2 certificate
(manifest forward -1.5..2.0, sideways ±1.0, turn ±0.7) and re-gated
over 40 commands drawn there: **40 / 40, [0.912, 1.000], passed** (the
earlier 38/40 had drawn from stage-one ranges; its two misses were
0.12 and 0.14 m/s commands). Not a like-for-like pair: different
draws, different envelopes.

Friction 21, and the rule behind it (2026-09-11, evening): **the
Studio re-derived a training curve from console text while the
trainer's own record sat in the run folder.** rsl_rl writes a
TensorBoard event file into every run: 39 series per iteration for
the Go2 - all thirteen reward terms, the curriculum's stage, the
policy's action noise, the collection and learning times, the
termination causes. Read as it is (TensorBoard's own reader, now in
the pipeline's `viz` extra), it answers what the console never could:
at iteration 5000 the two biggest losses were foot clearance (-0.38
to -0.56) and action rate (-0.33 to -0.50), ahead of the two tracking
terms (-0.16, -0.12) - the policy runs rougher at speed, not only less
accurately. `rq_pipeline.envs.tfevents` turns the file into the same
training record the cards read (the five console names kept; every
other series as `group/name`), the live loop prefers it and falls back
to the console log, and the viewer lays the series out in grouped
panels. The operator's rule, from this: be additive on top of the
libraries, use their features and their outputs as they are, keep our
modules for what they leave unsolved, and let the Studio and the data
piping tie it together.

What the loop still lacks on this robot: telemetry from a real Go2
(sys-id, an identified interval, monitoring). The next independent
gate is the reference's DDS simulator; then A7.
