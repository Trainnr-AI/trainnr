# The end-to-end test: ALOHA 2 through every stage

*Started 2026-08-26. The operator's call: run the whole pipeline, stage
by stage, on one robot — first a ViperX 300S, then (once the camera
question was answered) the full ALOHA 2 rig, because it has cameras.
This doc is the plan and the running record; each rung links its
evidence. The critical path (Paper 2 on the WSL card) is unchanged —
this is the platform proving it can swallow a second stranger, and the
first with cameras and someone else's measurement.*

## 0. Why ALOHA 2

- **It sits inside the wedge.** ViperX 300S arms run Dynamixel
  servos that report position, velocity, current and temperature —
  feedback the SG90 arm never had. Identification, drift telemetry and
  the HIL loop all have something to read.
- **It has cameras — the arm alone does not.** Menagerie's
  `trossen_vx300s/vx300s.xml` carries zero `<camera>` elements (checked
  in the file, pinned by `test_the_arm_alone_has_no_cameras`). The
  ALOHA rig carries six: two D405 wrist cameras on the gripper bodies,
  D405 overhead and worm's-eye cameras on the frame, and two viewpoint
  cameras. A bare arm is proprioception only — it knows where it is,
  never what is in front of it — so any task where the object moves
  needs the rig, not the arm.
- **Someone else measured it.** The ALOHA 2 team system-identified the
  model (11 real trajectories, box-constrained NLS on follower
  position/velocity). The standalone `vx300s` Menagerie file carries
  DIFFERENT, un-identified numbers for the same servos — 3–5× apart on
  shoulder damping and gains (table in `robots/aloha2-nominal/README.md`).
  Paper 1 on this robot is a replication with intervals they did not
  report, against a reference that exists.
- **The ecosystem is ready-made.** ACT was born on this rig; LeRobot
  drives ViperX/ALOHA natively for collection, ships ALOHA ACT
  configs and pretrained sim checkpoints, and `gym-aloha` has MuJoCo
  eval tasks (transfer cube, insertion).

## 1. Stage by stage

| Stage | On ALOHA 2 | Hardware? | Status |
|---|---|---|---|
| ② ONBOARD — bundle | `robots/aloha2-nominal/`: Menagerie `aloha` byte-identical (commit `da76818e`, BSD-3) + `aloha2.xml` sensor wrapper (28 sensors: 14 jointpos + 14 jointvel in actuator order); census pins 14 actuators, 28 sensors, **6 cameras**, 95 geoms; wrapper-purity and hold/travel tests | No | **DONE 2026-08-26** — `pipeline/tests/test_aloha2_bundle.py`, 6/6 |
| ② ONBOARD — identification | Excitation on the real arms → `mujoco.sysid` → intervals + identifiability verdicts, compared against ALOHA 2's published values (Paper 1 as replication). Rehearse first in sim exactly as Paper 0 did: true model → sweep → degraded telemetry → `identify()` recovers | Real: yes. Rehearsal: no | queued |
| ① SCAN | The scene IS the bundle's `scene.xml` (frame, table, cameras); task objects as cousins with measured mass/μ | No | queued |
| ③ VALIDATE | Gate A: paired sim/real rank correlation, certificate on the lower bound | Yes | after hardware |
| ④ COLLECT | LeRobot's ALOHA recorder (Dynamixel bus + cameras → LeRobotDataset) — not the Pico `.wire` | Yes (or scripted sim demos) | queued |
| ⑤⑥⑦ TRAIN | ACT via LeRobot's ALOHA config on the 3090 Ti; released sim checkpoints first | No | queued |
| ⑧ EVALUATE | Harness tasks on the bundle (rq_pipeline/tasks/aloha2.py (next rung)): transfer cube first, matching gym-aloha's protocol; paired trials → certificate dry run | No | next |
| ⑨ ENVELOPE | Dynamixel current/torque limits + software watchdog — no Pico Tier 0 here; the safety story is different and must be written | Yes | queued |
| ⑩⑪ OPERATE / TELEMETRY | Leader-arm interventions; present-current/temperature registers as drift telemetry | Yes | after hardware |

## 2. Decisions and findings, dated

**2026-08-26 — episodes start where the protocol says, not where the
model resets.** The first hold test drove both arms from qpos=0 (arms
straight up) to `neutral_pose` and found the shoulders stuck 1 rad
short: the elbows fold inward faster than the shoulders lean back, the
grippers meet at the top centre and jam at **over 1 kN** of contact
force (`left/gripper_base` ↔ `right/gripper_base`), both arms
symmetric. The real rig never traverses that pose. First fix tried —
make the backend's home the first keyframe — broke four SO-101 task
tests whose scripted picks were tuned from the zero start. Final fix:
`EpisodeProtocol.home` names the keyframe an episode starts from
(`None` = reset); the harness resolves it through
`backend.keyframe_state(name)`. SO-101 keeps its tuned start byte for
byte; ALOHA protocols declare `neutral_pose`. Where an episode starts
is a protocol fact, hash-stamped with the rest, never inherited from
the simulator's zero.

**2026-08-26 — the census counts cameras.** `ModelCounts.cameras`
(from `model.ncam`) joins actuators/sensors/geoms, because a vision
policy on a camera-less model fails silently — black frames score 0%
with no error, the failure class the census gate exists for.

**2026-08-26 — verified in the viewer.** All six cameras rendered at
`neutral_pose` through the offscreen renderer (contact sheet in the
session), each wrist camera looking down its own fingers at the
opposite arm; the jam pose rendered from the teleoperator viewpoint.

## 3. The training ladder (⑥/⑦) — started 2026-08-26

The operator's turn, verbatim: "now lets get to the training part,
from a robotics foundation model, to different policy training
techniques in simulation with physics and dynamics" — and the
addendum: "with evals of trying and combnining different policy
models or mixture of models to quickly train the robot to some
industrial task."

Stage ⑦ had zero code by design (census first); ALOHA 2 is where it
starts, because the whole public ecosystem for this rig already
exists — ACT was born on it, LeRobot ships its datasets and
checkpoints, and the identified dynamics give "physics" a meaning.
The addendum names the two things the ladder is FOR: the recipe
engine's second axis — *combining* models (seed ensembles, a VLM
planner over BC skills, merged fine-tunes), every combination judged
by the same harness as a single model — and the target: not the
cube-transfer toy the public data covers, but an industrial-shaped
task in our own scene (kitting: parts into slots, bimanual handover
where reach demands it), where demos are generated, not downloaded.
"Quickly" is the platform's promise; the certificate is what makes
the speed honest.

Principles carried in from the research (docs/e2e-research/20, 22,
30 §⑥⑦, and the sprint's 32): **adopt, do not build** (LeRobot is
the trainer; we write recipes, not optimizers); **ACT first, not to
ship but to test the data pipeline** (trains in under an hour); **the
foundation-model rung is a fine-tune** (SmolVLA fits a 3090 Ti
comfortably; π0.5's LoRA is borderline at 24 GB; MolmoAct2 LoRA
~20 GiB); **synthetic data is centred on identified values** (DR
around a guess trains robustness to the wrong distribution); and
**the judge is our harness**, never the trainer's own eval —
validation loss provably does not predict rollout success
(robomimic), and a recipe is a hash-stamped artifact whose winner is
chosen by a certificate.

| Rung | What | Data | Judge | Status |
|---|---|---|---|---|
| T0 | The train environment: `.venv-train` (Python 3.12.8, `lerobot[dataset,smolvla,training]` 0.6.1 + our sim extras, torch 2.11 + CUDA 13) beside the untouched 3.11 sim venv; the smoke `lerobot-train` proved the GPU path — ACT, 300 steps in 39 s at ~10 steps/s, batch 8, 2.1 GB VRAM, loss 3.23 → L1 0.505, checkpoint at `runs/t0-act-smoke/checkpoints/000300` | `lerobot/aloha_sim_transfer_cube_human` (50 human demos, 20k frames, the original ACT sim) | — | **DONE 2026-08-26** — the repo's first training run |
| T1 | Released checkpoint through OUR harness: `lerobot/act_aloha_sim_transfer_cube_human` via `evaluate.vision.lerobot_checkpoint_policy` on the ALOHA 2 bundle's transfer-cube scene (`pipeline/rq_pipeline/tasks/aloha2.py`, cameras matched to gym-aloha's `top`, `look="act_sim"` for their cosmetics). The gap between its published sim score and ours is a sim-to-sim dynamics + camera gap — Paper 2's shape, before any hardware. **Result 2026-08-26: 0/4 in the grey twin, 0/4 in the wood scene** (the Hub checkpoint needed LeRobot's processor migration first). The mapping was then proven by replaying human demo 0's actions open-loop: all six joints of both arms track the dataset's recorded states at r = 0.98–1.00, RMSE ≤ 0.06 rad — so the zero is not a sign flip. It exposed one channel: the gripper state floor (fixed to the measured closed position, 0.0078 m). Re-run with the fix: **still 0/4**. The filmstrip showed why: the right arm performs the whole reach–sweep–handover choreography and the cube never moves — a transfer of nothing. Then the kinematics were compared directly, the ACT sim's model beside ours at the same joint angles: **our fingertips sit 2.6–5.7 cm higher above the table** (neutral +5.7, mid-reach +5.3, lowest reach +2.6) — 2.1 cm because Menagerie mounts the ALOHA 2 bases above the tabletop where the ACT sim's are flush, the rest because the ALOHA 2 gripper's fingers are shorter. A joint-space policy trained on their chain closes above a 4 cm cube on ours. Not a dynamics gap yet — a KINEMATIC one, between two models of "the same robot" | released weights | harness, paired trials | **explained: 0/4 is kinematic** |
| T2 | ACT from scratch on the public demos → `policy@hash` → harness beside T1 and a limp floor: the recipe engine's first walk over one family. **Done 2026-08-26: 10k steps in ~20 min on the shared card, ten checkpoints each played live in `train-watch` (all fail — 10% of the released model's training), then the harness: ours 0/4, released 0/4, limp 0/4 (933 s).** A row of zeros the certificate correctly cannot rank: no joint-space ACT trained on the ACT sim's chain transfers to this one (T1's kinematic gap), and 10k steps is not a policy yet. The walk continues on OUR demos (T5) | public demos | certificate dry run | **DONE — zero row** |
| T3 | Foundation model: SmolVLA fine-tune on the same demos (fits), then MolmoAct2/π0.5 LoRA if VRAM allows — same harness, same protocol; the family axis of the recipe space | public demos | certificate dry run | queued |
| T4 | Combining models — the second axis: seed ensembles of T2/T3 checkpoints (action averaging), a VLM planner routing between BC skills, and merged fine-tunes (weight-space soups); each combination is a `recipe@hash` scored by the same protocol as a single model | public demos | certificate dry run | queued |
| T5 | The industrial task, ⑥ EXPAND: `kitting` in OUR ALOHA 2 scene (parts → slots, bimanual where reach demands), scripted demos generated with domain randomisation centred on the identified parameters → the mixture axis (ours / public / both) × the family axis × the combination axis — the recipe engine's first full walk, "quickly train the robot to an industrial task" as an experiment with a certificate at the end | our sim | certificate dry run | queued |
| T6 | Reinforcement learning in parallel worlds — the picture the operator asked for ("how can we see this multi-system sim", Isaac Lab's grid of learners). Two layers: `tools/show-many.py` (N rigs on a grid, CPU, replay — measured 16 worlds at 0.29× realtime, the argument for the GPU) and `tools/rl-watch.py`: MuJoCo Playground's `AlohaHandOver` (Menagerie's MJX-patched ALOHA) trained with brax PPO on MJX, thousands of worlds on the 3090 Ti, sixteen of them on screen driven by the current policy at every evaluation. Stack installed 2026-08-26 (`jax[cuda12]` 0.11.1 sees the RTX; `playground`; `mujoco-mjx`; `mujoco-warp` 3.12 — which is version-locked to MuJoCo 3.12 and fails through the FFI against this venv's 3.11, so `impl="jax"` for now). Exploratory: Playground's task and reward, not our harness's; the harness judges the result afterwards. The RLT-shaped refinement of a frozen generalist stays the destination | Playground sim | watched curve, then harness | **stack up, tool written** |

Tracking: wandb API surface only, off by default (`--wandb.enable=false`),
Trackio the escape hatch; every run stamped with the bundle, scene,
dataset and recipe hashes it was produced under.

## 4. Next rung (evaluation side)

rq_pipeline/tasks/aloha2.py (next rung) — transfer cube: cube on the table
between the arms, right arm picks, hands to left, success = cube held
by the left gripper clear of the table. Scripted reference policy,
graded ladder (pick / no-close / limp), paired trials, through the
harness to a certificate dry run — the same shape that proved the
SO-101 tasks, one rig up. T1 above needs it; they land together.
