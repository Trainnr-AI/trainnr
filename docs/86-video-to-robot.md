# 86 — From video to robot training data

*Proposed 2026-11-01, at Prakhar's request: a pipeline from raw video to
robot training data, built on NVIDIA's current stack, and project
templates in the Studio that start a project on it. This page holds the
evidence, the split it rests on, the core modules and the phases.
Branch: `feat/video-to-robot`.*

## 1. The proposal

A person has video: their own phone footage of someone dancing or
folding a cloth, an egocentric recording of hands at work, or the
cameras of a robot they teleoperated. They want a policy for their
robot out of it. trainnr takes the video in, as a stamped artifact like
every other, and carries it through the loop it already has: a motion,
a robot, a task, training, an evaluation with intervals, a deployment.

The split this rests on, and the one rule the pipeline keeps:

> **Video gives the motion and the intent. Physics gives the contact and
> the verdict.** Nothing estimated from video becomes a training label
> until a simulated robot has done it and a success criterion has said
> it worked.

A new project starts from a **template**: a named plan for one of these
paths, with NVIDIA's models in each step by default, which the Studio's
New project offers and the project's Overview then walks.

## 2. NVIDIA's video-to-robot stack (primary sources, read 2026-10-31)

| Piece | What it does | In → out | Runs on | Licence |
|---|---|---|---|---|
| **GEM-X** ([NVlabs/GEM-X](https://github.com/NVlabs/GEM-X)) | monocular whole-body 3D human pose and motion on the SOMA body model; hand pose added 2026-10-21 | video (dynamic camera, ego or exo) → `.pt` of SOMA motion, camera and world frame | CUDA (ONNX/TensorRT path); CPU and ARM64 inference listed | code Apache-2.0 with OpenMDW-1.1 exceptions; weights under their own terms (the Hugging Face card says the NVIDIA Open Model License) |
| **SOMA Retargeter** ([NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter)) | constrained IK from SOMA motion onto a humanoid, on Newton and Warp | SOMA `.bvh` → `.csv` of root pose and joint values | NVIDIA GPU | code Apache-2.0; its sample motions under NVIDIA's evaluation licence. "Active beta" |
| **mjlab tracking** (installed in `trainnr-mjlab`, `mjlab/tasks/tracking`) | BeyondMimic-style RL that tracks a reference motion, Unitree G1 config shipped; its `csv_to_npz` script turns a retargeted CSV into its motion file | `.npz` (joint and body poses and velocities) → policy | NVIDIA GPU | Apache-2.0 |
| **Isaac Lab 3.0** (GA, GTC Berlin, [2026-10-21](https://nvidianews.nvidia.com/news/nvidia-advances-isaac-platform-for-general-purpose-robotics-with-new-reference-humanoid-frameworks-simulation-pipelines-and-data)) | NVIDIA's own robot-learning framework, on Newton and PhysX | — | RTX GPU, Linux or Windows | open source |
| **Cosmos Transfer 2.5** ([nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)) and **Cosmos 3** (2026-06) | appearance variation of robot video that keeps the trajectory ("keep the robot trajectory, change the world it sees", [ROBOTIS](https://docs.robotis.com/docs/systems/aisapiens/resources/technical_story/building_humanoid_skills/)); Cosmos 3 folds Predict, Transfer, Reason and Policy into one model | robot video + its controls → varied video, same actions | large NVIDIA GPU | code Apache-2.0; weights NVIDIA Open Model License |
| **Cosmos Curator** (Physical AI Data Factory blueprint, GTC 2026) | filter, annotate and deduplicate large video sets | raw video → curated clips | NVIDIA GPU | open source |
| **GR00T N1.7** ([NVIDIA/Isaac-GR00T](https://github.com/Nvidia/Isaac-GR00T)) | 3B vision-language-action model, Cosmos-Reason2-2B backbone, pretrained with 20k hours of human egocentric video (EgoScale) in a relative end-effector action space | LeRobot **v2** dataset + `meta/modality.json`, tag `NEW_EMBODIMENT` → policy | fine-tune: 40 GB+ GPU; inference: 16 GB+ (Jetson Orin/Thor included) | code Apache-2.0; weights NVIDIA Open Model License |
| **GR00T-Dreams / DreamGen** ([NVIDIA/GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams)) | a video world model imagines the task; an inverse-dynamics model labels the video with actions ("neural trajectories") | — | thousands of GPUs | code Apache-2.0 |

NVIDIA's own end-to-end examples follow one of two shapes. The
whole-body one is video → GEM-X → SOMA Retargeter → RL tracking in
Isaac Lab → `policy.onnx` on a Jetson (ROBOTIS, AI Sapiens). The
manipulation one is teleoperated robot episodes → Cosmos Transfer
variations → GR00T fine-tune (ROBOTIS, AI Worker; Sereact at GTC Berlin).

## 3. What the evidence allows

trainnr's research already ruled on half of this
(`e2e-research/22-data-generation.md` §3, "Do not bother"): a video
world model used as a *data generator* produces plausible approach motion
and plausible semantics, and does not produce contact. Veo-Act's
video-plus-inverse-dynamics policy reached near-zero task success;
GAUGE measured the recovered dynamics as wrong, worst at impulsive
contact; DreamGen's real-robot gains cost about 81,000 GPU-hours. The
research note's rule stands, and this design keeps it:

> Every system that works uses them for high-level planning or appearance
> diversity and gets low-level actions from somewhere else.

So the pipeline uses video in the three ways the evidence supports, and
gets the actions from physics each time:

1. **Video as a motion reference** (whole-body). GEM-X estimates the
   person's motion, the retargeter maps it onto the robot's joints, and
   an RL policy learns, in simulation, to *track* it while staying up.
   The policy's actions come from its own rollouts against physics; the
   video only says what the motion should look like.
2. **Video as intent and kinematics** (manipulation). Hand and wrist
   trajectories from GEM-X are retargeted to the robot's end effector
   and **replayed in simulation with success filtering**: the robot attempts
   each one, the task's success criterion judges it, and only the
   successes become episodes. The video proposes; the success criterion decides.
   (`collect/press.py` already does exactly this for scripted and planned
   experts; the retargeted video is one more expert.)
3. **Video augmentation** (appearance only). Cosmos Transfer varies how a
   recorded episode *looks* while its actions stay the ones the robot
   actually took. Because the research note found appearance diversity
   worth little in some settings (MimicLabs; ROSIE lost to a green
   screen), an augmented dataset earns its place the trainnr way: a
   paired evaluation of the policy trained with it against the policy
   trained without it, before it is called better.

And the one it does not: **generating action data from imagined video**
(GR00T-Dreams, Cosmos Predict as a data source). It stays out of the
defaults until a measured result says otherwise; §8 keeps it as a seam.

## 4. Where trainnr stands (2026-11-01)

What exists and fits:

- **The loop's kinds and stages** (`project/kinds.py`, `project/index.py`
  `STATES`): recording, robot, task, batch, dataset, run, policy,
  certificate, deploy. The data stage is proved by a *batch*
  (`datasheet.md`); a reinforcement project marks it not needed.
- **Recordings** (`robots/recording.py`, `trainnr-recording/1`) with a
  `collection` word (teleop, mocap, wearable, …) and a `basis`; the
  `mocap` adapter already reads human BVH and says it is not retargeted
  (`docs/76` §5.1).
- **The press** (`collect/press.py`): the keep-only-successes loop with
  per-episode stamps; its export to LeRobot v3 with provenance
  (`collect/demo_export.py`).
- **Video handling** for scenes only (`scenes/capture.py`: ffprobe,
  ffmpeg, COLMAP; `scenes/tooling.py`: find an external tool or say how
  to install it).
- **RL training** through `trainnr-mjlab`, which ships mjlab's tracking
  task and its CSV-to-motion converter; **cloud GPUs** through
  `cloud/` and `tools/cloud-gpu.py`; long work as **jobs** (`mcp_jobs.py`).

What does not: any video kind, any pose estimator or retargeter, a
tracking task in trainnr, an export in GR00T's LeRobot-v2 flavour, an MCP
tool that makes a dataset from a batch, and any notion of a template
(the Studio's New project asks for a name only, `onboarding.rs`).

## 5. The architecture

### 5.1 The stages and their artifacts

```
 video ──ingest──▶ footage ──estimate──▶ motion ──retarget──▶ reference
 (mp4)             (new kind)            (recording,           (recording on
                                          collection "video")   the robot)
                                                                  │
            ┌─────────────── whole body ──────────────────────────┤
            ▼                                                     ▼ manipulation
   tracking task ──train (RL, mjlab)──▶ run ─▶ policy     replay in simulation
   (task)                                                  (successes kept)
                                                                  │
                                                                  ▼
                                   augment (optional) ◀── batch ──▶ export
                                   Cosmos Transfer,               LeRobot v3, or v2 +
                                   a new batch citing it          modality.json (GR00T)
                                                                  │
                                              fine-tune (GR00T N1.7, cloud) ─▶ policy
   policy ─▶ evaluate (intervals; paired when augmented) ─▶ certificate ─▶ deploy
```

- **footage** is the one new kind: `footage.json` (`trainnr-footage/1`)
  beside the video it describes, stamped over its bytes like any bundle,
  with what ffprobe says (duration, frame rate, size, codec), the camera
  if known (intrinsics, ego or exo), the `basis` (own, public) and where
  it came from. It proves the Telemetry stage, as a recording does.
- **motion** is a recording (`trainnr-recording/1`) whose `collection`
  gains one word, `video`, and whose manifest names the estimator by
  stamp (GEM-X's version and weights hash) and the footage it came from.
  Its signals are the SOMA joints and root over time, with the
  estimator's per-frame confidence.
- **reference** is a recording on the robot: joint positions and root
  pose over time for one robot bundle, naming the motion, the
  retargeter and the robot by stamp, with what it violated (joint
  limits, self-collision, foot skate) counted, never hidden.
- Everything after is the loop as it is: a task, a batch, a dataset, a
  run, a policy, a certificate, a deployment. A tracking task cites its
  reference; a filtered batch cites the references its expert replayed;
  an augmented batch cites the batch it varied and the augmenter.

### 5.2 The core modules

One new subpackage, `trainnr/trainnr/video/`, at tier 3 beside
`collect` and `scenes` (it imports `bundles`, `robots` and `scenes`'
tooling; nothing below it imports it):

| Module | Holds |
|---|---|
| `video/footage` | `ingest_footage`: probe, hash, copy or link the file, write `footage.json`; clip by time range; reuses `scenes/capture.py`'s ffprobe and ffmpeg |
| `video/estimate` | the `MotionEstimator` protocol and registry (entry point `trainnr.motion_estimators`); writes a motion recording; refuses a clip whose confidence is too low to use |
| `video/retarget` | the `Retargeter` protocol and registry (`trainnr.retargeters`); writes a reference; counts violations against the robot bundle's own limits |
| `video/quality` | the checks every motion and reference passes: confidence, jitter, foot skate, limits, self-collision; each check a named pass or fail in the record |
| `video/augment` | the `Augmenter` protocol and registry (`trainnr.augmenters`); writes a new batch citing the old one, its actions untouched and checked equal |
| `video/nvidia/` | the default implementations: `gem_x` (estimator), `soma` (retargeter), `cosmos_transfer` (augmenter). Each runs the NVIDIA tool **as an external program in its own environment**, found or installed the way `scenes/tooling.py` finds COLMAP, never as a trainnr dependency |
| `tasks/tracking` | the tracking task family: a reference, a robot, a success criterion (tracking error under a bound for the whole clip, never fallen) |
| `collect/retarget_feed` | the expert that replays a reference's end-effector path, so a manipulation batch is filtered for success like any other |
| `collect/gr00t_export` | a dataset in GR00T's LeRobot v2 flavour with `meta/modality.json`, written from the same episodes as the v3 export, with the same provenance |
| `trainnr_mjlab/tracking` | the bridge to mjlab's tracking task: reference → motion `.npz` (its `csv_to_npz`) → train; the run records the reference's stamp |

Protocols and registries follow `plugins.load_group` like every other
seam, so a different estimator (a phone-friendly one, a mocap suit) or
retargeter (MuJoCo IK on the CPU) is a plug-in, and the NVIDIA ones are
defaults, not requirements.

### 5.3 The tools

Verb first, as GOVERNANCE.md asks; long ones are jobs:

- `ingest_video(path, name)` → footage
- `estimate_motion(footage, name, estimator="gem-x")` → motion (job)
- `retarget_motion(motion, robot, name, retargeter="soma")` → reference (job)
- `create_task(...)` gains the tracking family, naming a reference
- `train_tracking(task, ...)` → run (job, mjlab)
- `replay_references(task, references, ...)` → batch (job)
- `export_dataset(batch, format="lerobot-v3" | "gr00t")` → dataset (the
  missing "make a dataset" tool the map found)
- `augment_batch(batch, augmenter="cosmos-transfer", ...)` → batch (job)
- `import_policy(path, robot, name)` → policy: one trained elsewhere
  (Isaac Lab, most often), from its ONNX file, joint order and gains,
  evaluated and gated like one of ours (§9)

Each refuses with its reason rather than guessing: no GPU, the tool not
installed (and how to install it), a licence the user has not accepted,
a clip too short or too unsure to use.

### 5.4 Compute, network and licences

- The NVIDIA models need an NVIDIA GPU, mostly a large one. A tool says
  what its step needs (`runs_on`) and runs locally when the machine
  has it, or on a rented GPU through `cloud/` when the user asks.
- **No download happens without the user asking** (AGENTS.md;
  SECURITY.md lists the network calls). Fetching GEM-X's weights or a
  Cosmos checkpoint is a step the user starts, and SECURITY.md gains
  each one.
- **Weights under the NVIDIA Open Model License never ship with
  trainnr.** trainnr runs NVIDIA's tools where the user installed them;
  the records name the model and its weights by hash so a result says
  what made it. The user accepts each licence once, and the tool says so.

## 6. Templates

A template is a named plan for one path through §5.1: the loop, the robot
it is designed for, and its steps, each naming its stage, what it does,
the trainnr tool that does it, the model used, and what it runs on.
`trainnr/trainnr/templates.py` holds them; the Studio carries a copy of
the list for its New project picker (checked equal by
`test_studio_mirrors`, like the samples).

- **Creating a project from a template** writes the template's name into
  `project.json` (`template`) and its loop. Nothing is downloaded and
  nothing runs.
- **The Overview walks it.** The index carries the template's steps with
  their state (done when the stage's artifact exists, else to do), and
  a step whose tool is not built yet says *planned*. The template's
  steps replace the three generic start steps.
- **A step names a tool only when the tool exists**, and a test checks
  every named tool is in the server, so a template cannot claim what
  trainnr cannot do.

The first three, NVIDIA's stack by default:

| Template | Robot | Loop | Path |
|---|---|---|---|
| **Humanoid motion from video** | Unitree G1 | reinforcement | video → GEM-X → SOMA Retargeter → tracking task → RL in mjlab → evaluate → ONNX deployment |
| **Manipulation from human video** | SO-101 | imitation | hand video → GEM-X hand pose → retarget to the gripper → replay in simulation, keep successes → dataset (GR00T flavour) → fine-tune GR00T N1.7 → evaluate |
| **More from your own demonstrations** | SO-101 | imitation | teleoperated episodes → Cosmos Transfer variations → fine-tune GR00T N1.7 → paired evaluation, with and without the variations |

## 7. The phases

1. **Templates and this design** (this branch): `templates.py`, the
   `template` argument to `create_project`, the template in the index,
   the Studio's picker in New project and the steps on the Overview.
2. **Footage**: the kind, `ingest_video`, the Recordings page showing a
   video's first frame and length.
3. **Whole body, end to end on a G1**: GEM-X, SOMA, the tracking task,
   `train_tracking` through mjlab, an evaluation with intervals. Measured
   as a record in `docs/findings/` before any claim.
4. **Manipulation on the SO-101**: hand pose, the retarget feed, success filtering,
   `export_dataset` with the GR00T flavour, a GR00T N1.7 fine-tune on a
   rented GPU.
5. **Augmentation**: Cosmos Transfer, and the paired evaluation that
   decides whether it stays.
6. **Policies trained elsewhere**: `import_policy`, then the same
   evaluation and deployment gate as ours. Independent of 2–5; it can
   come whenever someone brings an Isaac Lab policy.

## 8. What we do not do

- Train a video world model, or treat imagined video as action labels
  (§3). The augmenter seam takes appearance only; an "imagined
  trajectory" source, if a measured result ever earns it, is an
  expert like any other and passes the same success criterion.
- Ship NVIDIA weights, or call NVIDIA's services, or require an NVIDIA
  GPU for anything that does not need one: footage, templates and the
  Overview work on any machine.
- Train in Isaac Lab (§9). Its policies come in through `import_policy`.
- Hide a failed estimate or retarget: a clip that cannot be used is a
  record that says why, not a missing file.

## 9. Decisions and open questions

**Decided 2026-11-01: mjlab is the only trainer; Isaac Lab policies come
in, Isaac Lab training does not.** Reasons:

- The trainer is part of the instrument. A second one means the measured
  actuator models (BAM) and the linter ported and kept twice, another
  environment (Isaac Lab 3.0 pins its own PyTorch, CUDA, Warp and
  Newton), and every record measured twice or shown to agree.
- Nothing in phases 2–5 needs it: mjlab ships the tracking task and the
  CSV-to-motion converter; GR00T has its own trainer; GEM-X, SOMA and
  Cosmos are external programs.
- The physics is converging. Isaac Lab 3.0 installs without Isaac Sim on
  Newton, whose default solver is MuJoCo-Warp, the engine mjlab runs
  ([physics backends](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/index.html));
  Newton is still beta there. The PhysX backend needs Isaac Sim and its
  proprietary licence, so it stays off every default path.
- What travels out of Isaac Lab is an ONNX policy with its joint order
  and kinematics ([e2e-research/59](e2e-research/59-arena-composition-and-datagen.md)
  §3), and NVIDIA lists sim-to-real validated evaluation as a gap (§4). So
  `import_policy` (§5.3, phase 6) gives an Isaac Lab user trainnr's
  evaluation and gate without trainnr keeping their trainer.

Revisit when a user brings an Isaac Lab task that cannot be ported, or
when Newton leaves beta: then an Isaac Lab trainer on Newton, without
Isaac Sim, is cheap, and one re-measured record showing it agrees with
mjlab within the intervals admits it.

Still open:

- **The robot for the first whole-body run.** The SOMA Retargeter ships
  the G1, H2, Booster T1 and two AgiBot models; mjlab's tracking config
  is the G1. Is a G1 the target, or is there a robot you have?
- **GR00T fine-tuning cost.** 40 GB+ means a rented A100 or H100 for
  each fine-tune. Which provider and budget per run?
