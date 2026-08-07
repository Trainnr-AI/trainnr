# Data collection: where the real data comes from

Research date: **2026-08-08**. Question: what is the fastest and cheapest way
to collect enough real-world manipulation data to train a state-of-the-art
policy for an SO-101-class arm on a mobile base at a commercial site?

> **TL;DR.** Use the **SO-101 leader arm**. A joint-space leader beats VR
> controllers by **5× on success rate** and — the number that actually matters —
> produces trajectories with **78% fewer near-singular configurations**, because
> VR data is *poisoned*, not merely slower. Collect **~50 episodes per task and
> then stop**, switching to corrections, which are reported at **10× the data
> efficiency**. Buy **diversity, not repetition**: 32 environment×object
> combinations at 50 demos each reaches ~90% on novel objects; more repetitions
> per combination saturate. **Skip UMI** despite its 3× raw throughput.

---

## 1. Teleoperation methods, with numbers

### The decisive study

**BEHAVIOR Robot Suite / JoyLo** (arXiv 2503.05652, submitted **2025-03-07**,
CoRL 2025) ran a 10-participant user study comparing a **joint-space leader
device** against VR controllers and Apple Vision Pro, on a wheeled robot with
arms — architecturally the same problem as an SO-101 on a base.

| Measure | JoyLo vs VR controllers |
|---|---|
| Task success rate | **5× higher** |
| Median completion time | **23% shorter** |
| "Open dishwasher" (articulated object) | **+67% success** |
| Navigation speed | **71% faster than Apple Vision Pro** |
| **Singularity ratio** | **78% lower than VR, 85% lower than Vision Pro** |
| Cost | **< $500** |

All 10 participants rated it most user-friendly.

**The singularity ratio is the finding.** A *singularity* is an arm
configuration where the joint angles lose the ability to produce motion in some
direction — the wrist lines up with the elbow and the arm briefly cannot move
sideways at any joint speed. Near one, tiny changes in the desired hand position
demand enormous joint motion. A teleoperation method that spends more time near
singularities does not merely produce slower demonstrations; it produces
demonstrations with **wild, physically unreproducible joint velocities baked
into the action labels**. The policy learns those.

That is why this is not a comfort preference. **An SO-101 leader arm is a
JoyLo-class device** — kinematically isomorphic to the follower, joint-space,
no inverse kinematics anywhere in the loop, and therefore no singularity to be
near.

### The field

| Method | Cost | Demos/hr | Robot needed | LeRobot support |
|---|---:|---:|---|---|
| **SO-101 leader arm** | ~$110 (leader half of a $229.88 pair) | ~35–60 | Yes | ✅ `so_leader` — and **the only kind that unlocks DAgger** |
| JoyLo (BRS) | <$500 | — | Yes | ❌ (own stack) |
| GELLO | ~$300 (unverified) | — | Yes | ❌ |
| Quest 3 / OPEN TEACH | $500 | — | Yes | ❌ |
| Apple Vision Pro | ~$3,499 | — | Yes | ❌ |
| Phone (ARKit/ARCore 6-DoF wand) | $0 | low | Yes | ✅ `phone` — **`main` branch only, treat as experimental** |
| Gamepad / keyboard | $0–70 | very low | Yes | ✅ |
| Foot pedal | ~$25 | n/a | — | ✅ (PCsensor FootSwitch, Linux, documented) |
| AirExo-2 exoskeleton | "low-cost" | high | **No** | ❌ |
| **UMI handheld** | **$371** | **48** | **No** | ❌ |

LeRobot's full teleoperator list as of **2026-08-08**: `so_leader`,
`bi_so_leader`, `koch_leader`, `openarm_leader`, `openarm_mini`,
`bi_openarm_leader`, `bi_openarm_mini`, `rebot_102_leader`,
`bi_rebot_102_leader`, `omx_leader`, `gamepad`, `keyboard`, `phone`,
`homunculus`, `reachy2_teleoperator`, `unitree_g1`.

2026 additions worth knowing about, none of which changes the recommendation:
**DexDirect** (arXiv 2607.27784, 2026-07-30) — kinesthetic drag of a
gravity-compensated arm, **17.2× more successful demos than AnyTeleop**;
**AutoDex** (arXiv 2606.23689) — 500 trajectories in 10.3 h vs 49.4 h teleop;
**Human-as-Humanoid** (arXiv 2606.32009) — 4.8–7.2× raw throughput.

---

## 2. Robot-free collection, and why it is skipped

### UMI, honestly

Universal Manipulation Interface (arXiv 2402.10329, RSS 2024): a handheld
parallel-jaw gripper with a wrist camera, so a human collects demonstrations
with no robot at all.

The case *for*: **$73 gripper + $298 GoPro = ~$371**, and measured throughput of
**48 demos/hour vs ~16 for spacemouse teleop** on cup arrangement. Policy
results are real — cup arrangement 100% (20/20), dynamic tossing 87.5%
(105/120), bimanual cloth folding 70%, dish washing 70%, cross-embodiment
UR5→Franka 90%.

The case *against*, and it is decisive, from the authors' own README:

> **"ORB_SLAM3 is still the most fragile part of the UMI pipeline."**

> **"The policy doesn't work well under direct sunlight, since the dataset was
> collected during a rainy week at Stanford."**

The pipeline is ORB-SLAM3 (a UMI-specific Docker fork), plus camera–IMU
calibration, plus GoPro Labs firmware flashing with QR-code configuration, plus
an HDMI capture card. Every one of those is a thing that can be broken on
someone else's premises.

**A commercial site has windows.** A lighting-brittle collection pipeline whose
fragile component is visual SLAM is the wrong tool for a room with afternoon
sun.

**And the value proposition does not apply here.** Robot-free collection exists
so you can gather data where you cannot bring the robot. We can bring the robot.

### What robot-free is genuinely for, later

The strongest 2026 results come from **robot-isomorphic exoskeletons**, which
is what an SO-101 leader arm already is — except the leader arm gets native
action labels with zero calibration:

- **AirExo-2** (arXiv 2503.03081): a policy trained *solely* on adapted
  in-the-wild exoskeleton data **matches** one trained on teleoperated data.
- **ExoGS / AirExo-3** (arXiv 2601.18629, 2026-01-26): millimetre-level
  trajectories, reconstructs the scene as Gaussian splats for augmentation;
  *"significantly improves data efficiency and policy generalization compared to
  teleoperation-based baselines."*
- **XRZero-G0** (arXiv 2604.13001, 2026-04-14) publishes the only exchange rate
  found anywhere: **10:1 robot-free to real-robot data matches real-only
  performance at 1/20 the acquisition cost**, from a 2,000-hour corpus at 85%
  data validity.

That is the scaling answer for a later stage, not the pilot answer.

### Human video

- **EgoMimic** (arXiv 2410.24221): *"adding 1 hour of additional hand data is
  significantly more valuable than 1 hour of additional robot data"* — concretely,
  2 h robot + 1 h hand beat ACT on 3 h robot. But it requires a wearable giving
  3D hand tracking **and** a robot whose kinematics and wrist-camera view were
  deliberately matched to the human data.
- **EgoDex** (arXiv 2505.11709, Apple): **829 hours, 194 tasks**, native 3D hand
  tracking. Benchmarked for *hand trajectory prediction*, not end-to-end policies.
- **WIYH** (arXiv 2512.24310, v3 2026-03-15): >1,000 hours at millimetre
  accuracy with an auto-labelling pipeline; reports manipulation success in
  cluttered scenes going **8% → 60%** when human data is mixed in.

Promising, dependent on hardware we do not have, and not on the critical path.

---

## 3. The correction loop, which is the real multiplier

After roughly 50 demonstrations per task, collecting more demonstrations stops
paying. What pays is correcting the policy while it runs.

| Method | Date | Result |
|---|---|---|
| **RaC** (arXiv 2509.07953) | 2025-09-09 | *"outperforms the prior state-of-the-art using **10× less data collection time and samples**"*; performance **scales linearly** in the number of recovery maneuvers |
| **HIL-SERL** (arXiv 2410.21845) | 2024-10-29 | near-perfect in **1–2.5 hours**; **2× success, 1.8× faster** than imitation |
| **RECAP / π\*0.6** (arXiv 2511.14759) | 2025-11-18 | *"more than doubles task throughput and roughly halves the task failure rate"* on the hardest tasks |
| UniIntervene (arXiv 2606.12372) | 2026-06-10 | **+8.6% success, 57% fewer human interventions** |
| EvoHIL (arXiv 2608.03872) | **2026-08-04** | self-evolving success classifier; evaluated on Franka FR3 **and SO-101** under controlled lighting shifts |

RaC's motivating observation is the one to internalise: contact-rich,
deformable and long-horizon tasks *"plateau far below perfect execution, even
with thousands of expert demonstrations."* You cannot buy the last 20% with
more demonstrations. You buy it with recoveries.

**LeRobot implements this directly.** `lerobot-rollout --strategy.type=dagger`,
and the documentation states it follows RaC's recovery/correction decomposition.
Other strategies in the same tool: `base` (no recording), **`sentry`**
(continuous recording with automatic upload every N episodes — this is your
deployed-log collector), `highlight` (ring buffer, save on keystroke),
`episodic`.

Documented recipe from LeRobot's own example: pre-train **50,000 steps**, then
each correction round is **`--strategy.num_episodes=50`** and a **20,000-step**
fine-tune.

---

## 4. Autonomous logs as training data

Deployed robots generate data for free, but it is unlabelled. As of 2026 that is
a solved-enough problem to plan around:

- **GVL** ("VLMs are In-Context Value Learners", arXiv 2411.04549) — temporal
  progress estimates enabling **success detection across 300+ real-world robotic
  tasks with no training or fine-tuning**. A zero-cost reward labeller you can
  run over logs today.
- **NILS** (arXiv 2410.17772) — *"automatically labels uncurated, long-horizon
  robot data at scale in a zero-shot manner without any human intervention."*
- **Xiaomi-Robotics-1** (arXiv 2607.15330, **2026-07-16**) — the existence proof
  at industrial scale: **>100,000 hours** of real trajectories trained with
  *"a scalable auto-labeling pipeline that annotates trajectory clips with
  natural languages describing scene state transitions."*

**Practical answer:** deployment logs are not free training data on their own,
but with zero-shot labelling plus a lightweight classifier they become a usable
corpus at near-zero marginal labelling cost. Budget a human to confirm a sample
of the labels.

---

## 5. What the data must look like

All of the following are **verbatim from official LeRobot documentation**, and
all of them are cheap and commonly violated.

From *"What makes a good dataset"*:

- **Two camera views** preferred
- **Minimum 480×640, or 720p**; **~30 fps**
- Neutral stable lighting, no colour casts, consistent exposure, sharp focus
- Static, non-distracting background
- **"The leader arm and human limbs must NOT appear in frame"** — only the
  follower arm and the manipulated objects should move. ⚠️ *This is the single
  biggest silent dataset-killer, and it is the reason to mount the leader arm
  beside or behind the robot, never over it.*
- Feature naming `<modality>.<location>`: `images.top`, `images.wrist.left`.
  **Not** `images.laptop`
- Task strings **25–50 characters** in real English — *"Pick the yellow lego
  block and put it in the box"*. Never `task1`

From the imitation-learning tutorial:

- **"Record at least 50 episodes, with 10 episodes per location."**
- **"Keep the cameras fixed and maintain consistent grasping behavior."**
- **"A good rule of thumb: you should be able to do the task yourself by only
  looking at the camera images."**
- **"Avoid adding too much variation too quickly, as it may hinder your
  results."**

From the cameras guide: **"lighting matters more than resolution."** RealSense
supports fixed manual `exposure`, `gain` and `white_balance` — **use them at a
commercial site**, where auto-exposure will otherwise drift through the day.

Augmentations are **train-time only**, never baked into recordings, so they can
be retuned without re-recording. Start conservative: brightness 0.9–1.1.

### The variation ladder

From the **Data Scaling Laws** study (arXiv 2410.18647, 40,000+ demonstrations
and 15,000+ real rollouts — by a wide margin the largest evaluation budget in
this field):

- **32 environments, each with a unique manipulation object and 50
  demonstrations** → **~90% success in novel environments with unseen objects**.
  That is **1,600 demonstrations total** for genuine zero-shot generalisation.
- **"The diversity of environments and objects is far more important than the
  absolute number of demonstrations; once the number of demonstrations per
  environment or object reaches a certain threshold, additional demonstrations
  have minimal effect."**
- The power law is in **number of environments and objects**, not number of demos.
- *"Four data collectors working for one afternoon"* sufficed to reach 90% on two
  tasks.

So: **Phase A** — one zone, one object, 5 positions × 10 demos = 50 episodes,
everything else frozen. **Phase B** — after that trains, add *zones and objects*,
not repetitions, toward 32 combinations.

The complementary warning, **Curse of Precision** (arXiv 2607.23108,
**2026-07-25**): for high-precision closed-world tasks such as assembly and
insertion, the required number of demonstrations grows **super-exponentially**
as target precision approaches a system limit set by sensors, expert quality and
hardware — *adding a wrist camera measurably lowers that limit; adding
demonstrations does not*. **If the task needs sub-millimetre insertion, do not
try to buy it with data — go to HIL-SERL.**

---

## 6. Format and scale

**LeRobotDataset v3** (in `lerobot >= 0.4.0`) is file-based rather than
episode-based: many episodes per Parquet shard and per MP4, with episode
boundaries in metadata. Designed to scale to millions of episodes, and streamable
from the Hub with `StreamingLeRobotDataset` / `--dataset.streaming=true` — no
local copy needed.

```
meta/info.json        schema, fps, path templates
meta/stats.json       mean/std/min/max for normalisation
meta/tasks.jsonl      natural-language task strings → integer IDs
meta/episodes/        per-episode lengths, tasks, byte and frame offsets
data/                 Parquet shards, many episodes per file
videos/               MP4 shards per camera, many episodes per file
```

⚠️ **You must call `dataset.finalize()` before `push_to_hub()`** or the Parquet
footers are never written and the dataset is silently corrupt.

**Scale is a non-problem.** Measured, `lerobot/svla_so101_pickplace`:
**86.1 MB for 50 episodes / 11,939 frames**, 30 fps, 2 cameras at 480×640.
That is ~1.7 MB per 8-second episode. Extrapolating (*our arithmetic, not an
official figure*): **1,000 episodes at 30 s ≈ 6.5 GB**; 1,600 episodes ≈ 16 GB.

**Do not reduce resolution to save disk.** Disk is not the constraint.

### Open datasets, and a licence trap

| Dataset | Scale | Licence |
|---|---|---|
| Open X-Embodiment | 1M+ trajectories, 22 embodiments, 60 datasets | per-subset; **check individually** |
| DROID | 76,000 trajectories, 350 h, 564 scenes | commonly cited MIT (unverified) |
| **AgiBot World Beta** | >1M trajectories, 2,976 h, 48.1 TB | ⚠️ **CC BY-NC-SA 4.0 — non-commercial. Do not pretrain a deployed policy on it.** |
| RH20T | >110,000 contact-rich sequences | — |
| ArmnetBench v0.1 | **3,118 labelled SO-101 episodes** | — |

And a warning that belongs here rather than in
[22-data-generation.md](22-data-generation.md), because it is about *collection*
discipline: **MimicLabs** (arXiv 2506.13536, ICLR 2025) found that
*"all of the models we co-trained with all of DROID failed to learn the tasks"* —
0% — while retrieving roughly **one tenth** of DROID, aligned on **camera pose**,
scored 85% and 75%. Their ranked importance is
**camera pose ≫ spatial arrangement ≫ … ≫ object texture (limited impact)**.

**Vary your camera pose deliberately during collection.** It is free, it is the
top-ranked diversity axis, and it is what makes anyone else's data usable to you
later.

---

## 7. Sources

LeRobot docs <https://huggingface.co/docs/lerobot/index> ·
imitation-learning tutorial <https://huggingface.co/docs/lerobot/il_robots> ·
cameras <https://huggingface.co/docs/lerobot/cameras> ·
LeRobotDataset v3 <https://huggingface.co/docs/lerobot/lerobot-dataset-v3> ·
HIL data collection <https://huggingface.co/docs/lerobot/hil_data_collection> ·
"What makes a good dataset" <https://huggingface.co/blog/lerobot-datasets> ·
teleoperators <https://github.com/huggingface/lerobot/tree/main/src/lerobot/teleoperators> ·
BEHAVIOR Robot Suite <https://behavior-robot-suite.github.io/>, arXiv 2503.05652 ·
UMI <https://umi-gripper.github.io/>, arXiv 2402.10329, <https://github.com/real-stanford/universal_manipulation_interface> ·
AirExo-2 arXiv 2503.03081 · ExoGS arXiv 2601.18629 · XRZero-G0 arXiv 2604.13001 ·
EgoMimic arXiv 2410.24221 · EgoDex arXiv 2505.11709 · WIYH arXiv 2512.24310 ·
RaC arXiv 2509.07953 · HIL-SERL arXiv 2410.21845 · RECAP arXiv 2511.14759 ·
UniIntervene arXiv 2606.12372 · EvoHIL arXiv 2608.03872 ·
GVL arXiv 2411.04549 · NILS arXiv 2410.17772 · Xiaomi-Robotics-1 arXiv 2607.15330 ·
Data Scaling Laws <https://data-scaling-laws.github.io/>, arXiv 2410.18647 ·
Curse of Precision arXiv 2607.23108 · MimicLabs arXiv 2506.13536 ·
SO-ARM100 BOM <https://github.com/TheRobotStudio/SO-ARM100>
