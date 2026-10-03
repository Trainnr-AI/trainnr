# Data generation: making more data from less

Research date: **2026-08-08**. Question: is synthetic or generated robot data
worth using in 2026, and if so which methods produce measurable real-world gains
per dollar?

Re-swept **2026-08-15** via the arXiv API and citation graphs (search quota
exhausted; one GreenAug citation pull failed and is noted where it matters).
Two claims moved: the "missing baseline" now has a first run (§1), and the
contact negative narrowed to *vision-only* world models (§3).

> **TL;DR.** The measured wins come almost entirely from **curating the data you
> already have** and **cheap geometric and visual augmentation of your own
> demonstrations** — not from video world models, which have the best videos and
> the worst evidence. A **$50 green screen** beat diffusion-based generative
> augmentation, 91% to 75%. Removing bad demonstrations is worth **+15–35
> points** and costs nothing. And the field's most important negative result is
> that **vision-only world models cannot generate contact**.

---

## 1. Two warnings that apply to every number below

**Almost nobody reports confidence intervals** (narrowed 2026-08-19: the
methodology now exists in one vendor post and one arXiv paper — NVIDIA
RoboLab's Clopper-Pearson intervals and PhAIL's Kaplan-Meier with clustered
bootstrap, see [20 §2](20-policies-and-models.md) — but in none of the papers
ranked below, and in no shipping framework). MimicGen evaluates 50 rollouts per
cell; MimicLabs 20; the co-training paper ~20 per task. At N = 20 and a true
success rate near 0.5, one standard error is roughly **11 percentage points**.
**Any reported gain under about 15 points on a single task is not
distinguishable from noise.** This document flags the results that survive that
filter.

**Almost nobody runs the comparison that matters** — "generate synthetic data"
versus "collect 50 more real demonstrations." Baselines are deliberately
starved: MimicGen's real baselines are **0%**, DemoGen's is **one
demonstration**. So the rankings here are *inference about value per dollar*,
not measurement of it. This is the largest hole in the field and it is recorded
in [27-open-questions.md](27-open-questions.md).

**First crack in that hole, 2026-08-15: the comparison has now been run once,
and synthetic won.** **LEGS** (arXiv 2606.01458, 2026-05-31) — 3DGS
re-rendering plus parametrized motion primitives, no video model — runs a
**count-matched** LEGS(50 episodes) vs Teleop(50 episodes) comparison across
3 tasks × 3 VLA backbones on a Unitree G1, and matches or beats teleop in
**all 9 cells** (e.g. 9/10 vs 4/10; 3/10 vs 0/10). Effort is accounted too:
~0.5 GPU-hours vs 1.5 operator-hours per 50 episodes, and under combined
appearance shift the teleop-trained policy "fails entirely" while LEGS holds.
Apply this doc's own filter honestly: **10 rollouts per cell is below the
noise threshold for any single cell** — the signal is the consistent 9-for-9
direction, and the stated limits are one platform, static scenes,
lighting-sensitive, small similar-geometry objects. One run, not a settled
question — but "nobody runs it" is no longer true, and no *video-model*
pipeline has run it yet.

---

## 2. The ranking, by measured real gain ÷ cost

### Tier 1 — do these first; they cost almost nothing

**1. Curate the demonstrations you already have.**

| Method | Result |
|---|---|
| **Demo-SCORE** (arXiv 2503.03707) | trains a classifier on the policy's own successful vs failed rollouts to find bad demos → **+15–35 percentage points absolute**, in sim and real, with **no new data collected** |
| **CUPID** (arXiv 2506.19121, CoRL 2025) | influence functions on closed-loop performance → SOTA diffusion policies on **<33% of the data** |
| **ATHENA** (arXiv 2606.16208, 2026-06-15) | scales influence functions to billion-parameter VLAs (**313× speedup**) → matches or exceeds full-data fine-tuning on **50% of sim and 66.7% of real data**, across 6 real tasks |
| **PSD metric** (arXiv 2605.01544, 2026-05-02) | ranks demonstration smoothness by power spectral density — **no policy learning, no environment interaction, no expert labels**. Pure signal processing on the trajectory. Reports higher success than uncurated baselines; ⚠️ no percentages in the abstract |
| **WARP-RM** (arXiv 2606.28320, 2026-06-26, Goldberg lab) | self-supervised progress reward from time-warp augmentations of the demos themselves — no rollouts, no environment interaction — used to weight demos in BC. Real bimanual T-shirt folding with increasingly polluted demo mixes: **WARP-BC 19/20 vs vanilla BC collapsing to 2/20**. A fourth independent curation mechanism; sits between PSD (needs nothing) and Demo-SCORE (needs rollouts) |

Several independent methods — online-rollout classifiers, influence functions,
spectral smoothness, progress consistency — converge on the same conclusion.
**In 2026 the quality-versus-quantity question is settled in favour of
quality**, and there is no comparably-sized result showing "just add more
demos" wins.

Start with the PSD ranker, since it needs nothing. Graduate to Demo-SCORE once
you have rollouts.

**2. Align and diversify camera pose.** MimicLabs (arXiv 2506.13536, ICLR 2025),
real Franka, 10 target demos + 1,000 co-training demos, 20 rollouts per model:

- *"All of the models we co-trained with all of DROID failed to learn the tasks."* — **0%**
- Retrieval aligned on **camera pose**: serve-snack **85% vs 0%**, pour **75% vs 0%**
- Aligned retrieval used **~1/10th the data** and almost always beat using everything
- Ranked importance: **camera pose ≫ spatial arrangement ≫ … ≫ object texture (limited impact)**

Cost: a tripod and a decision. **Retrieve, do not dump.**

**3. Fine-tune a pretrained model rather than training from scratch.** The
cleanest head-to-head anyone has published, from the SmolVLA paper
(arXiv 2506.01844) — same architecture, same demonstrations, real SO-100:

| Task | SmolVLA pretrained | SmolVLA **no pretraining** |
|---|---:|---:|
| Pick-Place | 75% | 55% |
| Stacking | 90% | 45% |
| Sorting | 70% | 20% |
| **Average** | **78.3%** | **40.0%** |

**+38.3 points from pretraining alone**, because someone else already paid the
~30,000 GPU-hours. See [20-policies-and-models.md](20-policies-and-models.md)
for which checkpoint.

**4. GreenAug — a green screen and random textures.** *"Green Screen
Augmentation Enables Scene Generalisation in Robotic Manipulation"*
(arXiv 2407.07868, 2024-07-10). **850 training demos, 8,200 real evaluation
episodes** — by a wide margin the largest real evaluation budget in this survey,
which is why these numbers mean something.

Average over 8 tasks × 3 novel scenes:

| Method | Success |
|---|---:|
| No augmentation | 55% |
| Standard computer-vision augmentation | 70% |
| **Generative augmentation (ROSIE-style diffusion inpainting)** | **75%** |
| **GreenAug-Rand** | **91%** |
| GreenAug-Gen | 77% |
| GreenAug-Mask | 58% |

Texture ablation: solid colour 65%, Perlin noise 66%, **MIL textures 87%**, none
48%. Green-screen coverage 0 → 100% of the trajectory: **48% → 87%**.

**Note what happened: the expensive diffusion method (75%) lost to $50 of
fabric (91%).**

Re-checked 2026-08-15: **no replication and no contradiction found** — an
arXiv full-text sweep returned nothing 2026-dated on green-screen
augmentation, though the Semantic Scholar citation pull failed (HTTP 429),
so a citing replication could exist unseen. Watch item: the GreenAug lab
itself (Edward Johns) now publishes **SynthICL** (arXiv 2606.08154,
2026-06-06) — in-context imitation trained *entirely* on synthetic RGB,
claiming **79% average on 16 unseen real tasks** from one test-time demo.
Baselines and rollout counts are not in the abstract; verify before it moves
any ranking. The $50-of-fabric author going fully synthetic is a signal.

### Tier 2 — high value, real setup cost

**5. DemoGen — spatial demonstration replay.** (arXiv 2502.16932, 2025-02-24).
**One human demonstration per task**, edited in 3D point-cloud space:

| Task | Source (1 demo) | DemoGen |
|---|---:|---:|
| Spatula-Egg | 10.0 | 88.0 |
| Flower-Vase | 6.3 | 82.5 |
| Mug-Rack | 6.3 | 85.0 |
| Fruit-Basket | 25.0 | 90.8 |
| **Average (8 tasks)** | **11.0** | **74.6** |

Cost: **22 seconds to generate 2,214 trajectories** (~147k observation-action
pairs) — 0.00015 s per pair. Essentially free.

⚠️ Limits the abstract hides: it edits **3D point clouds**, so it needs
depth observations, not RGB alone; single-view synthetic demos always show the
same face of the object; and *"success rates diminished on configurations more
far away from the demonstrated ones."* It is a spatial-generalisation augmenter
for quasi-static pick-and-place.

**6. RoboSplat — 3D Gaussian splat demonstration generation.**
(arXiv 2504.13175, RSS 2025). One demonstration plus splat editing — Gaussian
replacement, equivariant transforms, appearance editing, novel-view synthesis —
reaches **87.8%** across six generalisation axes versus **57.2%** for baselines
trained on *hundreds* of real demos plus 2D augmentation. One of very few
results where synthetic data beat a genuinely well-fed real baseline.

The splat-editing line kept moving through mid-2026, evidence grades noted:
**LEGS** (§1) is its strongest result — the count-matched win over teleop.
**WANDA** (arXiv 2607.13154, 2026-07-14) extends the DemoGen/RoboSplat recipe
to *mobile* manipulation — one real demo, background splats, whole-body
trajectory rearrangement — claiming long-horizon robustness with **no numbers
in the abstract**. **PRISM** (arXiv 2607.04880, 2026-07-06) generates
digital-cousin scenes plus executable demos from a single image, "up to 100%"
on three real tasks — a weak claim form, rollout counts unknown. Neither
enters the ranking until full-paper numbers are read.

**7. Sim co-training.** Covered in
[23-simulation-and-real2sim.md](23-simulation-and-real2sim.md) §4, including
the mixing-ratio landmine. Headline: **45.3% → 83.2%** average, and an
off-the-shelf task-agnostic asset library gives **+31.5%** against **+35.8%**
for hand-built scenes — so you do not have to build the assets.

### Tier 3 — situational

**8. MimicGen** (arXiv 2310.17596, CoRL 2023). Headline: ~200 human demos →
50,000+ demos. The **real-robot** section is rarely quoted and is the honest one:

| Task | Source demos | Generated | Policy on source | Policy on MimicGen |
|---|---:|---:|---:|---:|
| Stack (D1) | 10 | 200 | **0%** (50 evals) | **36%** |
| Coffee (D1) | 10 | 100 | **0%** (50 evals) | **14%** |

Generation success rate: Stack 82.3% of attempts, Coffee 52.1% — so half the
compute is discarded on Coffee, for a 14% real policy. **MimicGen on a real
robot converts "impossible" into "occasionally works."** Worth it as the
generator feeding sim co-training; not as a direct real-robot pipeline.
(2026-08-15: the line is incremental and sim-only — MinInter, arXiv
2606.24078, curates trajectories *during* generation and beats SkillGen on the
MimicGen benchmark; no MimicGen 2 or DemoGen 2 exists, and no real-robot
numbers since.)

**9. RoVi-Aug** (arXiv 2409.03403, CoRL 2024 oral) — robot and viewpoint
augmentation. **+30 points** in the specific cross-robot / cross-viewpoint
condition (50% vs 20% under large camera-pose change). But the pipeline is
LoRA-SAM → ControlNet → inpainting → novel-view synthesis, the authors admit
*"artifacts can cascade"*, and no GPU hours are reported. **If you can
physically match the camera, do that instead** — it is free, and MimicLabs says
it is the thing that matters.

---

## 3. Do not bother, with reasons

**Video world models as data generators.** The evidence, in order of how much
it should worry you:

- **Veo-Act** (arXiv 2604.04502, **2026-04-06**) — the skeptic's paper, and the
  most important negative result in the field. A pure frontier-video-model plus
  inverse-dynamics-model policy achieves instruction-following of 0.23–0.77 but
  **overall task success of 0.00–0.07**. Failure analysis: **57 interaction-stage
  failures versus 6** for a hierarchical version. Their words: generated hand
  motion during contact *"frequently contains geometric distortion, unstable
  contact configuration, or temporally inconsistent finger motion,"* and *"even
  minor variations in robot appearance, camera viewpoint, or scene layout can
  lead to substantially different video predictions."*

  > **World models generate plausible approach motion and plausible semantics.
  > They do not generate contact.** Every system that works uses them for
  > high-level planning or appearance diversity and gets low-level actions from
  > somewhere else.

  **Re-verified and sharpened, 2026-08-15.** No rebuttal of Veo-Act exists —
  a full citation sweep found two citers, both accepting its premise rather
  than refuting it. The negative got a stronger, *instrumented* citation:
  **GAUGE** (arXiv 2608.05948, 2026-08-06) benchmarks 6 image-to-video models
  and 3 physics engines against real measured trajectories across 22 task
  families, and finds video world models "produce trajectories with the
  expected equation form while recovering incorrect accelerations, momentum
  transfer, and oscillation timing" — worst at **impulsive contact**. This
  upgrades the claim from "downstream tasks fail" to "the recovered dynamics
  are measurably wrong." **H2R-Bench** (arXiv 2608.13049, 2026-08-13)
  corroborates for human→robot video transfer specifically ("functional
  contact transfer" fails). The field is repositioning rather than fixing it:
  **ImageWAM** (arXiv 2606.19531, 2026-06-17) asks whether world-action
  models need video generation at all, and answers with target-frame image
  editing instead of video rollout — 1/6 the FLOPs, 1/4 the latency.

  **The one qualifier the claim needs narrowing to: vision-only.** A
  visuo-tactile line adds a channel that video alone lacks. **TACO**
  (arXiv 2607.02840, 2026-07-03) closes the loop with a tactile-grounded
  "Recognize-Imagine-Label" correction cycle and claims **+44 points
  absolute** over the base policy on real contact-rich manipulation — its own
  motivation concedes the point, that vision-only world models "produce
  visually plausible yet contact-inconsistent trajectories." **ViTacWorld**
  (arXiv 2607.22530) and **FeelWorld** (arXiv 2607.24267) are the same trend,
  earlier-stage. All three are abstract-grade evidence — no rollout counts,
  no task lists confirmed — so treat this as a narrowing to watch, not a
  reversal: **vision-only world models do not generate contact; whether a
  tactile channel changes that is an open, thinly-evidenced claim.**

- **DreamGen / GR00T-Dreams** (arXiv 2505.12705, NVIDIA) — the one honest
  real-robot data point, and it is small: GR1 humanoid **37% → 46.4%**, Franka
  **23% → 37%**, SO-100 **21% → 45.5%**. That SO-100 average decomposes into
  strawberry picking 21% → 26% (noise) and tic-tac-toe 25% → 65% (real). The cost
  line nobody quotes: its 240k-sample dataset took **54 hours on 1,500 NVIDIA L40
  GPUs ≈ 81,000 GPU-hours**. And it does **not** compare against simply
  collecting more real trajectories.

- **VLAW** (arXiv 2602.12063, 2026-02-12) — reports honestly and thereby
  deflates its own headline: **+39.2 points total** over the base policy, of
  which **only +11.6 is attributable to the synthetic rollouts**; the other ~27.6
  came from real-world iteration.

- **Wh0** (arXiv 2606.22136, 2026-06-20) — the strongest 2026 result: 400 real
  teleop trajectories + 50,000 generated egocentric videos → **38.9%** on 18
  dexterous tasks, against π0.5 at 7.78%. Note, though, that the *real*
  egocentric video baseline already reached 21.4%; generation added the rest.
  Generation cost 5.44 GPU-hours per 1,000 videos.

**Genie 3** (DeepMind, announced 2025-08-05) — 720p/24 fps, consistent for
several minutes, **limited research preview** to a small cohort. Robotics
evidence is an agent acting inside generated environments — a demo, not a metric.
**Zero real-robot policy results.** You cannot get access and there is nothing to
measure.

**NVIDIA Cosmos** — Predict 2.5 / Transfer 2.5 / Reason 2 released 2025-10-06;
code Apache-2.0, **weights under the NVIDIA Open Model License**, Ampere or
newer required, **no published VRAM table**. The Predict-2.5 README now says the
model is **"no longer under active development."** Cosmos 3 (arXiv 2606.02800,
2026-06-01) claims *"best policy model by RoboArena ranking at time of technical
report"* — which is a **leaderboard placement, not a measured gain from generated
data**; the report contains no real-robot success rates of its own.

**Training your own video world model** — DreamGen's 81,000 GPU-hours. You will
not out-train NVIDIA on one card, and the marginal real gain is single digits.

**Generative inpainting augmentation (ROSIE-style)** — lost head-to-head to a
green screen, 75% vs 91%, with plain computer-vision augmentation at 70%. You
would be paying GPU time to do worse.

**Object texture and appearance diversity** — MimicLabs, explicitly: *"object
texture alignments have limited impact in retrieval."* Spend that effort on
camera pose and spatial arrangement.

**Discrete action tokens** and **chain-of-thought conditioning** — Toyota
Research Institute's systematic study (arXiv 2602.01067, **2026-02-01**;
**89 policies, 58,000 sim rollouts, 2,835 real rollouts**) found discrete robot
action tokens gave *"no significant benefits"* and chain-of-thought conditioning
*"did not improve performance."* Its winners were **vision-language data** and
**cross-embodiment robot data**. It also found robot-only training **degraded
the backbone's visio-linguistic ability**, which co-training restored.

**Chasing precision with synthetic data** — Curse of Precision
(arXiv 2607.23108, 2026-07-25): required demonstrations grow **super-
exponentially** as target precision approaches a limit set by *sensors, expert
quality and hardware*. **Adding a wrist camera lowers that limit. Adding 10,000
synthetic demonstrations does not.**

---

## 4. What this means for a solo builder with one 24 GB GPU

In order, and the order is the point:

1. **Fix the hardware and the camera before touching data.** ArmnetBench and the
   SO-101 benchmark both find execution failures — grasp instability, repetition
   loops — dominate precision failures, and Curse of Precision says the ceiling
   is a hardware property. Add a wrist camera.
2. **Build the evaluation harness.** ≥50 rollouts per condition, or you cannot
   tell a 10-point gain from noise and will chase it for months.
3. **Collect ~50 demos per task, varying camera pose and object placement
   deliberately, and shoot on a green screen.** All three cost nothing extra and
   buy the two highest-ranked axes at once.
4. **Fine-tune a pretrained checkpoint. Do not train from scratch.**
5. **Curate before collecting more.** PSD ranker first, Demo-SCORE once you have
   rollouts. This is where the marginal hour belongs, over any generative method.
6. **Only then, spatial augmentation** — DemoGen if you have depth, RoboSplat if
   you can build a multi-view rig. Both fit in 24 GB.
7. **Sim co-training later**, and only with the mandatory ratio sweep from
   [23](23-simulation-and-real2sim.md) §4.
8. **Skip world-model generation entirely at this scale.**

---

## 5. Sources

MimicGen arXiv 2310.17596 · MimicLabs arXiv 2506.13536 ·
DemoGen arXiv 2502.16932, <https://demo-generation.github.io/> ·
DexMimicGen arXiv 2410.24185 · SkillMimicGen arXiv 2410.18907 ·
SoftMimicGen arXiv 2603.25725 · MoMaGen arXiv 2510.18316 ·
RoboCasa <https://robocasa.ai/>, arXiv 2406.02523 · RoboSplat arXiv 2504.13175 ·
RoVi-Aug arXiv 2409.03403 · GreenAug arXiv 2407.07868 · ROSIE arXiv 2302.11550 ·
DreamGen arXiv 2505.12705 · Cosmos 1 arXiv 2501.03575, Cosmos 3 arXiv 2606.02800,
<https://github.com/nvidia-cosmos/cosmos-predict2.5> ·
Genie 3 <https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/> ·
Wh0 arXiv 2606.22136 · VLAW arXiv 2602.12063 · Veo-Act arXiv 2604.04502 ·
Sim-and-real co-training arXiv 2503.24361 · TRI co-training study arXiv 2602.01067 ·
Demo-SCORE arXiv 2503.03707 · CUPID arXiv 2506.19121 · ATHENA arXiv 2606.16208 ·
PSD metric arXiv 2605.01544 · SmolVLA arXiv 2506.01844 ·
Curse of Precision arXiv 2607.23108 · Shortcut learning arXiv 2508.06426
