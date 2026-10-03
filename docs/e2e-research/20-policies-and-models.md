# Policies and models: what to actually train

Research date: **2026-08-08**. Question: for a low-cost SO-101-class arm on a
wheeled base, trainable and servable on **one 24 GB consumer GPU**, what is the
fastest path to a policy good enough to ship?

Re-swept **2026-08-15**. The search quota was again exhausted before the sweep
began, so discovery ran on the arXiv/HF APIs and direct primary-source fetches —
good coverage of arXiv- and LeRobot-hosted releases, under-sampled for
blog-only vendor news. Edits below carry that date.

> **TL;DR.** Train **ACT first for one week** — not to ship it, but because it
> trains in 30–60 minutes and therefore tests your *data pipeline* rather than
> your model. Then move to **π0.5**, which independent third-party benchmarks on
> this exact hardware put at ~2.5× ACT. Use **MolmoAct2** instead if π0.5's LoRA
> will not fit in 24 GB. Flow matching won; autoregressive action tokens cost
> latency. And **build your own evaluation harness before your second model**,
> because published rankings for the same model differ by 5×.

---

## 1. The number that should change your mind

**ArmnetBench v0.1** (arXiv 2607.24481, **2026-07-27**) is the first
third-party benchmark run on a *fleet* of SO-101 cells: 7 policies × 12 tasks,
single-arm and bimanual, **each policy trained or fine-tuned on exactly 50
demonstrations per task**, **2,518 policy rollouts + 600 reference demos**,
every episode three-way labelled (successful / suboptimal / failure). Hardware
cost per cell: $359 single-arm, $477 bimanual. Re-verified **2026-08-15**:
still v0.1 (single arXiv version, v0.1 datasets only on HF), and no
third-party SO-101 benchmark published since contradicts the ordering below.

Strict success rate across all 12 tasks:

| Policy | Success |
|---|---:|
| **π0.5** | **47.6%** |
| π0 | 35.1% |
| GR00T N1.7 | 29.4% |
| Diffusion Policy (from scratch) | 26.7% |
| **ACT (from scratch)** | **19.2%** |
| MolmoAct 2 | 18.9% |
| SmolVLA | 15.0% |

Corroborated independently at 100 demos/task (arXiv 2606.08881,
**2026-06-07**; its v2 of 2026-06-11 leaves every number unchanged, 4 tasks):
π0.5 **56.25%**, Wall-X 51.25%, ACT 33.75%,
SmolVLA 32.5%. That paper also reports **recovery rate after a failed grasp** —
π0.5 **30.77%**, Wall-X 20.51%, ACT **6.45%**, SmolVLA **3.23%** — and
concludes that *"execution-related failures (grasp instability, repetition
loops, state mismatches) occur substantially more frequently than precision
misalignment."*

Read that last sentence twice. **On cheap arms the bottleneck is low-level
robustness, not data volume.**

### And the number that should make you distrust all of the above

SmolVLA scores **78.3%** in its own paper (real SO-100, multi-task) and **15.0%**
in ArmnetBench. Same model family, same class of hardware.

| Source | SmolVLA | ACT |
|---|---:|---:|
| SmolVLA paper (arXiv 2506.01844, own eval) | **78.3%** | 48.3% |
| arXiv 2606.08881 (third-party, 100 demos) | 32.5% | 33.75% |
| ArmnetBench v0.1 (third-party, 50 demos) | **15.0%** | 19.2% |

Neither third-party benchmark reports confidence intervals. (⚠️ Narrowed
2026-08-19 — "nobody in this field does" was falsified: NVIDIA RoboLab
publishes Clopper-Pearson intervals with worked numbers, and PhAIL's paper
does Kaplan-Meier with clustered bootstrap. What survives: **no leaderboard
publishes them, and they are in no shipping framework** — zero hits for
`clopper`/`binom`/`confidence_interval` in Isaac Lab-Arena's `main`.)

At N = 20 rollouts and a true success rate near 0.5, one standard
error is roughly **11 percentage points** — so any single-task difference under
about 15 points is indistinguishable from noise, including your own.

**Conclusion: author-reported rankings do not transfer. Build your own
evaluation harness before you pick your second model.** That is not a
counsel of perfection; it is the only way any of these numbers become
actionable.

A live test of that rule arrived while this doc was being re-swept: **G0.5**
(arXiv 2608.11739, **2026-08-12**), a single-stream *autoregressive* VLA,
reports **76.7% vs π0.5's 53.3%** after real-robot fine-tuning — on R1lite/
R1pro hardware, not SO-101, author-reported, with no weights or code found.
Exactly the class of claim this section says to watch and not act on.

---

## 2. The landscape

Everything below verified against LeRobot documentation, HuggingFace model
cards, official repositories and arXiv, on **2026-08-08**.

| Model | Params | Licence | Train VRAM | SO-101 checkpoint |
|---|---:|---|---|---|
| ACT | 80M | Apache-2.0 | 2–6 GB | n/a (trained from scratch) |
| Diffusion Policy | ~150M | Apache-2.0 | 8–14 GB | n/a |
| **SmolVLA** | 450M | Apache-2.0 | 10–16 GB | ✅ pretrained on SO-100 community data |
| **π0 / π0-FAST** | 3B+ | Apache-2.0 code, **Gemma-term weights** | 24–40 GB | ❌ |
| **π0.5** | 3B+ | Apache-2.0 code, **Gemma-term weights** | **LoRA >22.5 GB** | ❌ no pretrained SO-101 checkpoint |
| **MolmoAct2** (2026-05-05) | ~4B | Apache-2.0 | **LoRA-VLM 20.2 GiB @ bs8** | ✅ `lerobot/MolmoAct2-SO100_101-LeRobot` |
| X-VLA | 0.9B | Apache-2.0 | 24–40 GB tier | ✅ `so101_bimanual` action mode |
| GR00T N1.7 | 3B | **NVIDIA Open Model License** | 40 GB+ | via `new_embodiment` |
| EO-1, EVO-1, WALL-OSS, Being-H0.5 | 1–4B | Apache-2.0 (mostly) | 24–40 GB | generic |
| **FastWAM** (LeRobot 2026-06-18) | ~5B (Wan2.2 backbone) | Apache-2.0 | no consumer figure — eval reproduced on an H20 140 GB | ❌ placeholder rollout command only |
| **LingBot-VA** (LeRobot 2026-06-05) | ~5B (Wan2.2 backbone) | Apache-2.0 | inference 18–24 GB; **full FT does not fit 24–32 GB, LoRA required** | ❌ fixed 30-dim EEF-pose layout — joint-space SO-101 data must be remapped |
| π*0.6 (2025-11-17), π0.7 (2026-04-16) | — | **Closed** | — | — |

Table footnotes from the 2026-08-15 re-sweep. The π licence cells are now
precise: openpi's *code* is Apache-2.0, but the *weights* inherit **Gemma
licence terms** via the PaliGemma backbone — surfaced by `lerobot/pi052_base`,
whose model card lists `license: gemma`. And `pi052_base` itself is **not a new
Physical Intelligence model**: it is π0.5 re-laid-out as an initialization
scaffold for a LeRobot "pi05.2"-style variant, without PI's additional
training. Fresh LeRobot-format conversions landed 2026-07/08 for `wall-oss-0.5`
(4B, LeRobot PR #4200), `hy_vla_*`, `being_h05_*` and `eo1-base` — none ships
an SO-101 checkpoint. FastWAM and LingBot-VA are world-action models; what they
mean for the world-model question is in §5.

### VRAM, from LeRobot's official Compute & Hardware Guide

Peak VRAM at batch size 8 with AdamW:

| Group | Policies | Peak VRAM |
|---|---|---|
| Light behaviour cloning | `act`, `vqbet`, `tdmpc` | **~2–6 GB** |
| Diffusion | `diffusion`, `multi_task_dit` | ~8–14 GB |
| Small VLA | `smolvla` | ~10–16 GB |
| Large VLA | `pi0`, `pi0_fast`, `pi05`, `xvla`, `wall_x` | ~24–40 GB — *"24 GB tight at BS 1"* |
| Multimodal | `groot`, `eo1` | ~24–40 GB |

Wall-clock, single 24 GB card, 5 epochs on ~50 episodes (~45k frames):

| Policy | Time |
|---|---:|
| **ACT** | **~30–60 min** |
| Diffusion | 2–4 h |
| SmolVLA | ~3–6 h |

**MolmoAct2 is the one model that breaks the "frontier VLA needs 40 GB" rule**,
and it publishes measured numbers (H100, bf16, 2 RGB cameras, chunk size 10,
gradient checkpointing on):

| Mode | bs=8 | bs=16 | bs=32 |
|---|---:|---:|---:|
| Inference (bs=1, CUDA graph) | **12.1 GiB** | — | — |
| Fine-tune, action-expert only | **16.5 GiB** | 18.3 GiB | 21.4 GiB |
| Fine-tune, LoRA on the VLM | **20.2 GiB** | 26.8 GiB | 41.3 GiB |
| Fine-tune, full model | 48.3 GiB | 49.8 GiB | 60.1 GiB |

⚠️ **SO-101 gotcha, documented:** the MolmoAct2 SO-100/101 checkpoint was
trained under the pre-LeRobot-0.5.0 joint calibration convention. It requires
`joint_signs: [1,-1,1,1,1,1]` and `joint_offsets: [0,90,90,0,0,0]`. The
converted `lerobot/…-LeRobot` checkpoint already carries the fix. **Skip it and
the arm moves in the wrong direction.**

---

## 3. Action representation, and why it is a latency decision

The policy has to turn its internal state into joint targets. How it does that
sets the inference cost.

| Representation | Used by | Forward passes | Character |
|---|---|---:|---|
| Autoregressive tokens (FAST) | π0-FAST, RDT2-VQ | many — **RDT2-VQ needs 27** | Trains ~5× faster, worst latency |
| Diffusion | Diffusion Policy | 10–100 denoise steps | Slow, mitigated by chunking |
| **Flow matching** | π0, π0.5, SmolVLA, X-VLA, GR00T, EO-1, EVO-1, MolmoAct2, WALL-OSS | **4–10** | **The 2026 consensus** |
| L1 regression + parallel decode | OpenVLA-OFT | 1 | 26× faster generation than base OpenVLA |

*Flow matching*: instead of denoising in many small steps, the model learns a
direct velocity field from noise to the action, so a handful of integration
steps suffices. LeRobot's default is 10; VLA-JEPA uses 4, MolmoAct2 8.

**Action chunking is universal and non-negotiable** — the policy emits a
sequence of future actions per inference and the robot plays them out while the
next inference runs. Chunk lengths: ACT ~100, π0/π0.5 50, X-VLA 32, GR00T N1.7
40 (up from 16 in N1.5), MolmoAct2 10.

**Practical anchor:** an RTX 4090 running a π0-class model with 10 flow-matching
steps and a 50-action chunk measures **~80–90 ms**. At a 50 Hz control rate
(20 ms per action) that is an inference delay of 4–5 steps — comfortably inside
the budget in [19-the-system.md](19-the-system.md) §2.

**Pick a flow-matching policy.** The FAST tokenizer's training speedup is not
worth 27 forward passes on a latency-sensitive product.

**Contested as of 2026-08-15 — on the quality half only.** G0.5 (arXiv
2608.11739, §1) claims a single autoregressive stream now *beats* π0.5 on real
robots. Author-reported and weightless, so nothing to act on — but if it
replicates, "flow matching won" narrows to "flow matching won among 2026's
open models." The latency objection stands unrebutted either way; G0.5's
abstract does not address it. (Adjacent: πR², arXiv 2607.26055, 2026-07-28,
pushes the chunking lever instead — reactive ~25 Hz replanning on GR00T-N1.7,
claimed +30% real-world; verified at the metadata level only.)

---

## 4. Reinforcement learning and interactive correction

This is where the field's best-evidenced real-robot result lives.

| Method | Date | Evidence | Result |
|---|---|---|---|
| **HIL-SERL** (arXiv 2410.21845) | 2024-10-29 | ✅ real robot | *"near-perfect success rates and fast cycle times within just 1 to 2.5 hours of training"*; **2× success rate, 1.8× faster execution** vs imitation learning |
| **RaC** (arXiv 2509.07953) | 2025-09-09 | ✅ real bimanual | *"outperforms the prior state-of-the-art using **10× less data collection time and samples**"*; performance **scales linearly** in number of recovery maneuvers |
| **RECAP / π\*0.6** (arXiv 2511.14759) | 2025-11-18 | ✅ real, charts only | *"more than doubles task throughput and roughly halves the task failure rate"* on the hardest tasks |
| UniIntervene (arXiv 2606.12372) | 2026-06-10 | ✅ real | **+8.6% success while reducing human interventions by 57%** |
| RIPT-VLA (arXiv 2505.17016) | 2025-05 | ❌ **LIBERO sim only** | 1 demo: 4% → 97% in 15 iterations |
| Q-chunking (arXiv 2507.07969) | 2025-07 | ❌ sim benchmarks | no real numbers |
| EvoHIL (arXiv 2608.03872) | 2026-08-04 | ✅ real — Franka FR3 + **SO-101**, six tasks under lighting shifts | self-evolving reward + flow-matched HIL RL; claims better success/smoothness than HIL and imitation baselines, **no quantitative numbers in the abstract**; code status unconfirmed |
| **RLT — "RL Token"** (Physical Intelligence, pi.website/research/rlt) | 2026 (read 2026-08-20 in full text; no arXiv ID sighted) | ✅ real — 4 sub-millimetre tasks on **π0.6** (screw, zip tie, Ethernet, charger) | Frozen VLA exposes a compact **"RL token"** readout (encoder-decoder bottleneck over the VLA's final-layer embeddings); a small TD3-style actor-critic learns over **action chunks (C=10)**, BC-regularised toward the VLA's own reference chunk with reference-dropout, human interventions + sparse binary human success labels, update-to-data ratio 5. **Screw success 20% → 65%; up to 3× critical-phase speedup; on Ethernet the RL policy's median (66 steps) beats every expert teleop demo (median 146)** — in 15 min–5 h of robot data. Ablations are the mechanism evidence: single-step baselines (HIL-SERL, PLD) fail outright at 50 Hz sparse reward — **chunk-level credit assignment is why corrections work at VLA control rates**. Author-reported, PI's own model, no third party |

The RLT row updates two standing claims (2026-08-20). The "beyond the
demonstrator ceiling" argument now has its cleanest exhibit — half the RL
episodes are faster than **all** of the teleoperated demonstrations, via an
emergent press-and-wiggle insertion strategy present nowhere in the demo
data. And §5's "π*0.6/π0.7 — no paper, no N" complaint narrows again: π0.6
now carries a model card, RECAP, and RLT with real Ns; **the weights remain
closed**, which is the half that matters for anyone else's pipeline.

**Honest summary of what RL adds over imitation:**

- On a narrow, short-horizon, contact-rich skill: **~2× success rate in 1–2.5
  hours.** Real, published, reproducible-in-principle.
- On a general product policy: **no open evidence.** The only such result is
  RECAP, which is closed and chart-only.
- RIPT-VLA's spectacular 4% → 97% is **simulation**. Do not budget against it.

### The under-hyped winner: DAgger, which is not RL at all

LeRobot ships `lerobot-rollout --strategy.type=dagger`, and its documentation
states outright that this implements **RaC's recovery/correction decomposition**.

The protocol: watch the policy run → pause (the robot holds position, the leader
arm servos to match it) → take over (leader torque off, recover, then correct)
→ hand back to the policy → repeat *within the same episode*, with no reset.
Autonomous and human segments land in one continuous trajectory; you fine-tune
on demonstrations and corrections combined. Frames during the pause are not
recorded.

Requirements: a teleoperator with **active motors** — `so_leader`,
`bi_so_leader`, `bi_openarm_mini`. An SO-101 leader arm qualifies. A foot pedal
is officially supported on Linux (PCsensor FootSwitch). For slow VLAs
(π0, π0.5, SmolVLA) you must add `--inference.type=rtc`.

**This has the best evidence-to-effort ratio in the entire report.** It requires
no reinforcement learning, no reward classifier, no new model, and it targets
exactly the failures your policy actually has.

---

## 5. Explicit rejections, with reasons

**π\*0.6 (2025-11-17) and π0.7 (2026-04-16) — closed, and the evidence is bar
charts.** π\*0.6 reports espresso throughput "more than doubled", box assembly
~7 → ~14 successes/hour, laundry ~40 → ~60/hour. π0.7 reports laundry 95–100%,
espresso 90–95%. **No paper, no N, no evaluation protocol, no weights.** Stop
reading those blog posts; you cannot use the models and you cannot check the
claims. Re-checked 2026-08-15: openpi still ships only π0, π0-FAST and π0.5;
PI's newest model post is still π0.7.

**LIBERO as a model-selection signal — saturated.** MolmoAct2 98.25%, π0.5
97.5%, EVO-1 96.65%, GR00T N1.7 96.5%, VLA-JEPA 96.5%, X-VLA 93%. **Zero
discriminative power.** LIBERO is a lifelong-learning benchmark whose own site
states no real-robot validation. Select on the SO-101 real-robot evidence, the
VRAM table, and the licence.

**JEPA and world models as planners — not yet, and the field has conceded it.**

| System | Real robot? | Numbers |
|---|---|---|
| V-JEPA 2-AC (2025-06) | ✅ Franka | reach 100%, **cup grasp 60%**, pick-and-place 80%; 62 h of robot data; planning latency **never published** |
| V-JEPA 2.1 (2026-03-16) | — | representation update; **no new action-conditioned checkpoint** |
| V-JEPA 3 | — | **does not exist** (re-checked 2026-08-15 across every arXiv "V-JEPA" paper) |
| DINO-WM (2024-11) | ❌ **sim only**, 6 environments | — |
| VLA-JEPA (2026-02-10) | LIBERO 96.5% + real claims | Apache-2.0 |

A 60% cup grasp with a model-predictive-control loop on a Franka is not
competitive with a 50-episode ACT policy. But the decisive detail is what the
2026 papers that *work* actually do: **VLA-JEPA uses the world model as an
auxiliary training loss only.** From LeRobot's own documentation:
*"Inference: Only Qwen + the action head are used. **The world model is not
needed at inference time.**"* and *"`enable_world_model=False`… **This is
sufficient for good action performance.**"*

**Narrowed 2026-08-15: "not needed at inference" is no longer universal, even
inside LeRobot.** FastWAM (paper arXiv 2603.16666; LeRobot 2026-06-18) keeps
video modelling during training and predicts actions directly at inference —
the thesis holds there, and its LIBERO 96.4% is one more saturation datapoint.
But **LingBot-VA** (LeRobot 2026-06-05, Apache-2.0, ~5B) *does* roll imagined
video latents at inference, closed-loop with a KV cache (~20 video + ~50 action
denoise steps). Its evidence is **LIBERO/RoboTwin simulation only**, its action
space is a fixed 30-dim EEF-pose layout, and full fine-tuning does not fit
24–32 GB. The advice below is unchanged; the generalisation now carries one
sim-only counter-example.

JEPA survived 2025–26 as a **representation-learning trick**, not as a
controller. Do not build model-predictive control over a learned world model for
a product.

**GR00T N1.7 — licence and churn.** NVIDIA states 40 GB+ for fine-tuning, and
the weights are under the **NVIDIA Open Model License, not Apache-2.0** — read
it before commercial deployment. Also, N1.5 support was *removed* from LeRobot
as a breaking change, forcing version pins. That churn is a real cost for a solo
builder.

**OpenVLA 7B — superseded.** The durable contribution was the OFT *recipe*
(action chunking + continuous actions + parallel decoding + L1 loss), and every
2026 model has absorbed it.

**RDT-2 — no quantitative success rates published at all**, and it requires
UMI-specific hardware (a particular camera, gripper, flange and printed
bracket). Not applicable to SO-101.

**Gemini Robotics — now version 2 (checked 2026-08-15), access model
unchanged.** The VLA remains trusted-tester-only (100+ testers, waitlist);
**Gemini Robotics-ER 2 *is* API-accessible** via AI Studio, and an On-Device 2
variant now exists. DeepMind's page shows no release date, and the announcement
could not be dated without search. The role is what it was at 1.5: a **planner
and visual-question-answering model, not a controller.** Correct use:
high-level task decomposition sitting *above* your policy. Wrong use: expecting
it to move a servo.

**Diffusion Policy as the workhorse — the awkward middle.** 2–4 hours of
training against ACT's 30–60 minutes, for no reliable gain at 50-episode scale.

---

## 6. The ladder

The ladder for a solo builder: full-time, one 24 GB GPU, six months, SO-101 on a wheeled base.

**Rung 0 — data infrastructure, not models.** Teleop, record, replay. Cameras
placed, lighting fixed, resets rehearsed. Exit when 50 clean episodes take under
two hours. Details in [21-data-collection.md](21-data-collection.md).

**Rung 1 — ACT.** 30–60 minutes of training. **Its purpose is to test the data
pipeline, not to ship.** If ACT cannot reach ~70% on your task with 50 episodes,
your cameras, task definition or reset procedure are wrong, and no VLA will
rescue them. Exit at ≥70% single-task; **do not proceed until you hit it**.

**Rung 2 — π0.5 (LoRA).** The third-party benchmarks put it at ~2.5× ACT on this
exact hardware. Once fine-tuning, note **FiberTune** (arXiv 2606.08653,
2026-06-07): preserving the teacher's visual residuals during action-supervised
fine-tuning reports **72.7% → 78.1%** on a physical SO-101 pick-place with zero
inference overhead — code release unconfirmed as of 2026-08-15.
⚠️ **LoRA needs >22.5 GB — a 24 GB card is right at the edge.**
Use bf16 and gradient checkpointing, or rent by the hour for these runs.
**If it will not fit, use MolmoAct2**: LoRA-VLM at 20.2 GiB has genuine headroom,
inference is 12.1 GiB, and it ships an actual SO-100/101 checkpoint. Its own
guidance matches this regime — *"for real-world dataset with less than 200
demonstrations, global batch 16–32… `enable_lora_vlm=true` or
`train_action_expert_only=true` is a practical choice."*

**Rung 3 — corrections, overlapping Rung 2.** `--strategy.type=dagger` with
`--inference.type=rtc`. This is the highest-return rung and almost nobody does
it: **RaC's 10× data efficiency, and it is what RECAP does at scale.** Exit when
you have gone from ~70% to 85–90% *without collecting a single new
demonstration*.

**Rung 4 — HIL-SERL, on exactly one sub-skill.** Not the whole product. One
5–10 second contact-rich step (insertion, precise grasp, connector mating) where
imitation has plateaued. Payoff: near-perfect in 1–2.5 hours. Cost: real setup —
workspace bounds, region-of-interest cropping to 128×128, a ResNet-10 reward
classifier, separate actor and learner processes. LeRobot's implementation is
SO-100/101-native and documents its constraints: **fps = 10**, tasks completable
in **5–10 seconds**, avoid long horizons.

**Serving, from Rung 2 onward and never changing:** the 24 GB GPU on the LAN
running LeRobot's `PolicyServer`, gRPC to a `RobotClient` on each robot, with
RTC. SmolVLA needs only ~2 GB at inference, π0 14 GB, MolmoAct2 12.1 GiB — one
card serves three robots. See
[25-deployment-and-fleet-ops.md](25-deployment-and-fleet-ops.md).

New measured edge datapoint (arXiv 2608.03938, **2026-08-04**): INT8-TensorRT
ACT runs a **bimanual SO-101 at 19/20 success with 12.65 ms inference on an
8 GB Jetson Orin Nano Super** — 9.0× vs the 114 ms full-precision baseline,
with only the ResNet18 backbone quantized (0 of 145 transformer layers), three
cameras, trained offline on an RTX 3070. The first rung runs at the edge on
the cheapest Jetson; weigh it against that device's price in
[24-compute-and-hardware.md](24-compute-and-hardware.md).

---

## 7. Episodes needed → success rate: all the real evidence

| Source | Episodes | Result |
|---|---|---|
| SmolVLA official docs, SO-100 pick-place | **25** | *"not enough, leading to bad performance"* |
| SmolVLA official docs | **50** (5 positions × 10) | works — the recommended recipe |
| LeRobot imitation-learning tutorial | **≥50** (10 per location) | *"high success rate"* |
| ACT docs | ~50 | *"often achieves high success rates"* |
| **ArmnetBench v0.1** | **50/task** | π0.5 47.6% … ACT 19.2% (table §1) |
| arXiv 2606.08881 | 100/task | π0.5 56.25%, ACT 33.75% |
| MolmoAct2 official guidance | **<200** | use LoRA or action-expert-only, global batch 16–32 |
| HIL-SERL | ~15–20 demos + **1–2.5 h** of corrections | near-perfect, 2× vs imitation |
| RaC | **10× less** collection time than scaling demos | beats prior SOTA |

**The rule that falls out: 50 episodes per task variation is the floor, 25 is
documented to fail, and past roughly 200 you should be collecting corrections
rather than demonstrations.**

---

## 8. Sources

LeRobot docs <https://huggingface.co/docs/lerobot/index> ·
Compute & Hardware Guide <https://huggingface.co/docs/lerobot/hardware_guide> ·
ACT <https://huggingface.co/docs/lerobot/main/en/act> ·
SmolVLA <https://huggingface.co/docs/lerobot/main/en/smolvla>, paper arXiv 2506.01844 ·
π0.5 <https://huggingface.co/docs/lerobot/main/en/pi05> ·
openpi <https://github.com/Physical-Intelligence/openpi> ·
MolmoAct2 <https://huggingface.co/docs/lerobot/main/en/molmoact2>, blog 2026-05-05 ·
X-VLA <https://huggingface.co/docs/lerobot/xvla> ·
GR00T <https://huggingface.co/docs/lerobot/main/en/groot> ·
VLA-JEPA <https://huggingface.co/docs/lerobot/main/en/vla_jepa> ·
HIL-SERL <https://huggingface.co/docs/lerobot/hilserl>, arXiv 2410.21845 ·
DAgger / HIL data collection <https://huggingface.co/docs/lerobot/main/en/hil_data_collection> ·
RaC arXiv 2509.07953 · RECAP arXiv 2511.14759 · RIPT-VLA arXiv 2505.17016 ·
ArmnetBench v0.1 arXiv 2607.24481 · SO-101 VLA benchmark arXiv 2606.08881 ·
V-JEPA 2 <https://github.com/facebookresearch/vjepa2> · DINO-WM arXiv 2411.04983 ·
Real-Time Chunking <https://www.pi.website/research/real_time_chunking> ·
G0.5 arXiv 2608.11739 · FastWAM <https://huggingface.co/docs/lerobot/main/en/fastwam>, arXiv 2603.16666 ·
LingBot-VA <https://huggingface.co/docs/lerobot/main/en/lingbot_va> ·
quantized ACT on Jetson arXiv 2608.03938 · EvoHIL arXiv 2608.03872 ·
FiberTune arXiv 2606.08653 · πR² arXiv 2607.26055 ·
Gemini Robotics <https://deepmind.google/models/gemini-robotics/> ·
pi052_base <https://huggingface.co/lerobot/pi052_base>
