# Auto-selection and data mixtures: prior art for the recipe engine

*Fourth pass, 2026-08-25. Researched by one agent against primary
sources (arXiv, GitHub, vendor pages), per the one-agent-per-field
discipline. Tests the verdict of [docs/30-the-full-loop.md](../30-the-full-loop.md)
§3.2: is a recipe engine — walking (policy family × data mixture ×
hyperparameters), judged only by statistically-gated evaluation —
novel, partially claimed, or crowded? The agent's report follows
verbatim.*

---

Legend: **VERIFIED** = abstract/page fetched directly. **CLAIMED** = from search snippets only, not independently confirmed.

## (a) Dated findings

### Q1 — AutoML / automatic policy-architecture selection

**Finding: no published system automatically selects a policy family/architecture for a given task+embodiment.** Four separate query formulations (AutoML robot learning, NAS-for-VLA, policy architecture search, LLM-agent recipe selection) returned nothing that does this end-to-end. What exists is adjacent:

- **robomimic study** (Aug 2021, [arXiv:2108.03298](https://arxiv.org/abs/2108.03298), [study page](https://robomimic.github.io/study/)) — the canonical large design-space sweep for imitation learning; also the canonical evidence that **validation loss does not predict rollout success** (best-val policy 50–100% worse than best policy; more val data doesn't help). VERIFIED via study page. This is the strongest published argument that recipe judgment *must* be rollout-eval-gated — i.e., the venture's premise.
- **AutoRL survey** (JAIR 2022, [paper](https://www.jair.org/index.php/jair/article/download/13596/26808/30767)) — AutoML for RL (hyperparameters + architectures), never extended to imitation/VLA. CLAIMED.
- **AutoLLMResearch** (May 2026, [arXiv:2605.11518](https://arxiv.org/abs/2605.11518)) — trains research agents to automate *LLM* experiment configuration with a multi-fidelity "learn from cheap, optimize expensive" gym. The recipe-engine concept exists in LLM land, not robotics. CLAIMED.
- **StarVLA** (Apr 2026, [arXiv:2604.05014](https://arxiv.org/pdf/2604.05014)) — "Lego-like" modular VLA codebase; enables walking the architecture space, doesn't automate the choice. CLAIMED.
- **TRI, "A Careful Examination of Large Behavior Models"** (Jul 2025, [arXiv:2507.05331](https://arxiv.org/abs/2507.05331)) — closest to principled architecture/recipe *comparison*: blind randomized trials, statistical CIs, sim+real; finds multitask pretraining predictably improves with scale/diversity. VERIFIED (fetched). Manual comparison, not automated selection.

### Q2 — Data-mixture optimization for robot policies

- **Re-Mix** (Aug 2024, [arXiv:2408.14037](https://arxiv.org/abs/2408.14037), Hejna et al.) — DoReMi-style distributionally robust optimization of domain weights on Open X-Embodiment using a small proxy policy; learned weights beat uniform by 38% and human-curated RT-X weights by 32%. **The** nearest neighbor for the mixture axis. VERIFIED.
- **DoReMi** (May 2023, [arXiv:2305.10429](https://arxiv.org/abs/2305.10429)) — the LLM original: 280M proxy sets weights for an 8B model (30× transfer), 2.6× fewer steps. The cost-aware trick Re-Mix imports. VERIFIED via snippets.
- **CUPID** (Jun 2025, CoRL 2025, [arXiv:2506.19121](https://arxiv.org/abs/2506.19121), [code](https://github.com/agiachris/cupid), Stanford/TRI) — influence functions over *rollout performance* to filter/subselect demos; <33% of curated data yields SOTA diffusion policies on RoboMimic; works on hardware and for generalist post-training. VERIFIED.
- **Robot Data Curation with Mutual Information Estimators (DemInf)** (Feb 2025, RSS 2025, [arXiv:2502.08623](https://arxiv.org/abs/2502.08623), Hejna et al., DeepMind/Stanford) — kNN mutual-information demo-quality scoring; +5–10% RoboMimic, gains on real ALOHA/Franka. VERIFIED via snippets.
- **SCIZOR** (May 2025, [arXiv:2505.22626](https://arxiv.org/abs/2505.22626)) — self-supervised curation removing suboptimal/redundant samples at scale. CLAIMED.
- **MimicLabs: "What Matters in Learning from Large-Scale Datasets"** (ICLR 2025, [arXiv:2506.13536](https://arxiv.org/abs/2506.13536), [site](https://robo-mimiclabs.github.io/)) — which diversity dimensions matter (camera pose + spatial arrangement dominate); retrieval strategies on DROID beat standard training by up to 70%. VERIFIED via snippets.
- **Data Scaling Laws in IL** (Oct 2024, ICLR 2025 oral, [arXiv:2410.18647](https://arxiv.org/abs/2410.18647), Lin et al.) — generalization scales with environment/object diversity, not raw demo count; demo count saturates. VERIFIED.
- **π0.5 mixture practice** (Apr 2025, [arXiv:2504.16054](https://arxiv.org/pdf/2504.16054), [blog](https://www.pi.website/blog/pi05)) — Physical Intelligence's mixture (MM/ME/CE/HL/web; 97.6% of data not from the target robot) is **hand-designed and ablated, not optimized** — no DoReMi-style weighting reported. VERIFIED via paper/blog snippets. Nothing found from PI/DeepMind on *learned* mixture weighting in 2025–26.
- **Shortcut Learning in Generalist Robot Policies** (Aug 2025, [arXiv:2508.06426](https://arxiv.org/html/2508.06426)) — dataset fragmentation/diversity failure modes; a reason mixtures need per-task judgment. CLAIMED.

### Q3 — Real+sim co-training since early 2026

- **"A Mechanistic Analysis of Sim-and-Real Co-Training"** (Apr 15 2026, [arXiv:2604.13645](https://arxiv.org/abs/2604.13645), Lei, Liu, Maddukuri, Jiang, Zhu — UT Austin/NVIDIA lineage of the original co-training recipe) — first *explanatory* account: performance is governed by structured representation alignment (primary) and importance reweighting (secondary); a **band of "balanced mixing ratios"** exists where alignment emerges implicitly, and domain discernibility must be preserved. Motivates a method that beats prior ratio-picking. VERIFIED (fetched). This directly reframes the "optimal ratio flips per task" landmine: the target is a detectable band, not a scalar.
- **RLinf-Co** (Feb 2026, [arXiv:2602.12628](https://arxiv.org/html/2602.12628v3)) — RL-based sim–real co-training for VLAs (interactive sim instead of static sim demos, preserving real capabilities). CLAIMED.
- **Grounding Sim-to-Real Generalization for VLAs** (Mar 2026, [arXiv:2603.22876](https://arxiv.org/pdf/2603.22876)) — empirical sim-to-real study with VLAs. CLAIMED.
- Baseline for both: **Sim-and-Real Co-Training** (Mar 2025, [arXiv:2503.24361](https://arxiv.org/abs/2503.24361), [site](https://co-training.github.io/)) — already in the standing knowledge (docs/20).

### Q4 — Recipe/hyperparameter search at VLA scale

- **What labs actually report: hand-designed recipes + ablations. No lab reports Bayesian optimization or automated sweeps at VLA scale.** π0.5 ([2504.16054](https://arxiv.org/pdf/2504.16054)) ablates mixture components; TRI LBM ([2507.05331](https://arxiv.org/abs/2507.05331)) does rigorous *evaluation* of manually chosen recipes; GR00T/Gemini-class reports likewise. VERIFIED absence within sources checked (not exhaustive — tech reports often omit tuning detail).
- Cost-aware search prior art lives in proxies: **DoReMi/Re-Mix proxy-model transfer** (above), **AutoLLMResearch's** multi-fidelity cheap→expensive scheme ([2605.11518](https://arxiv.org/abs/2605.11518)), and scaling-law extrapolation (**Neural Scaling Laws in Robotics**, [arXiv:2405.14005](https://arxiv.org/pdf/2405.14005), CLAIMED).
- **"What Matters in Orchestrating Robot Policies"** (Jun 2026, [arXiv:2606.10267](https://arxiv.org/abs/2606.10267)) — systematic design-choice study for hierarchical VLA agents; manual, but maps a recipe space. CLAIMED.

### Q4/Q5 bridge — the evaluation layer (certificate prior art)

- **SIMPLER** (May 2024, CoRL 2024, [arXiv:2405.05941](https://arxiv.org/abs/2405.05941), [site](https://simpler-env.github.io/)) — sim eval as real proxy; introduced **MMRV (Mean Maximum Rank Violation)**, the standard sim↔real *rank-consistency* metric; 1500+ paired sim/real evals. VERIFIED. The "rank-correlation certificate" has its metric already defined here.
- **SureSim / "Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators"** (Oct 5 2025, [arXiv:2510.04354](https://arxiv.org/abs/2510.04354), Majumdar group, Princeton) — **prediction-powered inference**: a small number of *paired* real+sim evals rectifies sim bias, non-asymptotic CIs on mean performance; ~20–25% less hardware effort at equal bounds. VERIFIED (fetched). **Nearest single neighbor to the certificate concept.**
- **"A Practical Recipe Towards Improving Sim-and-Real Correlation for VLA Evaluation"** (Jun 9 2026, [arXiv:2606.10366](https://arxiv.org/pdf/2606.10366), Yang Gao group) — systematic study of when sim eval predicts real for VLAs: ranking consistency, correlation, failure-pattern alignment; guidance on post-training data volume vs alignment. VERIFIED (fetched).
- **Beyond Binary Success** (Mar 2026, [arXiv:2603.13616](https://arxiv.org/html/2603.13616)) — sample-efficient, statistically rigorous policy comparison via **safe anytime-valid inference (SAVI)**: sequential testing with early stopping. CLAIMED.
- **"Is Your Imitation Learning Policy Better than Mine?"** (Mar 2025, [arXiv:2503.10966](https://arxiv.org/pdf/2503.10966)) — statistical framework for policy comparison under low trial counts. CLAIMED.
- **"Robot Learning as an Empirical Science"** (Sep 2024, [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)) + **TRI blog on statistical A/B testing** ([Medium](https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d)) — best-practice canon: Wilson intervals, paired tests, CLD summaries. CLAIMED.
- **PhAIL** (May 2026, [arXiv:2605.29710](https://arxiv.org/pdf/2605.29710)) — audits standard practice: modal n=10–20 trials, **0 of 13 recent real-robot VLA papers report CIs or paired tests**. CLAIMED. The field's gap is the opening.
- **AutoEval** (Mar 2025, CoRL 2025, [arXiv:2503.24278](https://arxiv.org/abs/2503.24278), [code](https://github.com/zhouzypaul/auto_eval), Berkeley) — 24/7 autonomous real-robot eval, >99% less human effort, public queue. VERIFIED. **RoboArena** (Jun 2025, [arXiv:2506.18123](https://arxiv.org/pdf/2506.18123)) — distributed real-world pairwise A/B eval. CLAIMED. **RoboChallenge/Table30** (Oct 2025, [arXiv:2510.17950](https://arxiv.org/abs/2510.17950)) — hosted real-robot fleet, third-party benchmark (the SO-101-class benchmark family the π0.5-vs-ACT number comes from). VERIFIED via snippets.
- World-model-based eval wave (2026): **WorldEval** ([2505.19017](https://arxiv.org/pdf/2505.19017)), **GigaWorld-1** (Jul 2026, [2607.02642](https://arxiv.org/abs/2607.02642)), **PolaRiS** (Dec 2025, [2512.16881](https://arxiv.org/pdf/2512.16881)), **RobotArena∞** ([2510.23571](https://arxiv.org/html/2510.23571v1)), **REALM** ([2512.19562](https://arxiv.org/html/2512.19562v1)). All CLAIMED.

### Q5 — Products

- **NVIDIA Isaac Lab-Arena + LeRobot** (2026, [product page](https://developer.nvidia.com/isaac/lab-arena), [HF blog](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot), [NVIDIA blog](https://developer.nvidia.com/blog/how-to-evaluate-general-purpose-robot-policies-for-real-world-deployment/)) — open-source large-scale *policy evaluation* framework, GPU-parallel, "closed-loop workflows" language; RoboLab features productizing **Aug 2026**. Eval infrastructure only — no automatic recipe search, no statistical gating. VERIFIED product existence.
- **Lightwheel** ([site](https://www.lightwheel.ai/lightwheel-platform), [World Labs case study](https://www.worldlabs.ai/case-studies/2-lightwheel)) — sim2real eval-as-a-service ("evaluation has become the limiting factor, not training"); partners: DeepMind, Figure, AgiBot, ByteDance. Services company, not an automated recipe engine. VERIFIED pages.
- **phospho** ([site](https://phospho.ai/), [docs](https://docs.phospho.ai/learn/ai-models), [GitHub](https://github.com/phospho-app/phosphobot)) — one-click cloud training of ACT / smolVLA / π0.5 / GR00T-N1.5 on SO-100/101. **The user picks the model; nothing selects it, nothing is eval-gated.** VERIFIED pages. Closest commercial UX neighbor, missing exactly the differentiator.
- No product found — commercial or OSS — that claims automatic policy/recipe selection judged by statistically gated evaluation.

## (b) Verdict: partially claimed per axis; the closed loop is unclaimed

Every pillar exists in isolation, none are composed:

| Pillar | State | Nearest neighbor |
|---|---|---|
| Mixture optimization | Claimed, strong | Re-Mix (2408.14037) — but pretraining-oriented, DRO objective, not eval-gated per task |
| Statistical certificate (CIs, paired sim+real) | Claimed, fresh | SureSim (2510.04354); SAVI comparison (2603.13616); MMRV metric (SIMPLER) |
| Auto evaluation infra | Crowded (2025–26 wave) | AutoEval, Isaac Lab-Arena, RoboChallenge, Lightwheel, world-model evals |
| Architecture/family selection | **Gap — nothing found** | robomimic/TRI manual studies; AutoLLMResearch is the LLM-domain analogue |
| The integrated recipe engine | **Unclaimed** | AutoLLMResearch (2605.11518) proves the concept is "in the air" — for LLMs |

The "recipe engine judged by certificates" is **novel as a system**: no one walks (policy family × mixture × hyperparameters) with the winner chosen solely by statistically certified paired evaluation. The moat is thinnest on the eval layer (crowded and productizing fast — NVIDIA's Aug 2026 Arena roadmap) and thickest on the search/optimizer layer sitting on top of it. Two supporting facts strengthen the premise: validation loss provably fails as a judge (robomimic), and 0/13 recent VLA papers even report CIs (PhAIL) — the field is not doing this itself.

## (c) Two techniques to implement first

1. **Anytime-valid sequential testing over paired sim+real evals (the certificate, made cheap).** Combine SAVI-style sequential comparison ([2603.13616](https://arxiv.org/html/2603.13616): stop a recipe's eval the moment evidence suffices) with SureSim's prediction-powered inference ([2510.04354](https://arxiv.org/abs/2510.04354): few paired real trials debias mass sim trials, non-asymptotic CIs, 20–25% hardware savings), and report MMRV ([2405.05941](https://arxiv.org/abs/2405.05941)) as the sim↔real rank certificate. Evidence this is the binding constraint: eval cost is what every 2026 player (NVIDIA, Lightwheel) calls the bottleneck, and a recipe walk multiplies eval count — sequential+paired inference is the only published way to afford it.
2. **Proxy-policy DRO mixture reweighting, Re-Mix style, upgraded with the 2026 co-training mechanism.** Re-Mix ([2408.14037](https://arxiv.org/abs/2408.14037)) shows learned domain weights beat human weights by 32% using a *small proxy policy* (DoReMi's 30× cheap-to-expensive transfer, [2305.10429](https://arxiv.org/abs/2305.10429)) — this replaces brute-force grid search over real:sim ratios, the measured landmine. The mechanistic analysis ([2604.13645](https://arxiv.org/abs/2604.13645)) says the per-task optimum is a detectable *band* (representation-alignment probes can find it without full training runs), turning "mixture is a hypothesis" into a cheap pre-test inside the engine. CUPID ([2506.19121](https://arxiv.org/abs/2506.19121)) is the natural third step (within-mixture demo filtering: <33% of data, SOTA policy) once the mixture weights are set.

Caveats: search coverage was arXiv + vendor pages; closed-lab internals (PI, DeepMind, Figure) may run undisclosed sweep automation — absence of reporting is not absence of practice. All 2606–2607 arXiv IDs are June–July 2026 preprints, mostly unrefereed.
