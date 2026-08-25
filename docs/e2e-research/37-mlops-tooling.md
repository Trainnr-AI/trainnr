# MLOps for robot-learning training: the 2026-08-25 survey

*Fourth pass, 2026-08-25. Researched by one agent against primary
sources (repo code at HEAD, license texts, pricing pages), per the
one-agent-per-field discipline. Answers
[docs/30-the-full-loop.md](../30-the-full-loop.md) §3.4's open
decision (tracking, orchestration, what stays custom). Headline:
wandb is the universal tracker in robot learning; keep the DAG
custom, execute on SkyPilot; the registry stays ours. The agent's
report follows verbatim.*

---

Researched 2026-08-25. **VERIFIED** = I read the code, license text, or vendor page directly today. **CLAIMED** = from press/search results, not independently confirmed.

## (a) Dated findings

### Q1 — What robot-learning codebases actually integrate (all VERIFIED at repo HEAD, 2026-08-25)

| Codebase | Tracker | How |
|---|---|---|
| **LeRobot** (HEAD 2026-08-25) | **wandb only**, optional | `WandBConfig` (`enable=False` default, `mode: online/offline/disabled`) in lerobot's src/lerobot/configs/default.py; `WandBLogger` in src/lerobot/common/wandb_utils.py; dep `wandb>=0.24.0,<0.28.0` as a `[training]` extra. No tensorboard/mlflow/anything else. |
| **openpi** (Physical Intelligence, HEAD 2026-08-24) | **wandb**, core dep, default ON | `wandb>=0.19.1` in `pyproject.toml`; `wandb_enabled: bool = True` in src/openpi/training/config.py; used in both scripts/train.py (JAX) and scripts/train_pytorch.py. |
| **OpenVLA** | **wandb + JSONL fallback** | prismatic/training/metrics.py defines a `Tracker` protocol with exactly two impls: `JSONLinesTracker` and `WeightsBiasesTracker`. |
| **NVIDIA Isaac-GR00T** | **wandb only**, optional (default off) | `use_wandb: bool = False` in gr00t/configs/training/training_config.py; pinned `wandb==0.23.0`. |

- Conclusion: **wandb is the universal tracker in VLA/imitation learning; nothing else has any footprint** in these codebases. Adopting a non-wandb tracker means writing and maintaining your own logger glue against LeRobot.
- Orchestration data point (VERIFIED in code): LeRobot HEAD ships src/lerobot/jobs/hf.py — submits training to **Hugging Face Jobs** (`lerobot-train --job.target=<flavor>`, flavors t4→a100-large, billed per second). CLAIMED: shipped in LeRobot v0.6.0 ([blog](https://huggingface.co/blog/lerobot-release-v060)); GPU rates ~$0.40/hr (T4) to ~$23.50/hr (8xL40S).

### Q2 — Tracker landscape in 2026

- **W&B** — [pricing](https://wandb.ai/site/pricing) (VERIFIED 2026-08-25): Free = 5 seats / 5 GB storage; Pro "starts at $60/mo" (10 seats, 100 GB, teams <50 employees); **self-hosted free tier is personal-use only — "Corporate use is not allowed"**; corporate self-host = enterprise license. Owned by **CoreWeave since 2025-05-05** (VERIFIED via [CoreWeave press release](https://investors.coreweave.com/news/news-details/2025/CoreWeave-Completes-Acquisition-of-Weights--Biases/default.aspx)).
- **Neptune — dead.** `neptune.ai/pricing` 308-redirects to [openai.com/index/openai-to-acquire-neptune](https://openai.com/index/openai-to-acquire-neptune/) (VERIFIED redirect today). Announced 2025-12-03; CLAIMED: standalone service sunset by **2026-03-05** ([Bloomberg](https://www.bloomberg.com/news/articles/2025-12-03/openai-agrees-to-acquire-neptune-to-improve-ai-model-training), [EdTech Hub](https://www.edtechinnovationhub.com/news/openai-moves-to-acquire-neptuneai-as-experiment-tracking-platform-prepares-three-month-shutdown)). Object lesson: standalone trackers are acquisition targets.
- **MLflow** — Apache-2.0, active (pushed today, 27.7k stars; VERIFIED via GitHub API). MLflow 3.0 (June 2025) pivoted the project's energy toward GenAI tracing/evals ([mlflow.org/blog/mlflow-3-0-launch](https://mlflow.org/blog/mlflow-3-0-launch)). Free self-host, but zero robot-learning ecosystem presence and you'd run the server yourself.
- **Aim** — Apache-2.0, alive (pushed 2026-08-24, 6.2k stars, VERIFIED) but small; no robot-learning adoption.
- **ClearML** — client Apache-2.0 and active, but **server is SSPL** (VERIFIED license text: "Server Side Public License, VERSION 1... Copyright 2025 ClearML Inc.") and the server repo's last push was 2026-03-24. Self-host legally fine for internal use, but slow server cadence is a flag.
- **Trackio** (Hugging Face, [github.com/gradio-app/trackio](https://github.com/gradio-app/trackio)) — MIT, active (pushed 2026-08-24, VERIFIED), launched 2025-07-29 ([HF blog](https://huggingface.co/blog/trackio)). **Drop-in wandb API compatible** (`import trackio as wandb`), local-first SQLite, <1k LOC, optional sync to HF Spaces. This is the escape hatch that makes W&B lock-in cheap.

### Q3 — Orchestration (all licenses/activity VERIFIED via GitHub API 2026-08-25; all Apache-2.0, all active)

- **NVIDIA OSMO** ([github.com/NVIDIA/OSMO](https://github.com/NVIDIA/OSMO)) — open-sourced 2025-10-01, Apache-2.0, pushed today, 215 stars. Purpose-built Physical-AI orchestration: whole pipeline in YAML, **tasks pass artifacts sim → train → eval**, heterogeneous pools (training GPUs, sim nodes, Jetson edge for hardware-in-loop). VERIFIED from README: self-hosted on Kubernetes; used internally for GR00T/Isaac Lab before release. Closest conceptual match to "stages exchange hash-stamped artifacts" — but young as OSS and requires running your own K8s.
- **SkyPilot** ([repo](https://github.com/skypilot-org/skypilot)) — 10.5k stars, Apache-2.0. Managed jobs = launch on cheapest of 25+ clouds/K8s, auto-recover from spot preemption, resume from checkpoint. CLAIMED: now a funded company with a commercial platform ([skypilot.ai blog](https://skypilot.ai/blog/skypilot-the-company)); managed API server on Nebius ([Nebius blog](https://nebius.com/blog/posts/managed-skypilot-api-server-tech-overview-and-setup)). Best-in-class for burst cloud GPUs without owning infrastructure.
- **Ray** — 43.6k stars. Ray Train V2 (Nov 2025, [Anyscale](https://www.anyscale.com/blog/ray-train-v2-unified-distributed-training-on-ray)) adds local mode + elastic training. Ray is a **within-stage scaling library** (batched sim rollouts, distributed training, Tune sweeps), not a pipeline DAG orchestrator.
- **Flyte 2** — GA **2026-08-04** ([Union.ai announcement](https://www.globenewswire.com/news-release/2026/08/04/3338397/0/en/Union-ai-Announces-General-Availability-of-Flyte-2-Bringing-Durable-Runtime-to-Open-Source.html)) — a ground-up rewrite (pure-Python authoring, durable runtime); Flyte 1 security-fixes only through end of 2026 (CLAIMED). Strong typed artifact-passing story, but three weeks post-GA of a rewrite = adoption risk, plus heavy K8s ops for a small team.
- **Dagster** — OSS Apache-2.0 free; its software-defined-assets model maps well to artifact-exchange, but Dagster+ repriced 2026-05-01: Starter $100/mo + **$0.035/credit with zero included credits, 1 credit per asset materialization** ([support note](https://support.dagster.io/articles/3171123463-dagster-solo-and-starter-pricing-updates-may-2026)) — hostile to high-churn pipelines. **Prefect** — OSS free; Cloud Starter ~$100/mo (CLAIMED via [pricing roundups](https://www.prefect.io/blog/dagster-vs-prefect-self-serve-plans-compared)); imperative flows, weakest artifact-awareness of the group.

### Q4 — LLMOps tooling: noise, with one transferable idea

- Searched for actual usage of LangSmith/Langfuse/W&B Weave in robot-policy pipelines: **no evidence found** (2026-08-25). Their data model is conversation traces/spans; robot training telemetry is high-rate time-series + video — wrong shape. Verdict: **noise for this platform**.
- The one pattern worth stealing is methodological, not a product: **automated eval judges** — world-foundation-model/VLM-as-judge scoring of rollout success (e.g., [Cosmos-Surg-dVRK, automated online eval of surgical robot policies](https://arxiv.org/pdf/2510.16240); NVIDIA's [Physical AI Data Factory blueprint](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Open-Physical-AI-Data-Factory-Blueprint-to-Accelerate-Robotics-Vision-AI-Agents-and-Autonomous-Vehicle-Development/default.aspx), March 2026). If ever adopted, judge verdicts are *inputs* to certified evals, never certificates themselves.

### Q5 — Robotics-specific MLOps products (all CLAIMED from vendor/press, 2026)

- **Foxglove** ([foxglove.dev](https://foxglove.dev/)) — repositioned as "agentic data platform for Physical AI" (Aug 2026): fleet data triage, semantic search over unlabeled robot data via NVIDIA Cosmos, dataset curation for training ([press](https://theaiinsider.tech/2026/08/18/foxglove-launches-agentic-data-platform-for-physical-ai-collaborates-with-nvidia-on-semantic-search/)).
- **Rerun 0.32** (~May 2026, [blog](https://rerun.io/blog/data-layer-for-robot-learning)) — "data layer for robot learning": SQL/dataframe queries over .rrd, **PyTorch dataloader training directly on .rrd files** (random access, DDP, multi-worker prefetch), plus **Rerun Hub** commercial catalog/storage in private preview. Directly relevant — the rig is already Rerun-native.
- **NVIDIA OSMO** (above) doubles as the robotics-specific orchestrator.
- These are fleet-data/curation platforms; none replaces an evidentiary artifact registry — none has per-value provenance or certification semantics.

## (b) Recommendation stack

**Tracking: W&B SaaS (free tier now, Pro $60/mo when >5 seats) — but coded strictly against the wandb API.**
Reasoning: it is the only tracker with ecosystem gravity in robot learning (VERIFIED in all four codebases); LeRobot's `WandBLogger` works out of the box (`--wandb.enable=true`); free tier covers a 1–2 person team today. The CoreWeave-ownership and Neptune-sunset risks are neutralized by two things to make policy: (1) the standing rule already in force — W&B holds only disposable high-churn debugging telemetry, all decision-grade data lives in the hash-stamped registry, so losing the tracker loses nothing evidentiary; (2) use only the `wandb.init/log/finish` surface so **Trackio (MIT, wandb-API drop-in) is a one-line migration** if pricing or ownership turns hostile. Wire the two systems by ID, not by data: log `robot@hash / scene@hash / policy@hash / recipe@hash` as wandb config+tags on every run, and store the wandb run URL in the bundle's provenance record. One-way references, no authority transfer.

**Orchestration: keep the DAG custom; execute with SkyPilot managed jobs; scale within-stage with Ray only when needed.**
Reasoning: the recipe@hash registry *is* the pipeline definition — the stage graph, inputs, and outputs already have provenance semantics that Dagster/Prefect/Flyte would duplicate and dilute (their artifact stores would compete with the registry's evidentiary authority). What's actually missing is *execution*: getting a stage onto a burst cloud GPU reliably. SkyPilot does exactly that (Apache-2.0, spot-preemption recovery with checkpoint resume, 25+ clouds, no K8s to operate) and its unit of work — "run this container/command on this hardware, mount these artifacts" — composes cleanly under a thin custom driver that resolves hashes → S3 paths before launch and stamps outputs after. For the very first cloud runs, `lerobot-train --job.target=a10g-large` on HF Jobs is zero setup and already in the framework. Ray Train/Tune enters later, *inside* the eval stage, when batched MuJoCo rollouts outgrow one node — not as pipeline infrastructure.

**Keep custom: the artifact registry, certified evals, per-value provenance, and stage wiring.** Nothing surveyed threatens or replaces it; every commercial "robot data platform" (Foxglove, Rerun Hub) is a curation/debugging layer, and every orchestrator's artifact store is weaker than what exists. The standing rule stands: W&B curves never reach a customer.

## (c) Defer, and until when

| Deferred | Revisit trigger / date |
|---|---|
| **Ray Train/Tune** | When sim-eval batches exceed one node, or first multi-node training run. Not before. |
| **NVIDIA OSMO** | When you operate your own K8s/GPU cluster or add hardware-in-loop stages to the pipeline. Check OSS maturity ~Q1 2027 — it is the long-term shape-match for sim→train→eval artifact pipelines. |
| **Flyte 2** | Q1 2027 at the earliest — let the ground-up rewrite (GA 2026-08-04) harden first. |
| **Dagster / Prefect** | Only if large non-ML data-engineering pipelines (dataset ETL, fleet ingest) appear. Dagster+'s per-materialization pricing (May 2026) is a poor fit for high-churn training. |
| **Trackio migration** | Only on W&B pricing/ownership deterioration. Preserve the option by never using wandb features beyond init/log/finish/config/tags. |
| **Rerun Hub** | Watch private preview → GA; natural fit given the Rerun-native viewers, and the .rrd PyTorch dataloader could later simplify eval-recording pipelines. |
| **LLM-judge auto-eval** | When human episode-labeling becomes the eval bottleneck; adopt as an input to certified evals, never as the certificate. |

Sources: [LeRobot](https://github.com/huggingface/lerobot), [openpi](https://github.com/Physical-Intelligence/openpi), [OpenVLA](https://github.com/openvla/openvla), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (all inspected at HEAD), [W&B pricing](https://wandb.ai/site/pricing), [CoreWeave/W&B close](https://investors.coreweave.com/news/news-details/2025/CoreWeave-Completes-Acquisition-of-Weights--Biases/default.aspx), [OpenAI/Neptune](https://openai.com/index/openai-to-acquire-neptune/), [Trackio](https://huggingface.co/blog/trackio), [MLflow 3](https://mlflow.org/blog/mlflow-3-0-launch), [OSMO](https://github.com/NVIDIA/OSMO), [SkyPilot](https://github.com/skypilot-org/skypilot), [Flyte 2 GA](https://www.globenewswire.com/news-release/2026/08/04/3338397/0/en/Union-ai-Announces-General-Availability-of-Flyte-2-Bringing-Durable-Runtime-to-Open-Source.html), [Ray Train V2](https://www.anyscale.com/blog/ray-train-v2-unified-distributed-training-on-ray), [Dagster pricing update](https://support.dagster.io/articles/3171123463-dagster-solo-and-starter-pricing-updates-may-2026), [Rerun 0.32](https://rerun.io/blog/data-layer-for-robot-learning), [Foxglove](https://foxglove.dev/), [LeRobot v0.6.0](https://huggingface.co/blog/lerobot-release-v060), [Cosmos-Surg-dVRK](https://arxiv.org/pdf/2510.16240), [NVIDIA Physical AI Data Factory](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Open-Physical-AI-Data-Factory-Blueprint-to-Accelerate-Robotics-Vision-AI-Agents-and-Autonomous-Vehicle-Development/default.aspx)
