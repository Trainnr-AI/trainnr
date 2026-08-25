# Fleet data planes as practiced, 2024–2026

*Fourth pass, 2026-08-25. Researched by one agent against primary
sources, per the one-agent-per-field discipline. Tests
[docs/30-the-full-loop.md](../30-the-full-loop.md) §3.3's tier-0/1/2
upload design against what deployed fleets actually do. Headline: the
tier structure is confirmed by convergent practice, with five
amendments — and the calibration/certificate measurement chain is
AHEAD of practice. The agent's report follows verbatim.*

---

Method note: VERIFIED = the primary page/abstract was fetched and states this. CLAIMED = primary source identified but only search-snippet/secondary confirmation (the permission system began denying WebFetch/curl mid-task for several corporate domains). All dates are publication dates.

## (a) Dated findings

### 1. Manipulation / humanoid fleets

**Physical Intelligence**
- **2025-11-17 — π*0.6 / RECAP** (https://www.pi.website/blog/pistar06, paper https://arxiv.org/abs/2511.14759) — VERIFIED. Exactly three data streams: demos, teleop corrections captured by "running our best current policy and taking over with manual teleoperation when the robot makes a mistake," and autonomous rollouts; a value function trained on deployment data + advantage conditioning extracts training signal even from failures; full RECAP roughly doubles throughput and halves failure rate (13h espresso runs, laundry in new homes, 59 factory boxes). The clearest published quantification of deployment-data + correction value. No upload/infra detail published.

**Figure**
- **2025-02-20 — Helix** (https://www.figure.ai/news/helix) — CLAIMED (primary page in results). ~500 h multi-robot multi-operator teleop; auto-labeling VLM generates hindsight language instructions.
- **2025-02-26 — Helix in logistics** (https://www.figure.ai/news/helix-logistics) — VERIFIED. 8 h of curated demos; slow/missed/failed demos excluded but **corrective behavior deliberately retained** when failure was environmental; curated model got 40% better throughput on 1/3 less data; visual-proprioception self-calibration for cross-robot policy transfer.
- **2025-06 — Scaling Helix** (https://www.figure.ai/news/scaling-helix-logistics) — CLAIMED. Demos scaled 10 h → 60 h (6x); memory module + force feedback; 4.05 s/package, ~95% barcode orientation.
- **2025-09-17 — Project Go-Big** (https://www.figure.ai/news/project-go-big) — CLAIMED. Passive egocentric human video from Brookfield's 100k+ residential units as pretraining; navigation transfer from human video alone.
- **2025-10-09 — Figure 03** (https://www.figure.ai/news/introducing-figure-03) — VERIFIED. Hardware designed around the data plane: **10 Gbps mmWave dock offload "allowing the entire fleet to upload terabytes," offloading during shift breaks**; palm cameras, 3-gram-threshold tactile; BotQ 12k robots/yr. This is the "Jetson-class robot ships nightly batches" tier made physical.

**1X**
- **2025-06-10 — Redwood AI** (https://www.1x.tech/discover/redwood-ai) — VERIFIED (curl). Trained on "a large dataset of teleoperated and autonomous episodes from EVE and NEO" gathered in offices and employee homes; 160M params, runs fully on NEO's onboard GPU (~5 Hz); **explicitly "learns from success and failure data"** — failure rollouts supervise auxiliary "cognitive" heads.
- **2025-10-28 — NEO launch / Expert Mode** (https://www.1x.tech/discover/neo-home-robot; press e.g. https://www.techradar.com/ai-platforms-assistants/neo-robot-sounds-like-the-answer-to-our-home-chore-prayers-but-also-a-potential-privacy-nightmare) — CLAIMED. Interventions are **scheduled, user-approved teleop sessions** (ring-light indicator, US-based operators, face blurring, no-go zones); every teleoperated task is recorded, labeled, and fed to Redwood. The consumer-consent version of Tier 1: intervention capture as a negotiated privilege, not a default.

**Agility**
- **2024-03-11 — Agility Arc launch** (https://www.agilityrobotics.com/content/agility-robotics-brings-operational-visibility-to-deployment-of-digit-fleets-with-the-launch-of-agility-arc-tm) — VERIFIED (curl). Cloud fleet platform: KPIs uptime, throughput, **MTBI**, robot status; alerts/troubleshooting; WMS/WES/MES APIs. Pure Tier-0 ops telemetry — the training pipeline is not publicly part of Arc.
- **2025–26 — deployment data claims** (https://www.nvidia.com/en-us/case-studies/agility-robotics-digit-humanoid-robot/; https://www.agilityrobotics.com/content/digit-moves-over-100k-totes) — CLAIMED. 65,000+ operational hours across 9 customer facilities framed as proprietary training data; 100k+ totes at GXO Flowery Branch.

**Others deploying**
- **2024-03 — Covariant RFM-1** (https://covariant.ai/covariant-introduces-rfm-1-to-give-robots-the-human-like-ability-to-reason/) — CLAIMED. 8B model on "tens of millions of trajectories" from the deployed picking fleet.
- **2025-01-07 — Ambi Robotics PRIME-1** (https://www.ambirobotics.com/media/ambi-robotics-deploys-prime-1-ai-foundation-model/) — CLAIMED. Pretrained on 20M+ images from **150,000 fleet operating hours**; production scaling laws; **trained on only ~1% of data collected** — retain broadly, retrain on a curated sliver.
- **2025-04-30 — Dyna DYNA-1** (https://www.dyna.co/research/dyna-1) — CLAIMED. 24 h autonomous run, ~99.4% success; claims a continuous deployment-RL loop; no per-data metrics.
- **2025-11-04 — Generalist GEN-0** (https://generalistai.com/blog/nov-04-2025-GEN-0) — CLAIMED. 270,000+ h manipulation data growing ~10,000 h/week (handheld UMI, not robot fleet); public data/compute power laws.
- **2026 — Skild** (https://www.skild.ai/blogs/series-c) — CLAIMED. "Data flywheel" across security/warehouse/factory deployments; marketing-level only, no data-plane engineering published.

### 2. AV prior art — the mature version

**Tesla**
- **2021-06 — Karpathy CVPR 2021 workshop** (writeups: https://bdtechtalks.com/2021/06/28/tesla-computer-vision-autonomous-driving/) — CLAIMED (concordant secondaries). The canonical data engine: **221 trigger types** (4 months to build), seven loops of train → deploy in **shadow mode** → trigger-flagged disagreement clips upload → auto-label → retrain; final dataset 1.5 PB / 1M 10-s clips.
- **2019-02 — firmware reverse engineering** (https://x.com/greentheonly/status/1096322821934993408) — CLAIMED. Server-pushed trigger specs to car subsets; **wifi-only uploads**; always-on low-rate per-drive summaries + disengagement reports; ~10 GB reserved on-vehicle snapshot buffer. The three tiers, in the wild, in 2019.
- **2021/2022 AI Days** (https://codecompass00.substack.com/p/tesla-data-engine-trigger-classifiers) — CLAIMED. Campaign example: cut-in trigger → ~10k 10-s clips in days, **hindsight auto-labeled** by what the cut-in car actually did; offline reconstruction so one labeled scene labels thousands of drives.
- **2025-10 — Elluswamy, ICCV WDFM-AD** (https://www.humanoidsdaily.com/feed/tesla-ai-chief-details-unified-world-simulator-for-fsd-and-optimus) — CLAIMED. Fleet = "~500 years of driving per day," "more data than we could ever store" — the binding constraint has moved from uplink to **storage and curation**.

**Waymo**
- **2020-10-30** (https://arxiv.org/abs/2011.00038) — VERIFIED. Every contact/disengagement reconstructed incl. **29 counterfactual "what-if" replays** — interventions retained with enough fidelity to simulate the untaken branch.
- **2021-03-08 — offboard auto-labeling** (https://arxiv.org/abs/2103.05073) — VERIFIED. Offline oversized models produce human-quality labels at fleet scale.
- **2022-10-15 — rare-example mining** (https://arxiv.org/abs/2210.08375) — VERIFIED. Feature-space density estimation mines the long tail from fleet logs, cost-aware.
- **2025-12-09 — Waymo Foundation Model** (https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/) — VERIFIED. Teacher→student distillation; a **Critic "automatically flags any suboptimal driving behavior"** as the outer data loop; 100M+ autonomous miles.
- **2026-08-20 — retention reporting** (https://nashvillebanner.com/2026/08/20/waymo-autonomous-vehicles-surveillance-data-retention/) — CLAIMED. No fixed retention schedule; storing random non-training footage "too costly"; footage deleted before a warrant arrived in one SF case.

**Wayve** — **2023** Azure fleet-learning loop, >12 PB/yr chauffeur fleet + "100s of PB" from third-party fleets (https://wayve.ai/thinking/scaling-machine-learning-from-garage-to-fleet-with-microsoft-azure/) — CLAIMED; **2025-03-26** GAIA-2 trained on ~25M two-second fleet sequences with a minimum temporal stride to kill redundancy — an explicit curation rule (https://arxiv.org/abs/2503.20523) — abstract VERIFIED.

**Cruise** — **2020-09** Continuous Learning Machine (https://medium.com/cruise/cruise-continuous-learning-machine-30d60f4c691b) — CLAIMED. Prediction-error active learning: **the observed future auto-labels the past**; only prediction-vs-reality gaps enter training.

**Zoox** — **2025-10/12** (https://siliconangle.com/2025/10/31/aws-fuels-zooxs-autonomous-robotaxis/) — CLAIMED. Up to **4 TB raw/hour**; cellular can't carry it, so vehicles **physically dock** at AWS Data Transfer Terminals (400 Gb/s); ~30 PB hot / ~1 EB cold.

**Mobileye REM** — current (https://www.mobileye.com/technology/rem/) — VERIFIED. The extreme edge-filtering pole: **10 KB per car per kilometer** uploaded after on-vehicle semantic compression.

**comma.ai — the best-documented two-tier design**
- **2019-12 → current** (https://raw.githubusercontent.com/commaai/openpilot/master/RELEASES.md; https://docs.comma.ai/concepts/logs/) — VERIFIED. Drives split into **1-minute segments**; per segment: full rlog + HEVC video held on device, plus always-uploaded **qlog (decimated rlog) + qcamera (low-res preview)** "small enough to upload instantly on slow internet"; route visible in the cloud after **0.2%** is uploaded; full logs wifi/on-demand; auto-delete at 90% disk.
- **2025-02-28 — Firehose Mode** (https://blog.comma.ai/098release/) — CLAIMED. Opt-in max-upload tier on unmetered power+wifi; ingestion pipeline scaled 100x; **2026**: fleet logs ~1M driving minutes/day, pipeline **ingests ~15% of drives** into training (https://blog.comma.ai/chestnut/) — CLAIMED.

### 3. Fleet-learning literature 2025–26 (most load-bearing; all arXiv abstracts VERIFIED unless noted)

- **RaC, 2025-09** (https://arxiv.org/abs/2509.07953) — performance **scales linearly with the count of recovery/correction maneuvers** in the data; matches SOTA with 10x less collection. The strongest argument that Tier 1 bytes are the most valuable bytes.
- **SOP, 2026-01-06** (https://arxiv.org/abs/2601.03044) — spot-checked first-hand: fleet streams on-policy experience + human intervention signals to a centralized cloud learner, asynchronously receives updated policies; near-linear scaling with robot count; instantiated with HG-DAgger and RECAP. The published blueprint closest to the platform's whole loop.
- **Sirius-Fleet, 2024-10** (https://arxiv.org/abs/2410.22689) — world-model anomaly predictors solicit help pre-emptively and **auto-tighten thresholds as autonomy improves**, so intervention burden decays.
- **ARMADA/FLOAT, 2025-10** (https://arxiv.org/abs/2510.02298) — a ~95%-accurate failure detector is the trigger for shared-control takeover across a multi-robot deployment.
- **RLIF, 2023-11/ICLR-24** (https://arxiv.org/abs/2311.12996) — the **occurrence/timing of an intervention is itself a reward label**, even without a corrective trace; SiLRI, 2025-12 (https://arxiv.org/abs/2512.24288) adds operator-uncertainty logging.
- **Fleet-DAgger, 2022-06** (https://arxiv.org/abs/2206.14349) — defines **Return on Human Effort**, the field's canonical improvement-per-supervision metric (nobody yet publishes improvement-per-GB).
- **Mirchandani et al., 2024-11** (https://arxiv.org/abs/2411.01813) — sobering: autonomous success-log collection under-delivers vs more human data; passive Tier-2 success logs are low-yield per byte unless value/advantage-labeled (RECAP) or auto-labeled against ground truth (Scanford flywheel, 2025-11, https://arxiv.org/abs/2511.19647 — measured per-deployment gains, e.g. book-ID 32.0→71.8%).
- **Sentinel/STAC, 2024-10** (https://arxiv.org/abs/2410.04640) — **two-tier failure detection**: cheap on-robot temporal-consistency statistic + expensive VLM video-QA in the cloud; the split catches 18% more failures. FAIL-Detect, 2025-03 (https://arxiv.org/abs/2503.08558) — conformal thresholds, no failure exemplars needed.
- **Hide-and-Seek, 2026-05** (https://arxiv.org/abs/2605.30834) — localizes *which timesteps* signal failure from episode-level labels: intervention mining proper, validated on π0/π0.5.
- **Continual VLA benchmark, 2026-05** (https://arxiv.org/abs/2605.26820) — tuned experience replay beats joint training at equal compute: **retention policy is a first-class design decision**. FLEET-MERGE, 2023-10 (https://arxiv.org/abs/2310.01362) is the opposite pole: upload merged weights (~MB/day), not episodes.
- **Curation:** DemInf RSS-25 (https://arxiv.org/abs/2502.08623), QoQ 2026-03 (https://arxiv.org/abs/2603.09056) — quality-scoring/influence ranking beats training on everything.

### 4. Bandwidth / retention economics

- **Foxglove, 2024-01-15** (https://foxglove.dev/blog/best-practices-for-recording-and-uploading-robotics-data) — VERIFIED. Robots record **>1 GB/s against 10–100 Mbps site uplinks**; hence selective segment/topic upload; **retain lightweight telemetry indefinitely, cull heavyweight sensor data**; if engineers touch data only ~2 weeks post-recording, retention past a month is waste.
- **ReductStore (~2025-26)** (https://www.reduct.store/blog/amr-fleet-data-infrastructure) — CLAIMED. One AMR: **30–100 GB/shift**; 100 robots: 3–10 TB/shift; FIFO volume-quota retention at the edge, replication filters decide what reaches cloud.
- **Formant docs** (https://docs.formant.io/docs/how-telemetry-streams-work) — CLAIMED. Cloud telemetry default **0.5 Hz/stream, configurable 0–5 Hz**; on-demand ingestion cuts cellular spend "upwards of 80%"; teleop ~3 Mbps symmetric via WebRTC. Platform costs cluster ~$200–500/robot/month (third-party quotes, Formant/InOrbit).
- Raw-vs-uploaded spread: 4 TB/h raw (Zoox) → 10 KB/km uploaded (Mobileye): **8–9 orders of magnitude**, bridged everywhere by the same architecture — always-on low-rate telemetry + trigger-emitted snippets + full logs discarded, pulled on demand, or physically couriered/docked (Zoox terminals; Figure 03's mmWave dock is the manipulation-world copy).
- Retention disclosure is a void: no robot/AV fleet publishes deletion windows (Waymo: "reasonably appropriate amount of time," case-by-case, 2026-08).

### 5. Standardization

- **MCAP** (https://mcap.dev/) — VERIFIED. Append-only, self-describing (schemas stored with data), chunk-indexed for partial remote reads; **rosbag2 default since ROS 2 Iron, May 2023** (https://foxglove.dev/blog/mcap-as-the-ros2-default-bag-format — VERIFIED); Wayve/Anduril named users. De facto on-robot recording standard.
- **LeRobotDataset v3.0, 2025-09-16** (https://huggingface.co/blog/lerobot-datasets-v3) — VERIFIED. Parquet + MP4 + relational metadata; multi-episode packing for **millions of episodes**; streaming without full download. De facto training-exchange format (RLDS/OXE is the legacy archival format). Used as a native recording format only for tabletop demo rigs; fleets convert post-hoc from MCAP — **the MCAP→LeRobot boundary is a bespoke pipeline at every company**.
- **Foxglove Foxlet** (https://docs.foxglove.dev/docs/data) — VERIFIED. Register-metadata-first, **import-on-demand** pull model + device auto-delete: the commercial embodiment of "hold locally, let the server retro-pull."
- **Protocols** — CLAIMED: MQTT for low-rate uplink, with **VDA 5050** (MQTT+JSON, `state`/`order`/`factsheet` topics) the only true standard, AGV-only; WebRTC universally for teleop; Zenoh Tier-1 in ROS 2 Kilted (2025-05); OpenTelemetry-for-robotics is one-person glue code — **no standard exists for robot health/status semantics**.
- **Calibration/config history as versioned telemetry: no format, no vendor, no convention.** MCAP's schema-with-data solves half; nobody publishes the other half.

## (b) Verdict on the Tier 0/1/2 design

**Confirmed by practice in structure.** The three-tier shape is exactly what every mature fleet converged on independently: comma (qlog always / interventions marked / full logs wifi-batched, since 2019), Tesla (drive summaries + disengagement reports / trigger snippets / wifi-only), Formant/Foxglove (0.5–5 Hz cloud streams / events / import-on-demand), Figure 03 (dock-batched offload), 1X (consent-gated intervention episodes). Treating takeovers as labeled demos is now the consensus of both industry (RECAP, 1X Expert Mode) and literature (RaC's linear scaling, SOP). The per-class transport split ($7 MCU = status only; Jetson = nightly batches) mirrors the Mobileye-to-Zoox spectrum precisely. The calibration/certificate measurement chain is **ahead of practice** — a published gap, so keep it as a differentiator.

**Five amendments:**
1. **Decimate Tier 0 on the wire.** 50 Hz is the right *local recording* rate, but nobody ships 50 Hz to the cloud continuously: practice is 0.5–5 Hz cloud streams (Formant) or per-episode summaries (Tesla buckets, comma qlog ≈ 0.2% of data). Make Tier 0 a 50 Hz on-robot ring buffer with a decimated/summarized uplink, burstable on anomaly.
2. **Tier 1 must fire without a human.** The 2025-26 literature's biggest addition is the autonomous failure detector as an upload trigger (Sentinel/STAC's cheap-on-robot + expensive-in-cloud split, FLOAT, FAIL-Detect's conformal guarantees, Sirius-Fleet's self-tightening thresholds). Operator takeover is one Tier-1 trigger among several; detector-flagged episodes with no takeover are equally food.
3. **Capture more than the correction trace.** Log the takeover *timestamp* even when no clean corrective trace exists (RLIF: timing alone is a reward label), the pre-intervention failure prefix (RaC rewinds to an in-distribution state — the approach matters), and operator confidence (SiLRI). An intervention episode = prefix + takeover marker + correction + outcome label.
4. **Add a campaign/retro-pull channel to Tier 2.** Batched-around-events is necessary but not sufficient: the server must be able to push new predicates and pull historical segments still sitting in device buffers (Tesla triggers, Foxglove import-on-demand, comma's prioritized interactive uploads). And label autonomous batches with outcome/value — unlabeled success logs are the lowest-yield bytes in the pipeline (Mirchandani; RECAP fixes this with value labels).
5. **Make retention a negotiated bundle parameter too.** Practice retains broadly and retrains on ~1% (Ambi), and continual-learning results show replay data is a compute asset (2605.26820) — so the bundle should carry a retention/replay budget per embodiment, including a weights-not-data fallback (FLEET-MERGE, on-device Redwood) for privacy-constrained deployments like homes.

## (c) The single most load-bearing AV pattern to copy

**The trigger-campaign loop against on-device full-rate buffers**: robots record everything locally into a FIFO ring buffer, always upload only the tiny index tier, and the server deploys *revisable* trigger predicates to the fleet that cause matching snippets — including ones recorded *before* the trigger existed — to be uploaded, hindsight-auto-labeled, and folded into the next training round (Tesla's 221-trigger engine; comma's qlog-first + on-demand pull; Foxglove's import-on-demand is the off-the-shelf version). It's load-bearing because it makes curation decisions reversible after deployment — you discover what mattered later, and the data is still there. And Tesla's 2025 lesson bounds it: at fleet scale the binding constraint is storage and curation, not uplink — which is exactly the constraint the tier design should be engineered around, since manipulation robots dock nightly on wifi (Figure 03 made that literal with a 10 Gbps offload dock) and bandwidth mostly stops being the problem.
