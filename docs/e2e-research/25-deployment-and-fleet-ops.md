# Deployment, fleet operations and telemetry

Research date: **2026-08-08**. Question: what does it take to run 2–3 robots at
a customer site — serving policies, logging, updating, and knowing whether it is
working?

> **Re-verified 2026-08-15** against the primary sources (a discovery sweep;
> web search was quota-exhausted, so vendor *news* was out of reach — prices
> and versions below were checked directly). Everything load-bearing held;
> the only version bump is Rerun 0.35.0 → 0.36.0. Deltas are marked inline
> with the date.

> **TL;DR.** Serve from a **LAN GPU over gRPC with real-time chunking**;
> **quantization is not the lever** at batch size 1. Log to **MCAP**, which has
> a mature Rust crate and would replace the archived rig's two bespoke formats.
> **Calibration is device state, not artifact state** — and every episode must
> record which calibration was active, or your training data is silently
> corrupted. Rollout goes offline-replay → shadow → canary → fleet, because you
> cannot A/B test a robot. And **there is no published industry number for a
> good intervention rate** — define it with the customer in week zero.

---

## 1. Serving

**LeRobot 0.6.1** (PyPI, **2026-08-03**; still the latest release as of
2026-08-15; requires Python ≥3.12). Install with `pip install -e ".[async]"`,
which pulls `grpcio`.

Two processes over gRPC:

```
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
python -m lerobot.async_inference.robot_client  --server_address=… --robot.type=…
```

The server starts as an **empty container** — the policy identity, checkpoint
and device are negotiated in the first client handshake. Operationally that means
one server binary serves many policies, and also that **nothing pins which model
version is running**. In production, invert it: the server should refuse
handshakes whose requested policy is not the one the fleet's release manifest
authorises, or a stale robot can pull an unapproved model.

Tunables that matter:

| Parameter | Typical | Effect |
|---|---|---|
| `actions_per_chunk` | 10–50 | More actions means less chance of queue starvation, but compounding error over a longer horizon |
| `chunk_size_threshold` | ~0.5–0.6 | Queue-fullness fraction at which the client sends a fresh observation. 0 is fully synchronous; 1 sends every step |
| `aggregate_fn_name` | `weighted_average` | How overlapping chunk regions are blended |
| `--debug_visualize_queue_size` | off | Plots action-queue depth live — **this is your latency instrument** |

**Real-time chunking** on top: `--inference.type=rtc`, with
`execution_horizon` 8–12 and `max_guidance_weight` 10.0 for 10-step flow
matching. Works with flow-matching policies (π0, π0.5, SmolVLA, GR00T) and
**needs no retraining**.

Measured: gRPC round-trip **sub-100 ms on a local network**, ~5× faster than an
equivalent REST design; observations are *streamed* because payloads routinely
exceed 4 MB. LeRobot reports **~2× speedup in task completion time at comparable
success rate**, and for SmolVLA specifically **9.7 s vs 13.75 s (~30% faster)**.

The ratio to watch is `c = environment_dt / inference_time`. **If `c` ≪ 1 you
have degraded to sequential control — lower your control rate rather than
pretending.**

Inference memory: SmolVLA **~2 GB**, MolmoAct2 **12.1 GiB**, π0 **14 GB**. One
24 GB card comfortably serves three robots.

### Quantization is not the lever

There is **no official ONNX, TensorRT or quantization export path** in LeRobot as
of 2026-08-08. The only thing found is third-party and immature —
`eddisonpham/LeRobot-Edge` (Apache-2.0, **1 GitHub star**), reporting on
SmolVLA: NF4 gives **1142 MB → 287 MB (3.97×)** memory reduction, and FP16 gives
**1.05× speedup at batch 4, minimal improvement at batch 1**.

**Batch 1 is what a single robot runs.** Quantization bought memory, not
latency. Your actual levers are (a) a smaller policy, (b) RTC and async to hide
the latency that remains, and (c) putting the GPU on the LAN. **Skip
quantization for a 3-robot pilot.**

### The link, and what fails

Design targets found for warehouse and retail WiFi:

- **≥ −65 dBm RSSI at cell edge** (−67 dBm is the other commonly cited target)
- **15–20% cell overlap** between adjacent access points — mobile robots need
  more than the voice-grade minimum
- **Handover latency < 150 ms** sustained
- Enable 802.11k/v; **test 802.11r against your specific client chipset** before
  committing

Failure modes in the order they bite:

1. **Sticky clients.** Oversized cells mean the robot clings to a distant access
   point that is "still acceptable". The counter-intuitive fix is to **lower
   transmit power** to force earlier handoff. Do not turn it up.
2. **Roaming gaps at aisle ends and dock doors** — geometry, not equipment. Racking
   full of metal or liquid is a different radio environment when full than when
   empty; **survey both**.
3. **"Full bars but the app hangs"** is retransmits and interference, not
   coverage. Do not chase signal strength.
4. **Retail specifically:** guest WiFi contention and captive portals. Put robots
   on their own SSID and VLAN with quality-of-service.

⚠️ *Assessment:* Wi-Fi 6E gives clean spectrum but 6 GHz propagates and
penetrates worse, needing denser access points. For three robots, 5 GHz with a
good cell plan is usually right; 6 GHz is a congestion fix, not a latency fix.

The failover ladder itself is in [19-the-system.md](19-the-system.md) §3, and its
one rule bears repeating: **the safe-stop must not depend on the link that just
failed**, and **RTC hides latency, not loss**.

---

## 2. Logging

### MCAP

- **Spec version 0.** Explicit stability promise: *"Any changes to this
  specification document will be binary backward-compatible within the major
  version."* Readers must ignore unknown fields.
- Layout: `Magic · Header · Data · [Summary] · [Summary Offset] · Footer · Magic`.
  **Always write the summary sections** — they are what give O(1) seeking.
- Compression: zstd, lz4 or none, at chunk level, so files stay indexable while
  compressed.
- SDKs: C++, Go, Python, **Rust**, Swift, TypeScript. The default log format in
  ROS 2.

**Rust crate `mcap` v0.25.0** (2026-06-11), **~8.88M total downloads**
(re-verified 2026-08-15; +170k since 2026-08-08), MIT, feature flags for
zstd/lz4 and async via tokio. Version cadence 0.23.3 (Aug 2025) → 0.24.0 (Dec
2025) → 0.25.0, still current. The spec is likewise still major version 0 with
the same compatibility promise. **That is a mature, widely-used crate — a
legitimate foundation, not a science project.**

For the archived rig specifically: MCAP is the obvious eventual replacement
for the two bespoke formats behind its hil-host wire recorder (crates/hil-host/src/wire.rs) and the
perception recorder ([Trainnr-AI/rig](https://github.com/Trainnr-AI/rig)). It keeps replay-as-regression-test, and gains Foxglove, Rerun and
rosbag2 compatibility **without adopting ROS 2**.

### Volume and cost

⚠️ *Estimate, not a sourced figure* — published references are all from
autonomous driving (10–14 TB/day per test vehicle), which is a wildly different
regime.

| Stream | Rate | Per 8-hour shift |
|---|---:|---:|
| 2 × 640×480 @30 fps, H.264 ~2 Mbps | 4 Mbps | ~14 GB |
| Joint states, odometry, actions @50 Hz | ~0.5 Mbps | ~1.8 GB |
| Policy I/O (observation hashes, chunks, latencies) | ~0.1 Mbps | ~0.4 GB |
| **Total, full fidelity** | | **~15–20 GB/robot/shift** |

⚠️ **That total is this table's own arithmetic, not an industry figure**,
and it is worth saying so because two independent research passes on
2026-08-10 went looking for a primary source and found none. Inverting it gives 4.2–5.6 Mbit/s, which is
consistent with one compressed camera plus lidar and odometry, so the
decomposition holds together. But do not cite it as a measured
industry-wide number, because nobody appears to have published one.

It is also **essentially all camera**: the same pass calculated the
scalar telemetry line here at ~6.5 MB/hour raw, which is roughly 500×
cheaper than the video and rounds to nothing in this total.

**Cloudflare R2** wins this workload decisively: **$0.015/GB-month** standard,
$0.010 infrequent access, and **free egress**. Three robots × 20 GB/day × 30 days
= 1.8 TB/month ≈ **$27/month**.

**Free egress is the real win** — you will pull this data back repeatedly for
training, and on a provider that charges egress that line would dwarf storage.

> **The real cost is not storage; it is deciding what not to log.** Keep the
> low-rate structured streams always on. Record full-fidelity video only around
> *events* — interventions, e-stops, task failures, confidence dips — plus a ~5%
> random sample for distribution monitoring. That drops you to a few hundred GB
> a month and, more importantly, makes the data searchable.

### Consoles

| Option | Reality, 2026-08-08 |
|---|---|
| **Foxglove** | Free: $0, 10 GB, 5 devices, 3 users. **Pro: $20/mo** + usage, 1 TB, then $20/device/mo and $42/user/mo. Remote access 300 min/device included then $0.05/min. ⚠️ **Open-source Foxglove Studio was discontinued 2024-03-11**, and **self-hosted data is Enterprise-only.** Excellent for *your* engineers; a vendor lock for a customer console. Their agent docs advise avoiding files >50 GB. Re-verified 2026-08-15: pricing and the Enterprise-only self-hosting unchanged; Pro now also includes unlimited view-only "Basic seats" and AI/MCP features at $5/user/mo included usage. |
| **Rerun 0.36.0** (2026-08-10) | SDK Apache-2.0/MIT, "open source forever". 0.35 added **MCAP time-windowed conversion and corrupted-file recovery**, a local catalog, HDF5 import; 0.36 adds **experimental 3D Gaussian-splat rendering**, `mcap info`/`mcap check` CLI tools, multi-sink gRPC, and a PyTorch dataloader manifest builder. **Still nothing about alerting, device health or ops dashboards** — re-confirmed in the 0.36 notes. Best-in-class for debugging and training-data curation; **not a fleet telemetry system.** Rerun Hub is a commercial data platform with no public pricing. |
| **InOrbit** | **Free Edition: unlimited robots, free forever.** Paid tiers on request. |
| Formant / Freedom Robotics | Freemium tiers exist; no public pricing. ⚠️ A third-party aggregator quotes "$50–150/vehicle/month" — **unverified hearsay, not a vendor price.** |

⚠️ None of the four is open source. Genuinely open fleet ops means assembling
MCAP + Grafana/Rerun + Mender OSS or RAUC/hawkBit + your own console — weeks of
work, not days.

**Recommendation for three robots: Foxglove for engineers plus a small custom
web console for the operator.** They are different products for different
people, and merging them produces something bad at both. The operator console
needs about six things: which robot, what task, autonomy versus intervention
state, one live camera, a big stop button, and a take-over button. **A week, not
a quarter.** Given the archived rig's Rust/WASM direction, a Rust→WASM console reading
MCAP-schema'd messages over WebSocket was a coherent fit.

---

## 3. Updates and fleet management

### OTA, verified pricing

| System | Plan | Price | Devices |
|---|---|---:|---:|
| **Mender** | Open Source | free | unlimited (self-host) |
| | **Basic** | **$34/mo** | ≤50 |
| | Professional | $291/mo | ≤250 (adds delta updates, scheduled deployments, filtering) |
| | Enterprise | custom | adds phased rollouts, RBAC, mutual TLS, on-prem |
| **balena** | Free | $0 | 10 |
| | Prototype | $159/mo | 30 |
| | Production | $1,439/mo | 110 |
| **Torizon** | Maker | free forever | ≤5 |
| | Developer | $249/mo or $2,500/yr | 50 |
| **RAUC + hawkBit** | self-hosted | $0 | unlimited |

Mender Basic $34/mo and Professional $291/mo re-verified 2026-08-15 (the ≤50-
device cap was not re-verifiable — the pricing page now uses a device-count
slider).

**For three robots: Mender Basic at $34/mo**, or Mender OSS self-hosted. balena
is the nicest developer experience and the worst value here ($159/mo for 30
devices when you have 3). Torizon is tightly coupled to Toradex modules, and
⚠️ its *"as low as $0.30/month per device"* claim does not reconcile with
$2,500/yr ÷ 50 devices = **$50/device/year**.

🚨 **RAUC must be ≥ 1.15.2** (released **2026-03-27**; still the latest as of
2026-08-15, and CVE-2026-34155 is still the project's only security advisory),
which fixes
**CVE-2026-34155**: plain-format bundles exceeding a 2 GiB payload hit an integer
overflow where **the signature covered only the initial portion of the payload**.
That is a signature-bypass class bug, and robot OTA images are easily over 2 GiB.
RAUC otherwise offers fail-safe atomic updates, x.509 signing with **PKCS#11 and
HSM support**, HTTPS streaming with no intermediate on-device storage, and
symmetric A/B slots. It explicitly *"does NOT intend to be a deployment
server"* — pair it with Eclipse hawkBit.

### The release manifest

⚠️ **Synthesized recommendation. No published robotics-specific standard for
this exists** — the only prior art is generic MLOps deployment strategy.

**Concept: a "robot release" is a single signed manifest, and nothing ships
outside it.**

```
release: 2026.08.14-rc3
  os_image:      sha256:…          # RAUC or Mender bundle
  mcu_firmware:  sha256:…          # per board
  policy:
    id:          smolvla-picking-v7
    checkpoint:  sha256:…
    input_spec:  {cameras: [top, wrist], res: 224x224, fps: 30}
    action_spec: {dim: 7, chunk: 50, hz: 30}
  runtime_config:
    actions_per_chunk: 50
    chunk_size_threshold: 0.55
  compat:
    min_mcu_protocol: 4
    max_mcu_protocol: 5
```

Four rules make it work:

1. **Calibration is not in the release.** Per-unit calibration is *device state*,
   not artifact state — it belongs on the robot, mirrored to the cloud keyed by
   device ID, and it must **survive an OTA and a factory reset**. And:
   **every recorded episode must log which calibration hash was active**, or a
   recalibration you forgot about silently corrupts your training data.
2. **Policy↔hardware compatibility is explicit.** The `compat` block is what
   stops you shipping a 7-DoF policy to a robot whose protocol carries 6. Refuse
   to boot into a mismatched release.
3. **Everything is content-addressed and signed**, with the policy checkpoint on
   the same chain as the OS bundle.
4. **Keep the policy artifact separable from the OS bundle**, so a model rolls
   back in seconds without a reboot. Automatic rollback of the OS is solved by
   A/B slots; **automatic rollback of the policy is not — you build that.**

### Staged rollout

The crucial disanalogy: **you cannot A/B test a robot the way you A/B test a web
service, because the failure mode is physical and the sample size is three.**

1. **Offline replay** — run the candidate against logged MCAP observations from
   the site. Compare its actions to the incumbent's and to the human
   interventions. Cheap, safe, catches the worst regressions.
2. **Shadow mode on-robot** — candidate runs on live observations, logging
   predictions and latency; **its actions are discarded** and the incumbent
   drives. Highest-value step, and it costs only GPU. Gate on the
   action-divergence distribution and latency p95.
3. **Canary on 1 of 3**, supervised, low-value tasks first, for a defined number
   of task-hours.
4. **Fleet**, with the previous release retained on the inactive slot.

Define the automatic rollback triggers **numerically, up front**: intervention
rate over the trailing N tasks exceeding k× the incumbent's baseline; any e-stop
attributable to policy behaviour; success rate below a floor; inference latency
p95 over budget (a starved action queue *is* degraded control). Policy output
that is NaN or outside the envelope should be a **hard local guard that rejects
the action**, not merely a rollback trigger.

### Identity and secrets

Established practice converges on hardware-backed key storage — TPM 2.0, an HSM
or a secure element — with **per-device identity and never a shared fleet
credential**. RAUC's PKCS#11 support means the same TPM can hold the update
signing keys, giving one root of trust for identity and update integrity.

For a three-robot pilot the pragmatic minimum is per-robot X.509 client
certificates from a private CA, mutual TLS to the policy server and telemetry
endpoint, short certificate lifetimes with automated rotation, and **no
long-lived cloud API keys on the robot**. A stolen robot must not grant fleet
access. Note a Raspberry Pi 5 has no TPM without an add-on.

---

## 4. Metrics

The metric *names* are well established. **Published numeric benchmarks for what
"good" looks like are essentially absent.**

**MTBI — mean time between interventions** — is the consensus headline metric,
defined in the literature as "the mean time that a human-robot system operates
nominally." The Silicon Valley Robotics Center's fleet-pilot guidance names the
right categories (technical: reliability, setup time, intervention frequency,
recovery quality; business: cycle time, learning speed, customer impact, whether
the pilot de-risks a larger purchase) and gives **exactly one hard number**:
pilot duration, *"2 to 6 weeks is a useful range."* It offers no intervention
thresholds, no acceptable-failure percentages, no ROI benchmarks, and it ends in
a structured **expand, revise, or stop** decision.

> **Be honest about this with the customer: there is no published industry
> number for a good intervention rate in a commercial mobile-manipulator pilot.
> Anyone quoting one is quoting an internal figure.** Define the target jointly
> in week zero, from their current manual baseline.

Log from day one:

- **Interventions per hour and per task, split three ways** — teleop assist,
  physical touch, e-stop. These have wildly different business meaning and
  averaging them destroys the signal.
- **MTBI** in wall-clock and in task count
- **Task success rate** — attempted → completed without intervention
- **Cycle time p50 and p95** against the human baseline
- **Autonomy ratio** — autonomous seconds ÷ total mission seconds
- **Time-to-recover after an intervention.** Underrated: a robot with mediocre
  MTBI and 15-second recovery is commercially viable; one with excellent MTBI and
  10-minute recovery is not. **This is what decides whether three robots need one
  babysitter or three.**

`lerobot-rollout --strategy.type=sentry --strategy.upload_every_n_episodes=5`
is the deployed-log collector — turn it on from the first day of deployment so
the pilot doubles as data collection. See
[21-data-collection.md](21-data-collection.md) §4 for turning those logs into
training data.

---

## 5. The customer-facing app

⚠️ *Product judgement informed by the pilot-metrics research above, not a
sourced finding.*

The mistake is showing the customer robot telemetry. They do not care about joint
torques. The customer app should answer exactly four questions:

1. **Did it do the work today?** Tasks completed against target. One number, big.
2. **How much did it cost us in babysitting?** Interventions today, minutes of
   human time consumed. **This is the number that kills or closes the deal** — if
   their staff spent three hours babysitting to save two hours of work, you both
   need to see it.
3. **Is it getting better?** Week-over-week trend on success rate and cycle time.
   This is what justifies extending the pilot.
4. **Was it safe?** Incident log, e-stop events, near-misses. Their EHS lead reads
   this first, and it must be *complete* — hiding a stop event once destroys the
   relationship permanently.

Plus a shareable weekly summary by email or PDF, because **the person who signs
the expansion order is not the person logging into your dashboard.**

---

## 6. Sources

LeRobot async inference <https://huggingface.co/docs/lerobot/async>, blog
<https://huggingface.co/blog/async-robot-inference>, RTC guide
<https://huggingface.co/docs/lerobot/main/en/rtc>, rollout
<https://huggingface.co/docs/lerobot/main/en/inference> ·
MCAP spec <https://mcap.dev/>, Rust crate <https://crates.io/crates/mcap> ·
Foxglove pricing <https://foxglove.dev/pricing> ·
Rerun releases <https://github.com/rerun-io/rerun/releases> ·
InOrbit free edition · Mender pricing <https://mender.io/pricing> ·
balena pricing <https://www.balena.io/pricing> ·
RAUC <https://rauc.readthedocs.io/> (CVE-2026-34155 fixed in 1.15.2, 2026-03-27) ·
Cloudflare R2 pricing <https://developers.cloudflare.com/r2/pricing/> ·
Formant heartbeat pattern (developer docs) ·
Silicon Valley Robotics Center pilot guidance
