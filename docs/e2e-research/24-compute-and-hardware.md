# Compute and hardware: what to run it on, what to build

Research date: **2026-08-08**. Prices verified on that date — they moved a lot
in 2025–26, so **verify at checkout rather than trusting this page**.
Context: one 24 GB RTX 3090 Ti already owned, in a Windows/WSL machine.

> **TL;DR.** The owned 24 GB card removes the largest line item, **but
> ManiSkill3's GPU simulation and rendering do not work under WSL** — dual-boot
> Ubuntu before this bites in month four. **Do not buy a Jetson**: prices roughly
> doubled on 2026-07-22, and a measured benchmark has SmolVLA running at
> **1.35 Hz on a $3,499 AGX Orin**. Run the policy off-robot, which is what a
> LeKiwi-class base is designed for. **Do not buy a depth camera** — every
> relevant policy is RGB-in. And **no arm at this price runs 8 hours a day**;
> that is a scoping decision, not a component choice.

---

## 1. The WSL finding — actionable today

| Workload | Under WSL2? |
|---|---|
| LeRobot training (PyTorch + CUDA) | ✅ |
| Policy serving (gRPC PolicyServer) | ✅ |
| MuJoCo, MJX, MuJoCo Playground | ✅ |
| `mujoco.sysid` | ✅ — it is CPU-batched anyway |
| **ManiSkill3 GPU simulation** | ❌ |
| **ManiSkill3 rendering** | ❌ |

ManiSkill3's own installation documentation publishes a platform matrix. WSL
reads: **"CPU Sim ✅ | GPU Sim ❌ | Rendering ❌"**. Windows is
**"CPU Sim ✅ | GPU Sim ❌ | Rendering ✅"**. The constraint comes from SAPIEN,
the underlying engine, and requires a Vulkan driver they say is best supported
on *"linux machines with NVIDIA GPUs, with limited support on other systems."*

Training and serving are unaffected. What is blocked is the **simulated
evaluation** path — and
[23-simulation-and-real2sim.md](23-simulation-and-real2sim.md) §5 argues that
evaluation is worth more than sim training, because a good simulated benchmark
predicts real performance better than a small real evaluation does.

**Recommendation: dual-boot native Ubuntu on that machine.** It is a few hours
now. Discovering it in month four is weeks.

---

## 2. What 24 GB actually fits

From LeRobot's official Compute & Hardware Guide — peak VRAM at batch size 8
with AdamW:

| Group | Policies | Peak VRAM | On a 24 GB card |
|---|---|---|---|
| Light behaviour cloning | `act`, `vqbet`, `tdmpc` | ~2–6 GB | trivial |
| Diffusion | `diffusion`, `multi_task_dit` | ~8–14 GB | comfortable |
| Small VLA | `smolvla` | ~10–16 GB | comfortable |
| Large VLA | `pi0`, `pi0_fast`, `pi05`, `xvla`, `wall_x` | ~24–40 GB | **"24 GB tight at BS 1"** |
| Multimodal | `groot`, `eo1` | ~24–40 GB | effectively no |

**π0.5 LoRA needs >22.5 GB** (openpi's own README: inference >8 GB, LoRA
>22.5 GB, full fine-tune >70 GB) — right at the edge of a 24 GB card.
**MolmoAct2's LoRA-VLM at 20.2 GiB @ bs8** has genuine headroom, and its
inference is 12.1 GiB. Full table in
[20-policies-and-models.md](20-policies-and-models.md) §2.

Training wall-clock on a 24 GB card, 5 epochs on ~50 episodes: **ACT 30–60
minutes**, diffusion 2–4 h, SmolVLA ~3–6 h.

**Rent the overflow rather than buying more card.** RunPod community pricing,
2026-08-08: RTX A5000 24 GB **$0.16/h**, RTX 3090 **$0.22/h**, RTX 4090
**$0.34/h**, A100 80 GB $1.19/h; secure tier roughly 2× that. Vast.ai runs
cheaper and less reliably (3090 **$0.07–0.15/h**). Hugging Face Jobs also takes
`lerobot-train --job.target=a10g-large` directly, billed per second.

Realistic usage — weekly training for six months with 4× overhead for failed
runs and sweeps — is **~100–150 GPU-hours**, i.e. **$25–105**. Budget **$150**.
Breakeven against a $1,000+ used 3090 is thousands of hours away.

---

## 3. Onboard compute: the 2026 price shock

NVIDIA raised Jetson prices on **2026-07-22**:

| Product | Was | Now | Change |
|---|---:|---:|---:|
| Jetson Nano module (1KU) | $99 | **$199** | +101% |
| **Orin Nano Super DevKit** | $249 | **$399** (street $409.99, 2026-08-06) | +60% |
| Orin NX 16 GB module | $599 | **$999** | +67% |
| AGX Orin DevKit | $1,999 | **$3,499** | +75% |
| AGX Thor DevKit | $3,499 | **$5,499** | +57% |

The DRAM shortage hit single-board computers too: Raspberry Pi 5 8 GB went
$80 → **$95** (2025-12-01), and secondary sources describe two further hikes in
Feb and Apr 2026 taking the 16 GB model to **~$205**. ⚠️ The official April post
did not resolve when fetched — **budget Pi 5 4 GB at $70–90 and 8 GB at
$95–130, and check the cart.**

### The measured number nobody quotes

`furuya02/jetson-smolvla-edge-bench`, running `lerobot/smolvla_base` on a
**Jetson AGX Orin 32 GB**, JetPack 6.2.1, unoptimised PyTorch:

| Power mode | Latency | Throughput |
|---|---:|---:|
| 30 W | 972.3 ms | **1.03 Hz** |
| MAXN | 739.2 ms | **1.35 Hz** |

**A $3,499 device running a 450M-parameter model at 1.35 Hz.** Expect maybe
2–4× from TensorRT and FP16, so call it 3–5 Hz optimised. The 275-TOPS figure
on the box is meaningless for transformer inference.

⚠️ Similarly, **Hailo-8's "26 TOPS at 2.5 W" is real for convolutional
networks**, and no evidence was found of transformer or VLA support in its
toolchain. Do not buy it for VLA work on the strength of the TOPS number.

### Why 1.35 Hz is nonetheless not the disqualifier

A VLA emits **action chunks**. SmolVLA's default chunk is 50 steps. At 1.35 Hz
inference with a 50-step chunk replayed at 30 fps, each inference buys **1.67 s
of motion for 0.74 s of compute** — that closes, with margin. So 1–3 Hz *is*
usable for a chunked policy.

**The disqualifier is the price, not the rate.** $399 × 3 units is $1,200 of a
$1–3k budget, for compute you do not need:

> **A LeKiwi-class base is already designed to run the policy off-robot.** The
> Raspberry Pi is a network bridge for motors and cameras; the policy lives on a
> host machine over WiFi. Onboard compute is **~$85 per robot, not $399** — and
> the latency budget in [19-the-system.md](19-the-system.md) §2 says that is
> fine.

Buy an Orin Nano Super only if a robot must operate with no WiFi at all.

---

## 4. Arms — and the durability finding

### Verified prices

Official BOM, `TheRobotStudio/SO-ARM100`:

| Configuration | US | EU | CN |
|---|---:|---:|---:|
| **Leader + follower pair** (12 servos, boards, PSUs, clamps) | **$229.88** | €226.30 | ¥1,343 |
| **Follower only** | **$121.94** | €124.30 | ¥682 |

Servos are **Feetech STS3215 7.4 V** at $13.89 each — follower is 6× at 1/345
gearing; the leader mixes 1/191, 1/345 and 1/147 so it can hold its own weight
while staying backdrivable. Excludes 3D printing. Sourced from Alibaba/Taobao,
so **4–8 week lead times**.

Retail kits that skip the printing:

| Vendor | Product | Price | Stock (as fetched) |
|---|---|---:|---|
| Seeed | SO-ARM101 Pro Kit, **assembled** | **$299** ($293 @10+) | in stock ⚠️ confirm whether leader+follower or follower only |
| Seeed | SO-ARM101 Pro servo motor kit | $260 | in stock |
| WowRobo | SO-ARM101, 1 leader + 1 follower + camera | **$199–239** | ⚠️ collection page says in stock; **product page showed every variant unavailable** |

### The durability finding, stated plainly

> **Neither Feetech's datasheet nor the SO-ARM100 repository publishes a duty
> cycle, a thermal rating, a payload specification, or an MTBF. That absence is
> the finding.**

This is a hobby servo with a magnetic encoder bolted on, not an industrial
actuator. Supporting evidence, all weak individually and consistent in direction:

- **LeRobot issue #1319** — a user asks how to read servo temperature, load and
  internal error. **Closed as "not planned", with no maintainer response.** There
  is no first-class servo health telemetry in LeRobot.
- LeRobot issue #2819 — SO-101 follower servo trouble.
- A public build log titled *"AI Robot Arm — LeRobot SO-101 — First Try and
  Burned It"*; another reporting two days debugging a dead servo board.
- A vendor forum thread explicitly framed around *"servo overheating and current
  limit settings"* — **a solicitation for community input, not data**. That
  educators needed to open it is itself the signal.
- AliExpress guidance warns that non-genuine STS3215 clones *"underperform during
  extended operation"* — **counterfeit servos are a real hazard on the cheap
  sourcing path.**

**No quantitative failure-rate or gear-wear data exists. Anyone who gives you a
number is making it up.**

⚠️ *Assessment, not a sourced claim:* the dominant failure mode is thermal
rather than gear wear, and it will hit **joint 2 (shoulder lift)** first, because
that motor holds the entire arm's static weight continuously — a stationary
loaded arm draws near-stall current and cooks. A 1:345 metal gearset will
outlast the winding.

### Mitigations that actually buy continuous duty

1. **Use 12 V servos, not the official BOM's 7.4 V.** The same task at a lower
   fraction of stall torque means dramatically less heat. Seeed "Pro" and WowRobo
   ship 12 V.
2. **Counterbalance joint 2** with a spring or gas strut so holding torque ≈ 0.
3. **Torque-disable when idle**, with a park pose between cycles. Probably the
   single biggest win.
4. **Poll the temperature and load registers yourself.** LeRobot will not.
   Feetech's protocol exposes them. **This is required work, and it belongs in
   the Tier 0 layer this repo already has** — see
   [19-the-system.md](19-the-system.md) §5.
5. **Treat servos as consumables.** $13.89 each; a full six-motor respare is $84.

### The honest conclusion

The cheapest arm with a genuinely industrial duty cycle is the **AgileX PiPER at
$3,999** (1.5 kg payload, ±0.1 mm repeatability, 626 mm reach) — **above the
entire budget for one arm**, and not natively supported by LeRobot (request
open as issue #1335).

> **Do not try to buy 8 h/day reliability. Buy SO-101s with 12 V servos,
> engineer the thermal mitigations, stock spares, and scope the pilot to 2–4 h
> of *actuated* time per day. Prove the value proposition, then spend $4k/arm
> on PiPER-class hardware with the pilot's revenue.**

### The rest of the field

| Arm | Price | DOF | Payload | Repeatability | LeRobot native |
|---|---:|---:|---|---|---|
| SO-101 | $122–299 | 6 | ~500 g practical (unspecified) | not published | ✅ flagship |
| Koch v1.1 | ~$250–480 | 6 | ~200 g | — | ✅ |
| LeKiwi (arm + base) | $482–499 BOM | 6 + 3 wheels | — | — | ✅ |
| myCobot 280 | $203–599 | 6 | 250 g | ±0.5 mm | ❌ |
| Unitree D1 | not published | 6 | 500 g | ±0.1 N force control | ❌ |
| **AgileX PiPER** | **$3,999** | 6 | **1.5 kg** | **±0.1 mm** | ❌ (issue #1335) |
| Trossen WidowX AI | from $2,995 | 6 | — | — | ✅ (ALOHA family) |
| Trossen Solo / Stationary / Mobile AI | $7,995 / $15,995 / $22,995 | — | — | — | ✅ |

⚠️ Trossen WidowX-250 and ViperX-300 are **legacy/discontinued** — the $3,550 and
$6,130 figures still circulating are stale.

---

## 5. Mobile bases

**LeKiwi**, verified BOM (`SIGRobotics-UIUC/LeKiwi`):

| Configuration | US | EU | CN |
|---|---:|---:|---:|
| Full 12 V (base + Pi 5 + 2 cameras + leader/follower arms) | **$482** | €545.80 | ¥2,891 |
| Base only, 12 V | **$251.50** | €307.80 | ¥1,501 |
| Base only, wired (no battery or Pi) | **$184** | €235 | ¥963.90 |

Or **Seeed's assembled LeKiwi Kit 12 V at $179** — base only, explicitly
excluding the arm, Pi and cameras. At $179 against $251.50 DIY that is a good
deal and skips the printing.

**XLeRobot** is $660 for dual arms on an IKEA RÅSKOG cart, assembly claimed
under 4 hours, with a $250 upgrade path from an existing SO-100 + LeKiwi. ⚠️ Note
$179.99 of that $660 is a power station, and the figure excludes printing, tools,
shipping and tax.

Above the hobby tier the numbers jump hard: AgileX LIMO from **$3,200**, Tracer
~$7,000, **Mobile ALOHA ~$32,000**, Trossen Mobile AI $22,995.

### Omniwheel versus differential drive

Manipulation from a mobile base is fundamentally a **base-positioning-error**
problem. A holonomic base (three omni wheels, as LeKiwi and XLeRobot use) can
translate sideways to fix its approach pose without re-orienting. Differential
drive forces arc-shaped repositioning, which small-dataset policies handle badly.

**Cost delta: about $30 for three omni wheels. Take them.** The real cost of
omni wheels is traction and odometry noise on carpet and thresholds — fine for
warehouse, lab and retail floors; bad for anything rough.

---

## 6. Sensors

### Depth: probably do not buy one

> **Every relevant policy is RGB-in.** SmolVLA, ACT, π0 and π0.5 all take RGB.
> The official LeKiwi BOM specifies **two $12.98 USB cameras**, not depth.
> XLeRobot's $660 configuration uses **one RGB camera**, with RealSense as a
> +$220 option. And LeRobot's own troubleshooting says the number-one camera
> problem is that **"lighting matters more than resolution"**.

**Two $13 webcams plus $40 of LED lighting beats one $334 depth camera for
imitation learning.** Buy depth only for base navigation and obstacle avoidance.

⚠️ **RealSense is not discontinued.** Widely-repeated claims to the contrary are
stale: Intel **spun RealSense out as an independent company in July 2025** and it
**raised $50M**. The D400 line is actively sold — D405/D415 $272, D435 $314,
D435i $334, D455 $419 — with a **tariff surcharge added 2026-02-03**, so the cart
will exceed those numbers.

**Orbbec Gemini 335 at $264** is the best value found: stereo, 0.1–20 m,
1280×800 @30 fps, ≤1.5% error at 2 m, IP5X, 97 g. Luxonis OAK-D Lite is $269,
OAK-D $329; the OAK-4 generation ($949–1,049) has priced itself out.

### The rest

- **Encoders: already included.** The STS3215 has magnetic absolute encoders on
  the bus. Nothing to buy.
- **IMU: get it bundled** — D435i, OAK-D Lite, or a $5 MPU-6050/BNO055.
- **Force/torque: skip.** Real 6-axis sensors are $2,000+. Proxy contact from the
  STS3215 present-load registers for free — **which you should be reading anyway
  for the thermal watchdog.**
- **Tactile: not worth it in 2026 at this budget.** No LeRobot policy natively
  consumes tactile input, so you would build the integration yourself. If you
  must, **9DTact** is the only fully open, genuinely cheap option; **AnySkin** is
  the other candidate. GelSight Mini does not publish pricing, which is an
  enterprise-quote signal.
- **UMI rig: skip.** See [21-data-collection.md](21-data-collection.md) §2.

---

## 7. Indicative BOM

⚠️ **Indicative, not a purchase plan** — research continues, and every compute
price on this page is moving.

**One unit plus safety hardware — the right starting point**

| Item | Qty | Each | Total |
|---|---:|---:|---:|
| SO-ARM101 assembled kit (Seeed) | 1 | $299 | $299 |
| LeKiwi 12 V base kit, assembled (Seeed) | 1 | $179 | $179 |
| Raspberry Pi 5 4 GB + cooler + PSU + microSD | 1 | ~$122 | $122 |
| USB webcams (wrist, top, front) | 3 | $13 | $39 |
| Powered USB hub | 1 | $16 | $16 |
| LED lighting + diffusers | 1 | $45 | $45 |
| **Spare STS3215 12 V servos** ⚠️ 4–8 wk lead — order day one | 6 | $13.89 | $83 |
| Spare control board, counterbalance strut, cabling | — | — | $96 |
| WiFi 6 access point | 1 | $110 | $110 |
| Green screen + foot pedal | — | — | $75 |
| **Dual-channel PLd Cat 3 e-stop + safety relay** (see doc 26) | 1 | — | **$600** |
| Cloud GPU overflow, 6 months | — | — | $150 |
| Contingency 12% | — | — | $217 |
| **Total** | | | **≈ $2,031** |

Scaling to three units adds roughly **$700–800 each** (arm, base, Pi, cameras,
spares) — so a three-unit fleet lands around **$3,500–3,700**, i.e. above the
$3k ceiling. **Build one, close the loop, then clone.** Cloning is exactly where
per-unit system identification and calibration earn their keep.

**Explicitly excluded, and why:** a training GPU (owned), Jetsons (policy runs
off-robot), depth cameras (policies are RGB-in), tactile and force/torque
sensors, and a UMI rig.

### Supply risk

| Risk | Severity | Mitigation |
|---|---|---|
| Feetech servos 4–8 week lead from Alibaba/Taobao | 🔴 | Order spares **day one**, in parallel |
| Counterfeit STS3215 clones | 🔴 | Buy through Seeed, not random marketplace sellers |
| Vendor stock claims contradicting their own product pages | 🟠 | Keep Seeed as the fallback at ~$100/arm more |
| DRAM crisis still active — Pi hiked 3× in 4 months, Jetson +101% | 🔴 | **Verify every compute price at checkout** |
| No servo health telemetry in LeRobot (#1319, closed) | 🔴 | Write the watchdog. Required, not optional |
| RealSense tariff surcharge since 2026-02-03 | 🟠 | Orbbec Gemini 335 at $264 instead |

---

## 8. Sources

Jetson price rises <https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/> ·
Orin Nano street price <https://pricehistory.app/p/nvidia-jetson-orin-nano-super-developer-kit-JKYHZuce> ·
Raspberry Pi price rises <https://www.raspberrypi.com/news/1gb-raspberry-pi-5-now-available-at-45-and-memory-driven-price-rises/> ·
SO-ARM100 BOM <https://github.com/TheRobotStudio/SO-ARM100> ·
LeRobot SO-101 <https://huggingface.co/docs/lerobot/so101> ·
Compute & Hardware Guide <https://huggingface.co/docs/lerobot/hardware_guide> ·
LeRobot issue #1319 <https://github.com/huggingface/lerobot/issues/1319> ·
LeRobot issue #1335 <https://github.com/huggingface/lerobot/issues/1335> ·
LeKiwi BOM <https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/BOM.md> ·
Seeed LeKiwi kit <https://www.seeedstudio.com/Lekiwi-Kit-p-6501.html> ·
Seeed SO-ARM101 <https://www.seeedstudio.com/SO-ARM-101-Assembled-Kit-Pro-p-6691.html> ·
XLeRobot <https://github.com/Vector-Wangel/XLeRobot> ·
SmolVLA on Jetson <https://github.com/furuya02/jetson-smolvla-edge-bench> ·
RunPod pricing <https://www.runpod.io/pricing> ·
RealSense spin-out <https://www.therobotreport.com/intel-spins-out-realsense-as-standalone-company/>, store <https://store.realsenseai.com/> ·
Orbbec Gemini 335 <https://store.orbbec.com/products/gemini-335> ·
Luxonis <https://shop.luxonis.com/collections/oak-cameras-1> ·
STS3215 specs <https://servodatabase.com/servo/feetech/sts3215> ·
AgileX PiPER <https://www.roboticscenter.ai/hardware/agilex-piper> ·
ManiSkill installation <https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html> ·
AnySkin <https://any-skin.github.io/> · 9DTact <https://linchangyi1.github.io/9DTact/>
