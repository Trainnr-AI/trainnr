# The end-to-end system: from this repo to a robot fleet

> **Archived code.** The Rust crates, firmware and emulator tools this page
> describes moved with their history to [Trainnr-AI/rig](https://github.com/Trainnr-AI/rig)
> on 2026-10-02; the paths below are relative to that repository.

Research date: **2026-08-08**. Context: what a commercial mobile-manipulator
pilot actually requires end to end — data gathering, data generation,
simulation, sim training, real training, app, deployment, telemetry — and
where `trainnr` fits in it.

> **TL;DR.** Four tiers. **Tier 0 (safety) and Tier 1 (real-time) stay Rust on
> the MCU and are what this repo is for.** Tier 2 (autonomy) and Tier 3 (fleet
> and learning) are Python, because LeRobot and MuJoCo are, and rewriting them
> would be the single most expensive mistake available. The policy runs **off
> the robot**, on a LAN GPU, because the latency budget allows it and onboard
> compute costs more than the robot. And the Tier 0 boundary turns out to be
> worth far more than it looked — see [26-safety-and-regulation.md](26-safety-and-regulation.md).

This document is the map. The twelve that follow it are the territory:

| Doc | Field |
|---|---|
| [20-policies-and-models.md](20-policies-and-models.md) | which policy to train |
| [21-data-collection.md](21-data-collection.md) | where real data comes from |
| [22-data-generation.md](22-data-generation.md) | making more data from less |
| [23-simulation-and-real2sim.md](23-simulation-and-real2sim.md) | how reality gets into the simulator |
| [24-compute-and-hardware.md](24-compute-and-hardware.md) | what to run it on, what to build |
| [25-deployment-and-fleet-ops.md](25-deployment-and-fleet-ops.md) | serving, logging, OTA, metrics |
| [26-safety-and-regulation.md](26-safety-and-regulation.md) | what the law requires |
| [27-open-questions.md](27-open-questions.md) | what we still do not know |
| [28-wifi-on-the-chip.md](28-wifi-on-the-chip.md) | why the MCU stays tethered |
| [29-the-company.md](29-the-company.md) | the business thesis and the value chain |
| [30-the-pipeline.md](30-the-pipeline.md) | the product: twelve stages, three gates |
| [31-defects.md](31-defects.md) | what review found wrong in all of the above |

---

## 1. The four tiers

Each tier is defined by **what it is allowed to fail**. That is the only
organising principle that survives contact with a real deployment.

```
┌ TIER 3 · FLEET — one GPU box on the LAN, plus cloud storage ─────────┐
│  policy inference for every robot · dataset ingest · training        │
│  simulation · evaluation harness · OTA · telemetry · consoles        │
│  MAY FAIL: yes. Seconds of outage are survivable.                    │
├ TIER 2 · AUTONOMY — a cheap Linux SBC on each robot ─────────────────┤
│  camera capture · compression · clock and session · link to Tier 3   │
│  MAY FAIL: yes, IF Tier 1 notices and stops.                         │
├ TIER 1 · REAL-TIME — the microcontroller. THIS REPO. ────────────────┤
│  joint and wheel control · odometry · 50–1000 Hz                     │
│  MAY FAIL: no. A missed deadline is a physical event.                │
├ TIER 0 · SAFETY — same MCU, cannot be overridden by anything above ──┤
│  command watchdog · velocity and torque envelope · e-stop            │
│  current and thermal limits                                          │
│  MAY FAIL: never. This is the certification boundary.                │
└──────────────────────────────────────────────────────────────────────┘
```

The arrow of trust points **downward only**. Tier 3 may *request* motion;
Tier 0 decides whether motion happens. Nothing above Tier 1 can widen a limit.

`trainnr` today is a complete Tier 1 with a partial Tier 0 — `CommandWatchdog`
exists and is tested, but no motor has ever been attached to it.

---

## 2. The latency budget, measured

The question that decides the whole hardware architecture: **can the policy
live off the robot?**

Measured breakdown, from a practitioner post on ROS Discourse (`chfritz`,
**2026-05-06**) plus LeRobot's async-inference documentation:

| Segment | Time | Note |
|---|---:|---|
| Camera sensor → USB → userspace | **~100 ms** | **dominant, and it is hardware** |
| H.264 encode | ~10 ms | |
| Network, on-premises LAN | 5–20 ms | |
| Policy forward pass | 50–150 ms | model-dependent |
| Decode + return | ~10–20 ms | |
| **Round trip** | **~150–300 ms** | |
| Local control loop | 1–10 ms | **must never depend on any of the above** |

Two conclusions follow, and both are counter-intuitive:

**The camera is the bottleneck, not the network.** ~100 ms of a ~200 ms budget
is spent before a single byte moves. Buying a faster network buys nothing; MIPI/CSI
or GigE cameras are the fix. A cloud GPU over a WAN, by contrast, spends 40–120 ms
on transit *and* has an unbounded jitter tail — so **on-site LAN beats cloud** for
latency, cost and data egress at this scale.

**200–300 ms is affordable, if you hide it.** Physical Intelligence's Real-Time
Chunking (RTC, **2026-06-09**) reports holding performance with **+200 ms of
injected latency** (total inference delay >300 ms), where synchronous inference
collapsed from **~1.0 to 0.2 tasks/minute**. LeRobot ships this as
`--inference.type=rtc`.

*Action chunking* — the policy emits a **sequence** of future actions per
inference rather than one, and the robot plays them out while the next inference
runs. Typical chunk lengths: ACT ~100, π0/π0.5 50, X-VLA 32, MolmoAct2 10.
*RTC* is the refinement that blends the overlap between consecutive chunks so
the seam is not a visible discontinuity.

So: **thin robots, one shared GPU on the LAN.** The economics of that choice are
in [24-compute-and-hardware.md](24-compute-and-hardware.md); the short version
is that onboard VLA compute costs more per robot than the robot.

---

## 3. Failover: the rule people get wrong

> **The safe-stop must not depend on the link that just failed.**

That sentence is the whole design. It is also exactly why Tier 0 lives on the
microcontroller and not on the SBC that holds the network socket.

| Layer | Mechanism | Fires within |
|---|---|---|
| **L0** hardware | Physical e-stop mushroom, dual-channel, PLd Category 3. Cuts actuator power independently of *all* software | instant |
| **L1** MCU watchdog | Microcontroller stops motion when setpoints stop arriving | **50–200 ms** |
| **L2** on-robot supervisor | Local process sees an empty action queue or a stale chunk → commanded stop | 200–500 ms |
| **L3** server heartbeat | Application-level keepalive over the policy link | ~1 s |

**L1 is `CommandWatchdog`.** It already exists in this repo, in `no_std` Rust,
on the MCU, with no allocator and no `unsafe`. That is the correct place for it
and the correct language for it.

Two further rules, both learned the hard way by others:

- **RTC hides latency, not loss.** A 200 ms delay is fine. A 2 s outage means the
  last chunk is exhausted and the robot is running blind on a stale world model.
  Do not "finish the chunk" — decelerate to a controlled stop, hold, and require
  an explicit operator resume.
- **Heartbeat tuning is a real tradeoff.** Formant's published pattern is a
  heartbeat every **50 ms** per connection over SCTP at **<0.004 Mbps** —
  fast detection, but it needs sensitivity tuning against false positives from
  ordinary network lag. Polling APIs are more reliable and take *seconds*, which
  is far too slow for a moving arm.

---

## 4. The loop that makes it a system rather than a pipeline

Data does not flow one way. The thing that compounds is the loop.

```
   ┌──────────── real robot ──── system identification ────────┐
   │            (see 23 §2: fit the simulator to THIS robot)   │
   │                                                            ▼
   │  real site ── scan + photograph ──────────────────► SIMULATOR
   │            (23 §3: appearance from splats,           │
   │             collision hand-authored, mass                  │
   │             from a scale, friction from a tilt test)       │
   │                                                            │
   │  human demos ── leader-arm teleop ──────────────► DATASET   │
   │            (21: ~50 episodes per task, then stop)     │     │
   │                                                       ▼     ▼
   │                                                     TRAINING
   │                                                       │
   │                        ┌──────────────────────────────┤
   │                        ▼                              ▼
   │                  SIM EVALUATION               REAL EVALUATION
   │            thousands of trials, cheap        tens of trials, slow
   │            → this is the GATE                → this is the TRUTH
   │                        └──────────────┬───────────────┘
   │                                       ▼
   │                                    DEPLOY
   │                                       │
   │        ┌──────────────────────────────┼──────────────────┐
   │        ▼                              ▼                  ▼
   │   failures become            human corrections      sim/real gap
   │   new sim scenarios          become training        grows → the sim
   │        (→ simulator)         data (→ dataset)       has drifted
   └────────────────────────────────────────────────────────────┘
                                                    (→ re-run sysid)
```

Three things in that diagram are the difference between a system and a demo:

1. **Corrections, not more demos.** After roughly 50 demonstrations per task,
   collecting more stops paying. Corrections during autonomous operation are
   reported at **10× the data efficiency**. See [21](21-data-collection.md).
2. **Sim evaluation as the gate, real evaluation as the truth.** Sim eval
   correlates with reality better than a small sloppy real eval does — a genuinely
   surprising result, with numbers in [23](23-simulation-and-real2sim.md) §5.
3. **The sim/real gap is a live fleet metric, not a one-time calibration.** If
   the fitted model starts diverging from the robot, the robot has changed — a
   worn gearbox announces itself as a drifting fit.

---

## 5. What `trainnr` becomes

This needs saying plainly, because it is the part that is easy to get wrong out
of attachment.

### The ML stack is Python and must stay Python

LeRobot, `mujoco.sysid`, every policy, the dataset format, the evaluation
benchmarks. There is no Rust path to any of it, and building one would consume
the entire six months to arrive at a worse version of something free. The
"end-to-end Rust" goal in the README survives *below* the policy, not above it.

### What survives, and is worth more than it looked

| Asset in the repo | Its role in the product |
|---|---|
| `CommandWatchdog`, `no_std`, `#![forbid(unsafe_code)]`, zero panics in the robot path | **L1 in the failover table**, and the deterministic safety layer that keeps the ML out of Notified Body assessment — see [26](26-safety-and-regulation.md) |
| Record/replay with divergence checking (`crates/hil-host/src/wire.rs`) | **Two products**: the system-identification harness ([23](23-simulation-and-real2sim.md) §2) and the policy evaluation harness ([21](21-data-collection.md)) |
| `RobotSpec` and its written provenance table (`crates/sim-core/src/spec.rs`) | **The parameter vector system identification writes into** — today four geometry numbers, eventually dynamics |
| `tools/verify.sh`, `tools/check-docs.py`, `crates/hil-host/examples/chip_probe.rs`, the host-vs-silicon diff | **The technical file.** This is literally what a safety assessor or a customer's EHS lead asks to see |
| The habit of shipping known mismatches as *failing tests* rather than TODOs | The reason the above is trustworthy |

The missing piece that this research surfaced: **a servo temperature and current
watchdog**. LeRobot does not have one — issue **#1319** (how to read servo
temperature and load) was closed by a **stale-bot, not a maintainer decision**
(the user did reply; see doc 24's correction), and an open **PR #3456**
implements the reading. So the gap is real but narrower than "not planned":
the primitives may land upstream, while the *watchdog policy* — what to do
when the servo runs hot — remains unbuilt anywhere, and it belongs exactly
where `CommandWatchdog` already lives. Details in
[24-compute-and-hardware.md](24-compute-and-hardware.md).

### What is superseded, and should be let go without ceremony

- **`DiffDrive` kinematics.** A LeKiwi-class base is three omni wheels
  (holonomic), not differential drive. The omni wheels are worth their ~$30
  premium for a reason beyond mobility: a base that can translate sideways lets
  the policy learn a far simpler action distribution than one that must perform
  arc-shaped repositioning to fix its approach pose.
- **`crates/vision`'s CoreML detector**, for data collection. LeRobot owns the
  camera stack, and the perception work in this repo is macOS-only. It remains
  valuable as the thing that taught the pipeline, and possibly for navigation.
- **The two bespoke recording formats.** MCAP is the fleet standard and has a
  mature Rust crate — see [25-deployment-and-fleet-ops.md](25-deployment-and-fleet-ops.md).

None of that is wasted. The Stage 0–2 work bought the two things that cannot be
bought later: a Tier 0/1 that is actually trustworthy, and the verification
reflex that makes every number in these documents something we would check
rather than believe.

---

## 6. Sources

- Latency breakdown: `chfritz`, ROS Discourse, **2026-05-06**
- Real-Time Chunking: Physical Intelligence, **2026-06-09** — <https://www.pi.website/research/real_time_chunking>
- LeRobot async inference: <https://huggingface.co/docs/lerobot/async>
- Formant heartbeat pattern: Formant developer documentation
- LeRobot issue #1319 (servo health telemetry; stale-bot closed, PR #3456 open): <https://github.com/huggingface/lerobot/issues/1319>
