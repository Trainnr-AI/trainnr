# Open questions: where this research stops

> **Archived.** Historical (written 2026-08-08, second pass 2026-08-15, last touched
> 2026-08-28). The open questions of the 2026-08 research pass, in their
> state at that date; several were answered since. The rig they refer to
> lives in [Trainnr-AI/rig](https://github.com/Trainnr-AI/rig); `trainnr` in the text means that rig codebase. §5
> (actuator identifiability) was answered by [53](53-bam-actuator-identification.md),
> [docs/26](../26-sts3215-synthetic-identifiability.md) and the findings
> ledger ([docs/68](../68-findings.md)); Warp under WSL by [49](49-gpu-path-mjxwarp.md).
> Kept as the record.

Research date: **2026-08-08**; **second pass 2026-08-15** (six field agents,
one per doc). Entries below carry their post-second-pass state. This is the
agenda for the next pass, so it starts where the last one stopped rather than
re-treading it.

Each entry says what is unknown, why it matters, and what would settle it.

---

## 0. A limit on everything above

> **WebSearch quota was exhausted (200/200) before any research agent in this
> session issued a query.** Every finding in
> [19](19-the-system.md)–[26](26-safety-and-regulation.md) came from direct
> fetches of primary sources — official repositories, model cards, arXiv,
> vendor pages, standards bodies — plus a plain-HTML search proxy for discovery.

That is *good* source quality. It is *poor* discovery coverage: broad sweeps
were not possible, so **mid-2026 releases may simply be missing**.

**Second pass, 2026-08-15: the quota was exhausted *again* (200/200) before any
agent queried.** The workaround this time was systematic rather than
URL-guessing — date-sorted arXiv API listings, the Hugging Face and GitHub REST
APIs, Semantic Scholar citation graphs (partially blocked), plus direct
primary-source fetches. That covers **arXiv-indexed and HF/GitHub-hosted work
well** — the biggest gap of the first pass is closed — but leaves one standing
blind spot: **announcements that exist only as vendor blogs or trade press**
(commercial teleop kits, price moves, non-arXiv product launches).

**First action of the next pass, as planned 2026-08-15: a vendor/commercial-news sweep with search actually
available** — teleop kit vendors, Jetson street prices, RealSense/Orbbec/PiPER
stores (fetches to those were denied this pass), and anything in
[20 §5](20-policies-and-models.md) still marked "not yet."

---

## 1. Almost nobody runs the comparison that decides everything

**Unknown:** how does synthetic data generation compare against *the same effort
spent collecting real demonstrations*?

**Why it matters:** it is the only comparison a solo builder actually faces, and
the entire ranking in [22-data-generation.md](22-data-generation.md) is
*inference about* value per dollar rather than measurement of it.

**The state of the evidence:** baselines in this literature are deliberately
starved. MimicGen's real-robot baselines are **0%**. DemoGen's is **one
demonstration**. DreamGen explicitly does not compare against collecting more
real trajectories. A generative method beating a 1-demo baseline tells you
nothing about whether it beats 50 more demos.

**Updated 2026-08-15 — one datapoint now exists, and synthetic won.** LEGS
(arXiv 2606.01458, 2026-05-31) runs the comparison count-matched: 50 generated
episodes (3DGS re-rendering + parametrized motion primitives — splats and
primitives, *not* a video model) versus 50 teleoperated episodes, across 3
tasks × 3 VLA backbones on a Unitree G1. Synthetic **matched or beat teleop in
all 9 cells**, at ~0.5 GPU-hours versus 1.5 operator-hours, and held under
appearance shift where the teleop-trained policy "fails entirely." Caveat by
this doc set's own standard: 10 rollouts per cell is below the noise floor for
any single cell — the signal is the 9-for-9 direction.

**What remains open:** nobody has run it for a *video-model* generation
pipeline (the expensive kind), and nobody has run it dollar-matched on a cheap
arm. **An A/B of our own — fix a task, N hours of augmentation versus N hours of
collecting, ≥50 rollouts per arm — was still worth running as of 2026-08-15,
with LEGS as the published precedent to compare against; it has not been run.**

---

## 2. No published simulation↔real exchange rate

**Unknown:** how many simulated demonstrations are worth one real one?

**Why it matters:** it is the number that decides how much simulator effort is
justified at all.

**State of the evidence:** searched across the robot data-scaling-law
literature — Data Scaling Laws (arXiv 2410.18647), Curse of Precision
(arXiv 2607.23108), and the co-training papers — and **no paper publishes a
conversion.** Re-verified 2026-08-15: two fresh arXiv sweeps surfaced nothing
quantifying N sim ≈ 1 real (nearest: SimWeaver, arXiv 2606.15338, uses 200 sim
demos/task zero-shot with no conversion stated). The nearest anchors:

- arXiv 2503.24361 used **50 real + 10,000 simulated per task** at a sampling
  ratio of 0.99 — a *volume* ratio of 1:200 and a *sampling* ratio of 1:99, both
  purely empirical, and removing the real data collapses it. (Still v2,
  unchanged, re-checked 2026-08-15.)
- **XRZero-G0** (arXiv 2604.13001, 2026-04-14) publishes the only exchange rate
  found anywhere: **10:1 robot-free human data to real-robot data matches
  real-only performance at 1/20 the acquisition cost** — and that is *human*
  data, not simulation. Still unreplicated as of 2026-08-15 (full citation
  sweep: six citers, none a replication).
- New claim to track (2026-08-15): **HiFi-UMI** (arXiv 2607.25895) goes
  further than XRZero-G0 — it claims sufficiently high-fidelity robot-free
  capture removes the need for **any** real-robot anchor at post-training
  ("zero-robot post-training"). Abstract-grade evidence; if it replicates, the
  exchange-rate question changes shape entirely.

---

## 3. The co-training ratio is a knife edge with no rule

**Unknown:** how to set α, the probability that a training minibatch is drawn
from simulation.

**Why it matters:** measured on one task, **α = 0.99 gave 95% success and
α = 0.995 gave 60%.** Half a percentage point cost 35 points. There is no
principled setter, and the source paper says only *"carefully tune."*

**State of the evidence:** neither replicated nor refuted at scale. Toyota
Research Institute's 89-policy study (arXiv 2602.01067) is the largest
controlled work in the field and **does not report ratio sensitivity** — its
mixing ratios are described as *"fixed based on ablation experiments"*. Two
2026 follow-ups work *around* the problem — optimal-transport domain adaptation
(arXiv 2509.18631) and RL-based co-training (arXiv 2602.12628) — rather than
solving it.

Re-verified 2026-08-15: all four papers checked for new versions (TRI's is
still v1, abstract silent on ratio; 2602.12628 v4 adds an RL stage, no ratio
guidance), and an arXiv query for co-training + mixing ratio in cs.RO returned
**zero results**. ⚠️ Coverage caveat: the Semantic Scholar citation sweep was
blocked this pass, so a paper not matching those query terms could exist
unseen.

**Practical stance until it is solved:** treat a sweep over
{0.9, 0.95, 0.98, 0.99, 0.995} as a **mandatory, budgeted cost** of any
co-training, not an optimisation.

---

## 4. Does the simulated-evaluation correlation hold for cheap arms?

**Unknown:** whether SIMPLER's **Pearson r = 0.924** between simulated and real
policy ranking survives on a compliant $122 servo arm.

**Why it matters:** [23 §5](23-simulation-and-real2sim.md) makes simulated
evaluation the primary justification for building a simulator at all. If the
correlation is a property of rigid, well-characterised robots, that argument
weakens considerably.

**Reasons for doubt:** SIMPLER measured 6 policies from **one family** on **one
rigid robot**; "Visual Matching" means the scene was **hand-tuned** to match
reality — labour, not automation; and the comparison against RoboArena's
r ≈ 0.60 for conventional real evaluation juxtaposes **two different papers with
different setups**, so it is suggestive rather than proven.

**Updated 2026-08-15 — the one-family/hand-tuned doubt is retired; the
cheap-arm doubt stands.** **SimFoundry** (arXiv 2606.28276) reports mean
Pearson **0.911** and mean maximum ranking violation 0.018 across **7
manipulation tasks and 5 policy architectures**, from *automated* zero-shot
real-to-sim scene construction — no hand-tuning. Robot platform unnamed in the
abstract, so whether it is a cheap compliant arm is unverified. Simulated
evaluation is visibly becoming a subfield (also PolaRiS, arXiv 2512.16881;
soft-body splat evaluation, arXiv 2511.04665; ManipArena; RoboSnap). Watch
item, vendor-grade: NVIDIA's livestream claim that policy rankings are
preserved *across simulator types* (NuRec reconstruction vs the OmniDreams
world model) appears in **no paper** as of 2026-08-15 — if it ever lands in
print it is the first cross-simulator-type ranking-stability datapoint.

**What would settle it:** rank 4–5 policies in simulation and on the real arm
with ≥50 rollouts each, and compute the rank correlation ourselves. **This is
directly downstream of the evaluation harness and is genuinely publishable if
it holds.** And it got cheaper on 2026-08-15: **both halves now exist
separately in the SO-101 ecosystem** — ArmnetBench (arXiv 2607.24481) is the
real-arm evaluation farm with data released in LeRobot format, and Squint
(arXiv 2602.21203) ships a ManiSkill3 SO-101 task set. Joining them is mostly
assembly.

---

## 5. What does system identification actually recover for an STS3215?

**Unknown:** which parameters `mujoco.sysid` can recover for a Feetech STS3215
servo, and how much closer that gets the simulator.

**Why it matters:** [23 §2](23-simulation-and-real2sim.md) argues this is the
open technical position — Menagerie's own README says model grading *"will be
applied to each model once a proper system identification toolbox is created"*,
so **no Menagerie model is dynamically validated**, and `lerobot-calibrate` does
no dynamics at all.

⚠️ **Sharpened 2026-08-14 by re-reading the sources.** The engine is not the
gap: MuJoCo's sysid toolbox is free and shipping, and
`iit-DLSLab/sim2real-robot-identification` already wraps it with a published
excitation recipe — **chirp trajectories between two named keyframes**, base
fixed in air for quadrupeds or bolted to a table for manipulators. It reports
**five of six robots identified: A2, Aliengo, GO2, Piper and Z1, with HyQReal2
unfinished.**

**Every one of those is a quadruped or an industrial-grade arm** — real
encoders, torque sensing, thousands of dollars. **Nothing has been published
for a hobby-servo arm** — narrowed 2026-08-15 to: nothing *parametric*. Two
flanking results now exist. **NeuralActuator** (arXiv 2607.11734, 2026-07-13)
models actuator dynamics **on the SO-101 itself** — but as a learned
transformer surrogate, not a parameter vector ("actuator dynamics remain
underexplored and can be a major source of sim-to-real error, particularly on
low-cost platforms" — their words, this doc's thesis). And **Squint** (arXiv
2602.21203) achieves zero-shot sim-to-real manipulation on a real SO-101 by
**heavy domain randomization with no identification at all** (6–15 min
training on one RTX 3090). Identify-first-randomise-second remains unclaimed —
but the gap is being approached from both flanks, so the head start is months,
not years. Re-verified the same day: six MuJoCo releases past 3.5.0
(→ 3.11.0), none touching sysid; `trs_so_arm100` in Menagerie unchanged; the
IIT roster unchanged (HyQReal2 still unfinished). One useful addition in
MuJoCo's unreleased changelog: a **PID actuator with integral action and
setpoint rate limiting** — directly relevant to servo modelling.

So the question is not "can this be done" but specifically: *does parametric
identification converge on a machine with backlash, no torque feedback and a
plastic gearbox?*

**Answered, in part (2026-08-28, [53-bam-actuator-identification.md](53-bam-actuator-identification.md)).**
[BAM](https://github.com/Rhoban/bam) (ICRA 2025, Apache-2.0) is a published
identification pipeline — pendulum bench, CMA-ES fit, a six-model friction
hierarchy beyond Coulomb-Viscous — that ships a working fit for **Feetech
STS3215 (7.4V)** among six other servos, oscilloscope-measured firmware
constants included, raw trajectories downloadable. It converges: real
parameters, real convergence, on hardware in exactly this class. What it does
NOT answer: whether identification converges on *our* arm, in *our* rig
configuration, with *our* electronics — BAM's numbers are from their own
bench, not ours. The narrower question (can OUR configuration be identified)
stays open; the broader one (has anyone even tried, does it converge at all)
is closed.

**The nonlinearity budget now has numbers (added 2026-08-23).** From a
third-party bench test of one STS3215-12V (a video summary,
not archived — re-grade when linked):

| Quantity | Measured | Against |
|---|---|---|
| Encoder resolution | 12-bit, 4,096 counts/rev = **0.088°/count** | 4.3× finer than the rig drivetrain's 960 ticks/rev — the planned identifiability paper's (which became [docs/26](../26-sts3215-synthetic-identifiability.md)) ~2%/~5% quantization bias should shrink accordingly |
| Backlash | **0.0151 rad ≈ 0.87°** (~10 counts) | ~2× the < 0.5° datasheet spec |
| Firmware dead zone | **10 encoder counts ≈ 0.88°** | motions inside it are *invisible to the encoder output by firmware choice*, not physics |
| Repeatability | ±0.17° (~2 counts) on a 10 cm arm | the effective noise floor |
| Low-speed behaviour | ~7% speed fluctuation with oscillation | velocity-dependent, matters for chirp design |
| Overload governor | throttles to **~20% of rated torque** at ~2/3 rated load | **a hidden actuator clamp: any excitation that crosses it poisons the fit silently** |

Consequence for the sub-questions below: backlash and the dead zone stack to
**~1.7–1.8° of nonlinearity — a 20-count blind band against 0.088° resolution**
— so excitation amplitudes must dwarf it, and the deadband question is now
two questions (mechanical backlash AND a firmware dead zone, different
mechanisms, similar magnitude, both ~10 counts).

**Specific sub-questions:**
- Is **backlash** capturable as a combination of joint damping, `armature` and
  friction loss, or does it need an explicit model?
- Does **deadband** — the duty below which the motor does not turn at all — need
  a separate term, or does the fitted P/D gain absorb it?
- Do the six joints need six independent fits, and does **left/right asymmetry**
  matter enough to model?
- How stable is the fit over time? **A drifting fit should be a worn gearbox** —
  is it, measurably?

**Blocked on:** real encoders. See §6.

---

## 6. The rig's recordings could not be used for system identification yet

*This section is about the rig's own recordings (`recordings/`), made by the
archived crates; the platform identifies robots from their telemetry
instead ([docs/76](../76-the-loop.md)).*

**Unknown:** nothing — this one is known and blocking, and it is written here so
it is not rediscovered.

Every `S` line in the existing `.wire` recordings is a **simulated** encoder
tick produced by `crates/sim-core`. The hardware-in-the-loop rig has a real brain
and a simulated body. **Fitting a model against data generated by that same
model is circular.**

Three prerequisites, all real work:

1. **Real wheels and real encoders on a real motor** — H4 of [23](23-simulation-and-real2sim.md).
2. **Absolute timestamps in the recorder.** Today time is inferred from a fixed
   tick interval, which holds under lock-step and breaks under free-running
   hardware.
3. **A parameter vector to fit into.** `RobotSpec` in
   `crates/sim-core/src/spec.rs` holds four numbers, all geometry. No mass, no
   inertia, no friction, no torque constant.

---

## 7. Which wedge

**Unknown:** the use case, and therefore the object set, precision requirement,
duty cycle and safety case.

**Why it matters:** it is gating. [21](21-data-collection.md)'s variation ladder
cannot be aimed without an object set, and [26](26-safety-and-regulation.md)'s
risk assessment cannot be written without a workspace.

*Context, 2026-08-15:* the plan of this date sketched the
loop the wedge must serve — customer scan → sim-first training →
on-site teleop calibration, hardware-agnostic via a per-robot identification
onboarding step. The wedge question is unchanged by it, but any candidate
wedge should now also be scored on how well it fits that loop (scan-able site,
demo-able task, intervention-tolerant workflow).

**Two disqualifiers already established by the research, which usefully narrow
the search:**

- **Sub-millimetre insertion or assembly.** Curse of Precision
  (arXiv 2607.23108): required demonstrations grow **super-exponentially** as
  precision approaches a limit set by sensors and hardware. If the task needs
  this, go to HIL-SERL rather than data — or pick a different task.
- **Deformables** — cloth, laundry, bedsheets. Not solved in 2026. The only
  credible tooling (PhysTwin) is videos-of-deformables-only with undisclosed
  cost.

**And one hard constraint from [24 §4](24-compute-and-hardware.md):** no arm at
this price runs 8 hours a day. **The wedge must tolerate 2–4 hours of actuated
time per day**, or it is the wrong wedge for this hardware generation.

---

## 8. Smaller open items

**Does MuJoCo Warp (and `mjwarp-render`) run under WSL2?** (filed 2026-08-23;
**answered 2026-08-27**: Warp runs on the WSL GPU, [49](49-gpu-path-mjxwarp.md)
postscript 2). It replaced the ManiSkill3-WSL finding whose force decayed when
Gate A (the go/no-go gate on simulated-versus-real ranking) moved to MuJoCo
([24 §1](24-compute-and-hardware.md)). The related question, whether a fitted
model runs under Warp's supported actuator subset or falls back to CPU
rollouts, was answered by the identified-actuator work in
[docs/76](../76-the-loop.md).

**Is π0.5 LoRA actually trainable on a 24 GB card?** **Updated 2026-08-15:**
still unresolved, and the evidence tilts worse. openpi's README is unchanged
(LoRA >22.5 GB), and openpi issue #677 is a live OOM report **on a 4090
running the LoRA variants**, with `gradient_checkpointing=True` as the
maintainers' standard remedy — no published user config confirms a
comfortable fit. **MolmoAct2 strengthened as the fallback** (not the default: the later
read, [24 §2](24-compute-and-hardware.md) of 2026-08-23, holds — on ArmnetBench
it scores 18.9% against π0.5's 47.6%):
its numbers now live in official LeRobot docs, an even cheaper
action-expert-only fine-tune exists at 16.5 GiB @ bs8, and a ready-made
zero-shot SO-100/101 checkpoint (`lerobot/MolmoAct2-SO100_101-LeRobot`) runs
via `lerobot-rollout` — see [20 §6](20-policies-and-models.md) and
[24](24-compute-and-hardware.md). Still settled only by trying π0.5 LoRA
directly; budget for it to fail.

**What is the real WiFi behaviour at the actual site?** Every design target in
[25 §1](25-deployment-and-fleet-ops.md) is generic. A site survey with the
racking both full and empty is the only way to know, and it is cheap.

**Does the Foxglove vendor lock matter?** Open-source Foxglove Studio was
discontinued **2024-03-11**, and self-hosted data is Enterprise-only. Whether
that becomes a problem depends on whether a customer's IT department demands
air-gap. Worth asking early.

**What does a Notified Body actually quote for an ML-driven mobile
manipulator?** The €5–15k figure is from consultancy marketing pages. Nobody in
the sources had precedent for this product class. **Updated 2026-08-15:** the
first official procedure-level guidance now exists — RfU CNB/M/00.514 Rev 02
([26 §2](26-safety-and-regulation.md)) settles *who* assesses ML safety
components (a machinery Notified Body, not a full AI Act one) — but no actual
quote or assessment example for this product class has been found anywhere.
One real quote would still be worth more than all of the marketing-page
estimates in [26 §7](26-safety-and-regulation.md).

**Is there servo health telemetry worth standardising?** **Updated 2026-08-15:**
the upstreaming calculus improved. LeRobot issue #1319 was closed by a
**stale bot**, not a maintainer decision — nobody championed it, rather than
LeRobot rejecting it — and an **open, unmerged PR (#3456, since 2026-04-24)**
already implements it: opt-in `record_telemetry` reading Feetech velocity
(register 58), load (register 60) and temperature (register 63), costing
~15–30 ms per 6-motor bus. If we write our own watchdog against those exact
registers — [24 §4](24-compute-and-hardware.md) says we must — reviving #3456
with our measurements attached is now a real option, not a cold pitch.

---

## 9. Things this research changed its mind about

Recorded because they are the entries most likely to be wrong again.

- **Simulation is a consumer of real data, not a postscript.** The first draft
  of this research treated it as an augmentation step at the end. It is the
  spine, and the three channels in [23 §1](23-simulation-and-real2sim.md) are
  the structure that was missing.
- **The simulator's first job is evaluation, not training.** A simulated
  benchmark appears to predict real policy quality *better* than a small real
  evaluation does — which is the opposite of the intuition everyone starts with.
- **Third-party benchmarks contradict author self-reports by 5×.** SmolVLA:
  78.3% in its own paper, 15.0% in ArmnetBench. This is why
  [20 §1](20-policies-and-models.md) leads with "build your own harness."
- **Curation beats generation.** Removing bad demonstrations is worth +15–35
  points for free; a $50 green screen beat diffusion-based augmentation 91% to
  75%.
- **The Tier 0 safety work has a regulatory value nobody was designing for.**
  Keeping the ML out of the safety path is the difference between self-
  certification and a Notified Body, and it resolves the AI Act at the same
  time. That was built for engineering reasons in the archived rig and turned out
  to carry a regulatory value of its own.
- **Regulatory claims decay fastest of anything measured in this doc set.**
  Doc 26's own in-force-date warning ("unsettled… verify before relying on
  it") settled within a week, and the underlying mechanism it described
  changed structurally within the month (Regulation (EU) 2026/1744 moved the
  Machinery Regulation to Annex I Section B — [26 §1](26-safety-and-regulation.md)).
  The engineering conclusion survived; the legal mechanics did not. Treat
  every regulatory citation here as the most perishable claim in the doc set.
- **"Splatting buys appearance, never physics" needed a product-reality
  update within the same week it was written.** Niantic's Scaniverse now ships
  a splat with a co-registered collision mesh in one file
  ([23 §3](23-simulation-and-real2sim.md)). The underlying physics claim still
  holds — the splat itself is not the physics — but a practical objection this
  doc treated as settled turned out to be a shipping-product away from
  obsolete. Re-verify "impossible" claims before repeating them, not just
  "unlikely" ones.
