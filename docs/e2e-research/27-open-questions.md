# Open questions: where this research stops

Research date: **2026-08-08**. This is the agenda for the next pass, so it
starts where this one stopped rather than re-treading it.

Each entry says what is unknown, why it matters, and what would settle it.

---

## 0. A limit on everything above

> **WebSearch quota was exhausted (200/200) before any research agent in this
> session issued a query.** Every finding in
> [19](19-the-system.md)–[26](26-safety-and-regulation.md) came from direct
> fetches of primary sources — official repositories, model cards, arXiv,
> vendor pages, standards bodies — plus a plain-HTML search proxy for discovery.

That is *good* source quality. It is *poor* discovery coverage: broad sweeps
were not possible, so **mid-2026 releases may simply be missing**. Anything
below dated after roughly June 2026 was found by guessing a URL, not by
searching.

**First action next pass: re-run discovery with search available**, specifically
for policies, teleoperation hardware, and anything named in
[20 §5](20-policies-and-models.md) as "not yet."

---

## 1. Nobody runs the comparison that decides everything

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

**What would settle it:** our own A/B — fix a task, spend N hours on
augmentation versus N hours collecting, evaluate both on the same held-out
suite with ≥50 rollouts per arm. **This is cheap to run once the evaluation
harness exists, and nobody has published it.**

---

## 2. No published simulation↔real exchange rate

**Unknown:** how many simulated demonstrations are worth one real one?

**Why it matters:** it is the number that decides how much simulator effort is
justified at all.

**State of the evidence:** searched across the robot data-scaling-law
literature — Data Scaling Laws (arXiv 2410.18647), Curse of Precision
(arXiv 2607.23108), and the co-training papers — and **no paper publishes a
conversion.** The nearest anchors:

- arXiv 2503.24361 used **50 real + 10,000 simulated per task** at a sampling
  ratio of 0.99 — a *volume* ratio of 1:200 and a *sampling* ratio of 1:99, both
  purely empirical, and removing the real data collapses it.
- **XRZero-G0** (arXiv 2604.13001, 2026-04-14) publishes the only exchange rate
  found anywhere: **10:1 robot-free human data to real-robot data matches
  real-only performance at 1/20 the acquisition cost** — and that is *human*
  data, not simulation.

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

**What would settle it:** rank 4–5 policies in simulation and on the real arm
with ≥50 rollouts each, and compute the rank correlation ourselves. **This is
directly downstream of the evaluation harness and is genuinely publishable if
it holds.**

---

## 5. What does system identification actually recover for an STS3215?

**Unknown:** which parameters `mujoco.sysid` can recover for a Feetech STS3215
servo, and how much closer that gets the simulator.

**Why it matters:** [23 §2](23-simulation-and-real2sim.md) argues this is the
open technical position — Menagerie's SO-101 model has invented actuator gains,
and `lerobot-calibrate` does no dynamics at all.

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

## 6. This repo's recordings cannot be used for system identification yet

**Unknown:** nothing — this one is known and blocking, and it is written here so
it is not rediscovered.

Every `S` line in the existing `.wire` recordings is a **simulated** encoder
tick produced by `crates/sim-core`. The hardware-in-the-loop rig has a real brain
and a simulated body. **Fitting a model against data generated by that same
model is circular.**

Three prerequisites, all real work:

1. **Real wheels and real encoders on a real motor** — H4.
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

**Is π0.5 LoRA actually trainable on a 24 GB card?** openpi says LoRA needs
>22.5 GB. That is a margin of about 1.5 GB. Settled by trying it with bf16 and
gradient checkpointing. If not, MolmoAct2 at 20.2 GiB is the fallback — see
[20 §6](20-policies-and-models.md).

**What is the real WiFi behaviour at the actual site?** Every design target in
[25 §1](25-deployment-and-fleet-ops.md) is generic. A site survey with the
racking both full and empty is the only way to know, and it is cheap.

**Does the Foxglove vendor lock matter?** Open-source Foxglove Studio was
discontinued **2024-03-11**, and self-hosted data is Enterprise-only. Whether
that becomes a problem depends on whether a customer's IT department demands
air-gap. Worth asking early.

**What does a Notified Body actually quote for an ML-driven mobile
manipulator?** The €5–15k figure is from consultancy marketing pages. Nobody in
the sources had precedent for this product class. One real quote would be worth
more than all of [26 §7](26-safety-and-regulation.md).

**Is there servo health telemetry worth standardising?** LeRobot issue #1319 was
closed "not planned." The Feetech protocol exposes temperature and load
registers. If we write that watchdog anyway — and
[24 §4](24-compute-and-hardware.md) says we must — it may be worth upstreaming.

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
  time. That was built for engineering reasons and turns out to be the most
  commercially valuable thing in the repository.
