# End-to-end research: prototype → shippable robot fleet

Research date: **2026-08-08**. A field-by-field pass on what it would take to
build, deploy and operate a small commercial fleet of **mobile manipulators** —
data gathering, data generation, simulation, sim training, real training, the
app layer, deployment and telemetry.

**These are research notes, not a plan.** Nothing here has been built, bought or
committed to. Every claim carries a date and a source so the next pass can see
what has decayed, and [27-open-questions.md](27-open-questions.md) records what
this pass could *not* settle.

**Second pass: 2026-08-15.** Docs 20–27 re-swept field by field against primary
sources (search quota was again exhausted, so discovery ran on the arXiv/HF/
GitHub APIs — blog-only vendor news remains under-sampled). Each doc carries a
dated re-verification note; [29-the-company.md](29-the-company.md) records the
business thesis that pass was tested against.

Read [19-the-system.md](19-the-system.md) first; it is the map.

| Doc | Field | The one thing to remember |
|---|---|---|
| [19](19-the-system.md) | The system end to end | Four tiers, and **Tier 0/1 is what this repo is for** |
| [20](20-policies-and-models.md) | Policies and models | Third-party benchmarks contradict author self-reports by **5×** — build your own harness |
| [21](21-data-collection.md) | Data collection | Leader-arm teleop, **~50 demos per task, then switch to corrections** |
| [22](22-data-generation.md) | Data generation | **Curation beats generation.** A $50 green screen beat diffusion augmentation |
| [23](23-simulation-and-real2sim.md) | Simulation and real→sim | Three channels; **the simulator's first job is evaluation, not training** |
| [24](24-compute-and-hardware.md) | Compute and hardware | **ManiSkill3 GPU sim does not work under WSL**; no arm at this price runs 8 h/day |
| [25](25-deployment-and-fleet-ops.md) | Deployment and fleet ops | MCAP, signed release manifests, and **calibration is device state** |
| [26](26-safety-and-regulation.md) | Safety and regulation | **Keep the ML out of the safety path** — it resolves two regimes at once |
| [27](27-open-questions.md) | Open questions | What the next pass should start with |
| [28](28-wifi-on-the-chip.md) | WiFi on the chip | cyw43 can't join WPA2-Enterprise, secure boot bypassable on A2 silicon — **the chip stays tethered; WiFi belongs on a Linux node** |
| [29](29-the-company.md) | The company thesis | Scene and engine are commodities; **the unclaimed layer is the customer's robot's own dynamics** |
| [30](30-the-pipeline.md) | The pipeline design | Eleven stages, two gates — **a number from an unvalidated simulator is not evidence** |

## The five findings that changed the picture

1. **Simulation is a consumer of real data, not a postscript.** Reality reaches
   a simulator through three separate channels — the robot, the scene, the
   behaviour — and conflating them is why "just use a simulator" fails.
2. **Simulated evaluation appears to predict real policy quality better than a
   small real evaluation does** (r ≈ 0.92 versus r ≈ 0.60). That inverts the
   usual intuition about what a simulator is for.
3. **The published numbers disagree with each other by 5×** for the same model
   on the same class of hardware, and confidence intervals live in exactly one
   vendor post and one arXiv paper — in no leaderboard and no shipping
   framework (narrowed 2026-08-19).
4. **Curation and cheap augmentation beat everything generative.** World models
   generate plausible approach motion and cannot generate contact.
5. **Keeping the ML out of the safety path is worth €5–15k and 3–6 months**, and
   it is a description of the Tier 0 architecture this repository already has.

## Caveat on coverage

**WebSearch quota was exhausted before any research agent in this session issued
a query.** Everything came from direct fetches of primary sources — official
repos, model cards, arXiv, vendor pages, standards bodies. Good source quality,
poor discovery coverage: **mid-2026 releases may be missing.** See
[27 §0](27-open-questions.md).
