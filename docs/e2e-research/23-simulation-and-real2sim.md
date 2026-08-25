# Simulation and real→sim: how reality gets into the simulator

Research date: **2026-08-08**. Question: what is the fastest, cheapest path
from a real commercial site and a real low-cost mobile manipulator to a
simulator good enough that using it improves real-world performance?

Re-verified and extended **2026-08-15** (second discovery pass). ⚠️ WebSearch
quota was exhausted again before the sweep started; discovery ran on the arXiv
API, GitHub/HF APIs and direct primary-source fetches, and Semantic Scholar was
blocked — so citation-graph coverage is incomplete and blog-only announcements
are under-sampled. Changes are folded in place, dated.

> **TL;DR.** Reality reaches a simulator through **three separate channels**
> that get conflated and shouldn't be: the **robot** (system identification),
> the **scene** (geometry, appearance, mass, friction), and the **behaviour**
> (demonstrations). Channel A is the cheapest, the highest-value, and the one
> nobody has done for an SO-101 — **MuJoCo shipped a first-party system
> identification toolbox in 3.5.0 (2026-02-12)** and Menagerie's own README
> says its grading *"will be applied to each model once a proper system
> identification toolbox is created"* — so **no Menagerie model is dynamically
> validated**, SO-101 included. (Re-checked 2026-08-15: still true for
> *parametric* sysid — but the gap is now being approached from the
> learned-surrogate and domain-randomisation sides; see §2.) **Splatting buys
> appearance, never physics** — though as of mid-2026 Niantic ships a splat
> *plus* an auto-derived collision mesh in one file, so the practical objection
> has narrowed (see §3). Precise digital twins of objects are *worse* than
> approximate ones.
> And the strongest argument for building a simulator at all is that
> **simulated evaluation predicts real performance better than a small real
> evaluation does.**

---

## 1. The three channels

Different things flow from the real world into a simulator. They have different
mechanisms, different costs and different cadences, and "just use a simulator"
fails because it treats them as one thing.

```
  CHANNEL A — THE ROBOT               CHANNEL B — THE SCENE
  ────────────────────────            ─────────────────────
  motor lag, deadband,                room layout, object shapes,
  gearbox backlash, joint             MASS, FRICTION, lighting,
  friction, encoder scale,            textures
  left/right asymmetry,
  loop latency                        how: phone capture + hand-authored
                                           collision + a kitchen scale
  how: drive a known signal,               + a tilt test
       log commanded input and        cost: an afternoon per site,
       measured output, FIT                ~2 min per object
  cost: a few hours, then free        cadence: once per site
  cadence: once per unit, then
           re-checked on a schedule
       ┌────────────────────────────────────────┐
       │        CHANNEL C — THE BEHAVIOUR       │
       │  what a good trajectory looks like     │
       │  how: leader-arm teleop, then          │
       │       corrections (see doc 21)         │
       └────────────────────────────────────────┘
```

**Channel A gates the others.** Domain randomisation — deliberately varying
simulator parameters so the policy is robust to the ones you got wrong — is
*worse than useless* when it is centred on a guessed value, because it teaches
robustness to the wrong distribution. **Identify first, randomise second.**

---

## 2. Channel A — system identification, and why it is the wedge

### The tool exists, is first-party, and is new

**MuJoCo 3.5.0 (2026-02-12) added an official System Identification toolbox**
in Python, with a Colab notebook. From `python/mujoco/sysid/README.md`:

- **Method:** nonlinear least squares with box constraints, Gauss-Newton,
  finite-difference Jacobians. Every parameter perturbation runs as a **single
  batched `mujoco.rollout` call parallelised across threads** — fast on CPU,
  **no GPU needed**.
- **Fits:** contact friction coefficients, joint damping, body mass and full
  inertia (pseudo-inertia Cholesky parameterisation, which keeps the inertia
  matrix physically valid and non-singular), **actuator P/D gains**, and
  measurement-model parameters — **sensor delays, measurement gains, biases**.
- **Output:** an interactive HTML report with videos, measured-versus-predicted
  comparisons, parameter tables, **and confidence intervals**.

Sensor delay + P/D gains + damping + friction is almost exactly the parameter
set that dominates a cheap servo arm.

### Nobody has done it for an SO-101, and the gap is documented

- **MuJoCo Menagerie's `trs_so_arm100`** README describes it as a
  **"simplified"** MJCF derived from the public URDF, with position actuators
  added in a manual conversion. **No system identification, no experimental
  validation.**

  ⚠️ **Corrected 2026-08-14.** This bullet previously read *"its actuator gains
  are invented and its inertias are inherited from CAD."* Re-fetching the README
  does not support that: it says **nothing whatsoever about where any parameter
  came from.** "Invented" was an inference stated as a quote.

  The accurate claim is weaker to write and stronger to hold: **the README is
  silent on parameter provenance, so the numbers are unattributable.** You
  cannot tell whether a gain was measured, guessed or inherited — and an
  unattributable number is not usable as a prior, because you do not know which
  direction it is wrong in. The sourced evidence for the same conclusion is the
  next bullet, which is a direct quote and covers every model in the repository.
- Menagerie's top-level README states its model-quality grading system
  *"will be applied to each model once a proper system identification toolbox is
  created"* — i.e. as written, **essentially no Menagerie model is dynamically
  validated.**
- **LeRobot's `lerobot-calibrate` is pure kinematic homing**: move to mid-range,
  sweep each joint through its range, record the limits. There is **zero**
  dynamics identification anywhere in the official SO-101 documentation — no
  torque constant, no friction, no lag, no gain fitting.

So everyone in the SO-101 ecosystem is running an unvalidated dynamics model.
That is the open, unclaimed technical position, and it is the one this repository
is already shaped for.

### Re-checked 2026-08-15: still unclaimed, but no longer untouched

- **MuJoCo shipped six releases past 3.5.0 (3.6.0 → 3.11.0, 2026-07-27) and
  none of them touch sysid** — the toolbox commits since March 2026 are
  stabilisation only. One unreleased changelog item matters for servo modelling:
  a **PID actuator with integral action and setpoint rate limiting**.
- Still **no published `mujoco.sysid` application to any hobby servo**
  (STS3215, SO-100/101, SG90-class); Menagerie's `trs_so_arm100` has seen XML
  formatting commits only; the IIT sim2real-identification roster is unchanged
  (five quadrupeds/industrial arms, HyQReal2 still unfinished).
- But two 2026 papers now approach the same gap from other directions.
  **NeuralActuator** (arXiv 2607.11734, 2026-07-13) publishes a *learned*
  actuator-dynamics model validated on the SO-101 — its own framing is ours:
  *"actuator dynamics… can be a major source of sim-to-real error, particularly
  on low-cost platforms."* A transformer surrogate, not a parametric fit — no
  interpretable friction/damping/gain numbers to carry into a simulator.
  **Squint** (arXiv 2602.21203, 2026-02-24) gets zero-shot sim-to-real
  manipulation onto a real SO-101 via **heavy domain randomisation with no
  identification at all**, training visual SAC in 6–15 min on one RTX 3090.

**The narrow claim that survives: nobody has run parametric system
identification on a hobby-servo arm.** The identify-first-randomise-second
position is still open — but the ecosystem is circling it, so it is a head
start measured in months, not years.

### What `robotiq` already has, and the one thing that blocks it

The audit finding: **the repo already logs the exact input/output pair system
identification needs.** From `crates/hil-protocol/src/lib.rs`, every 20 ms tick
carries

- `M duty_l duty_r` — the **commanded** motor duty, chip → host
- `S dl dr` — the **measured** encoder tick deltas, host → chip

lock-stepped 1:1 at 50 Hz, formatted to four decimals, and already parsed,
recorded and replayed by `crates/hil-host/src/wire.rs` with a mutation-verified
divergence checker. Commanded input and measured output, aligned in time, is
precisely what a fit consumes.

**Three things block using it today, and they should be written down plainly:**

1. **The existing recordings are circular.** Every `S` line in
   `recordings/rp2350-utrap.wire` is a *simulated* encoder tick produced by
   `crates/sim-core` — the HIL rig has a real brain and a simulated body.
   Fitting a model against data generated by that same model proves nothing.
   **Real wheels and real encoders are a hard prerequisite.**
2. **There are no timestamps.** Time is inferred from a fixed `tick_seconds`.
   Under free-running hardware that assumption breaks.
3. **There is nothing to fit into.** `RobotSpec` in `crates/sim-core/src/spec.rs`
   holds **four numbers, all geometry** — wheel radius, track width, ticks per
   revolution, maximum wheel speed. No mass, no inertia, no friction, no torque
   constant. The parameter vector has to exist before it can be identified.

And the actuator model that would be fitted is, today, eight lines in
`crates/sim-core/src/motor.rs`: a velocity clamp and a first-order lag. Its time
constant is `0.15 s` — **an unsourced guess sitting in a defaults literal**, with
no deadband, no stall, no acceleration limit, no back-EMF and no left/right
asymmetry. `docs/10-hil-protocol.md` already says so out loud:
*"motor dynamics are a first-order lag model, not a real motor."*

**The honest summary: this repo has the mechanism for real→sim and has never
pointed it at reality.** H4 — measuring the real robot when a motor arrives — is
therefore not a chore. It is the first system identification, and it is worth
building as a repeatable procedure with a written acceptance threshold rather
than as an afternoon of tuning.

---

## 3. Channel B — the scene, in a counter-intuitive order

### Do not reconstruct manipulable objects precisely

**ACDC / Digital Cousins** (arXiv 2410.07408, CoRL 2024) is the load-bearing
result. Input: **one RGB image**. Pipeline: depth estimation and segmentation
detect the objects, then each is matched to a **nearest-neighbour asset from a
library** — retrieval, not reconstruction — producing a fully interactive scene.
Generate several variants and you have *digital cousins* rather than a digital
twin.

> Reported zero-shot sim-to-real success: **90% for digital cousins versus 25%
> for the digital twin** (IKEA cabinet door opening, 50/20 trials).

The authors' framing is the insight: an approximate scene generated *as a
distribution* provides **implicit domain randomisation for free**. Precision is
worse *and* slower. ⚠️ It depends on OmniGibson and the BEHAVIOR asset dataset,
and wants **24+ GB VRAM**.

### Gaussian splatting gives appearance, never physics

This needs saying because the opposite is widely assumed.

> **2026-08-25 re-check — the verdict is amended.** The architecture
> claim stands (no manipulation training simulator computes contact on
> Gaussian kernels; every shipped system keeps a mesh/particle proxy),
> but as strategy it now misleads: the proxy is generated, co-registered
> and in the best work co-trained *through* the splat pipeline — Isaac
> Sim 6.0 ships splat+collision-proxy pairs natively, GASE closes the
> scan-to-policy gap to <10%, and DeepMind jointly optimizes splats +
> physics meshes in differentiable MuJoCo. Full findings, licences and
> the amended wording: [34-physics-splats-2026.md](34-physics-splats-2026.md).

| Tool | What it produces | Collision geometry? |
|---|---|---|
| **gsplat** v1.5.3 (Apache-2.0) | CUDA rasterisation of Gaussians; recent work is HiGS, MCMC, LiDAR rasterisation, 3DGUT | **No. Mesh extraction is not documented in the repo at all.** |
| **SplatSim** (arXiv 2409.10161) | replaces mesh *rendering* with splats **inside an existing simulator** — physics still comes from conventional meshes | No (inherits the simulator's) |
| **PhysGaussian** (arXiv 2311.12198) | custom material-point-method on Gaussian kernels, "what you see is what you simulate" | A *graphics* result; no robotics collision story |
| **Splat-MOVER** | semantics + grasp affordance + scene editing; 95–100% grasp success on specific objects | **No physics** |

SplatSim's headline — **86.25% average zero-shot sim2real versus 97.5% for
policies trained on real data** — is the strongest splat-based number in the
literature, and it is a *rendering* result layered on conventional physics.

**Practical reading: splatting is a phase-7 visual-gap optimisation, after you
have a working policy.** For collision, hand-author primitives from a floorplan
and a tape measure. A warehouse aisle *is* boxes; thirty minutes of measuring
beats an hour of mesh cleanup and produces something a solver can actually use.

#### Mid-2026 update: the collision-mesh objection has been productised away

The headline above survives in letter — the splat itself is still never the
physics — but the table's practical objection ("no collision geometry") is now
answered by a shipping product. Verified at nianticspatial.com/robotics,
2026-08-15:

> *"The USDZ export in Scaniverse turns a five-minute 360 capture into a
> simulation-ready environment for NVIDIA Isaac Sim and Isaac Lab; a Gaussian
> splat and an aligned mesh in a single file."*

**Niantic's Scaniverse export co-packages the splat (RGB layer) with a
co-registered collision mesh** — *"The mesh, derived from the same capture,
supplies the collision geometry it navigates"* (nianticspatial.com/robotics,
re-verified 2026-08-17). Because both layers come from the same geometry there
is no registration error for a policy to learn as false signal.

⚠️ **Attribution corrected 2026-08-17.** The **MVSAnywhere** mechanism and the
"gravity-aligned, metric-scale, collider-ready" phrasing come from **Flexion's
vendor post, not from Niantic** — a re-fetch of the Niantic robotics page finds
none of those four terms on it. City-block-scale tiling remains livestream-only
and appears on no public page.

The first policy result inside such a twin is **Flexion's** (vendor post,
flexion.ai, 2026-07-20 — **no paper**): massively parallel RL on rendered RGB
inside the office reconstruction, zero-shot to the real robot. Grade the
evidence carefully: the quantitative numbers are **in-sim** (RGB 97.8% vs
depth 93.8% in their office; 75.0% vs 70.9% in Niantic's), the post itself says
real-world performance is *"on par with a depth-based policy"*, and the
glass-door / thin-structure / semantic-hazard wins are **demo-video grade**.
The direction — reconstruct realism instead of randomising toward it — is
notable; the margins are not yet measured anywhere citable.

#### 2026-08-17: splats standardised, and the licence trap is the opposite way round

A dedicated sweep of OpenUSD ↔ MuJoCo ↔ 3DGS found the interoperability story
had changed under us, in our favour, and found the risk sitting somewhere other
than where this doc had been pointing.

**Gaussian splats are now a core OpenUSD schema, not a vendor extension.**
OpenUSD **v26.03 (2026-02-24)** added the `UsdVolParticleField` family
including **`ParticleField3DGaussianSplat`** — verified present in
`pxr/usd/usdVol/schema.usda` at v26.03 through v26.08 and **absent at v25.11**.
The same release shipped an open-source reference splat renderer
(`extras/imaging/examples/hdParticleField`) plus PLY→USD and SPZ→USD
converters; v26.05 added a scene-index filter that **degrades particle fields
to plain points** for any renderer that cannot splat. OpenUSD's licence is the
Tomorrow Open Source Technology License 1.0 — Apache-2.0 verbatim except the
trademark clause. **And NVIDIA is deprecating its own format in favour of it**:
the `3dgrut` export README states *"NuRec is going to be deprecated and
replaced by `ParticleField`. Prefer `ParticleField` for new assets."*

**The permissive rendering column is real.** `gsplat` (Apache-2.0, actively
maintained) ships a full robot-sensor stack — OpenCV pinhole, f-theta and
fisheye projections, **five rolling-shutter modes**, and spinning-LiDAR
projection, in differentiable CUDA kernels. NVIDIA's own `3dgrut` (3DGRT +
3DGUT) is Apache-2.0; `brush` is Apache-2.0 and needs no CUDA; Niantic's `spz`
container is MIT.

> ⚠️ **The licence trap is the mesh path, not the appearance path.** This doc
> has been warning about NVIDIA lock-in for *rendering*. But rendering has
> permissive options, while **SuGaR, 2DGS and GOF — the entire splat→mesh
> state of the art — all ship the INRIA research-only licence**: *"THE USER
> CANNOT USE, EXPLOIT OR DISTRIBUTE THE SOFTWARE FOR COMMERCIAL PURPOSES."*
> The path that *sounds* physics-respecting is the one that is commercially
> unusable.

**And their output is not simulation-grade anyway, by their authors' own
words.** SuGaR runs Poisson reconstruction over a sampled level set and
concedes *"splatted depth maps are not exact"* — Poisson output is watertight
*by construction*, so watertightness there is an artifact of the algorithm, not
evidence about the scene. 2DGS concedes failure on *"semi-transparent surfaces,
such as glass"* and *"fine geometric structures"* — precisely the two classes
Flexion's demo videos claim as wins. The 2026 successor (Manifold-GS,
arXiv 2608.00214) exists specifically to name the failure mode: **watertight
mesh hallucination**, inventing surface in unobserved regions.

**The finding that matters most for evaluation: off-trajectory decay.** A
policy under evaluation moves the camera to novel poses *by definition*, and
that is exactly where a splat is weakest. Measured: on genuinely
out-of-distribution splits, 3DGS scores **PSNR 17.66 / FID 113.84** (Difix3D+,
arXiv 2503.01774), repairable to FID ~42 — still not photorealistic. EVPGS
quantifies the deviation as **~25° of pitch** off the training views, which is
a trivially small move for a wrist camera. **An evaluation harness whose
fidelity is worst exactly where the policy is most interesting will produce
ranking noise and call it a result** — the failure §5 exists to prevent.

**Two working MuJoCo+3DGS precedents exist, both permissive**, which adds a
third option to the fork in [30 §①](30-the-pipeline.md): **DISCOVERSE**
(MIT, IROS 2025, MuJoCo physics + 3DGS rendering, no INRIA or NVIDIA code in
its dependency chain) and **MuGS** (Apache-2.0; MuJoCo renders robot and
objects, gsplat renders background, alpha-composited; **515 Hz** end-to-end at
160×120 on one RTX 4090). Both are 3–5 months stale, so budget maintenance.

**What MuJoCo itself does and does not have.** It gained a real **USD importer**
(`plugin/usd_decoder`, ~2,800 lines) promoted out of experimental in **3.5.0**,
so a `.usdz` can be dropped straight into `simulate`. It has **zero** splat
code — a code search for "gaussian" and "splat" across the repository returns
nothing, with no PR and no roadmap item. Consequence, from reading the importer
source: it gates geometry on `UsdGeomGprim`, and `ParticleField` is not one —
so **dropping a Niantic USDZ into MuJoCo today should yield the collision mesh
and silently skip the splat**, which is arguably exactly what we want. ⚠️
Inferred from the source gate, not observed at runtime — worth one test.

### The two numbers nobody automates

**Mass** and **friction** dominate manipulation-sim fidelity, and there is no
turnkey 2026 tool for either. Mass comes from a kitchen scale. Friction comes
from a tilt test — raise one end of a board until the object slides, and the
coefficient is the tangent of that angle. Roughly $20 of equipment for the two
numbers that matter most.

### Other approaches that do produce physics

- **Articulate-Anything** — text, image or **video** → Python programs compiling
  to **articulated URDF**. **75% joint-prediction success on PartNet-Mobility
  against 8.7–12.2% for prior methods**, validated by training in simulation and
  transferring to a real Franka. The best video→articulated-asset tool found.
- **RialTo** (arXiv 2403.03949) — scan, construct the scene in a GUI, transfer
  real demos into simulation, fine-tune with RL, distil back with real
  co-training. **>67% increase in policy robustness** over imitation baselines
  across 7 tasks. ⚠️ Per-scene construction time is **not reported anywhere** —
  a conspicuous omission, and the GUI step implies substantial manual work. Take
  the idea, not the tooling.
- **PhysTwin** — sparse videos of *deformables* under interaction → geometry
  plus dense physical properties. Reconstruction time undisclosed. Deformables
  only.
- **EmbodiedGen** (arXiv 2506.10600) — image/text→3D, articulated object and
  scene generation with **URDF export carrying physical properties and real-world
  scale**. ⚠️ No quantitative benchmarks in the abstract; treat as a convenience
  toolkit, not a validated method.
- **RoboCasa365 v1.0.1** (2026-05-12) — 2,500 kitchen scenes, 3,200+ objects
  across 150+ categories, 600+ h human and 1,600+ h synthetic demos. **No
  sim2real transfer results reported.** An *asset library* to mine, not a method
  to adopt.

### What the AV domain shipped (labelled: AV, not manipulation)

Autonomous driving runs the same Channel-B pipeline at industrial scale, and
its mid-2026 state previews where robotics tooling goes. NVIDIA's own framing
is a useful taxonomy: **sim 1.0** artist-built scenes → **sim 2.0** neural
reconstruction (NeRF, splats) → **sim 3.0** generative world models — with the
caveat that manipulation *contact* is still stuck between 2.0 and 3.0
([22-data-generation.md](22-data-generation.md) §3).

- **The cost collapse has a first-party number — but it is livestream-grade.**
  M City (Univ. of Michigan) rebuilt its 30-acre test-track twin with NuRec for
  **~$2,300 in 2 days** against **$150k and 6 months** for the hand-built
  original ($90M extrapolated to the city of Ann Arbor). ⚠️ Stated on an NVIDIA
  livestream by the M City team, 2026-08; their repo
  (`mcity/mcity-digital-twin`, MIT) contains **no NuRec writeup** — no citable
  source exists yet.
- **Splat degradation off the capture trajectory has a productised repair
  stack**, verified on Hugging Face 2026-08-15: `nvidia/difix` (DiFix3D+,
  arXiv 2503.01774; **~720k combined downloads**; ⚠️ that figure is difix's, not
  instant-nurec's — see below), `nvidia/asset-harvester`
  (3D asset from one image; licence "other"), `nvidia/Harmonizer` (relighting
  inserted assets; licence "other"), `nvidia/instant-nurec` (**335 downloads**, licence "other"; feed-forward splat,
  PLY in <2 min, **NVIDIA Open Model License — commercial use allowed**). The
  reconstruct-the-backdrop-inject-synthetic-actors workflow is assembled and
  shipping in AV; robotics gets it second-hand.
- **"Cosmos Streams" is publicly OmniDreams** (`nv-tlabs/omni-dreams`,
  Apache-2.0; arXiv 2606.03159): a real-time action-conditioned autoregressive
  video world model for closed-loop AV simulation. ⚠️ The livestream claims
  that matter most here — 30 fps on an RTX 6000 Pro, and **policy stack
  ranking preserved between the NuRec simulator and the world-model
  simulator** — are in **neither the paper abstract nor the README**. The
  ranking claim would be the first cross-simulator-type ranking-stability
  datapoint if it ever appears in print; until then it is a vendor slide.
  Watch-item for §5.

---

## 4. Does sim training actually work? Split the answer

### Locomotion and navigation: yes, zero-shot, and fast

From the **MuJoCo Playground** paper (arXiv 2502.08844): Unitree Go1, Berkeley
Humanoid, Unitree G1 and Booster T1 all achieve **zero-shot joystick locomotion
transfer** across multiple terrains. Training times: **Go1 flat terrain ~5
minutes on 2× RTX 4090**, Berkeley Humanoid <15 min, G1 and Booster T1 <30 min.
Throughput on an A100: **417,451 steps/s** for state-based Go1 joystick.

### Contact-rich manipulation: only in a narrow band

Also from Playground:

| Task | Result | Training |
|---|---|---|
| Franka + Robotiq, non-prehensile block reorientation | **100% best case, 85.7% ± 12.2 mean over 35 real trials** | 10 min on 16× A100 |
| Franka + depth camera, pick-cube **from pixels** | **12/12 real trials** | **10 min on a single RTX 4090** |
| LEAP Hand, in-hand cube reorientation | median **3.5 consecutive rotations** over 10 trials | ~30 min on 2× 4090 |

Note the pattern: **every successful zero-shot manipulation result is a rigid,
geometrically simple object handled by a rigid, well-characterised arm.**
Nothing here is deformable or cluttered. Note also the third row's framing —
"consecutive rotations", not success rate — which is a weak result presented
well.

⚠️ **Corrected 2026-08-15.** This paragraph previously added "or on a cheap
compliant servo arm." That is no longer true: **Squint** (arXiv 2602.21203)
transfers zero-shot sim-to-real manipulation onto a **real SO-101** — eight
ManiSkill3 tasks, heavy domain randomisation, visual SAC in 6–15 min on one
RTX 3090. No identification, no sim-real correlation reported; but the
cheap-arm barrier itself has been crossed, via DR rather than sysid.

Encouraging for a solo builder: LeapCubeReorient takes **~2,080 s on 1× RTX 4090
versus ~670 s on 8× H100**. One consumer card gets you ~35 minutes for a
genuinely hard dexterous task. **A 24 GB card is sufficient.**

### Domain randomisation has been reframed, not retired

Three independent lines say the same thing. ACDC: cousins beat twins **90% vs
25%** (the one-task IKEA cabinet result from §3, corroborated by SimFoundry's
cousin gains in §5), and the authors call cousin distributions *"implicit
domain randomisation."* SplatSim: photorealistic rendering closes most of the visual
gap without visual randomisation. And co-training beats both.

### Co-training real + sim: the numbers, and the landmine

*"Sim-and-Real Co-Training: A Simple Recipe"* (arXiv 2503.24361, 2025-03-31):

| Task | Real only | Real + digital cousins | Real + generic prior | Real + both |
|---|---:|---:|---:|---:|
| CounterToSinkPnP | 44% | 67% | 58% | 72% |
| CounterToCabPnP | 38% | 72% | 53% | 72% |
| CloseDoor | 10% | 100% | 100% | 100% |
| CupPnP | 65% | 95% | 80% | 85% |
| MilkPnP | 50% | 70% | 80% | 80% |
| Pouring | 65% | 85% | 70% | 90% |
| **Average** | **45.3%** | **81.1%** | **76.8%** | **83.2%** |

Two findings that *reduce* your risk:

- **Camera alignment is not critical here.** Misaligned cameras degrade
  67 → 56%, but do not collapse. (Contrast MimicLabs in
  [22-data-generation.md](22-data-generation.md), where camera pose *was*
  dominant — the difference is that sim co-training carries an intentional
  domain gap either way.)
- **Task-aligned simulation is not required.** Task-aware digital cousins give
  **+35.8%**; a generic task-agnostic asset library gives **+31.5%**. You
  capture ~88% of the benefit from off-the-shelf assets you did not build.

And one finding that is a genuine landmine. Let α be the probability that a
training minibatch is drawn from simulation rather than reality. On CupPnP with
20 real and 1,000 simulated demonstrations:

| α | Success |
|---|---:|
| 0.50 | suboptimal — **a 1:1 mix is wrong** |
| **0.99** | **95% (best)** |
| 0.995 | **60%** |
| 0.999 | worse still |

**Half a percentage point of mixing ratio cost 35 points of success, and as of
August 2026 nobody has published a principled way to set α.** The paper says
only "carefully tune." **Budget a mandatory sweep** over
{0.9, 0.95, 0.98, 0.99, 0.995} as a non-negotiable cost of co-training.

Follow-ups work *around* the problem rather than solving it: optimal-transport
domain adaptation (arXiv 2509.18631) reports **up to +30%**; RL-based
sim-real co-training (arXiv 2602.12628, 2026-02-13) reports **+24% on OpenVLA
and +20% on π0.5** and explicitly beats the supervised 2025 recipe. Toyota
Research Institute's 89-policy study (arXiv 2602.01067) reports real
language-following **47.7% → 69.4%** but **does not report ratio sensitivity**,
so the α problem is neither replicated nor refuted at scale.

Re-checked 2026-08-15: **still no principled α setter and no knife-edge
replication anywhere reachable.** All four papers above show no relevant new
versions; an arXiv query for co-training mixing ratios in cs.RO returned zero
results. (Semantic Scholar was blocked this pass, so a paper not matching those
terms could have been missed.) The mandatory-sweep advice stands. The published
sim↔real exchange rate (N sim demos ≈ 1 real) also still does not exist.

**Honest bottom line: for a cheap servo arm at a commercial site, simulation's
highest-value role is co-training augmentation and evaluation — not standalone
zero-shot policy training.**

---

## 5. The strongest argument for building a simulator: evaluation

This is the finding that inverts intuition.

**SIMPLER** (arXiv 2405.05941), Visual Matching, Google Robot, 6 policies
(3 RT-1 checkpoints, RT-1-X, RT-2-X, Octo-Base), ~1,500 evaluation episodes each
from real and simulation:

| Metric | Pick Coke Can | Move Near | Drawer | **Average** |
|---|---:|---:|---:|---:|
| **Pearson r** | 0.976 | 0.855 | 0.942 | **0.924** |
| MMRV (lower is better) | 0.031 | 0.111 | 0.027 | **0.056** |

*MMRV* is Mean Maximum Rank Violation, introduced by the same authors because
Pearson correlation only measures linear fit while what you actually care about
is whether the simulator **ranks** policies the same way reality does.

Against that, **RoboArena** (arXiv 2506.18123) measured how well *real* robot
evaluation predicts an exhaustive-evaluation oracle, across 7 institutions and
600+ pairwise real episodes:

| Evaluation method | Correlation with oracle |
|---|---:|
| Task-aware RoboArena (distributed, double-blind) | r ≈ 0.95 |
| Standard Bradley-Terry | r ≈ 0.90 |
| **Conventional centralised real evaluation** | **r ≈ 0.60** |

> **A well-built simulated benchmark (r ≈ 0.92) is a more reliable estimator of
> true policy quality than a small, sloppy real-robot evaluation (r ≈ 0.60).**

⚠️ **These are two different papers with different setups; this juxtaposition is
suggestive, not proven.** SIMPLER used 6 policies from one family on one rigid
robot, and "Visual Matching" means the scene was hand-tuned to match reality —
labour, not automation. RoboArena does not compare itself to SIMPLER.

### 2026-08-15: the correlation replicated, and sim-eval is becoming a subfield

**SimFoundry** (arXiv 2606.28276, v1 2026-06-26) is the second
SIMPLER-magnitude datapoint, and it removes the two biggest doubts above:
**mean Pearson 0.911 and mean maximum ranking violation 0.018 across 7
manipulation tasks and 5 policy architectures**, with scenes built by
**automated zero-shot real-to-sim construction from video** — five
architectures, not one family; automation, not hand-tuning. (It also reports
17/21/40% real gains from object/scene/task cousins, independently
corroborating ACDC's digital-cousins direction.) ⚠️ The robot platform is not
named in the abstract — **the cheap-arm question stays open.**

Around it, simulated evaluation is visibly turning into a subfield: **PolaRiS**
(arXiv 2512.16881 — neural-reconstruction eval for generalist policies, claims
stronger real correlation than existing sim benchmarks, no coefficient in the
abstract), a Gaussian-splat **soft-body** evaluation paper (arXiv 2511.04665 —
plush packing, rope routing; splats render, physics is separate, consistent
with §3), **ManipArena** (arXiv 2603.28545, paired real-to-sim environments)
and **RoboSnap** (arXiv 2607.06699, one-shot real-to-sim with sim-real
correlation).

And the cheap-arm experiment this doc keeps asking for now has **both halves
sitting unjoined in the SO-101 ecosystem**: ArmnetBench (arXiv 2607.24481) is
the real half — 2,518 human-scored rollouts across 7 policies, data released —
and Squint's ManiSkill3 SO-101 task set is the sim half. Nobody has computed
the correlation between them. That is mostly assembly now, and still
publishable.

But the direction is clear enough to act on: **the simulator's first job is to
be an instrument.** ManiSkill3 ships the SIMPLER-derived environments and claims
*"100× faster real-world policy evaluation via GPU simulation"* — though its own
docs carry no correlation numbers; that evidence lives in the SIMPLER paper.

**LIBERO and Meta-World are not this.** Neither publishes real-world
correlation; both are algorithm-comparison benchmarks. LeRobot can evaluate on
them out of the box, which is convenient and is not evidence about your robot.

---

## 6. Engine choice, 2026

| Engine | Version / date | Licence | GPU | macOS | Verdict |
|---|---|---|---|---|---|
| **MuJoCo** | **3.11.0, 2026-07-27** | Apache-2.0 | CPU-native | ✅ | **Primary.** Best contact model for a compliant arm; **only stack with first-party sysid** |
| **MuJoCo Playground** | active; RSS 2025 outstanding demo | Apache-2.0 | CUDA 12 | ❌ | **Best solo-builder on-ramp** for RL |
| MJX | ships with MuJoCo | Apache-2.0 | JAX/XLA | marginal | via Playground |
| MuJoCo Warp | **3.6.0+ (2026-03-10)** | Apache-2.0 | **NVIDIA required** | ❌ | ⚠️ **Row corrected 2026-08-17: it now ships a first-party GPU batch renderer** (`mjwarp-render`, exposed as `mjx.render`) — RGB/depth/segmentation, textures, shadows, heterogeneous multi-camera, per-world domain randomisation. Still no IMPLICITFAST/PGS/PLUGIN actuators |
| **ManiSkill3** | v3.0.0+, RSS 2025 | Apache-2.0 code, **assets CC BY-NC 4.0** | Linux + NVIDIA | ❌ | ⚠️ **"Best evaluation harness" is stale (2026-08-17)** — MuJoCo's own batch renderer now does 186k FPS at 64×64 / 48k at 128×128 on a robot-arm scene, first-party and in-stack. ManiSkill3's 30,000+ FPS RGBD figure no longer leads |
| NVIDIA Newton | **v1.0.0 2026-04-13, v1.4.0 2026-07-16** | **Apache-2.0** | NVIDIA | ❌ | Released, out of beta — but a *backend*, not a workflow |
| Isaac Sim / Isaac Lab | **Sim 6.0.1 GA; Lab stable 2.3.2, Lab 3.0 is Beta** (2026-08-15) | **Apache-2.0 code** (corrected 2026-08-17) + proprietary Kit/RTX binaries / BSD-3 | NVIDIA | ❌ | **Skip.** ≥16 GB VRAM plus rendering headroom, Ubuntu 22.04 or Win 11 only |
| Genesis | **v1.3.2, 2026-08-07** | Apache-2.0 | CUDA/ROCm/**Metal** | ✅ | Materially stabilising (determinism, differentiable rigid body, elliptic friction cone) — but **no public reconciliation of the original benchmark claims**. Revisit in 6 months |
| Drake | v1.55.0, 2026-07-15 | BSD-3 | CPU | ✅ | Best *physical* contact model (hydroelastic), wrong tool for RL throughput |
| PyBullet | citation stops 2021, tracker closed | Zlib | — | ✅ | **Legacy** |

**MuJoCo 3.11.0 (2026-07-27)** added things that matter for a commercial site
specifically: `geom/surfacevel` (conveyors, turntables), `geom/adhesion` and
`pair/adhesion` (suction and magnetic grippers), gyroscopic derivatives in
`implicitfast`, and Union-Find replacing quadratic flood-fill for contact
islands.

### NVIDIA's own architecture slide, and what it settles (2026-08-17)

An **NVIDIA SIGGRAPH 2026 "Architecture Overview" slide** (shared by the
operator; a photographed slide, not a fetched URL — grade accordingly) draws
the whole stack, and it is the clearest confirmation yet of the conclusions
above. Reading it top-down:

```
                        ┌───────────── OpenUSD ─────────────┐
                        │   (interchange for everything)     │
                        └──┬────────┬──────────┬─────────┬──┘
                           ▼        ▼          ▼         ▼
                       ┌──────┐ ┌────────┐ ┌────────┐ ┌──────┐
                       │Isaac │ │ Newton │ │ MuJoCo │ │ Warp │
                       │      │◀│        │◀│        │ │      │
                       │Sim   │ │geometry│ │Playgrd │ │Warp  │
                       │Lab   │ │sensors │ │MJX     │ │CUDA  │
                       │      │ │ik      │ │MuJoCo  │ │      │
                       │      │ │solvers:│ │  Warp  │ │      │
                       │      │ │ MuJoCo │ │        │ │      │
                       │      │ │ Kamino │ │        │ │      │
                       │      │ │ canon. │ │        │ │      │
                       │      │ │ custom │ │        │ │      │
                       │NVIDIA│ │DeepMind│ │DeepMind│ │NVIDIA│
                       │      │ │ Disney │ │        │ │      │
                       │      │ │ NVIDIA │ │        │ │      │
                       │Apache│ │ Apache │ │ Apache │ │Apache│
                       │2.0/  │ │  2.0   │ │  2.0   │ │ 2.0  │
                       │BSD   │ │        │ │        │ │      │
                       └──────┘ └────────┘ └────────┘ └──────┘
        Arrows: Warp → Newton, MuJoCo → Newton, Newton → Isaac.
```

Four things this settles or adds:

1. **OpenUSD is the interchange layer for NVIDIA's entire stack** — drawn
   above all four pillars, feeding each. ⚠️ Our format decision is NOT that
   one, and this line once claimed it was: the slide is a photograph of a
   vendor's architecture, while reading both halves of the converter pipe
   (the section below, with its four-item loss table) retracted "USD as the
   sole interchange format" the same day it was written. The standing
   decision is the **split**: **MJCF is canonical for the robot** (sensors,
   cameras, defaults and actuators survive nowhere else), **USD for the
   scene**, where composition arcs earn their keep.
2. **Everything below Isaac is Apache-2.0**, on NVIDIA's own slide — Newton,
   MuJoCo and Warp all labelled Apache 2.0, Isaac labelled "Apache 2.0 / BSD".
   This independently corroborates the licence correction above; the lock-in
   was never the licences, only the Kit/RTX binaries.
3. **MuJoCo is becoming a solver inside Newton.** "MuJoCo Solver" sits in
   Newton's solver column beside a **Kamino Solver**, canonical solvers, and a
   custom-solver slot, with the dependency arrows running Warp → Newton and
   MuJoCo → Newton → Isaac. That is the same direction of travel as MuJoCo
   deprecating its `mjc:` USD attributes in favour of `newton:` ones — and it
   means **staying MuJoCo-native does not strand us**: our solver is a
   first-class component of the stack everyone else is assembling.
4. New names not previously recorded here: **Kamino Solver**, and the
   `newton.geometry` / `newton.sensors` / `newton.ik` module split. **Disney
   Research** appears as a third Newton partner alongside DeepMind and NVIDIA.

⚠️ Slide-grade evidence: no version numbers, no dates, and the module list may
be aspirational. Treat the *architecture* as confirmed and any individual
component as unverified until fetched.

### What USD actually costs us, read from both halves of the pipe (2026-08-17)

A follow-up sweep read the writer (`mujoco-usd-converter`) and the reader
(`plugin/usd_decoder/usd_decoder.cc`, 2,798 lines) rather than the marketing.
**The answer splits cleanly, and the split should become our architecture.**

**USD is description-only, by its authors' explicit choice.** UsdPhysics'
own overview calls itself *"a baseline initial extension to USD that enables
the minimum set of common concepts required to represent rigid body physics"*,
and on solver behaviour it says the quiet part out loud: *"the precise
deactivation rules are an implementation detail… we prefer to keep this as a
hidden implementation detail."* **USD cannot make two engines agree; it can
only make them read the same file.** That is the whole layer boundary.

**More survives an MJCF→USD trip than expected.** MuJoCo's `mjcPhysics` schema
family carries solver contact parameters (`solref`, `solimp`, `solmix`,
`margin`, `gap`, `condim`), the full actuator gain/bias/dyn parameterisation,
tendons with wrap paths, joint armature/springref/frictionloss, equality
connect/weld/joint, and the whole `<option>`/`<compiler>` block. Roughly 80%
is there, and somebody thought hard about it.

**But four things are lost, each for a different reason:**

| Lost | Why |
|---|---|
| **All sensors** | No `MjcSensor` schema exists — not in mjcPhysics, not in UsdPhysics. The importer reads only the global on/off flag. The `<sensor>` block evaporates. |
| **All cameras and lights** | Zero references in the importer; the converter's own changelog says camera and light conversion *"is not implemented."* **For a vision pipeline this is the worst item on the list — camera extrinsics *are* the calibration.** |
| **`<default>` / `class` / `childclass` structure** | *"baked down"* by the converter. Values survive; authorial intent does not. A one-line edit becomes an N-line diff — and it is gratuitous, because USD's `inherits` over `class` prims is the exact native equivalent, simply unused. |
| **Keyframes, `contype`/`conaffinity` filtering, contact pairs, ellipsoids, heightfields, SDFs, plugins** | Read by the importer, **never written** by the only converter that exists. They die at the write step. |

> ⚠️ **The failure mode to design against — and it is this repository's
> recurring bug shape exactly.** The importer has **zero** references to
> `UsdPhysicsDriveAPI`, `LimitAPI`, `DistanceJoint`, or
> `UsdPhysicsArticulationRootAPI`. So a well-formed *generic* USD robot —
> authored in Isaac, Blender or Houdini, with its motors expressed as
> `UsdPhysicsDriveAPI` — imports into MuJoCo **with no actuators and no
> error.** A plausible file and a silently inert robot: two things that must
> agree, with nothing comparing them. **Any USD robot we ingest must be
> checked for actuator count after import, not assumed.**

**So USD buys file portability, not semantic portability** — the surviving 80%
travels inside `mjcPhysics`, a plugin schema only MuJoCo reads. And the return
trip is worse than lossy: the one third-party USD→MJCF converter that exists
(`usd2mjcf`, Apache-2.0) emits `inertiafromgeom=True` with a literal
`#TODO: Add Inertia` in its source — it **fabricates mass properties from
geometry** instead of translating the authored ones, which for a pipeline
whose product *is* measured dynamics is disqualifying. The only other return
path is MuJoCo's own importer, which the converter itself warns *"may alter
Prim names, mesh topology, and other properties."* You do not get your MJCF
back — you get *a* MJCF back.

**Composition arcs are the genuinely valuable part, and they answer our
versioning needs natively:** variant sets express "same scene, three robot
models"; **payloads defer a heavy splat** so a bundle can ship it without every
consumer paying to load it; and sublayers give base-scene + per-site +
per-calibration overrides as independently hashable, independently diffable
units — which is the shape our bundle store already wants. ⚠️ But the importer
opens the *composed* stage (zero `Payload` or `Variant` references), so
**composition is a build-time authoring convenience, never a runtime
capability.** MuJoCo receives a flattened result either way.

**Governance is thinner than the adoption implies.** The AOUSD Physics working
group was chartered 2024-12-16 and **has ratified nothing** in the 20 months
since; deformables remain an unmerged PR; physics is explicitly out of scope
for the ratified Core Specification. And the splat schema — the most
consequential recent addition for our domain — is credited to an *"AOUSD
Emerging Geometry IG"* that has **no public charter, no public page, and no
published proposal**. The physics and splat layers are de-facto standards, not
de-jure ones.

**Adoption outside NVIDIA is thin:** Google DeepMind's MuJoCo is the only real
independent centre of gravity; Open Robotics' `gz-usd` has been dormant since
October 2024 and supports only USD v24.08; robot vendors ship USD as a third
format whose target is Isaac. No robotics company sits in AOUSD's founding
tier.

On the Isaac row's licence: Isaac Sim's source has been on GitHub since
May 2025 and NVIDIA calls it "open-source", but GitHub classifies the licence
as **"Other"** (a custom NVIDIA licence), and the application runs on
**proprietary prebuilt binaries** — the Omniverse Kit SDK and the RTX renderer.
Isaac Lab is genuinely BSD-3; **Newton is Apache-2.0** (NVIDIA + Google
DeepMind + Disney). So even NVIDIA's open physics future converges toward the
MuJoCo lineage, while the rendering stays theirs. Build *on* MuJoCo; *use*
Isaac/NuRec rendering where photorealism pays, without making it load-bearing.

**Updated 2026-08-19 — the binaries escaped Kit, and the lock-in got easier to
adopt, not looser.** NVIDIA unbundled Omniverse into pip-installable
agent-callable libraries ("Omniverse Libraries", part of NVIDIA Agent Toolkit).
Licence files read, not marketing labels: **`ovrtx`, `ovstage`, `ovstream`,
`ovui`, `ovstorage` all carry the NVIDIA enterprise Software License
Agreement** (*"non-exclusive, non-transferable, non-sublicensable… subject to
payment of applicable fees"*; PyPI: `LicenseRef-NvidiaProprietary`), and the
GitHub repos are **open shells around binary cores** — headers, bindings and
examples with no renderer source. The page's *"ovphysx — open source"* claim is
**false as of today**: no `ovphysx` repo exists in any NVIDIA org; it ships
only as a proprietary-licence PyPI wheel (the *old* PhysX SDK is BSD-3; the new
agent-era runtime is closed). Two practical details: **ovphysx runs on CPU
(AVX), no NVIDIA GPU needed**; ovrtx rendering still requires one. The
genuinely Apache-2.0 column is real and useful — `usd-search`, `usd-exchange`,
`usd-optimize`, `usd-validation-nvidia`, `usd-convert-asset`, and the
`mujoco-usd-converter`/`urdf-usd-converter` already in this doc. The verdict
above survives verbatim: the RTX renderer is now `pip install`-able into any
app — same binaries, same licence, friendlier packaging.

⚠️ **ManiSkill3's GPU simulation does not work under WSL** (re-verified
2026-08-15, install-matrix unchanged: "WSL | ✅ CPU | ❌ GPU Sim | ❌
Rendering") — see [24-compute-and-hardware.md](24-compute-and-hardware.md).
This is the single most actionable infrastructure finding in the whole research
pass.

This supersedes nothing in [../05-simulation-ros2-wasm.md](../05-simulation-ros2-wasm.md);
it extends it. That document's conclusions — MuJoCo adopted, Isaac skipped for
lack of macOS support, Genesis "watch, don't build on" — have all held up.

---

## 7. Not worth doing in 2026

1. **Isaac Sim / Isaac Lab.** You would spend week one on installation.
2. **Splat→physics as the real2sim path.** None of SplatSim, PhysGaussian,
   Splat-MOVER, gsplat, SuGaR, 2DGS or GOF produce collision geometry.
   (2026-08-15: Niantic's Scaniverse USDZ now co-packages an auto-derived
   collision mesh — §3. The objection has narrowed from "impossible" to "one
   vendor pipeline, backdrops only, Isaac-targeted"; hand-authored primitives
   remain the right call for *our* MuJoCo-based stack.)
3. **Precise digital twins of manipulable objects.** ACDC: **90% cousins vs 25%
   twins.** Precision is worse *and* slower.
4. **Drake as the trainer.** Excellent physics, CPU-bound, built for
   optimisation-based control rather than RL throughput.
5. **MuJoCo Warp / mjlab as the starting point.** Migrate later if throughput
   binds — it will not at this scale, when Playground trains pick-cube-from-pixels
   in 10 minutes on one 4090.
6. **Waiting for Newton to mature.** It already did; it just is not a workflow.
7. **RoboVerse or RoboCasa as a real2sim *method*.** RoboVerse reports no
   sim2real numbers; RoboCasa365 reports none either. Mine RoboCasa for assets.
8. **LIBERO / Meta-World as product signal.** No published real-world
   correlation.
9. **Deformables.** PhysTwin is the only credible option, videos-only, with
   undisclosed cost. Out of scope.

---

## 8. Sources

MuJoCo releases <https://github.com/google-deepmind/mujoco/releases> ·
sysid toolbox <https://github.com/google-deepmind/mujoco/tree/main/python/mujoco/sysid> ·
Menagerie <https://github.com/google-deepmind/mujoco_menagerie> ·
Playground <https://playground.mujoco.org/>, arXiv 2502.08844 ·
MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp> ·
Newton <https://github.com/newton-physics/newton/releases> ·
Isaac Lab <https://isaac-sim.github.io/IsaacLab/> ·
ManiSkill <https://maniskill.readthedocs.io/> ·
Genesis <https://github.com/Genesis-Embodied-AI/Genesis/releases> ·
Drake <https://drake.mit.edu/> ·
SIMPLER <https://simpler-env.github.io/>, arXiv 2405.05941 ·
RoboArena arXiv 2506.18123 · LIBERO <https://libero-project.github.io/> ·
ACDC <https://digital-cousins.github.io/>, arXiv 2410.07408 ·
RialTo <https://real-to-sim-to-real.github.io/RialTo/>, arXiv 2403.03949 ·
Sim-and-real co-training arXiv 2503.24361 · domain adaptation arXiv 2509.18631 ·
RL co-training arXiv 2602.12628 · TRI study arXiv 2602.01067 ·
SplatSim arXiv 2409.10161 · PhysGaussian arXiv 2311.12198 ·
Splat-MOVER <https://splatmover.github.io/> · gsplat <https://github.com/nerfstudio-project/gsplat> ·
PhysTwin <https://jianghanxiao.github.io/phystwin-web/> ·
Articulate-Anything <https://articulate-anything.github.io/> ·
EmbodiedGen arXiv 2506.10600 · RoboCasa <https://robocasa.ai/> ·
LeRobot SO-101 docs <https://huggingface.co/docs/lerobot/so101> ·
NeuralActuator arXiv 2607.11734 · Squint arXiv 2602.21203 ·
SimFoundry arXiv 2606.28276 · PolaRiS arXiv 2512.16881 ·
splat soft-body eval arXiv 2511.04665 · ManipArena arXiv 2603.28545 ·
RoboSnap arXiv 2607.06699 · ArmnetBench arXiv 2607.24481 ·
Niantic Spatial robotics <https://www.nianticspatial.com/robotics> ·
Flexion post <https://www.flexion.ai/news/niantic-spatial-flexion-and-nvidia-closing-the-sim2real-gap-for-humanoids> ·
Isaac Sim source <https://github.com/isaac-sim/IsaacSim> ·
DiFix3D+ arXiv 2503.01774, <https://huggingface.co/nvidia/difix> ·
instant-nurec <https://huggingface.co/nvidia/instant-nurec> ·
OmniDreams arXiv 2606.03159, <https://github.com/nv-tlabs/omni-dreams> ·
Mcity twin <https://github.com/mcity/mcity-digital-twin>
