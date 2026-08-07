# Simulation and real→sim: how reality gets into the simulator

Research date: **2026-08-08**. Question: what is the fastest, cheapest path
from a real commercial site and a real low-cost mobile manipulator to a
simulator good enough that using it improves real-world performance?

> **TL;DR.** Reality reaches a simulator through **three separate channels**
> that get conflated and shouldn't be: the **robot** (system identification),
> the **scene** (geometry, appearance, mass, friction), and the **behaviour**
> (demonstrations). Channel A is the cheapest, the highest-value, and the one
> nobody has done for an SO-101 — **MuJoCo shipped a first-party system
> identification toolbox in 3.5.0 (2026-02-12)** and Menagerie's SO-101 model
> still has invented actuator gains. **Splatting buys appearance, never
> physics.** Precise digital twins of objects are *worse* than approximate ones.
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
  validation.** Its actuator gains are invented and its inertias are inherited
  from CAD.
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

| Tool | What it produces | Collision geometry? |
|---|---|---|
| **gsplat** v1.5.3 (Apache-2.0) | CUDA rasterisation of Gaussians; recent work is HiGS, MCMC, LiDAR rasterisation, 3DGUT | **No. Mesh extraction is not documented in the repo at all.** |
| **SplatSim** (arXiv 2409.09016) | replaces mesh *rendering* with splats **inside an existing simulator** — physics still comes from conventional meshes | No (inherits the simulator's) |
| **PhysGaussian** (arXiv 2311.12198) | custom material-point-method on Gaussian kernels, "what you see is what you simulate" | A *graphics* result; no robotics collision story |
| **Splat-MOVER** | semantics + grasp affordance + scene editing; 95–100% grasp success on specific objects | **No physics** |

SplatSim's headline — **86.25% average zero-shot sim2real versus 97.5% for
policies trained on real data** — is the strongest splat-based number in the
literature, and it is a *rendering* result layered on conventional physics.

**Practical reading: splatting is a phase-7 visual-gap optimisation, after you
have a working policy.** For collision, hand-author primitives from a floorplan
and a tape measure. A warehouse aisle *is* boxes; thirty minutes of measuring
beats an hour of mesh cleanup and produces something a solver can actually use.

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
Nothing here is deformable, cluttered, or on a cheap compliant servo arm. Note
also the third row's framing — "consecutive rotations", not success rate — which
is a weak result presented well.

Encouraging for a solo builder: LeapCubeReorient takes **~2,080 s on 1× RTX 4090
versus ~670 s on 8× H100**. One consumer card gets you ~35 minutes for a
genuinely hard dexterous task. **A 24 GB card is sufficient.**

### Domain randomisation has been reframed, not retired

Three independent lines say the same thing. ACDC: cousins beat twins **90% vs
25%**, and the authors call cousin distributions *"implicit domain
randomisation."* SplatSim: photorealistic rendering closes most of the visual
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
| MuJoCo Warp | active | Apache-2.0 | **NVIDIA required** | ❌ | Not yet feature-complete (no IMPLICITFAST, PGS, PLUGIN actuators) |
| **ManiSkill3** | v3.0.0+, RSS 2025 | Apache-2.0 code, **assets CC BY-NC 4.0** | Linux + NVIDIA | ❌ | **Best evaluation harness.** 30,000+ FPS RGBD on a 4090 |
| NVIDIA Newton | **v1.0.0 2026-04-13, v1.4.0 2026-07-16** | **Apache-2.0** | NVIDIA | ❌ | Released, out of beta — but a *backend*, not a workflow |
| Isaac Sim / Isaac Lab | Sim 6.0 (Jun 2026); Lab 3.0-beta pins Sim 5.1 | proprietary / BSD-3 | NVIDIA | ❌ | **Skip.** ≥16 GB VRAM plus rendering headroom, Ubuntu 22.04 or Win 11 only |
| Genesis | **v1.3.2, 2026-08-07** | Apache-2.0 | CUDA/ROCm/**Metal** | ✅ | Materially stabilising (determinism, differentiable rigid body, elliptic friction cone) — but **no public reconciliation of the original benchmark claims**. Revisit in 6 months |
| Drake | v1.55.0, 2026-07-15 | BSD-3 | CPU | ✅ | Best *physical* contact model (hydroelastic), wrong tool for RL throughput |
| PyBullet | citation stops 2021, tracker closed | Zlib | — | ✅ | **Legacy** |

**MuJoCo 3.11.0 (2026-07-27)** added things that matter for a commercial site
specifically: `geom/surfacevel` (conveyors, turntables), `geom/adhesion` and
`pair/adhesion` (suction and magnetic grippers), gyroscopic derivatives in
`implicitfast`, and Union-Find replacing quadratic flood-fill for contact
islands.

⚠️ **ManiSkill3's GPU simulation does not work under WSL** — see
[24-compute-and-hardware.md](24-compute-and-hardware.md). This is the single
most actionable infrastructure finding in the whole research pass.

This supersedes nothing in [../05-simulation-ros2-wasm.md](../05-simulation-ros2-wasm.md);
it extends it. That document's conclusions — MuJoCo adopted, Isaac skipped for
lack of macOS support, Genesis "watch, don't build on" — have all held up.

---

## 7. Not worth doing in 2026

1. **Isaac Sim / Isaac Lab.** You would spend week one on installation.
2. **Splat→physics as the real2sim path.** None of SplatSim, PhysGaussian,
   Splat-MOVER, gsplat, SuGaR, 2DGS or GOF produce collision geometry.
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
SplatSim arXiv 2409.09016 · PhysGaussian arXiv 2311.12198 ·
Splat-MOVER <https://splatmover.github.io/> · gsplat <https://github.com/nerfstudio-project/gsplat> ·
PhysTwin <https://jianghanxiao.github.io/phystwin-web/> ·
Articulate-Anything <https://articulate-anything.github.io/> ·
EmbodiedGen arXiv 2506.10600 · RoboCasa <https://robocasa.ai/> ·
LeRobot SO-101 docs <https://huggingface.co/docs/lerobot/so101>
