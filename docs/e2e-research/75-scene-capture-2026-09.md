# Scene capture for the loop: the 2026-09-22 pass

*Sixth pass on captured scenes, and the first one asked as a product
question rather than a physics one. Six agents, one per field, against
primary sources on 2026-09-22 (the web-search budget was exhausted that
day, so every agent worked from direct fetches of arXiv, GitHub, PyPI,
readthedocs and vendor pages; VERIFIED means the number was read in the
primary source, CLAIMED means an abstract or README asserted it). The
prior verdict (docs/e2e-research/34, 2026-08-25): splats buy appearance,
physics needs a mesh proxy, standardize on the pair. This pass keeps
that and corrects one of its facts, then answers the question the
operator put on 2026-09-22: what does a captured scene add to
collection, augmentation, generation, telemetry and the loop that
improves the model.*

## 0. The six answers in one place

1. **Capture to scene is minutes, and the chain is permissive.** Phone
   video, poses from a feed-forward model with a commercial licence
   (VGGT-1B-Commercial, or Depth Anything 3 Small/Base under Apache),
   a splat trained in 3–5 minutes on one consumer GPU (ACE-GS class,
   REPORTED; gsplat Apache-2.0). The phone apps are the non-permissive
   part: Scaniverse's commercial export is $50 a month, Polycam's
   splat export is not listed, Luma's capture app is on its way out.
2. **Splat to physics is still the weak link, and Isaac does not have
   it either.** The prior pass recorded Isaac Sim 6.0's "splat plus
   paired collision proxy". VERIFIED on 2026-09-22 against the model
   card: Asset Harvester outputs a splat PLY only, "does not include
   physics attributes, collision proxies, or USD export"; the NuRec
   docs never mention collision for splat assets; Harmonizer is a
   Replicator randomization constraint. The pairing is a 3DGRUT export
   script that packs a mesh the user supplies. Every splat-to-mesh
   method with published accuracy (2DGS, PGSR, GOF, SuGaR, TopoSurfel,
   AnyGS2Mesh) is non-commercial or unreleased; the permissive route
   (gsplat's 2DGS mode into a TSDF mesh into CoACD) has no published
   DTU number. Watertightness is never reported, and MuJoCo's exact
   inertia refuses a mesh that is not.
3. **A splat scene ranks policies; it does not predict their absolute
   real success.** Calibrated twins reach Pearson 0.89–0.98 against
   real trials, always at ≤7 tasks, ≤30 real trials per cell, usually
   after real co-finetuning. For legged robots the paired evidence is
   twenty trials per task (Neverwhere, Go1: exact on hurdles, 25 points
   optimistic on stairs) and one warning that decides everything: two
   G1 assets with equal sim scores gave 4/40 and 33/40 on hardware
   (R2S-EGO).
4. **Our stack already has the renderer.** mujoco_warp merged a
   Gaussian-splat path into its batch ray tracer on 2026-08-19
   (PR 1585): splats composited front-to-back with mesh hits, depth
   written, per-world splat groups, static splats only, no spherical
   harmonics. Rerun 0.36 ships a `GaussianSplats3D` archetype
   (experimental). mjlab's camera sensor does not pass the splat
   arguments through: a twenty-line patch. Our pins are mujoco_warp
   3.11; the renderer is in 3.13, which needs MuJoCo ≥3.12.
5. **Nobody identifies scene physics with an interval.** Every method
   that fits physics through a splat (PhysTwin, PIN-WM, PAC-NeRF,
   MonoPhysics, D-REX, RealSimLoop) returns point estimates; PIN-WM
   sets its randomization width to a hand-picked ±10 % around the
   point. The one method that turns measured uncertainty into a
   randomization span (DROPO) never touched a captured scene.
   `mujoco.sysid` (3.5.0, 2026-02) ships linearized confidence
   intervals for robot-side parameters. No published loop re-judges
   scene parameters over time. That slot is the one our actuator
   identification and drift machinery already fills.
6. **What a captured scene provably adds to the data loop.** One phone
   scan plus one demonstration stands in for about 150 teleoperated
   demonstrations once a thousand episodes are rendered (R2R2R, MIT,
   n=15 per cell; RoboSplat; FTE) and delivers viewpoint, lighting,
   background and embodiment invariance that 2D augmentation does not.
   Below a thousand rendered episodes, real demonstrations win at every
   matched count. Every co-training study needs real data (0 % real
   collapses). No published system closes the loop from deployed
   failures back into the scene; the closest (Real-is-Sim, TAIL-Safe)
   update object Gaussians online but train from the twin, not from
   deployment logs.

## 1. Reconstruction: capture to scene

| Pipeline | Capture to scene | GPU | Quality evidence | Licence |
|---|---|---|---|---|
| Video → VGGT-1B-Commercial or DA3 poses → gsplat | poses "under a second" (CLAIMED) + 3–5 min train (REPORTED, ACE-GS 2606.21244; 2607.03209 ~3 min, 23.6 dB) | one consumer NVIDIA | PSNR only; no geometry number for the Apache chain | VGGT Licence v1 / Apache + Apache |
| Video → DA3 direct splat head | seconds (CLAIMED) | <12 GB | no table vs optimised | Apache (Small/Base), CC BY-NC (Large/Giant) |
| Video → COLMAP 4.2 global mapper → any trainer | "1–2 orders faster" than incremental (GLOMAP claim) | CPU/GPU | classical | BSD-3 |
| Video → SimFoundry → OmniGibson USD with physics | "under an hour" (README) | one NVIDIA + 250 GB | sim–real Pearson 0.911 (REPORTED) | Apache core; gated SAM3, Hunyuan3D-2, Gemini |
| Phone → Scaniverse on device → SPZ/PLY | minutes (CLAIMED) | none | none published | commercial export $50/month |
| Image → SAM 3D Objects / Track-Articulate-Act / PhysX-Anything → movable or articulated asset | unpublished | unstated | 5:1 human preference (SAM 3D) | SAM (commercial OK) / none / non-commercial |

New since August (VERIFIED abstracts unless noted): VGGT-Ω
(2605.15195, training code 2026-09-18, non-commercial weights),
MapAnything (Apache code and weights, metric output), WorldMirror-2.0
(Tencent licence excludes the EU and UK), BLASt3R (NAVER,
non-commercial), COLMAP 4.2.0 with the global mapper and learned
features (BSD-3, 2026-09-01), gsplat 1.6.0 (sparse, multi-GPU, LiDAR
rasterization, Apache), LichtFeld Studio 0.5.3 (GPLv3), Brush
(Apache, Rust/wgpu, "faster than gsplat" CLAIMED). Nerfstudio's last
release is 2024-11-11. Geometry-accurate trainers (2DGS/PGSR class):
nothing new after February 2026. Object and articulation: SimFoundry
code (Apache core, 2026-08-14), Track-Articulate-Act (2609.19119,
video to MuJoCo articulated object, code without a licence file),
RORA (joint axes 0.22°–1.43°, code unfound), NeoWorld-Pro, DEXTERA,
InstanceSplat, GroupForward. Not verifiable: any wall-clock inference
time on a named GPU for the feed-forward models; RealityScan and
Apple's tools (login-walled or JavaScript-only pages).

## 2. Splat to physics: proxies and properties

**Mesh.** AnyGS2Mesh (2609.03304, 2026-09-03): feed-forward, DTU
Chamfer 0.47–0.67, TnT F1 0.42–0.70, 30–45 s a scene, code unreleased.
TopoSurfel (2608.20687): DTU 0.51 CLAIMED, custom non-commercial
licence. "When 3DGS Recovers Real Surfaces" (2608.30054): surfaces are
identifiable only when the angular capacity is bounded; cap the
spherical-harmonic degree when geometry matters. 2DGS, PGSR, GOF and
SuGaR all carry the Inria or ZJU non-commercial licence (LICENSE files
read). gsplat exposes `rasterization_2dgs` and depth rendering under
Apache, so a 2DGS → TSDF chain can be built commercially clean; nobody
publishes its accuracy.

**Proxy.** CoACD 1.0.14 (MIT, 2026-08-28) is the decomposer; V-HACD is
archived in its favour. MuJoCo 3.13.0 (2026-09-08): mesh collision is
the convex hull, `maxhullvert` caps it, no built-in decomposition, and
`inertia="exact"` errors on a mesh that is not watertight. Newer
decomposers: VisACD (Apache, needs OptiX), NVIDIA's feature-field
decomposition (2603.09285, Apache, runs on point clouds and splats,
concavity 0.097 vs CoACD 0.110, ~18 s), Convex Primitive Decomposition
(unreleased). **The audit nobody does:** collision-mesh poisoning
(2609.18122) shows that altering only the collision geometry drops
real pick success 61.7 points while the simulation looks unchanged,
and Chamfer or IoU between the visual and collision meshes misses six
of nine poisoned assets because legitimate gaps vary that widely. A
scene record must carry the gap it measured.

**Properties.** Thin. RORA gives joint axes, no mass. KinemaForge
(RGB-D to URDF, 2.83° axis error, code pending). QuadVerse: friction as
a GPT-4 prior plus a ±0.2 grid search, base error 0.70 → 0.12 m.
"Before the Tipping Point": mass and centre of mass under 5 % from a
force sensor tipping the object (CLAIMED). Scalable Real2Sim (MIT,
Drake + CoACD) identifies mass, centre of mass and inertia from robot
pick-and-place, errors unpublished. DEXTERA's mass and friction are
language-model guesses. Every method under 5 % error has a robot
touching the object; passive capture stays guesswork.

## 3. Real-robot results from splat scenes (since 2026-08-01)

| Paper (arXiv, date) | Robot | Task | Sim-trained success on real | Against | Code | Grade |
|---|---|---|---|---|---|---|
| Neverwhere 2609.16443, 09-14 | Go1 | hurdles / stairs | 15/20 / 12/20 | sim 75 % / 85 % | MIT + data | VERIFIED |
| R2S-EGO 2608.06827, 08-07 | G1 | sit | 33/40 | 4/40 with a baseline asset at equal sim score | none | VERIFIED |
| GaussianFactory 2609.21112, 09-17 | UR10e | pick and place ×3 | 84.2 % of 120 | sim 95.1 % | none | VERIFIED |
| DREAM 2608.29078, 08-29 | xArm7 | block into bowl | 14/15 (π0.5, 1000 synthetic) | 13/15 on 100 teleop | none | VERIFIED |
| DEXTERA 2609.21045, 09-17 | two dexterous platforms | six contact tasks | 61.9 % co-trained | 29.2 % sim-only; 96 % sim → 30 % real on one task | none | n per cell unstated |
| GS-Playground v2 2604.25459, 08-04 | Airbot | pick cube | 18/20 | 0/20 for three baselines | MIT preview | VERIFIED |
| HumanoidVLN 2608.12860, 08-13 | G1 | navigation pilot | 20 paired episodes | r = 0.935 on navigation error | "soon" | VERIFIED |
| Video2DoorTraversal 2608.20251, 08-20 | A2-W + Z1 | doors (mesh twin) | 169/175; 85/105 unseen | sim 98.4 % | none | VERIFIED |
| R2S-Eval 2609.03276, 09-03 | arm (mesh) | evaluation | 6 VLAs × 7 tasks, 20 real each | mean gap 2.13 points, r = 0.978 | none | VERIFIED |
| LEGS 2606.01458 | G1 | loco-manipulation ×3 | up to 10/10; 1,110 trials | 200 splat demos beat 50 teleop on 9/9 cells | "soon" | VERIFIED |
| QuadVerse 2606.07118 | Go2 | grass navigation | 21/25 | sim 92 % | link dead | VERIFIED |
| GaussGym 2510.15352 | A1 | stairs | no trial count anywhere | — | GitHub | demo only |

Failure modes the papers name: baked static lighting (auto-exposure
off, sensitivity outside the scan's lighting); reflective and
transparent surfaces fail reconstruction; rigid bodies only; contact
and actuator dynamics ("inaccurate contact models, actuator dynamics,
material properties"; sim over-estimates stairs); ego-view render
quality off the capture trajectory. Of the post-August splat papers,
only Neverwhere and GS-Playground have a downloadable repository.

## 4. The renderer in our stack

| Component | Gives us | Licence | Version | Effort |
|---|---|---|---|---|
| mujoco_warp splat renderer (PR 1585, 2026-08-19) | static splat scene + mesh robot, RGB and depth, thousands of worlds, occlusion inside one ray tracer | Apache-2.0 | 3.13.0 | small: pass four arrays |
| mjlab `CameraSensor` | the splat arguments into its render context | Apache-2.0 | main | small patch, upstream it |
| Rerun `GaussianSplats3D` | the scene in the Studio | Apache/MIT | 0.36.0+, experimental | small |
| gsplat | movable per-body splats, fisheye, depth modes | Apache-2.0 | 1.6.0 | medium |
| DISCOVERSE `gaussian_renderer` | MuJoCo body-to-splat binding, 240 FPS RGB-D on a 3060 (REPORTED) | MIT | 0.2.0 | medium |
| 3DGRUT | capture to PLY/USD, mesh packed as collider | Apache-2.0 | 2.0.0 | medium (CUDA training) |
| wgpu-3dgs-viewer / brush | splats in the egui shell itself | MIT/Apache | 0.8.0 / main | large |
| Isaac NuRec runtime | reference only; the RTX renderer is a closed binary | Kit | Isaac Sim 6.1 | — |

Details VERIFIED in mujoco_warp's source: `create_render_context`
takes `splat_position`, `splat_rotation` (wxyz), `splat_scale`,
`splat_rgba`, `splat_adr`, `splat_group_id`; ray-splat analytic
intersection, up to 32 hits, transmittance composited with mesh hits;
splat depth reaches the depth buffer; segmentation ignores splats;
`refit_splat_bvh` exists but is never called, so splats are static.
Our pins: mujoco 3.11.0, mujoco-warp 3.11.0, mjlab 1.6.0, warp 1.17.0,
rerun 0.36.3 (the archetype is there). The renderer needs mujoco-warp
3.13, which needs mujoco ≥3.12: a version step on the walk package's
instrument. Other simulators: Genesis via its optional Nyx path tracer
(Apache, CUDA 12.9+); ManiSkill3, SAPIEN, Habitat, TDW, Newton: none.
Not verified: whether Rerun's archetype depth-tests against meshes in
the same view; mujoco_warp's frame rate with splats.

## 5. The loop: capture to identified physics

| Method | Parameters | Error reported | Uncertainty | Tool / licence | Grade |
|---|---|---|---|---|---|
| PhysTwin 2503.17973 | dense spring stiffness, damping, collision | Chamfer 0.005–0.012 | none | Warp / MIT | VERIFIED |
| PIN-WM 2504.16693 | mass, inertia, friction, restitution | 1.7 cm one-step; 75 % / 65 % real (n=20) | none; DR = ±10 % fixed | differentiable LCP + 2DGS | VERIFIED |
| PAC-NeRF 2303.05512 | continuum material | 5–10 % synthetic | none | MPM | VERIFIED |
| Phys2Real 2510.11689 | centre of mass | 57 % vs DR 24 % | yes: ensemble mean and variance, to condition the policy | Isaac Lab + splats | VERIFIED |
| DROPO 2201.08434 | masses, friction, centre of mass | 2 cm push | **yes: Gaussian (μ, Σ) that IS the randomization span** | MuJoCo / MIT | VERIFIED |
| ADDF 2308.01001 | friction, mass, centre of mass, inertia | nRMSE 0.18–0.20 | yes: filter covariance | PyBullet | VERIFIED |
| ASID 2404.12308 | friction, mass distribution | 7/10 vs 3/10 | Fisher information for exploration only | MuJoCo | VERIFIED |
| SIMPLER / ASAP | PD gains / mass, centre of mass, gains | MMRV 0.056; MPJPE 50 vs 61 mm | none / ± std | SAPIEN / Isaac Gym | VERIFIED |
| `mujoco.sysid` (3.5.0, 2026-02-12) | friction, damping, inertia, gains, sensor delay | user-dependent | **yes: t-intervals from (JᵀJ)⁻¹**, linearized | MuJoCo / Apache | VERIFIED |
| RealSimLoop 2609.09828 | elastic moduli per sliding window | 4.6 mm reconstruction | none | custom, pending | VERIFIED |

Where it closes: robot-side parameters with intervals
(`mujoco.sysid`; ASAP-class fits) and one object's rigid parameters
with a posterior (DROPO, ADDF), and DROPO alone turns the posterior
into a span. Where it does not: every capture-driven scene method is a
point estimate, none re-runs on later data, and the gap metrics
(SIMPLER, ASAP) diagnose a mismatch without saying which parameter to
widen. Recurring calibration exists as pose sync at 60 Hz (Real-is-Sim)
or per-window material refits (RealSimLoop), never as interval-based
drift judged across a fleet. Not verified: which Newton and Genesis
solvers deliver gradients through rigid contact; coverage of
`mujoco.sysid`'s intervals beyond the linearized Gaussian assumption.

## 6. The data loop around a captured scene

| Method | Date | Demos in → out | Real success of the trained policy | What varied | Code | Grade |
|---|---|---|---|---|---|---|
| R2R2R 2505.09601 | 2025-05 | 1 video → 1000 in 14–38 min on a 4090 | 8–13/15 per task ≈ 150 teleop | pose, lighting, camera ±2 cm/5° | MIT | VERIFIED |
| RoboSplat 2504.13175 | 2025-04 | 1 → 1800–6400 | 94.7 % mean (n=30) vs 200 real demos | six axes incl. embodiment | Apache (generation only) | VERIFIED; per-cell n missing |
| DemoGen 2502.16932 (no splat) | 2025-02 | 1 → 100–432 | 40–91 %, 530 rollouts | object pose only | MIT | VERIFIED |
| FTE 2605.01232 | 2026-05 | 1 → 256 | 37/40, 30/40, 25/40 | pose, view, goal | none, non-commercial | VERIFIED |
| LEGS 2606.01458 | 2026-05 | 0 teleop → 200 | 9/10 vs teleop 4/10 | background, objects | none | VERIFIED |
| ExoGS 2601.18629 | 2026-01 | 60 → ×20 | loses on seen objects (50 % vs 72 %), wins on new (76 % vs 0 %) | view, colour, light, pose | GitHub | VERIFIED |
| WANDA 2607.13154 | 2026-07 | 1 → mobile-manipulation worlds | 54.8 % progress (n=10) | poses, base, drift | coming soon | progress score |
| 1001 Demos 2606.19586 | 2026-06 | 89 → thousands | 5 → 100 % (n=10–20) | view + action | page only | weak n |
| Real-is-Sim 2504.03597 | 2025-04 | 30 real + 30 in the twin | 57 → 80 % (n=60); 60 real demos comparable | failure states | none | VERIFIED |
| TAIL-Safe 2605.01195 | 2026-05 | ~500 twin rollouts | 20–25 → 100 % (n=50) | perturbations | none | VERIFIED |
| DreamGen 2505.12705 (no 3D) | 2025-05 | frame → 240k | 37 → 46 %; 0 → 28.5 % new environment | — | page | VERIFIED |
| Sim-and-real co-training 2503.24361 | 2025-03 | cousins, 1k–10k | +10–20 points at 40–400 real | cousin scenes | none | VERIFIED |

Diminishing returns, with numbers: R2R2R's scaling table has synthetic
near 0–50 % at 50–150 episodes and teleop winning at every matched
count, catching up only at 1000. Co-training: the optimal real fraction
is small but never zero (α=0.99; α≥0.995 collapses to 60 %); RoboSnap's
best mix is 60 % real, 0 % real collapses to 17.3 %. Not verifiable:
any video generator conditioned on splat renders with a real success
number (Cosmos-Transfer2.5 documents depth, segmentation, edge and blur
conditioning only); a deployed loop that re-scans from telemetry.

## 7. The verdict, amended

The 2026-08-25 verdict stands with one correction and one addition.
The correction: the vendor-standardized "splat plus collision proxy
pair" does not exist as a shipped product; Isaac ships the splat and a
script that packs a mesh you supply, so the pairing is ours to build
and to audit, and the audit (the visual–collision gap, measured and
recorded) is what the field is missing. The addition: the renderer is
in our own simulator now, so the cost of a captured scene in our stack
is a version step and a twenty-line patch, not a second renderer.

What a captured scene is for, in this product: a data source that
turns one demonstration into a policy worth about 150 teleoperated
ones, an evaluation surface that ranks policies with a real
correlation, and a place to mine failure states — never a substitute
for real data, and never a predictor of absolute real success. The
piece no one has built is the one our loop is shaped for: scene
physics identified with an interval where a robot touches, declared
with a stated span where it does not, the randomization width set from
the interval, and a drift check that re-judges the scene from fresh
telemetry. The experiment is docs/78.

## 8. Not verified, collected

Wall-clock times for every feed-forward reconstruction model on a named
GPU. Any accuracy number for the Apache splat-to-mesh chain. Isaac's
collision-approximation option list. Absolute mass and friction errors
for Scalable Real2Sim, TwinAligner, BendTwin. Rerun's splat depth test
against meshes. mujoco_warp's splat frame rate. GaussGym's licence and
any real trial count. Code for LEGS, VLK, QuadVerse, R2S-EGO,
GaussianFactory, DREAM, DEXTERA (all "soon" or absent). Which Newton
and Genesis solvers are differentiable through contact. Any deployed
system that re-scans a scene from telemetry.
