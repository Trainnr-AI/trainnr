# Physics on gaussians: the 2026-09-22 pass

*The operator's rule, 2026-09-22: make sure we are researching the state
of the art in physics for Gaussian-splat simulation training. The
previous pass (docs/e2e-research/75) inherited the standing verdict that
physics needs a mesh proxy and researched the proxy chain hard; this one
asks the opposite question on purpose — what runs physics on the splat
representation itself, and does any of it train robots. Six agents, one
per field, primary sources on 2026-09-22 (web search was exhausted that
day; every agent worked from direct fetches of arXiv, GitHub and docs;
VERIFIED means the number was read in the primary source, CLAIMED means
an abstract, README or project page asserted it).*

## 0. The six answers

1. **Continuum physics on gaussians is real for tabletop deformables
   and still graphics for everything else.** The spring-mass and XPBD
   twin line (PhysTwin, Embodied Gaussians, Real2Sim-Eval) reports
   centimetre tracking against real video and sim-to-real policy
   correlations above 0.9, with MIT code. The MPM-on-gaussians line
   (PhysGaussian to i-PhysGaussian, the scene-level heterogeneous
   solver) reports stability gates and user studies, never a
   ground-truth trajectory error; contact is a grid clamp; a frame
   costs seconds to minutes; no method runs more than a handful of
   parallel environments. Nothing new appeared in August or September.
2. **At reinforcement-learning scale, gaussians are a renderer, and
   two engines now render them natively.** mujoco_warp (since
   2026-08-19) renders static splats per world with mesh occlusion and
   depth, unlit, no shadows, no harmonics, collision from MJCF geoms.
   Newton (1.6.0, 2026-09-10) attaches gaussians to bodies as shapes
   with their own BVH, refit from forward kinematics each step, and
   builds a collision proxy for them (convex hull, alpha shape, points)
   — the cleanest design in the field. Neither publishes a frame rate
   with splats. The only 10⁴ FPS claim (GS-Playground) sits on a closed
   physics engine with its collision mechanism unstated. No system
   anywhere moves object-level gaussians by physics at a thousand
   environments with a real-robot number.
3. **Gradients through everything are not a practical path to a
   scene's friction and contact.** No shipped engine differentiates
   contact make and break: mujoco_warp's hybrid-analytic PR is unmerged
   and freezes the active set, Newton's experimental rigid contact is
   first-order in body poses with the narrow phase frozen, Genesis
   returns zero or undefined through contact, MJX needs one solver
   iteration. Every pixel-to-physics work on real data recovers
   kinematics or mass only; the one credible full loop (PIN-WM) needs
   known geometry and reports point estimates; the PhysTwin successor
   dropped gradients for a sampling fit because contact "prevents
   reliable gradient computation." No differentiable-identification
   paper reports an interval.
4. **Deformables in splat sims are benchmarked, batchable, and not yet
   a training surface.** Deform360 (MIT, 198 objects, 41 cameras plus
   tactile) benchmarks the twins at 1–4 cm Chamfer; Boba batches the
   PhysTwin model to 4,534 instance-steps a second on one 4090
   (CLAIMED). Nobody has trained a policy in one and deployed it; the
   real cloth wins (SIM1, SILR) are mesh stacks from scans, calibrated
   by eye. Granular and fluid in reconstructed scenes: zero real-robot
   numbers.
5. **Learned dynamics on gaussians is a short-horizon aid, not a
   simulator.** Centimetre accuracy at half a second; real-robot gains
   as a policy-side auxiliary (GaussianDream++ 63/120 vs 35/120,
   PhysMani 52/64) and as a residual over analytic physics (PGRD 8/10
   vs 2/10, QuadVerse 21/25); nobody has trained a policy in a pure
   gaussian world model and moved it to hardware; no gaussian-dynamics
   evaluator reports a real-success correlation.
6. **The field measures policy rank, not physics.** Reconstructed
   scenes rank policies at r 0.88–0.99 across six papers; no paper
   reports contact timing or force matching for a reconstructed scene;
   pose after interaction ranges 0.4–14 cm by pipeline; absolute
   success is biased 7–14 points. One sensitivity number decides how
   the gap must be measured: moving a device mesh by 20 mm and 5°
   dropped a contact task from 30 % to 0 % while the image similarity
   did not move (VERIFIED, simulation only, 39 scenes). A certification
   protocol can be assembled from what exists (§7).

## 1. Continuum physics on gaussians

| Method | Date | Physics | Contact | Speed / GPU | Accuracy reported | Parallel | Robot | Code | Grade |
|---|---|---|---|---|---|---|---|---|---|
| i-PhysGaussian 2602.17117 | 2026-02 | implicit MPM | grid clamp | unstated, 4090 | drift gates only | no | no | none | VERIFIED |
| Scene-Level Heterogeneous 2606.21753 | 2026-06 | MPM + SPH + PBD (Genesis) | SDF impulse, Coulomb | <10 min a sequence, offline | none | no | no | MIT | VERIFIED |
| FastPhysGS 2602.x / GaussianFluent 2601.x | 2026 | MPM | grid | 1 min a scene; 0.4–5 s a frame, 1–27 M particles | CLIP score; user study | no | no | none | VERIFIED |
| MonoPhysics 2605.30320 | 2026-05 | differentiable MPM | ground plane | 2 h a scene | log-E MAE 0.33–0.52 | no | no | none | VERIFIED |
| Embodied Gaussians 2406.10788 | 2024 | XPBD + shape matching | sphere pairs | 5 ms, 1k particles, 3090 | 1.4–2.1 cm real tracking | no | yes | MIT (rigid part) | VERIFIED |
| PhysTwin 2503.17973 | 2025-03 | spring-mass, Warp | impulse | real time | Chamfer 0.012 future | no | planning | MIT | VERIFIED |
| Real2Sim-Eval 2511.04665 | 2025-11 | PhysTwin | force-threshold grip | 5–30 FPS | sim-real r 0.90–0.94 | 200 episodes on 8 GPUs | evaluation | MIT | VERIFIED |
| Boba (ECCV 2026) | 2026 | batched surrogate spring-mass | inherits | 3,310 FPS aggregate, 4090 | "consistent with PhysTwin" | batch 64 | sim RL only | Apache-2.0 | CLAIMED |
| GaussGym 2510.15352 | 2025-10 | Isaac Gym rigid; splats render | Isaac | >100k steps/s | — | thousands | A1 stairs, no trial count | GitHub | abstract |

New since August, all thin: LaGSplat (learned Lagrangian, gaussians as
decoder), ChainSplat (deformable linear objects by screw chains, "SOTA"
without numbers), SplashSplat (liquids, a 20-scene benchmark), Wind on
Trees (frequency recovered, damping "not recovered at all"). Stability:
i-PhysGaussian holds 20× larger steps at the cost of a nested Newton
solve per step; the mesh-topology conversion for physics still loses
65–80 % of geometry (2606.00444). Admitted failures: pick-up and physics
misalignment, fast robot motion, occlusion-dependent synchronisation,
no automated parameter identification.

## 2. Splat-native rigid simulators at RL scale

| Simulator | Physics | Splat attachment | Robot-to-scene collision | Throughput | Robot result | Licence | Last commit | Grade |
|---|---|---|---|---|---|---|---|---|
| Newton `add_shape_gaussian` | Newton (MuJoCo-Warp, XPBD…) | per-body shape, local BVH, FK refit (`bvh_refit_shapes`) | a proxy built from the gaussians (convex hull, alpha shape, points) or a user mesh; gaussians never collide | none published with gaussians | none | Apache-2.0 | 2026-09-21 (1.6.0) | VERIFIED in code |
| mujoco_warp render | MuJoCo-Warp | world-static, per-world groups, BVH built once | MJCF geoms only | 2048 worlds × 4 cameras × 64² at 111 ms a step on a 5090 WITHOUT splats; none with | none | Apache-2.0 | 2026-09-21 (3.13.0) | VERIFIED in code |
| GS-Playground (RLGK) 2604.25459 | MotrixSim (closed binary) | per-link clusters, batched transform + gsplat | MJCF mesh (implied; mechanism unstated) | 2048 × 640×480 ≈ 10⁴ FPS on a 4090 | Airbot 18/20 | MIT sim, closed engine | 2026-06-11 | CLAIMED |
| DISCOVERSE | MuJoCo | per-body binding | MJCF mesh | 240 FPS RGB-D on a 3060 | prior | MIT | 2026-02-24 | stale |
| Genesis Nyx | Genesis | static light field, no runtime placement | n/a | unspecified | none | Apache-2.0 | 2026-06-08 | VERIFIED docs |
| Isaac Sim NuRec | PhysX / Newton | static background | mesh | unspecified | none | closed renderer | 2026-09-11 | CLAIMED |
| GSWorld (ICRA 2026) | SAPIEN | gaussians bound to URDF links and objects | mesh | unspecified | "zero-shot", unquantified | no licence file | 2026-02-27 | partial |
| Neverwhere | MuJoCo 3.1.6 CPU | static | OpenMVS mesh | one env | paired real trials (docs/75 §3) | MIT | 2026-09-16 | VERIFIED |

Camera fidelity, VERIFIED in code: mujoco_warp composites
`hit_color = splat_color + hit_color · transmittance`, splats unlit and
geoms lit, splats occluded by geoms (a test says so), depth from
splats, segmentation for geoms only, no robot shadow on the scene.
Newton returns the gaussian hit distance for depth, per-shape semantic
colour for segmentation, and its shadow rays have mesh and primitive
branches only. Object-level gaussians moved by physics with a real
number in 2026: GaussianFactory re-poses clusters kinematically with no
engine in the loop (UR10e 84.2 %); RoboSnap renders objects as meshes
over the splat; WANDA replays poses in Isaac. None at a thousand
environments.

## 3. Gradients through physics and rendering

| Work | Date | Differentiated | Target | Real-data error | Interval | Engine + rasterizer | Code | Grade |
|---|---|---|---|---|---|---|---|---|
| PIN-WM 2504.16693 | 2025-04 | both | mass, inertia, friction, restitution | 1.7 cm one-step; 75 % of 20 | none (±10 % fixed) | custom LCP + 2DGS | MIT | VERIFIED |
| Splatting Physical Scenes 2506.04120 | 2025-06 | both, kinematics only | geometry, poses, joints | Chamfer 3–7 mm; TCP 3.8 mm | none | MJX + 3DGS | none | VERIFIED |
| D-REX 2603.01151 | 2026-03 | physics; splats detached | mass | 4.8–12 % on six objects | none | Brax/MJX | none | VERIFIED |
| RigPI 2606.25212 | 2026-06 | physics; no render | mass, inertia, friction | mass 1.5–3.9 %, inertia 4–16 % | spread over 5 seeds, no interval | Newton | none | VERIFIED |
| PersistGS 2606.03479 | 2026-06 | both | friction, velocity | synthetic only | none | rigid + 4DGS | none | VERIFIED abstract |
| PGRD 2607.13451 | 2026-07 | none: CMA-ES by choice | spring-mass parameters | 2.6–2.9 cm | none | spring-mass + 3DGS | public | VERIFIED |
| SGPS 2609.20575 | 2026-09 | physics; depth detached | a Go2 policy | demo, no trial count | none | MJX | none | abstract |
| GRaD-Nav 2503.03984 | 2025-03 | physics; images detached | a drone policy | 6–7 of 10 | none | PyTorch + 3DGS | GitHub, no licence | VERIFIED |
| mujoco_warp PR 1535 | open since 2026-07-19 | implicit-function backward, active set frozen | friction, damping, inertia | author's finite-difference check | none | MuJoCo-Warp | unmerged | CLAIMED |

Engines, VERIFIED in docs and source: mujoco_warp's README says
differentiability "is not yet available"; MJX's constraint solver
blocks reverse mode unless one iteration; Newton 1.6.0 marks
differentiable rigid contact experimental, first-order in body poses,
narrow phase frozen, "validate case by case"; Genesis documents zero or
undefined gradients through some contact paths. gsplat returns
camera-pose gradients, not intrinsics; 3DGRUT's pose gradients could not
be found. Every policy trained with first-order gradients detaches the
renderer. Admitted failures: gradients through contact "diverge or
escape a good local minimum"; sensitivity to initialisation; the robot's
shadow corrupting the render loss.

## 4. Deformables, granular, fluid

| Method | Date | Material | Reconstruction accuracy | Simulation | Policy trained, real | Speed | Code | Grade |
|---|---|---|---|---|---|---|---|---|
| SIM1 2604.08544 | 2026-04 | cloth (scans, not splats) | sub-millimetre CLAIMED | AVBD on Newton | yes: 27, 24, 28, 28 of 30 | ~15 FPS, one env | Apache-2.0 | VERIFIED |
| SILR + FLASH 2606.24552 | 2026-06 | cloth (RGB to mesh) | Chamfer 1.9–3.7 mm | mesh, Newton contact | yes: 48/60 | 19 ms a step at 64 | none | VERIFIED |
| PhysTwin on Deform360 2607.05390 | 2026-07 | rope, cloth, plush | Chamfer 0.014 | spring-mass, Warp | evaluation only | 2–37 FPS | MIT | VERIFIED |
| Boba | ECCV 2026 | same | "consistent with PhysTwin" | batched spring-mass | sim-only RL | 4,534 instance-steps/s | Apache-2.0 | CLAIMED |
| PGRD 2607.13451 | 2026-07 | rope, cloth, plush | 1.3–4.3 cm | spring-mass + residual | MPC 8/10 | unstated | public | VERIFIED |
| BendTwin 2608.06164 | 2026-08 | non-cloth | Chamfer 0.0047 | spring + bending, Warp | no | 667 substeps a frame | none | VERIFIED |
| ChainSplat 2608.28570 | 2026-08 | deformable linear objects | "SOTA", no numbers read | screw chains | trajectory optimisation, no count | "real time" | pending | CLAIMED |
| RealSimLoop 2609.09828 | 2026-09 | silicone (FEM, not splat) | 4.6 mm | reduced FEM | shape control | 1–2 Hz | pending | VERIFIED |
| PhysGS 2511.18570 | 2025-11 | fabric properties | with calibrated uncertainty | none | no | — | CC BY | VERIFIED |
| granular or fluid in a reconstructed scene | — | — | none | — | none | — | — | absent |

Material recovery reports errors (MonoPhysics log-E MAE 0.33–0.52,
ViTacPhys stiffness 5.5–9.1 %) and, in one case (PhysGS), a calibrated
uncertainty; the twins that robots use calibrate by eye.

## 5. Learned dynamics on gaussians

| Method | Date | Representation | Learned | Error, horizon | Real robot | Speed | Code | Grade |
|---|---|---|---|---|---|---|---|---|
| PhysMani 2607.01938 | 2026-07 | 3DGS velocity field | online field + policy | 0.074 m at 0.5 s | 52/64 vs 40/64 | 205 ms a frame | CC BY-NC-SA | VERIFIED |
| GaussianDream++ 2608.25659 | 2026-08 | gaussian tokens inside π0.5 | auxiliary prediction | undisclosed | 63/120 vs 35/120 | +44 ms | Apache-2.0 | VERIFIED |
| PGRD 2607.13451 | 2026-07 | spring-mass + 3DGS | residual | 1.7–2.7 cm at 37 steps | 8/10 vs 2/10 | unstated | public | VERIFIED |
| QuadVerse 2606.07118 | 2026-06 | 3DGS + mesh + actuator residual | residual torque | 0.043 rad replay | 21/25 | unstated | promised | VERIFIED |
| MRO-GWM 2606.01950 | 2026-06 | object gaussians | rigid transforms | 0.45 cm, 5.5° at 0.4 s | none | 0.65 s per 30 rollouts | none | VERIFIED |
| ContactGaussian-WM 2602.11021 | 2026-02 | gaussian spheres + contact | mass, friction, stiffness | 1.2 cm, 0.07 rad | PSNR only | ~40 Hz | none | VERIFIED |
| GWM 2508.17600 | 2025-08 | latent DiT to gaussians | dynamics | PSNR 28 | 13/20 | unstated | GitHub | VERIFIED |
| video world models (Pelican-Sim, RoboWorld, Hydra-0) | 2026 | video, no 3D | video | r 0.99 (one vs a simulator) | some | 2 s a clip | mixed | mixed |

Nothing conditions a Genie- or Cosmos-class video model on a
reconstructed splat with a robot result. No paper compares evaluation in
a world model against evaluation in a splat simulator on the same real
ground truth.

## 6. Measuring a reconstructed scene's physics

| Paper | Date | Measures | Number | n | Grade |
|---|---|---|---|---|---|
| DEXTERA 2609.21045 | 2026-09 | replay, reconstruction | 343/390 replays; Chamfer 7.6 mm | 20 real a task | VERIFIED |
| Dex-X 2609.07747 | 2026-09 | wrist tracking sim to real | 2.83 cm | 30 a task | VERIFIED |
| R2S-Scene 2608.30821 | 2026-08 | geometry only | F 0.924; no physics, no trials | 9 scenes | VERIFIED |
| SCAPE 2608.19425 | 2026-08 | conformal coverage of sim-informed real estimates | 0.942–0.955 at nominal 0.95 | 95 paired Go2 | VERIFIED |
| Digital-twin clinics 2608.21416 | 2026-08 | contact-margin sensitivity, simulation | ±20 mm, ±5° → 30 % to 0 %; SSIM unchanged | 39 scenes | VERIFIED |
| Agentic Real2Sim 2607.19190 | 2026-07 | pose replay | ADD-S 107 mm; 48 % of episodes convert | 500 | VERIFIED |
| QuadVerse 2606.07118 | 2026-06 | joint and base replay | 0.043 rad; 0.12 m with per-region friction vs 0.70 m uniform | 10 a task | VERIFIED |
| RigPI 2606.25212 | 2026-06 | mass, inertia, friction | 1.5–7.2 % mass; 2–5 cm endpoint | 6 objects × 5 | VERIFIED |
| Practical Recipe 2606.10366 | 2026-06 | correlation vs real | Pearson 0.785 → 0.878 with 5–10 fine-tuning demos | 1,115 real | VERIFIED |
| TwinAligner 2512.19390 | 2025-12 | pose after push | ADD 1.39 cm | 15 a task | VERIFIED |
| GaussTwin 2603.05108 | 2026-03 | pose tracking | 0.43 cm, 3.3° | 4 tasks | VERIFIED |
| SureSim 2510.04354 | 2025-10 | prediction-powered intervals | 14 % narrower | 60 paired | VERIFIED |

No published protocol derives a randomisation range from a measured
gap; every reconstructed-scene paper since August hand-sets its ranges
and gives none numerically. Statistical practice: a minority reports
intervals (SureSim, SCAPE, Practical Recipe at 55 real trials a cell);
DEXTERA, DREAM, Video2DoorTraversal report none; seeds are fixed by two
papers.

## 7. The verdict, and what it does to docs/78

**Confirmed, now with evidence rather than inheritance:** for a legged
or contact-rich policy trained at thousands of environments, the only
working recipe in 2026 is rigid physics on a proxy with gaussians as the
renderer. Physics on the gaussians is a deformable-manipulation twin
line, honest and benchmarked, that nobody has trained a deployed policy
in. Gradients through contact do not ship. Learned gaussian dynamics is
a residual and an auxiliary, not a simulator. docs/78 §4 E2 stays on the
proxy.

**Amended, three things:**

1. **The gap is measured where the task touches, and the proxy is
   perturbed to find the cliff.** A 20 mm move of collision geometry
   zeroed a contact task while the image did not change. The scene
   record's four numbers over the footprint stay; a task adds a
   contact-site gap (the same four numbers within the task's contact
   regions) and a perturbation assay (the proxy shifted ±20 mm and ±5°,
   the task's success re-measured), and the cliff, not a visual metric,
   sets the collision tolerance and the span. docs/78 §3 and §4 E2.
2. **Movable objects will follow Newton's design, not a second
   renderer.** Gaussians attached to bodies as shapes with their own
   BVH refit from forward kinematics, a proxy built from the gaussians,
   the gaussians never colliding. The bounded port into mujoco_warp
   (a body id per splat group, intersection in body-local space, a
   per-step refit like flex) is the route when a task moves an object
   the camera must see; until then objects stay meshes. docs/78 §5.
3. **E5's shape is settled: a sampling fit with intervals over a
   tracked-pose residual, never gradients.** The PhysTwin successor
   reached the same conclusion; RigPI's 1.5–7 % on mass is the bar to
   beat; a gaussian pose tracker is the front end, `mujoco.sysid`'s
   Gauss-Newton with intervals or a DROPO-style posterior the fitter,
   and the span comes from the interval. docs/78 §4 E5.

**The certification protocol, assembled from what exists** (docs/78
§9): identify the object's or floor's friction, mass and centre of mass
by a scripted interaction and report the spread over at least five
repeats; replay at least ten recorded real trajectories open-loop and
report the pose after interaction (the field's good pipelines sit at
0.4–1.5 cm) and the joint-tracking error; run the perturbation assay on
the task's contact sites and report the cliff; evaluate at least three
policies with paired real and simulated trials, at least twenty a cell,
reporting the correlation, the rank-violation rate, a prediction-powered
or conformal interval, and the bias in points; fix and publish the
seeds. Every step but the real trials runs here today; the real trials
are where hardware enters.

## 8. Not verified, collected

Any gaussian-scene frame rate for Newton or mujoco_warp. GS-Playground's
collision mechanism and per-step cost. Boba's accuracy and licence
beyond its project page. Whether Newton's MuJoCo solver path is
differentiable and whether it lights gaussian hits. 3DGRUT's camera
gradients. mujoco_warp PR 1535's friction recovery. ChainSplat's tables.
Any scene-level force or contact-timing validation in the field (none
found across five sweeps). Licences for DEXTERA, DREAM,
Video2DoorTraversal, SCAPE, R2S-Scene, RigPI, GWM, PGRD.
