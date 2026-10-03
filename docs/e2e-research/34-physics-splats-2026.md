# Physics-carrying Gaussian splatting: the 2026-08-25 re-check

*Fourth pass, 2026-08-25. Researched by one agent against primary
sources, per the one-agent-per-field discipline. Re-tests
[23-simulation-and-real2sim.md](23-simulation-and-real2sim.md) §3's
verdict ("splats buy appearance, never physics", last checked
2026-08-17) for the scene-scanning stage of the loop. **The verdict is
amended — see (b).** The agent's report follows verbatim.*

*Historical (2026-08-25). Superseded by the 2026-09-22 passes
[75](75-scene-capture-2026-09.md) and [76](76-physics-on-gaussians-2026-09.md)
and by what was built, [docs/78](../78-the-scene-loop.md) (the scene loop);
kept as the record of when the proxy-through-the-splat path first appeared.*

---

Method note: VERIFIED = I fetched the primary source (arXiv abs page, GitHub repo, project page, vendor page) today. CLAIMED = surfaced only in search snippets/secondary coverage; primary not fetched.

## (a) Dated findings

### Q1 — Does anything give splats a real physics story now?

**GaussTwin (ICRA 2026; Cai, Jansonnie, de Farias, Arenz, Peters — Jan Peters' group)** — VERIFIED via project page https://6cyc6.github.io/gstwin/. Anchors Gaussians to physical primitives driven by Position-Based Dynamics + discrete Cosserat rods, with SE(3) updates corrected by photometric error; validated on a real Franka Research 3 doing push tasks, handles rigid AND deformable bodies. Why it matters: first peer-reviewed system where the physics solver and the splats share state on real manipulation hardware — but it is a tracking/digital-twin system, not a policy-training simulator.

**Real-to-Sim Policy Evaluation with GS Simulation of Soft-Body Interactions (ICRA 2026; Columbia + SceniX + Google DeepMind)** — VERIFIED https://real2sim-eval.github.io/. Builds soft-body digital twins (PhysTwin lineage) from real videos, renders with 3DGS, and shows sim-vs-real policy success rates "tightly clustered along the diagonal" on toy packing, rope routing, T-block pushing — explicitly stronger sim-to-real correlation than IsaacLab. Code public: github.com/kywind/real2sim-eval. Why it matters: physics-coupled splats delivering a measurable robotics outcome (trustworthy policy evaluation), with code.

**i-PhysGaussian (arXiv 2602.17117, submitted 2026-02-19, U. Sydney)** — VERIFIED via search + arXiv listing https://arxiv.org/abs/2602.17117. Implicit MPM integrator on 3DGS; stable at ~20x larger timesteps than explicit PhysGaussian-style solvers, handles stiff/quasi-static regimes. Why it matters: fixes exactly the stiffness/timestep failure mode that made PhysGaussian-class MPM useless for contact-rich manipulation rates — still a graphics paper, no robot.

**Scene-Level Heterogeneous Physics Simulation with 3DGS (arXiv 2606.21753, 2026-06-19, CVPR 2026 Findings)** — VERIFIED https://arxiv.org/abs/2606.21753. Unifies 3DGS, meshes, and fluids into one particle set with a solver-agnostic kernel and two-way collisions against scene boundaries. Why it matters: the "everything becomes particles" abstraction is the cleanest bridge yet from splats to a solver — but it is graphics-targeted, not robotics.

**Real-Time Physics with Dynamic Mesh-Gaussian Reconstructions (arXiv 2606.00444, 2026-05-30, Waterloo)** — VERIFIED https://arxiv.org/abs/2606.00444. Key negative result: converting high-fidelity varying-topology reconstructions into fixed-topology physics meshes costs 65–80% geometric degradation; appearance fidelity and physics-compatible structure are "fundamentally distinct objectives." Why it matters: independent evidence that the appearance/physics split is structural, not a tooling gap.

### Q2 — SplatSim successors and sim2real numbers

**DISCOVERSE (IROS 2025 Oral, Tsinghua)** — VERIFIED https://air-discoverse.github.io/ and https://github.com/TATP-233/DISCOVERSE. MuJoCo physics + 3DGS rendering, 650 FPS RGB-D at 640x480 across 5 cameras, ROS plugins, MIT LICENSE (verified in repo). Claims superior zero-shot sim2real on contact-rich tasks vs existing simulators (headline % not in README). Why it matters: this is architecturally the platform — MuJoCo carries physics, splats carry pixels — already built, MIT-licensed.

**GSWorld (arXiv 2510.20813, 2025-10-23)** — VERIFIED https://arxiv.org/abs/2510.20813. "Gaussian-on-Mesh" GSDF asset format binding splats to URDF/objects; closed-loop DAgger, benchmarking, zero-shot sim2real RL. Why it matters: a concrete asset-format answer for splat+physics co-registration.

**GASE (arXiv 2606.17520, 2026-06-16)** — VERIFIED https://arxiv.org/abs/2606.17520. Automated splat-scan-to-simulator pipeline (segmentation, inpainting, independent foreground/background reconstruction); real-robot manipulation and navigation with **<10% gap vs policies trained on real data**. Why it matters: SplatSim's 86.25%-vs-97.5% gap (~11.5 points) is now roughly matched/beaten by an automated end-to-end scan pipeline.

**AOMGen (arXiv 2512.18396, 2025-12-20, rev 2026-03-13)** — VERIFIED https://arxiv.org/abs/2512.18396. Physics-consistent photoreal demo generation for articulated objects from a single scan+demo; VLA fine-tuning goes 0% → 88.7% on unseen objects/layouts (sim-vs-real ambiguity in abstract). **TwinAligner (arXiv 2512.19390, 2025-12-22)** — VERIFIED https://arxiv.org/abs/2512.19390 — pixel-level visual alignment (SDF + editable 3DGS) plus rigid-dynamics identification from interaction; zero-shot claims, numbers not in abstract, code promised. RL-GSBridge (2409.20291, ICRA 2025) remains the mesh-bound-GS RL entry; no headline number beats SplatSim's framing.

### Q3 — Permissively-licensed splat→mesh

Still **no fully-permissive turnkey pipeline**, verified against actual repos today:
- **MILo** (SIGGRAPH Asia 2025, the current SOTA mesh-in-the-loop method): repo https://github.com/Anttwo/MILo — under the **Gaussian-Splatting (INRIA non-commercial) license**. VERIFIED.
- **GS2Mesh** https://github.com/yanivw12/gs2mesh: own code **Apache-2.0**, but depends on the INRIA-licensed 3DGS component; TSDF-fusion output via Open3D. VERIFIED.
- **GauStudio** https://github.com/GAP-LAB-CUHK-SZ/gaustudio: **MIT "except the rasterizer"** — the tainted part is precisely the INRIA rasterizer. VERIFIED.
- **gsplat** (Apache-2.0): VERIFIED via https://docs.gsplat.studio/main/ — still **no 2DGS mode and no mesh extraction**.
- Practical implication: a clean-room permissive path exists but you assemble it yourself — gsplat (Apache) depth rendering → Open3D (MIT) TSDF fusion → marching cubes, i.e., the GS2Mesh recipe with the INRIA parts swapped out. Nobody ships this as a product repo yet.

### Q4 — NVIDIA and Google/DeepMind

**Isaac Sim 6.0 GA, 2026-06-04, ships NuRec Gaussian splatting natively** — VERIFIED via https://developer.nvidia.com/omniverse/nurec and https://radiancefields.com/nvidia-s-isaac-sim-6.0-ships-with-nurec-gaussian-splatting. Splat scenes (built on open-source gsplat; USDZ via 3DGRUT; 3DGS/3DGUT rendering through Fabric Scene Delegate) come with **paired collision proxies** so the same captured scene serves perception AND physics; includes Asset Harvester (object extraction) and Harmonizer (artifact cleanup). Why it matters: "splat + collision proxy pair" is now the vendor-standardized asset architecture — the exact pattern the platform would ingest.

**Newton physics engine** — VERIFIED repo https://github.com/newton-physics/newton: **Apache-2.0**, MuJoCo Warp as primary backend, and an **in-tree MPM solver** (granular, snow, multi-material examples) plus VBD cable examples. Newton 1.0 GA at GTC 2026 (March) with large speedups over MJX — CLAIMED (secondary: developer.nvidia.com/newton-physics surfaced, Medium/blockchain.news coverage). README has **no splat/neural-rendering integration**; Isaac Lab's Newton integration (VERIFIED, https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html) is experimental (Isaac Lab 3.0 Beta) and also rendering-silent. A "Newton closed-loop with Warp+gsplat" claim circulates in secondary coverage (pebblous.ai blog, 2026-04) — CLAIMED only.

**Google DeepMind, "Splatting Physical Scenes" (arXiv 2506.04120, 2025-06-04; Moran, Comi, Byravan, Bohez, Erez, Li, Hasenclever — the MuJoCo/robotics roster incl. Tom Erez)** — VERIFIED https://arxiv.org/abs/2506.04120. End-to-end real-to-sim: splats for appearance + explicit meshes for physics, **jointly optimized through differentiable rendering and differentiable MuJoCo** from raw imperfect ALOHA 2 trajectories — geometry, appearance, robot poses, and physical parameters in one loop. Why it matters: DeepMind's own answer keeps meshes as the physics carrier but makes the physics geometry a trained quantity — and it is MuJoCo-native. Plus the ICRA 2026 soft-body evaluation work above (DeepMind co-authored, code released).

**Standards**: Khronos announced the glTF **KHR_gaussian_splatting release candidate 2026-02-04**, ratification expected Q2 2026 (press URL https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release; CLAIMED — not fetched, multiply corroborated). Adds a second interchange standard beside OpenUSD's ParticleField3DGaussianSplat.

## (b) Verdict

**"Gaussian splats buy appearance, never physics" — STANDS as an architecture claim, but is now MISLEADING as a strategy claim, and should be amended.**

- What stands: as of today, no manipulation *training* simulator computes contact on Gaussian kernels. Every shipped or near-shipped system — Isaac Sim 6.0 NuRec (paired collision proxies), DISCOVERSE (MuJoCo meshes), GSWorld (Gaussian-on-Mesh), DeepMind's Splatting Physical Scenes (explicit meshes) — keeps a mesh/particle proxy as the physics carrier. The Waterloo paper (2606.00444) gives independent evidence the split is structural (65–80% geometric degradation when converting appearance-grade geometry to physics-grade topology).
- What changed since the 2026-08-17 check: (1) the splat→physics-scene path is now **automated and productized** — Isaac Sim 6.0 ships it, GASE closes the sim2real gap to <10% from a scan; (2) physics is **jointly optimized with** splats in differentiable MuJoCo at DeepMind — physics geometry from a splat pipeline is now a learned output, not a hand-built asset; (3) direct solver-on-splat coupling reached **real robot hardware** (GaussTwin, PBD/Cosserat, ICRA 2026) and **released-code policy evaluation with soft bodies** (Columbia+DeepMind, ICRA 2026); (4) i-PhysGaussian removed the explicit-MPM timestep barrier. The corpus line should read: "splats still don't carry contact, but the proxy that does is now generated, co-registered, and in the best work co-trained — the physics story arrives *through* the splat pipeline, not despite it."

## (c) Recommendation for the scene-scanning stage

Standardize the scanner's output as a **splat + watertight-collision-proxy pair** (matching what Isaac Sim 6.0 made the industry format), and build the proxy leg permissively: gsplat (Apache-2.0) depth renders → Open3D (MIT) TSDF fusion → marching cubes — the GS2Mesh recipe with the INRIA-licensed pieces swapped out; explicitly avoid SuGaR/2DGS/GOF/MILo in the commercial path. Before building the sim side, benchmark against **DISCOVERSE (MIT, MuJoCo + 3DGS, IROS 2025 Oral)** — it is the closest existing implementation of our exact architecture — and track DeepMind's differentiable-MuJoCo joint-optimization (2506.04120) as the upgrade path, since Newton (Apache-2.0, MuJoCo Warp backend, in-tree MPM) already contains the solver pieces to eventually couple deformables to splat appearance in-house.
