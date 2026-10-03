# Research reads

Dated reads against primary sources: repositories read code-first, papers,
vendor pages, standards. Each document carries the date it speaks from
and a status: **current** (the facts still hold), **snapshot** (true at the
date; versions and release states have moved since), or **historical**
(superseded; kept as the record, with a banner that names what replaced
it). Numbers missing from the sequence are the maintainers' working
notes, which are not part of this edition.

Start with [the loop](../76-the-loop.md), then
[23 Simulation and real-to-sim](23-simulation-and-real2sim.md), then the
evaluation group.

## The field survey, 2026-08

Written for a planned three-robot pilot, before anything was built.
The findings about the field hold; the plans around them were overtaken
by the loop.

| Doc | What it covers | Status |
|---|---|---|
| [19 The system end to end](19-the-system.md) | Four-tier safety architecture and latency budget for the archived rig; policy off-robot on a LAN GPU | historical; the rig lives in Trainnr-AI/rig |
| [20 Policies and models](20-policies-and-models.md) | Which policy to train on a cheap arm and one 24 GB GPU; third-party benchmarks against author claims | current |
| [21 Data collection](21-data-collection.md) | Teleop methods with numbers, the 50-demo floor, corrections over demos, LeRobot dataset rules | current |
| [22 Data generation](22-data-generation.md) | Curation and cheap augmentation beat generative data; vision-only world models cannot generate contact | current |
| [23 Simulation and real-to-sim](23-simulation-and-real2sim.md) | Three channels from reality into simulation; identification first; simulated evaluation predicts real ranking; engine and USD choices | current, with dated notes where the field moved |
| [24 Compute and hardware](24-compute-and-hardware.md) | 2026-08 hardware prices, the Jetson price shock, STS3215 duty-cycle limits, a pilot's indicative bill of materials | historical; the servo durability section still useful |
| [25 Deployment and fleet operations](25-deployment-and-fleet-ops.md) | Serving over gRPC with real-time chunking, MCAP logging, signed release manifests, staged rollout, pilot metrics | current |
| [26 Safety and regulation](26-safety-and-regulation.md) | EU Machinery Regulation, the AI Act, the standards gap for mobile manipulators: keep ML out of the safety path | current, regulatory claims as of 2026-08-15 |
| [27 Open questions](27-open-questions.md) | What the 2026-08 research could not settle; several items since answered | historical |

## Prior art for the platform, 2026-08-25

| Doc | What it covers | Status |
|---|---|---|
| [32 Recipe engine prior art](32-recipe-engine-prior-art.md) | Evaluation-gated automatic recipe selection: mixtures, co-training ratio, statistical certificates; not built | current |
| [33 Agent-written engineering](33-agentic-engineering.md) | Agents writing robot engineering behind gates they cannot touch; reward-hacking evidence | current |
| [34 Physics-carrying splats](34-physics-splats-2026.md) | Splats still never carry contact; the collision proxy arrives through the splat pipeline | historical; superseded by 75, 76 and [the scene loop](../78-the-scene-loop.md) |
| [36 Newton status](36-newton-status.md) | Newton's GA status and its relation to MuJoCo | historical; superseded by 47, 50, 51 |
| [37 MLOps tooling](37-mlops-tooling.md) | Trackers and orchestrators in robot learning; what shipped here differs (TensorBoard, no SkyPilot) | historical |
| [38 Fleet data planes](38-fleet-data-planes.md) | How deployed robot and vehicle fleets move data: three upload tiers, trigger campaigns, retention | current |

## Evaluation: Isaac Lab Arena, LeRobot, the ecosystem, 2026-08-26 to 2026-08-31

The Arena reads were made from a repomix bundle of the repository; the
bundle is not shipped and the line numbers are its.

| Doc | What it covers | Status |
|---|---|---|
| [39 Arena: metrics and progress](39-arena-metrics-and-progress.md) | How Arena scores success, progress chains and per-episode rows; no interval anywhere; what was adopted | snapshot |
| [40 Arena: experiments and runner](40-arena-experiments-and-runner.md) | The experiment YAML, the runner, the per-episode record, the multi-node collector | snapshot |
| [41 Arena: variations and sensitivity](41-arena-variations-and-sensitivity.md) | Variation samplers and the sensitivity posterior; the paired, trial-indexed alternative with exact main effects | snapshot |
| [42 Arena: placement and relations](42-arena-placement-and-relations.md) | The relation solver and placement validators; the validators were built on MuJoCo contacts, the solver skipped | snapshot |
| [43 Arena: the policy interface](43-arena-policy-interface.md) | How Arena drives VLA policies: chunk schedulers, executed horizons, remote adapters | snapshot |
| [44 Arena: environment spec and agentic generation](44-arena-environment-and-agentic-generation.md) | The YAML environment spec and its critic loop; which load gates transfer to an agent writing task specs | snapshot |
| [45 LeRobot's evaluation contract](45-lerobot-eval-contract.md) | `lerobot-eval`'s env contract (0.6.1): four keys, plugin discovery, the per-episode record; the plan that became the LeRobot plugin | current |
| [46 The ecosystem's evaluation interfaces](46-ecosystem-eval-interfaces.md) | Gymnasium, openpi, GR00T, LIBERO, SimplerEnv, EnvHub and robomimic compared; none reports an interval | current |
| [59 Arena, second pass](59-arena-composition-and-datagen.md) | Composition, the data factory, its USD-only walls, zero identification code | snapshot |
| [62 The paired study](62-paired-study.md) | The protocol for guessed-versus-identified randomization on the lift; two honest nulls | historical; superseded by the lift records and the paper |

## Simulation engines and the GPU path, 2026-08-27

| Doc | What it covers | Status |
|---|---|---|
| [47 Newton, read docs-first](47-newton-docs-review.md) | The solver roster, a parity ledger, a measured MuJoCo solver sweep on two CPUs | snapshot, 1.6.0.dev0 |
| [48 The solver landscape](48-solver-landscape.md) | MuJoCo's Newton, CG and PGS solvers, integrators and diagnostics, mapped to the scenes; fits are fits of the discretization | current |
| [49 The GPU path, probed](49-gpu-path-mjxwarp.md) | MJX-Warp on CPU and an RTX 3090 Ti: coverage, model batching, float32 divergence, sizing traps | snapshot |
| [50 Newton delta and live probe](50-newton-delta-probe.md) | Newton 1.5.0 on Apple Silicon: solvers run on CPU, the PyPI-name trap, the sensor block dropped on import | snapshot |
| [51 SolverMuJoCo round-trip, measured](51-solvermujoco-roundtrip.md) | Sensors, keyframes and visual geoms lost through Newton's importer; servo damping zeroed | snapshot; the most useful Newton finding |
| [52 Warp determinism and mujoco_warp](52-warp-determinism-mjwarp.md) | Warp's deterministic mode against mujoco_warp's atomics; the GPU probe blocked by a sensor kernel | snapshot |

## Actuators, identification and trainers, 2026-08-28 to 2026-09-06

| Doc | What it covers | Status |
|---|---|---|
| [53 BAM actuator identification](53-bam-actuator-identification.md) | Rhoban's BAM identifies the Feetech STS3215 among other servos; the friction models M1 to M6; the vendored actuator library | current |
| [53 microduck_rl, read code-first](53-microduck-rl.md) | Pollen's walking recipe: BAM in a training loop, a backlash twin, mjlab task design, the deployment contract | historical, archived 2026-09-07 |
| [56 mjlab, read at the source](56-mjlab.md) | Isaac Lab's shape on MuJoCo Warp: a clean actuator interface without measurement provenance, curricula that mutate config, no evaluation layer; what was taken | current as of 2026-08-31 |
| [57 BAM's code and its first production consumer](57-bam-source-and-microduck.md) | BAM's models, fitter and artifact format, and microduck's production use; where provenance and randomization are left to the reader | current as of 2026-08-31 |
| [58 The cross-framework setup](58-cross-framework-setup.md) | Three artifacts: the certified actuator bundle, a maintained mjlab consumer, the deployment manifest; who adopts each | current, with a status line per step |
| [60 The data press](60-the-data-press.md) | Referee-gated synthetic data from an MJCF plus a certified bundle, with provenance sidecar and datasheet | current, with a status line per step |
| [63 The flagship](63-the-flagship.md) | microduck's walk rebuilt through the certified stack: stamped bundle, declared bases, the G-series runs, the two-instrument certificates | current |
| [65 Unitree's own mjlab stack](65-unitree-rl-mjlab-review.md) | unitree_rl_mjlab read code-first: two task families, no plugin schema, a steal list, gains derived from motor physics | current as of 2026-09-02 |
| [71 SmoothRL, read against this stack](71-smooth-rl.md) | Astribot's SmoothRL in full; the latency-budget certificate shows neither teacher nor student survives 20 ms | current |
| [72 BAM's raw bench logs](72-bam-raw-logs-and-the-fitted-interval.md) | The public XL330 logs refit and bootstrapped: the torque constant is pinned, the friction exponent is not | current |

## Scenes, the simulation window and assets, 2026-09

| Doc | What it covers | Status |
|---|---|---|
| [74 The simulation window](74-the-simulation-window.md) | How Isaac Sim, Isaac Lab, MuJoCo's simulate, Gazebo, Genesis and mujoco-rs lay out the runtime; the Studio's simulator page followed | current |
| [75 Scene capture for the loop](75-scene-capture-2026-09.md) | Phone capture to splat to proxy in minutes; a splat scene ranks policies and never predicts absolute success | current |
| [76 Physics on gaussians](76-physics-on-gaussians-2026-09.md) | Continuum, splat-native, differentiable and learned physics on gaussians; the proxy stays; a certification protocol | current |
| [77 Native USD import](77-usd-import-2026-09.md) | Isaac's 2F-85 USD imported through Newton into a stamped bundle, audited equal to the source | current |

## How this corpus was made

Six passes, each one agent per field against primary sources, every
claim dated: the field survey (2026-08-08, re-swept 2026-08-15), a
commercial sweep (2026-08-16 to 20, kept privately), the prior-art
sprint (2026-08-25), Isaac Lab Arena code-first (2026-08-26), the
evaluation layer (2026-08-26), and the topical reads that followed each
build step. Search quota was short in the first two passes, so discovery
ran on direct fetches of repositories, model cards, arXiv and vendor
pages; mid-2026 releases may be under-sampled.

Five findings changed the picture: simulation is a consumer of real data,
through three separate channels; simulated evaluation predicts real
policy quality better than a small real evaluation does; published
numbers for the same model disagree by five times and confidence
intervals live almost nowhere; curation and cheap augmentation beat
everything generative; keeping the ML out of the safety path resolves two
regulatory regimes at once.
