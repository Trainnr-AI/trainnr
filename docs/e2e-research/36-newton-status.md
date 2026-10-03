# Newton physics engine: shipping status, 2026-08-25

*Fourth pass, 2026-08-25. Researched by one agent against primary
sources, per the one-agent-per-field discipline. Tests
[docs/22-pipeline-architecture.md](../22-pipeline-architecture.md) §3's
position ("MuJoCo is the loop; Newton a backend, not a foundation").
Headline: the
position stands and is STRENGTHENED — and the GPU path should be
MJX-Warp, which consumes the very mjModel our sysid identifies. The
agent's report follows verbatim.*

*Historical (2026-08-25). Superseded by the measured reads
[47](47-newton-docs-review.md), [50](50-newton-delta-probe.md) and
[51](51-solvermujoco-roundtrip.md), which ran Newton; kept because the paper
cites it ([docs/paper/references.md](../paper/references.md)).*

---

## (a) Dated findings

### Release state (Q1)

| Finding | Date | Status | Source |
|---|---|---|---|
| Newton v1.0.0 — CHANGELOG entry "Initial public release" dated **2026-03-10**; GitHub release object published 2026-04-13 (backfilled). GA announced at GTC March 2026 | 2026-03-10 | VERIFIED (GitHub API + raw CHANGELOG.md) | https://github.com/newton-physics/newton/releases, https://raw.githubusercontent.com/newton-physics/newton/main/CHANGELOG.md |
| Monthly cadence since GA: v1.1.0 (Apr 15) → v1.2.0 (May 12) → v1.2.1 (Jun 5) → v1.3.0 (Jun 11) → v1.4.0 (Jul 16) → **v1.5.0 (Aug 11, 2026, latest)**. All marked non-prerelease | 2026-08-11 | VERIFIED (GitHub API, exact timestamps) | https://api.github.com/repos/newton-physics/newton/releases |
| Past alpha/beta: yes — GA/1.x. But **no formal API-stability guarantee anywhere** in docs or README; every minor release ships breaking changes (v1.3.0 removed writable joint-target attrs; v1.5.0 removed `joint_target_pos/vel` aliases; v1.4.0 changed contact-gap semantics with new `rigid_gap = 0.1 m` default). Deprecate-then-remove pattern via towncrier changelog | Aug 2026 | VERIFIED (releases pages + docs fetch) | https://github.com/newton-physics/newton/releases, https://newton-physics.github.io/newton/stable/ |
| Governance: Linux Foundation project, "community-built and maintained," initiated by Disney Research, Google DeepMind, NVIDIA. Apache-2.0 | Aug 2026 | VERIFIED (repo README) | https://github.com/newton-physics/newton |
| Ongoing migration: v1.3.0 began transition to coordinate-aligned joint-target layout (`newton.use_coord_layout_targets`); legacy DOF layout deprecating over future releases — API churn is not finished | 2026-06-11 | VERIFIED (release notes) | https://github.com/newton-physics/newton/releases |

### MuJoCo-solver integration (Q2) — this is real and deep

| Finding | Date | Status | Source |
|---|---|---|---|
| MJWarp README: "MJWarp is maintained by Google DeepMind and NVIDIA **as part of the Newton project**." Newton README: MuJoCo Warp is Newton's "**primary backend**" | Aug 2026 | VERIFIED (both READMEs fetched) | https://github.com/google-deepmind/mujoco_warp, https://github.com/newton-physics/newton |
| mujoco_warp releases are now **version-locked to MuJoCo**: tags v3.8.0…v3.12.0 released within hours of the matching MuJoCo releases (both v3.12.0 on 2026-08-20). MJWarp docs are hosted on mujoco.readthedocs.io under `/mjwarp/` | 2026-08-20 | VERIFIED (GitHub API timestamps for both repos) | https://api.github.com/repos/google-deepmind/mujoco_warp/releases |
| MuJoCo itself ships **MJX-Warp**: `pip install mujoco-mjx[warp]`, `impl='warp'` in `mjx.put_model()` — MJWarp behind the MJX API, presented alongside MJX-JAX; docs call MuJoCo Warp "the most fully-featured implementation of MuJoCo for hardware accelerated devices." Caveat: **no autodiff, no immediate plans** | current stable docs | VERIFIED | https://mujoco.readthedocs.io/en/stable/mjx.html |
| MuJoCo changelog: Warp added as MJX backend in 3.3.5 (beta); sysid officially released in **3.5.0 (2026-02-13)**; MJX-Warp batch rendering in 3.6.0; MuJoCo tracks **Newton USD schemas (upgraded to 0.4.0)** | 2026 | VERIFIED (changelog.rst fetched) | https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/changelog.rst |
| MJWarp known gaps: no IMPLICITFAST integrator, no PGS/noslip solvers, no PLUGIN actuators, no Warp differentiability; Flex "experimental" | Aug 2026 | VERIFIED (README) | https://github.com/google-deepmind/mujoco_warp |

### Ingestion: MJCF and USD (Q3)

| Finding | Date | Status | Source |
|---|---|---|---|
| `ModelBuilder.add_mjcf()` (backed by newton/_src/utils/import_mjcf.py in the Newton repository) supports joints, actuators, equality constraints (→ loop joints), materials/textures, meshes (file + inline), defaults/classes hierarchy, includes, heightfields; v1.5.0 added inline mesh-data import; importers "preserve materials, textures, velocity, collision filtering, and equality constraints with greater fidelity" | 2026-08-11 | VERIFIED (source file + release notes) | https://raw.githubusercontent.com/newton-physics/newton/main/newton/_src/utils/import_mjcf.py |
| **Menagerie coverage measured directly**: the Newton-project `mujoco-usd-converter` benchmarks all of Menagerie — **85/85 models convert, 100% success, including `trs_so_arm100/so_arm100` (verified, 1 minor warning)**, run dated 2026-03-09. Converter itself is explicitly "Alpha" | 2026-03-09 | VERIFIED (benchmarks.md fetched) | https://github.com/newton-physics/mujoco-usd-converter (benchmarks.md) |
| Newton's own robot examples (G1, H1, ANYmal, Allegro, UR10, Panda) increasingly load **structured USD** via `download_asset()`, not raw MJCF — USD is the favored packaging in Newton-land, MJCF import is the compatibility path | Aug 2026 | VERIFIED (example sources read) | https://github.com/newton-physics/newton/tree/main/newton/examples/robot |

So yes: an SO-ARM100 Menagerie MJCF loads today, either directly via `add_mjcf` or via the (alpha) USD conversion pipeline. The agent could not run it then (no GPU in its session; [47](47-newton-docs-review.md), [50](50-newton-delta-probe.md) and [51](51-solvermujoco-roundtrip.md) ran it later) — "runs correctly" (contact behavior, actuator parity) remains untested.

### GPU batched rollouts and adoption (Q4)

| Finding | Date | Status | Source |
|---|---|---|---|
| Isaac Lab 3.0: v3.0.0-beta **2026-03-17**, beta2 2026-06-17, beta2.patch1 2026-07-02. Multi-backend architecture with `isaaclab_newton` package "powered by MuJoCo-Warp." **Still beta; no stable 3.0** | 2026-07-02 | VERIFIED (GitHub API) | https://github.com/isaac-sim/IsaacLab/releases |
| Isaac Lab's own docs on Newton integration: "**experimental and under active development**… only a limited set of classic RL and flat terrain locomotion examples… you are likely to encounter breaking changes… we do not expect to be able to provide official support until the framework has reached an official release." Locomotion validated sim-to-sim and sim-to-real; **manipulation workflows not there yet** | current (main docs) | VERIFIED | https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html |
| Isaac Lab dev update (2026-01-06): dual-backend future — PhysX **and** Newton, PhysX not deprecated | 2026-01-06 | VERIFIED | https://github.com/isaac-sim/IsaacLab/discussions/4339 |
| Public users: Isaac Lab (beta), **mjlab** — "exposes the Isaac Lab manager-based API directly on top of MJWarp" (skipping Newton), MuJoCo Playground via MJX-Warp | Aug 2026 | VERIFIED (MJWarp README) | https://github.com/mujocolab/mjlab |
| MJX-Warp throughput: **2.96M–3.35M steps/s** on MuJoCo's own test scenes, far above MJX-JAX on contact-rich scenes | current docs | VERIFIED | https://mujoco.readthedocs.io/en/stable/mjx.html |
| "Newton 475x faster than MJX for manipulation / 252x locomotion on RTX PRO 6000 Blackwell"; "~50M steps/s on RTX 4090"; "GA adopted by manufacturers" | GTC Mar 2026 | **CLAIMED only** — secondary blogs (vnrobo, byteiota, blockchain.news); fetches of blogs.nvidia.com and the mjwarp nightly-benchmark page were blocked in the session | https://vnrobo.com/en/blog/nvidia-newton-physics-engine |

### Adapter cost behind a physics-backend seam (Q5)

VERIFIED API surface (docs 1.5.0 + example code): `ModelBuilder` → `add_mjcf/add_usd/add_urdf` → `replicate(xforms=...)` for batched multi-world → `SolverMuJoCo` (plus XPBD/VBD/Featherstone/Kamino/MPM) → `State`/`Control`/`Contacts` → `ArticulationView` selection. RL-relevant machinery landed through 2026: in-place solver resets with `StateFlags` (v1.3.0), deterministic bit-exact rollouts (v1.4.0), masked per-world resets and vectorized impedance controllers (v1.5.0, experimental). The Python API is workable for an adapter today — estimate: a thin adapter at days, not weeks.

The real cost is not code, it's **parity**: Newton rebuilds the model through its own `ModelBuilder` semantics (its own defaults, `rigid_gap=0.1` propagation since v1.4.0, custom-attribute registration for `SolverMuJoCo`, coordinate-layout migration in flight) — so Newton-side physics is not guaranteed to match the `mjModel` that `mujoco.sysid` identified. By contrast, **mujoco_warp / MJX-Warp consume the actual mjModel** (`mjx.put_model(m, impl='warp')`), version-locked to the MuJoCo release you identify against.

## (b) Verdict on the standing position: STANDS (with one refinement)

- "MuJoCo is becoming a solver inside Newton; one converging stack, not a fork risk" — **STANDS, strengthened.** MJWarp is co-maintained by DeepMind+NVIDIA as part of Newton, release-locked to MuJoCo versions (v3.12.0 both sides, 2026-08-20), documented on mujoco.readthedocs.io, and MuJoCo upstream tracks Newton USD schemas. This is one organism, not a fork.
- "MJCF stays source of truth; USD carries scenes" — **STANDS.** The Newton project's own converter treats Menagerie MJCF as the input of record (85/85 convert) and emits standalone USD; Newton imports both.
- "Newton slots in behind a physics-backend seam for GPU rollouts" — **STANDS as architecture, premature as action.** Newton is GA (v1.0.0 Mar 2026, v1.5.0 Aug 2026, monthly cadence) but still breaks API every minor release, and its flagship consumer (Isaac Lab 3.0) is explicitly experimental with no manipulation RL support yet.
- **Refinement:** the convergence means the platform doesn't need Newton to get the Newton-era GPU solver. MJWarp is reachable three ways — raw `mujoco_warp`, `MJX-Warp (impl='warp')`, or Newton — and the first two preserve mjModel/sysid parity for free.

## (c) Recommendation

**Stay MJX/mujoco_warp for GPU batched manipulation rollouts; do not build the Newton adapter yet.** Concretely: implement the GPU backend (a seam that is not in the tree) against **MJX-Warp** (`mujoco-mjx[warp]`, `impl='warp'`) — same mjModel the sysid produces, version-locked to the CPU MuJoCo, 3M+ steps/s verified on contact scenes, and mjlab exists if an Isaac-Lab-style manager API is later wanted without Newton. Note the one trade: MJX-Warp has no autodiff — if differentiable rollouts are needed, that stays on MJX-JAX.

**Revisit Newton when X happens, where X =** Isaac Lab 3.0 reaches stable (non-beta) with Newton manipulation-RL environments supported — that is the signal that Newton's API churn has settled and the MuJoCo-solver path is production-hardened for arm workloads. (Secondary trigger, whichever comes first: the platform needs what only Newton has — cables/cloth/MPM multi-physics coupling, hydroelastic contact, RTX tiled-camera sensors, or USD-native scene rollouts.) Isaac Lab 3.0 stable is plausibly a late-2026 event given the beta started 2026-03-17.

Session caveat: fetches of blogs.nvidia.com and the mjwarp nightly-benchmarks page were permission-blocked, so all headline speedup multipliers (475x/252x vs MJX) remain CLAIMED from secondary coverage of GTC 2026; everything else above was fetched from primary sources on 2026-08-25.
