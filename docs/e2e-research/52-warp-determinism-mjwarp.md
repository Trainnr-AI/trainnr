# Warp determinism × mujoco_warp: read at the source, one probe from settled

*2026-08-27. Source: local repomix packs of NVIDIA/warp (main; the
determinism guide `docs/user_guide/execution_and_performance/
deterministic_execution.rst`) and google-deepmind/mujoco_warp (main),
both reviewed here. This resolves docs/48's open flag #2 and upgrades
docs/49's determinism row. Everything is from their source unless
marked INFERENCE.*

## 1. What Warp's deterministic mode actually is

A **module-level compile option**, not a runtime switch: set
`wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN` **before
the kernels' modules are created** (or per-module via
`wp.set_module_options`) and Warp rewrites the supported
floating-point atomic patterns into sort-then-reduce (CUB radix sort +
reduce-by-key):

- **Pattern 1, accumulation**: `wp.atomic_add/sub/min/max` (and
  `arr[i] +=`) whose return is ignored — scatter records, sort by
  destination + thread index, reduce in fixed order.
- **Pattern 2, slot allocation**: `slot = wp.atomic_add(counter, i, n)`
  — count-scan-write, deterministic slots.
- NOT transformed: `atomic_cas`, `atomic_exch`, tile atomics.

Modes: `RUN_TO_RUN` = bit-exact repeats **on the same GPU
architecture** (fastest); `GPU_TO_GPU` = across architectures, much
slower. CUDA graph capture is supported (captured launches sort the
full fixed-capacity buffer — correct, slower; overflow checks off
during capture, so `deterministic_max_records` must be sized
conservatively).

**The cost is workload-shaped, not a constant** (their RTX 4090
measurement): high-contention atomics get FASTER under RUN_TO_RUN
(0.13×–0.76× the time — sorting beats a hot atomic address), while
low-contention cases pay 10–24×. Slot allocation pays ~6–8×.

## 2. mujoco_warp against that machinery (from its source)

| Question | Answer, from the pack |
|---|---|
| Which solvers exist? | their types module: `SolverType = {CG, NEWTON}` — "unsupported: PGS"; the solver module raises `NotImplementedError("noslip solver not implemented")`. **Our measured requirement (mjSOL_NEWTON) is first-class**; elliptic cones are first-class throughout (dedicated `ConstraintState.CONE` regime, tests parameterize PYRAMIDAL × ELLIPTIC) |
| implicitfast? | **RESOLVED (docs/48 flag #2): supported.** Regression tests exist ("implicitfast + tendon on serial chain must not NaN"), and their own benchmark scenes run `integrator="implicitfast"`. Only the *midpoint feature* of implicitfast is unsupported — the README's ambiguous line meant the sub-feature, not the integrator |
| Where does non-determinism live? | Hot-path reductions are `wp.atomic_add`: constraint ×52, smooth ×33, solver ×14, passive ×11, sensor ×9 (their _src modules) — **all Warp Pattern 1**. Contact/constraint row allocation is slot-style — **Pattern 2**. Exactly ONE `atomic_cas` in the tree: the island module's union-find, which is result-deterministic by construction (union-by-min converges to the same forest regardless of race order) — INFERENCE, from reading the kernel |
| Do they acknowledge it? | Their own tests sort efc rows before comparing ("Precompute sorting for efc fields to avoid non determinism") — constraint-row ORDER varies today under the default `NOT_GUARANTEED` mode. An RK4 repeatability test exists for the integrator side |
| Benchmarks | Nightly, published per scene (aloha_pot/clutter/sdf/cloth, unitree_g1, myoarm, franka, humanoid) at google-deepmind.github.io/mujoco_warp/nightly — the page fetches were blocked in earlier sessions; the harness ships in their repo (benchmarks/run) so the WSL box can produce OUR numbers directly |

## 3. The upgraded verdict on GPU determinism (revises docs/49)

docs/49 said "GPU non-determinism → certificates go statistical." The
source-level read upgrades that to: **run-to-run deterministic
MJX-Warp rollouts are structurally plausible** — the engine's
non-deterministic sites are precisely the atomic patterns Warp's
deterministic mode rewrites, and the one uncovered atomic is
result-deterministic. What remains between plausible and settled is
one WSL GPU probe:

1. `wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN` set
   **before importing mujoco_warp/mjx** (it is a compile option; the
   modules bake it in),
2. two identical kitting-scene rollouts, assert bit-equality,
3. the throughput cost measured on the same run (it may even be
   negative on contact-heavy scenes — high contention is where
   RUN_TO_RUN wins),
4. caveat to carry: same-GPU-arch only, and graph-captured launches
   sort full-capacity buffers.

If the probe passes, certificates from the GPU instrument can be
bit-reproducible on a named architecture — the instrument stamp
already carries the device, so the claim slots in with zero schema
change. If it fails, docs/49's statistical treatment stands.

**MEASURED the same day (`tools/determinism-probe.py`): the two
decisions are COUPLED.** Our pinned stack (mujoco~=3.11 lockstep →
warp-lang 1.14.0) predates the feature — `wp.DeterministicMode` does
not exist there; it ships with warp ≥ 1.16, which arrives with the
mujoco 3.12 pin. So deterministic GPU rollouts are unlocked by the
same deliberate 3.11→3.12 re-identification event docs/49's
postscript priced (nine broken tests' worth of physics drift to
re-referee). The probe degrades gracefully on 1.14 and reported the
CPU smoke: NOT_GUARANTEED already bit-equal on the sequential CPU
device, kitting 4 worlds × 1000 steps in 8.5 s warm — incidentally
the ALOHA bundle's first run through our own adapter
(naconmax=128, njmax=512 suffice).

## 4. Also settled by the packs

- **PGS/noslip absent from mjwarp is harmless**: our solver sweep
  (docs/47 §7) already disqualified both on the kitting referee.
- The ALOHA benchmark suite in mjwarp (pot/clutter/sdf/cloth scenes
  with the real meshes) is ready-made comparison material for our
  ALOHA bundle's `naconmax`/`njmax` sizing and throughput expectations
  on the 3090 Ti.
- Warp's deterministic mode extends to Tape backward passes —
  irrelevant today (mjwarp has no autodiff) but it means the
  determinism story will survive if differentiability ever lands.
