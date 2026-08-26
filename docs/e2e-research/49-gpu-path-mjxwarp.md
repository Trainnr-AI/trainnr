# The GPU path, probed: MJX-Warp runs here, batches models, costs float32

*2026-08-27, one agent against primary sources AND a live install
probe on this Mac (Apple Silicon, no CUDA; scratch venv, versions:
mujoco 3.12.0, mujoco-mjx 3.12.0, warp-lang 1.16.0). Companion to
[47](47-newton-docs-review.md)/[48](48-solver-landscape.md). The
standing recommendation of [36](36-newton-status.md) survives contact
with the machine and gets cheaper than expected.*

## Headline, all VERIFIED by execution on 2026-08-27

- **`uv pip install "mujoco-mjx[warp]"` works on the Mac.**
  mujoco_warp is vendored inside mujoco-mjx; Warp's CPU backend runs
  the full pipeline on Apple Silicon (no Metal, CPU device — the
  mujoco_warp README: NVIDIA GPU "for fast simulation, but supports
  CPU for development and debugging").
- **Both real bundles load and step** under `impl='warp'`:
  aloha2-nominal (nu=14, nsensor=28, ngeom=95 — needs the documented
  `naconmax`/`njmax` sizing knobs, warnings name the numbers) and
  so101-nominal (nu=6, nsensor=12, ngeom=31).
- **Feature coverage of our contact regime is total.** MJX support
  table: cone "All" (elliptic in), condim "All", solvers all except
  PGS/noslip, sensors all except PLUGIN, geoms/joints "All". Our
  measured requirement (mjSOL_NEWTON + elliptic, docs/47 §7) is
  fully covered; the gaps (PGS, noslip, plugins, flex) touch nothing
  we run.
- **Per-world MODEL batching is first-class** — MEASURED: vmap over
  `mx.tree_replace` on the exact fields `physics/variations.py`
  scales (`actuator_gainprm[:,0]` + `biasprm[:,1]`, `dof_damping`,
  `body_mass`/`body_inertia`) produced distinct trajectories across
  4 worlds. MJX docs: "Batch dimensions are a natural way to express
  domain randomization (in the case of `mjx.Model`)." Strictly better
  than CPU `mujoco.rollout`'s homogeneous-model-sequence trick for
  our scalar DR. Raw `mujoco_warp.put_model(mjm, batch_sizes=...)`
  offers the same per-field batching without JAX.
- **Measured sim-to-sim divergence** (warp-CPU vs CPU MuJoCo, box
  drop with impact under our contact options): max 2.7e-5 m at step
  100, settling 1.4e-5 m at 0.5 s — float32-scale, far under our
  tightest tolerance (±2 mm), but real; chaotic contact sequences
  amplify it.

## The costs, named

| Cost | Consequence |
|---|---|
| **float32** (MJWarp FAQ: floats vs MuJoCo's doubles) | divergence is expected, statistical treatment required |
| **GPU non-determinism** (FAQ: "ordering or small numerical differences"; CPU device for determinism) | certificates from the GPU path are statistical by construction — batch statistics, never single-trajectory reproduction. CPU MuJoCo stays the metrology instrument; MJX-Warp is the throughput instrument |
| **No `rollout()`** | the adapter writes the loop: `set_state` → per-tick ctrl write → `step` → `get_state`/sensordata. Raw mujoco_warp's `set_state/get_state` speak concatenated mjtState rows with an `active` mask — a direct FULLPHYSICS analogue of our `Stepper` seam. Days, not weeks; Mac-testable |
| **Version lockstep** (mjwarp 3.12.0 released within an hour of MuJoCo 3.12.0; mjx pins warp-lang==1.16.0) | every MuJoCo upgrade is a re-identification event; the instrument stamp must be `mjwarp-x.y.z+warp-a.b.c+device` |
| **The batch renderer is a different camera** | MJX-Warp batch rendering (MuJoCo 3.6.0+) is a BVH **raycaster** — it can render per-step observations in batched rollouts (nworld fixed at context creation; known vmap(scan) issue), but its frames are not our eye-calibrated OpenGL frames. Batched-vision scores need their own camera-match pass before any comparison (inference, flagged) |
| WSL gotcha (already recorded in rl-watch.py) | without `LD_LIBRARY_PATH=/usr/lib/wsl/lib` Warp **silently falls back to CPU** — a WarpBackend must assert `wp.get_cuda_device_count() > 0` when GPU is intended |

## What the repo already proves

MJX-Warp is not a candidate here — it is the **incumbent** GPU path
with a production data point: T6 PPO on the 3090 Ti under WSL2 at
36k env-steps/s across 2,048 worlds on a mesh-contact ALOHA scene
(20× MJX-JAX), JAX→Warp FFI working, `spawn` not fork, viewers in
their own process. The open work is the adapter/env shim plus sizing
knobs — feasibility is settled. Per the repo's own seam decision
(physics/backend.py, 2026-08-26: the Protocol was removed), **the
gymnasium env over the same tasks is the integration point**, not a
resurrected backend protocol.

## Partial resets and API notes (verified)

- MJX under warp impl: `jax.tree.map(where, ...)` does NOT work;
  use `data.where(done, reset_data)`.
- `make_data(..., naconmax, njmax)`: size for the busiest world and
  scale `naconmax` with batch size (MJX docs); ALOHA at defaults
  overflows with named numbers in the warning.
- Throughput on MuJoCo's own scenes (their docs): 2.96–3.35M steps/s
  pure Warp; Aloha-pot 2.33–2.45M. GTC's 475×/252× multipliers
  remain CLAIMED (secondary sources only).

## Not verified

GPU-side behavior of the probes (needs a one-hour WSL re-run — probe
kept in the session scratchpad); Warp-CPU vs Warp-GPU agreement;
batch-render quality on our camera specs.
