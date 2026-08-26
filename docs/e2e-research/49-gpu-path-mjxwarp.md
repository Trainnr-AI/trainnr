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
| **GPU non-determinism** (FAQ: "ordering or small numerical differences"; CPU device for determinism) | certificates from the GPU path are statistical by construction — UPGRADED 2026-08-27 ([52](52-warp-determinism-mjwarp.md)): Warp's RUN_TO_RUN deterministic mode covers exactly the atomic patterns mjwarp's hot paths use; bit-reproducible same-arch GPU rollouts are one WSL probe from settled. Until then, statistical stands; CPU MuJoCo stays the metrology instrument |
| **No `rollout()`** | the adapter writes the loop: `set_state` → per-tick ctrl write → `step` → `get_state`/sensordata. Raw mujoco_warp's `set_state/get_state` speak concatenated mjtState rows with an `active` mask — a direct FULLPHYSICS analogue of our `Stepper` seam. Days, not weeks; Mac-testable |
| **Version lockstep** (mjwarp 3.12.0 released within an hour of MuJoCo 3.12.0; each mjx pins its warp-lang) | every MuJoCo upgrade is a re-identification event; the instrument stamp must be `mjwarp-x.y.z+warp-a.b.c+device` |
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

## Postscript (2026-08-27, same day): the adapter landed

`rq_pipeline/physics/mjx_backend.py` — `MJXWarpBackend`, the loop plus
two sizing knobs this report promised: consumes the SAME compiled
mjModel (`load_model` — one compile, two instruments), returns
FULLPHYSICS rows referees read unchanged, `naconmax`/`njmax` passed
through for busy scenes. The acceptance gauntlet
(`tests/test_mjx_backend.py`) admits it by contract: census parity
through the same gate, shape-refusal parity, instrument stamp
(`mjx-warp-3.12.0+warp-1.16.0+cpu` measured here), and a divergence
bound against the reference instrument — MEASURED on the Mac:
**3.52e-07 max over 3 worlds x 50 steps** on the pendulum scene
(bound set at 1e-3: a conversion-layer bug is orders of magnitude, not
float32 noise). All four tests green on the Warp CPU backend, exactly
as §1 said they could be. Remaining for the WSL card: the same
gauntlet on CUDA, ALOHA-scene sizing, and the vectorized env wrapper.

**And the re-identification cost got measured the same hour.** The
first draft of the `mjx` extra used an open `>=3.12` floor; uv's
relock dragged the whole project from locked mujoco 3.11.0 to 3.12.0,
and NINE tests broke — pybind-enum-vs-numpy comparisons (a hinge
stopped looking like a hinge to arm_ik) AND physics outcomes (the
stack ladder lost a trial, 3/4; the kitting choreography failed
mid-reach). This is the version-lockstep row of the cost table,
demonstrated on our own suite. Consequences applied: the sim extra
now PINS `mujoco[sysid]~=3.11.0` with the doctrine in a comment
(upgrades move by decision + full referee re-run, never as a
dependency side effect), the mjx extra pins `~=3.11.0` to match, and
arm_ik compares joint types through `int()` so the next binding-layer
drift cannot silently refuse hinges. Final stamp on this Mac:
`mjx-warp-3.11.0+warp-1.14.0+cpu`, divergence unchanged at 3.52e-07.
NOTE this also corrects the report above: the repo runs LOCKED
mujoco 3.11.0 (3.12.0 is upstream latest) — the probes' 3.12 numbers
came from a scratch venv resolving fresh.

## Postscript 2 (2026-08-27, the WSL box: the RTX 3090 Ti, MJX 3.12 + Warp 1.16)

The adapter's first run on a GPU and on a real scene, both the same
night. The pendulum gauntlet passes on the card with the same
divergence the Mac measured on Warp's CPU backend (3.52e-07). The
kitting bundle (105 geoms, 23 bodies) with MJX's default contact and
constraint capacities **dumped core** — not a warning, not an
exception: the process died. With `naconmax=4096, njmax=8192` it runs:
4.0 s to compile, 25 ms per call for 2 worlds × 20 steps from the
neutral pose, and it diverges from CPU MuJoCo by **9.6e-4 at step 9**,
only 7 of 20 steps within 1e-6. Two consequences, applied:
`MJXWarpBackend` refuses any scene above 32 geoms without explicit
sizing (a Python error naming the knobs, pinned on the kitting spec),
and the gauntlet's docstring carries the real-scene number beside the
pendulum bound. The "costs, named" above were right and are now
measured: the GPU path is a different instrument (its stamp,
`mjx-warp-3.12.0+warp-1.16.0+gpu+x86_64`, says so), and on a contact
scene the difference is a millimetre within ten steps — certificates
from it are statistical claims about the GPU instrument, never bitwise
comparisons with the CPU reference.

## Postscript 3 (2026-08-27) — the batched stepper

`MJXWarpBackend.stepper` exists: batched, every world in lockstep, one
jitted program per substep count, the forward pass recomputed after
each step so sensors and poses describe one instant (the CPU
stepper's R7 rule). On the pendulum its rows agree with the
whole-episode `rollout` to 1e-4 and with the CPU stepper to the
gauntlet's 1e-3, world by world. The vectorized gymnasium env over it
is the remaining piece of this report's plan.

