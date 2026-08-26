# Newton delta + live probe: it runs on the Mac, and it drops our sensors

*Fifth Newton pass, 2026-08-27 — one agent, primary sources plus a
live install on this Mac (Apple Silicon, no CUDA, scratch venv).
Companions: [47](47-newton-docs-review.md) (docs review, main @
1.6.0.dev0), [48](48-solver-landscape.md), [49](49-gpu-path-mjxwarp.md).
Everything below VERIFIED on 2026-08-27 unless flagged.*

## Release state: v1.5.0 (2026-08-11) is still the latest

Main is active (~20+ commits) with no release-prep and no flagged
SolverMuJoCo/MJCF/CPU breaking changes; monthly cadence puts v1.6.0
around mid-September. The coordinate-layout migration is unchanged:
`use_coord_layout_targets` still opt-in default-False, while Newton's
own examples already set it True.

**Install trap, recorded:** the PyPI package is **`newton`** — the
"obvious" name `newton-physics` is a stale placeholder frozen at 1.0.0
(2026-02-27). Installing by the wrong name gets a 6-month-old engine.

## The open question of docs/36, settled by running it

`uv pip install "newton[sim]"` is clean on Apple Silicon (newton 1.5.0
+ warp-lang 1.16.0 + mujoco/mujoco-warp **3.11.0** — one minor behind
the 3.12.0 we run, confirming docs/47's version-lag row). **All five
stable solvers step a box-on-plane scene on Warp's CPU backend,
including SolverMuJoCo** — the 36-report's "presumably needs CUDA"
worry was wrong. Measured (1000 steps, warm cache, one machine, one
run):

| Solver | steps/s | rest height (truth 0.100) |
|---|---|---|
| SolverSemiImplicit | 2553 | 0.0922 (penalty penetration) |
| SolverFeatherstone | 1699 | 0.0922 |
| SolverXPBD | 701 | 0.1000 |
| SolverVBD | 455 | 0.0845 |
| SolverMuJoCo | 378 | 0.1000 |

## Our robot through `add_mjcf()`: counts match, sensors vanish

- Bare `newton[sim]` fails on our so101.xml (`No module named
  'trimesh'` — mesh geoms need the importers extra; trimesh+scipy is
  the minimal fix).
- With that: **loads with zero warnings**, and every count matches
  MuJoCo's compile of the same file (bodies 7+world, nv=6, nq=6,
  ngeom=31; Newton adds one fixed world-weld joint).
- **The `<sensor>` block is silently ignored** — the installed MJCF
  importer module contains zero occurrences of "sensor". Our 12
  jointpos/jointvel sensors — the entire reason so101.xml wraps the
  Menagerie file — vanish without a warning. This verifies docs/47's
  parity-ledger row by execution: the robot-is-an-artifact contract
  (policies observe sensordata only) has no representation on the
  Newton side of `add_mjcf`.
- Throughput on the arm: SolverMuJoCo default config ~**290 steps/s**
  on CPU — but plain `mujoco` steps this model class at ~10⁵ steps/s,
  so Warp-CPU is roughly **300–1000× slower** for single-scene arm
  work, and Newton's own collision pipeline with mesh colliders is
  ~1 step/s on CPU (891 ms/step narrowphase). Mac development stays
  on plain MuJoCo; Newton-on-CPU is an API-compatibility checker,
  nothing more.

## The revisit trigger has NOT fired — and docs/47 §5 needs a correction

Isaac Lab's latest release is still v3.0.0-beta2.patch1 (2026-07-02);
its Newton-integration page (updated 2026-08-26) still says "only a
limited set of classic RL and flat terrain locomotion reinforcement
learning examples," **no manipulation/arm/gripper environments**, and
"surface gripper workflows remain backend-specific or unsupported."
Newton's own FAQ (main) claims "basic manipulation" among initial
Isaac Lab environments — **the two primary sources disagree, and
Isaac Lab's own docs are the authority on Isaac Lab's contents**.
docs/47 §5's "inching closer" read is hereby downgraded: the trigger
shows no sign of firing before late 2026.

## Net effect on the standing recommendation

Unmodified: stay MJX/mujoco_warp; revisit Newton at Isaac Lab 3.0
stable with manipulation support. The delta adds one plus (a future
Newton adapter could be smoke-tested on the Mac, no CUDA needed) and
one minus (the sensor drop is now verified by execution, not just
documented — sensor plumbing would be a rebuild against Newton's own
Sensor classes, strengthening the mjModel-parity argument for
MJX-Warp, which keeps sensors for free).

## Caveats

CPU timings are single-run, one machine, no GPU comparison possible;
the docs site's per-solver pages were not individually fetched (solver
roster read from installed v1.5.0 source); Isaac Lab develop-branch
activity not audited beyond releases + the integration page. Probe
scripts live in the session scratchpad only.
