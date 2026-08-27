# SolverMuJoCo round-trip, measured: not the same robot

*2026-08-27, the queue's "probe SolverMuJoCo (save_to_mjcf diff,
deterministic mode)" item, run on this Mac in the newton-probe scratch
venv (newton 1.5.0, mujoco 3.11.0 — matching the repo's lock).
Method: `so101.xml` → `ModelBuilder.add_mjcf` →
`SolverMuJoCo(use_mujoco_cpu=True, save_to_mjcf=...)` → compile BOTH
files with the same MuJoCo and diff the compiled arrays. Everything
below MEASURED unless flagged.*

## The fidelity table

| Field | Source | Round trip | Verdict |
|---|---|---|---|
| nq / nv / nu / nbody / njnt | 6 / 6 / 6 / 8 / 6 | identical | SAME |
| dof damping / armature / frictionloss, jnt_range | — | identical | SAME |
| actuator gainprm (kp=50), ctrlrange, forcerange | — | identical | SAME |
| timestep, impratio, tolerance, iterations, cone, solver | — | identical | SAME |
| **nsensor** | 12 | **0** | the observation contract, gone (docs/47 §3, now round-trip-measured) |
| **nkey** | 2 | **0** | the reset mechanism, gone |
| **ngeom** | 31 | **18** | all 13 visual meshes (group 2, contype 0) dropped — physics-neutral, but a rendered policy would see a skeleton robot |
| **integrator** | Euler | **implicitfast** | the silent opinionated default, now written INTO the emitted MJCF |
| **body_mass[Base]** | 0.5625 kg | **0.1568 kg** | −72% on the root body (static here, so dynamically inert — but the authored value did not survive) |
| **actuator_biasprm kv** | −4.47…−5.13 | **0** | see below — the decisive row |
| Passive trajectory (200 steps, zero ctrl) | — | max qpos gap **1.26e-3 rad** | vs 3.5e-7 for MJX-Warp on the same model |

## The decisive row: servo damping does not survive IMPORT

The zeroed `kv` is not an export artifact. Probing the Newton model
directly after `add_mjcf`:

```
joint_target_ke: [50 50 50 50 50 50]   # kp imported
joint_target_kd: [ 0  0  0  0  0  0]   # kv DROPPED
```

Newton 1.5.0's MJCF importer carries the position actuator's `kp` into
`joint_target_ke` and silently discards its `kv` — so **Newton
simulates this robot with undamped position servos**, and
`SolverMuJoCo`'s internal mjModel (built from those Newton fields)
inherits the loss. Passive joint damping (`dof_damping`) does survive,
which is why the zero-ctrl trajectory gap stays small; under active
control the servo response is a different actuator. The actuator
damping is one of the parameters `mujoco.sysid` fits — the identified
quantity most directly erased by the conversion.

(Not verified at the time: whether some import flag or the `mujoco:*`
custom-attribute namespace can preserve kv.)

**Verified the next day, from the source** (the operator's pack of
`newton` main, 1.6.0.dev0 — *newton/_src/utils/import_mjcf.py*
`parse_actuators`, *newton/_src/solvers/mujoco/solver_mujoco.py*): the
importer does read a `<position>` actuator's `kv` and stores it in
`joint_target_kd` when `kv > 0`. Our servos never state one. Every
SO-101 servo is authored `<position kp="50" dampratio="1"/>`, and
**`dampratio` is resolved by MuJoCo's compiler** — `mj_setConst`
computes `kd = dampratio · 2 √(kp · acc0)` from each joint's reflected
inertia, which is why the compiled `biasprm[2]` reads −4.47…−5.13, six
different numbers the XML never contains. Newton's importer takes the
`dampratio` branch instead: it stores the ratio as a *positive
placeholder* in `biasprm[2]` and leaves `joint_target_kd = 0`. On the
way back, `SolverMuJoCo` recognises a positive `biasprm[2]` and calls
`mj_setConst` to recompute it — but only for the `CTRL_DIRECT`
actuator path; a `JOINT_TARGET` actuator is rebuilt from
`joint_target_ke`/`joint_target_kd`, and `kd` is the zero. So the
drop is specific and fixable upstream: the importer should resolve
`dampratio` the way MuJoCo does (it has the compiled model to read
`acc0` from) or carry the ratio through to the target-mode rebuild.
`mujoco.rst`'s own "Unsupported MuJoCo features" list names sensors,
cameras, lights, keyframes, composites, skins, plugins and user data —
and not this; it is a bug, not a documented limit. Filed against
nothing yet; the issue text is this paragraph.

Combined verdict, sharpening docs/47 §3: crossing into Newton (engine)
costs sensors, keyframes, visual geoms, the integrator choice, the
root body's authored mass, and the servo damping. "Not an adapter, a
port" now has a table.

## Deterministic mode: the mechanism, from their source

From the pack (docs/concepts + `newton/tests/determinism/`), for the
day the certificates want deterministic GPU rollouts:

- Warp ≥1.16 carries `wp.DeterministicMode` (e.g. `RUN_TO_RUN`) as a
  global/per-module kernel option; Newton solvers accept
  `deterministic=` and set it on their kernel modules — including
  `SolverMuJoCo`.
- Newton's own collision pipeline takes
  `CollisionPipeline(deterministic=True)`: fingerprint tiebreaks in
  contact reduction + a radix sort over `(shape_a, shape_b, sub_key)`,
  giving "bit-exact repeatability across runs on the same GPU
  architecture when the consuming solver is deterministic as well."
  Same-arch only, and it changes the hydroelastic contact set.
- **The caveat that matters:** their SolverMuJoCo determinism test
  runs with `disable_contacts=True` — articulation only. Determinism
  of the mujoco_warp CONTACT path is not established even by Newton's
  own suite.
- **Open question for our stack** (flagged for the WSL GPU probe):
  whether `wp.config.deterministic = RUN_TO_RUN` set before kernel
  compilation gives MJX-Warp (`impl='warp'`, no Newton) run-to-run
  repeatability on contact scenes, and at what throughput cost. If
  yes, docs/49's "certificates go statistical" softens to
  "deterministic mode available, same-arch, at a measured cost."
