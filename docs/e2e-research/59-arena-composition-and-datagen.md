# Arena, second pass: composition, the data factory, and the walls — read for the cross-framework design

*2026-08-31. Source: a repomix pack of `isaac-sim/IsaacLab-Arena` (v0.3.0-dev,
992 files), three field agents — composition/portability, data
generation/policy runtimes, platform/positioning. EXTENDS docs 39–46
(Arena's evaluation internals, variations, placement, runner — not
re-covered here). Foreign paths carry an `arena:` prefix. Feeds the
consolidated design in [58](58-cross-framework-setup.md).*

## 1. What Arena is, in their words and numbers

"An open-source extension to NVIDIA Isaac Lab for simplified task
curation and robotic policy evaluation at scale" — a **config
compiler**: Scene/Embodiment/Task composed into Isaac Lab's
`ManagerBasedRLEnvCfg`. Apache-2.0 code atop a proprietary runtime
("requires Isaac Sim, which includes components under proprietary
licensing terms"), Linux x86-64 only by resolver pin, Python 3.12,
Isaac Lab 3.0.0 **beta** + Isaac Sim 6.0. Self-described "Alpha … Do
not use this in production." Contributors: NVIDIA + LightWheel only.
209 test files, 80-minute GPU CI on self-hosted runners; exactly ONE
test file in the core declares itself sim-free. Ships its own Claude
Code skills for install/eval ops — NVIDIA drives Arena through agents
too.

## 2. Composition: what actually decouples

- **Slot-merge composition** (`arena:isaaclab_arena/environments/
  arena_env_builder.py`): scene, embodiment, task, placement and
  variations each write into named MDP slots
  (`combine_configclass_instances("EventsCfg", …)`); actions come from
  the embodiment ALONE. Adding an axis never touches the others.
- **`ArenaEnvGraphSpec`** (`arena:isaaclab_arena/environment_spec/`):
  a Pydantic scene/task graph with ZERO simulator imports — validates
  registry names, relations, task params before any engine boots. Task
  YAMLs are overlays on scene YAMLs (robolab: 20 scenes × ~38 task
  overlays). The single most portable artifact in the repo.
- **Typed plugin registry**: `@register_asset/@register_task/
  @register_policy…` into singletons; `PolicyBase[MyCfg]` generics give
  a bidirectional cfg↔impl map by reflection. But discovery is a
  hardcoded import list — no entry points — and integration is
  officially "vendor Arena as an unmodified git submodule" (no pip
  package). External code enters via a `module:Class` dotted-path CLI
  flag, tested end to end.
- **The honest caveat**: of 58 shipped environment graphs, 56 pin the
  same single embodiment. "Swap the robot without touching anything
  else" is architecturally real and practically unexercised.

## 3. The walls, measured

- **USD in, nothing else**: `get_placement_geometry_source()` asserts
  `spawn.usd_path is not None`; no URDF or MJCF importer exists. URDF
  appears only as a runtime *export* (USD→URDF for the Pink IK
  controller; pinocchio for the G1). 26 of 35 task files import
  isaaclab; the codebase is import-*ordered*, not import-clean —
  comments warn that touching `pxr` before `SimulationApp()` segfaults.
- **The accidental portability contract** is exactly two seams: the
  remote-policy process boundary (`get_action(env, obs) → Tensor` over
  websocket/zmq, per-robot adapters, LeRobot data out — the policy
  never sees Isaac), and the G1 whole-body controller — ONNX weights +
  joint-order YAML + URDF kinematics via pinocchio, a stack that runs
  anywhere (they even rewrite the ONNX graph for dynamic batch).
- **The solver convergence**: `ArenaPhysicsCfg.newton` selects
  **MuJoCo-Warp via Newton inside Isaac Lab** (`MJWarpSolverCfg`,
  implicitfast, elliptic cone). The GPU/EULA tax is Kit and USD, not
  the physics — the physics is becoming the same engine mjlab runs and
  we instrumented (docs/49, 52).

## 4. No measurement layer here either

`sysid`/`identification`: **zero hits in 992 files**. Actuator gains
are hand-entered `ImplicitActuatorCfg` numbers — a Galbot arm ships
`stiffness = 1745.32922 * 1e3` with no source; a GR1 trunk is welded
with `stiffness = 1e9`. The one place gains provably mattered, the fix
was to copy the *training-time* values, not to measure the hardware:
the G1 docstring records that Arena's default stiffness (2–4× the
policy's training values) produced visible yaw drift. And the README
lists "Sim-to-real validated evaluation methods" under contributions
they want — the gap, stated by NVIDIA.

Reproducibility, from their own docs: `--placement_seed` defaults to
None ("placement stays non-reproducible across runs"); "there is no
variation seed … build-time variations are not locked by either seed";
episode records carry seed + variation draws but no config hash, no
git SHA, no Isaac version stamp — `get_isaac_sim_version()` exists in
the tree and is never written into a result.

## 5. The data factory (the part our synthetic-data phase reads twice)

Teleop (keyboard/spacemouse/OpenXR incl. Vision Pro) → Isaac Lab's
`arena:record_demos.py` with an Arena callback → **success-only HDF5**
(auto-reset on success, failures never saved) → `arena:annotate_demos.py`
subtask marking → **Isaac Lab Mimic** generation: per-subtask
`SubTaskConfig` (object ref, termination signal, offset range,
`nearest_neighbor_object` source selection, `action_noise=0.005`,
interpolation steps), success-gated loop (`generation_guarantee=True`,
`generation_keep_failed=False`, retry until N successes, abort at 25
failures). Published ratios: 10 demos → 50–100 generated, 30 min–4 h.
Export: HDF5 → LeRobot v2.1 via one declarative key-mapping dataclass
+ three joint-order YAMLs → GR00T finetune (delegated to GR00T's own
trainer; Arena trains nothing itself beyond rsl_rl passthrough).
Published zero-shot GR00T numbers in their own docs are honest and
sobering: 0.0 success on most DROID pick-place objects.

Realism knobs in the generation path: isotropic pose noise and
retargeting. **No dynamics randomization, no distributional check
that generated demos stay near measured behavior, and a dataset whose
provenance is a seed.**

## 6. What this read settles for the design (pointers into 58)

1. The cross-framework manifest should look like `ArenaEnvGraphSpec`:
   typed, sim-import-free, validated before any engine starts.
2. The plugin mechanics to copy are Arena's typed registries + dotted
   external-class path; the discovery to fix is entry points (both
   Arena and mjlab lack them).
3. The portable policy seam is settled ecosystem-wide: out-of-process
   policy servers + ONNX-with-metadata + LeRobot datasets. Every
   framework already meets us there.
4. MuJoCo-Warp is the converging solver across BOTH ecosystems — a
   solver-level actuator mechanism (per-step dof writes, 57 §2) has a
   path into each.
5. The synthetic-data shapes to borrow and the provenance/identified-
   dynamics gaps to fill are enumerated — 58 §8 carries them forward.
