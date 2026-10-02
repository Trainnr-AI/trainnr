# Arena's object and robot placement: relations, solver, validation — mapped to our harness

*Sprint pass, 2026-08-26. One agent, one field, one primary source: a repomix
bundle of the whole `isaaclab_arena` repository (132,303 lines). Every Arena
claim is cited as `file.py:N` — *file.py* the path inside the repo, `N` the
LINE IN THE BUNDLE (re-read with `Read offset=N`) — with a short quote.
Nothing is from memory of Arena. Our side is cited by repo path and line.*

Terms, defined once: **layout** = (x, y, z) plus yaw for every placed object;
**anchor** = an object the solver never moves (the table); **AABB** =
axis-aligned bounding box, the smallest world-axis-aligned box containing an
object; **candidate** = a solved layout not yet accepted; **validator** = a
pass/fail check on a candidate; **seed** = the integer fixing a random draw
so it replays; **paired trials** = trial *k* of every policy starts from the
identical initial state.

---

## 1. What Arena does

### 1.1 The relation schema — "where things start", declared

An object carries a list of relations (`PlaceableAsset.relations`,
`relations/placement_asset.py:40732`), each a `RelationBase` subclass
(`relations/relations.py:44394`). The vocabulary and constructor defaults:

| Class | Meaning and knobs (bundle line) |
|---|---|
| `IsAnchor()` | fixed reference; "won't be optimized … must have an initial_pose set" (`relations.py:44617-44619`) |
| `On(parent, clearance_m=0.01, edge_margin_m=0.05)` | child footprint inside the parent AABB top, inset by the margin; child bottom = parent top + clearance (`:44551-44573`) |
| `NextTo(parent, side, distance_m=0.05, cross_position_ratio=0, tolerance_m=1e-2)` | beside the parent on `Side.POSITIVE_X/NEGATIVE_X/POSITIVE_Y/NEGATIVE_Y` (`:44500-44533`, `Side` at `:44385`) |
| `NotNextTo(parent, side, tolerance_m)` | keep-out half-plane past one edge, clipped to the parent footprint (`:44579-44584`) |
| `AtPosition(x, y, z)` | pins chosen world axes; `None` axes stay free for `On` (`:44815-44820`) |
| `PositionLimitsBox(x_min..z_max)` / `PositionLimitsCylindrical(cx, cy, r_min, r_max)` | world box limits, each bound optional (`:44856-44863`); XY annulus (`:44904-44910`) |
| `FaceTo(target)` | yaw so local +X points at the target, "applied … post-solve" (`docs/.../relations.rst:5688`) |
| `RequiresReachability()` | "does not affect placement geometry … but rejects unreachable placements" (`:44644`) |
| `RandomAroundSolution(…half_m/rad)` / `RotateAroundSolution(roll, pitch, yaw)` | turn the solved pose into a reset range (`:44660-44673`) / add a fixed rotation (`:44758-44770`) |

The same schema is YAML: `- kind: 'on'  subject: mug  reference: table`
(`relations.rst:5735-5743`); a Droid robot is placed `on: floor`,
`next_to: right_counter_top` (`side: negative_y, distance_m: 0.15`),
`rotate_around_solution: yaw_rad: 1.57` (`solver.rst:5867-5884`). Collision
avoidance is not a relation: "integrated into placement and is not expressed
as a relation" (`:5750`). Two facts shape the rest: **positions only** —
state "always stored as (batch_size, num_objects, 3)"
(`relation_solver_state.py:43720`), yaw being zero, a marker, a `FaceTo`
heading, or with `random_yaw_init=True` "a random fixed yaw … Not optimized"
(`object_placer_params.py:39749-39751`); and **AABB footprints** — `On`
"samples positions within the anchor's axis-aligned bounding box footprint
… For non-rectangular surfaces … may fall outside the actual surface"
(`object_placer.py:39873-39877`), deep `On` chains a TODO (`:40299`).

### 1.2 Losses: every relation is a hinge in metres

Each relation maps to a loss strategy (`relation_solver.py:43943-43946`) built
from four primitives in *loss_primitives.py*: `single_boundary_linear_loss`
"ReLU-style loss: zero at boundary, linear growth" (`:38702`),
`linear_band_loss` zero inside [lo, hi] (`:38757`), `single_point_linear_loss`
"V-shaped" (`:38784`), `interval_overlap_axis_loss` "slope * overlap length"
(`:38821`). Loss 0.3 at slope 10 means 3 cm of violation; nothing is squared.
`OnLossStrategy` = X band + Y band (parent extent minus child extent, inset by
the margin) + Z point at `parent_z_max + clearance_m - child_bbox.min_z`
(`relation_loss_strategies.py:43264-43291`). `NoCollisionLossStrategy`
penalises the **overlap volume** of two AABBs after growing the obstacle by
`clearance_m`: `slope * (overlap_x * overlap_y * overlap_z)`
(`:43412-43423`). Default slopes: NextTo 10, On 100, NotNextTo 10 with 0.1 m
margin, AtPosition/PositionLimits 100 (`relation_solver_params.py:
43641-43646`); no-overlap 10000 "so overlap avoidance dominates"
(`relation_solver.py:43961`). NextTo geometry is one shared function used by
loss and validator "so loss and validation can't disagree"
(`relation_loss_strategies.py:43004-43005`).

### 1.3 The solver: batched Adam, seeded per candidate

`RelationSolver.solve()` picks `"cuda" if torch.cuda.is_available() else
"cpu"` (`relation_solver.py:44127`), runs
`torch.optim.Adam([state.optimizable_positions], lr=…)` (`:44185`) for
`max_iters=600` at `lr=0.01`, stopping below `1e-4`
(`relation_solver_params.py:43654-43660`). Anchors must be identical across
the batch (`relation_solver_state.py:43774-43780`). Movable-vs-fixed pushes
only the movable; movable-vs-movable is scored twice with the other side
`.detach()`ed (`no_overlap_aabb.py:39089-39090`, `:39166-39168`); `On` pairs
are skipped (`:39112-39117`). `ObjectPlacer` wraps it: "1. Random
initialization … 2. Running the RelationSolver on all candidates in one batch
3. Validating … 4. Ranking … (valid first, then by loss) 5. Applying the best
layout" (`object_placer.py:39864-39869`), `max_placement_attempts=10`
candidates per environment (`object_placer_params.py:39753`).

**Seeding is per candidate.** A `torch.Generator` exists only when
`placement_seed` is set (`object_placer.py:39999-40002`) and is re-seeded
`generator.manual_seed(self.params.placement_seed + candidate_idx)` (`:40038`).
An `On` child starts uniform in the feasible band
`[parent_min - child_min, parent_max - child_max]` per axis (`:40374-40378`),
Z "so the bottom face lands on the parent top" (`:40345-40346`); others start
at the first anchor's centre (`:40176`). Object-set identity uses
`placement_seed + variant_set_idx` (`bounding_box_helpers.py:38515`); refills
advance the seed by the candidate count (`pooled_object_placer.py:
42678-42683`); per-environment RNGs come from one seeder via
`getrandbits(64)` (`utils/random.py:88060-88070`). Same definition + seed +
env count ⇒ same layouts (`pooled_placement.rst:5509-5511`); `--seed` is sim
randomness, `--placement_seed` placement randomness (`:5524-5525`).

### 1.4 Validation: pass/fail over the loss, required vs optional, cheap before expensive

"A low loss does not guarantee that a relation holds exactly … Validators are
the pass/fail layer" (`validation.rst:5925-5928`). Checks are the
`PlacementCheck` enum: `no_overlap`, `on_relation`, `next_to`, `not_next_to`,
`face_to`, `physics_settled` (run-time), `ik_reachable`
(`placement_validation.py:41264-41290`). Verdicts are
`PlacementValidationResults` = `dict[str, bool]` + `required_checks`, `None`
meaning every check that ran (`:41300-41310`). A validator is a registered
class with `validate_batch(positions, orientations, bboxes,
collision_objects) -> list[bool]` (`placement_validators.py:41472-41479`) and
a flag `run_after_inexpensive_checks` so "an expensive check (e.g. IK
reachability) never runs on a layout rejected on cheaper geometry"
(`:41454-41457`; driver `object_placer.py:40447-40497`). `on_relation`:
footprint inside the parent inset by the margin, `child_bottom` in
`(parent_top - eps, parent_top + clearance + eps]`, `eps =
on_relation_z_tolerance_m = 5e-3` (`:41586-41605`, `object_placer_params.py:
39766-39768`), refusing a margin the surface cannot honour (`:41569-41584`).
`no_overlap`: AABB overlap with `margin = clearance_m - 1e-6`
(`:41876-41885`), or sphere-vs-signed-distance-field in `MESH` mode, failing
if `sdf < radius + tolerance` (`:42074-42081`). Ranking sorts by
`(required_failures, optional_failures, loss)` (`object_placer.py:
40117-40123`). `debug_visualize=True` streams candidates to Rerun, or an
`.rrd` headless (`object_placer_params.py:39793-39797`).

`physics_settled` is **not** a build-time gate. A separate audit
(`placement_pool_validation.py:41117`; script at bundle line 45878) writes
each stored layout into the sim, steps `num_steps=5`, and stamps settled iff
every movable object's speed ≤ 0.1 m/s and ≤ 0.1 rad/s
(`physics_settle_params.py:40678-40686`, `utils/physics_settle.py:
87858-87862`). The reset writer sets pose + zero velocity and notes "the sim
will still apply gravity and other forces from collisions"
(`placement_events.py:40964-40966`, `:40988-40989`).

### 1.5 Collision handling

`CollisionMode.BBOX` (AABB, default) vs `MESH` ("Bounding spheres queried
against collision-mesh geometry", `collision_handling.rst:5054-5061`,
`num_spheres=30`, `relation_solver_params.py:43675`). Pairs checked:
movable/movable, movable/anchor, movable/passive obstacle — "Except when one
is directly `On` the other"; fixed/fixed never (`:5158-5178`). Background
geometry is a **passive obstacle**: "contributes collision geometry but the
solver does not move it" (`:5192`). In BBOX mode an anchor's rotation snaps
to quarter turns (`placement_asset.py:40832-40841`) — an AABB cannot hold a
table at 30°.

### 1.6 Pooled placement, resets, homogeneous vs heterogeneous

`PooledObjectPlacer` keeps one `EnvLayoutPool` queue per environment
(`pooled_object_placer.py:42576-42600`), `num_envs *
min_unique_layouts_per_env` (default 5) layouts (`relation_solver_interface.py:
32890`, `object_placer_params.py:39774`); each reset consumes the next
(`:42805-42822`); `resolve_on_reset=False` freezes one `PosePerEnv`
(`:33012-33017`). If a refill yields no valid layout, by default the
best-loss failure is stored anyway (`allow_best_loss_fallbacks=True`,
`:42700`, `:42757-42767`) and reset prints "Writing best-loss fallback
placement" (`placement_events.py:41024-41027`). An explicit pose on a
relation-placed object is an assertion error (`relation_solver_interface.py:
33020-33030`). **Homogeneous** = one registered object fills a role
everywhere; **heterogeneous** = a `RigidObjectSet`, "Each environment
receives one member" (`homogeneous_and_heterogeneous_placement.rst:5283-5289`),
identities fixed once per build (`object_set.py:25266-25270`), each member's
own AABB fed to the solver (`:25290-25302`). Robots join if they "provide
placement bounds" — the Droid uses only its stand footprint
(`solver.rst:5850-5853`, `:5892-5893`).

---

## 2. What we already have

- **Paired, deterministic starts by construction.** `EpisodeProtocol.perturb(trial_index, home_state)` "receives the trial index (not an RNG) so trials are paired across policies" (`trainnr/trainnr/evaluate/harness.py:49-51`); `score_policies` runs every policy through `protocol.perturb(trial, home)` (`:133-139`). Arena gets the same property from seed + candidate index (`object_placer.py:40038`); both make trial *k* independent of the trial count.
- **A declared home.** `home` names the start keyframe, resolved by `backend.keyframe_state` (`harness.py:54-67`, `:93-97`; the 1 kN jam behind it, `docs/31-aloha2-e2e.md:53-67`). Arena's analogue is the anchor's fixed `initial_pose`.
- **Spawn bands = `PositionLimitsBox` + `On`, hand-coded.** `PART_SPAWN` per arm (`trainnr/trainnr/tasks/aloha2/kitting.py:399-402`), `CUBE_SPAWN_X/Y` (`:62-63`), parts resting at `z = PART_HALF` (`:403-406`), the box body in `_add_free_box` with keyframes extended (`:259-283`, `:192-199`). `_corner_fraction(trial, inset)` cycles four corners (`:313-320`); kitting uses one draw for both parts (`:486-494`).
- **Seeded uniform draws + rejection by the referee.** `tools/kitting-demos.py` seeds `np.random.default_rng(SEED)` (`:86`), draws in the proven front half (`:110-120`), keeps an episode "ONLY if the task's own referee scores it a success" (`:15-22`), and writes `seed/attempt/draws` to a manifest (`:168-175`). That is a validator — but it costs a 28 s episode and judges the demo, not the placement.
- **Missing:** overlap, on-support, reachability and settle checks; placement verdicts in the record; a schema. "Where a part starts" is four constants and a slice index (`PART_STATE_SLICE`, `aloha2.py:409`).

---

## 3. What to ADOPT (on MuJoCo/MjSpec) — and what to skip

### ADOPT

**A1. A relation schema in the protocol: positions only, three relations.**
Frozen dataclasses in `trainnr/evaluate/placement`: `Anchor(body)`,
`On(body, parent, clearance_m=0.0, edge_margin_m)`, `Limits(body, x, y)`, and
a `RequiresReachability(body, arm)` marker. A task declares
`placements=[On("part_right", "table"), Limits("part_right",
*PART_SPAWN["right"]), …]` — today's four numbers, hash-stamped like Arena's
YAML `relations:` block (`relations.rst:5735-5743`). Body AABBs come from the
compiled model's geom sizes (our parts and table are boxes), so the schema
adapts when `PART_HALF` or the table changes — Arena's own motivation
(`concept_object_and_robot_placement.rst:7489-7516`). Skip `NextTo`, `FaceTo`,
cylinders until a task needs them ("Avoid specifying more relations than the
environment needs", `relations.rst:5710`).

**A2. Seed-per-trial sampling, pairing kept.** Adopt `seed + candidate_idx`
(`object_placer.py:40038`) as `np.random.default_rng([protocol_seed, trial,
attempt])`; sample each `On` child uniformly in Arena's feasible band
`[parent_min - child_min, parent_max - child_max]` per axis, intersected with
`Limits`, Z = parent top + clearance − child min (`:40345-40378`). That is
Arena's *initialiser*; for boxes on a rectangle with disjoint bands it is
already the whole solver (S1). Test the invariant: trial *k*'s start is
unchanged by `trials` and identical for every policy. Keep `_corner_fraction`
as `design="corners"` — a stratified design the uniform draw is not.

**A3. A validation layer: required/optional, cheap before expensive, no
silent fallback.** `PlacementVerdicts = dict[str, bool]` + `required`,
mirroring `PlacementValidationResults` (`placement_validation.py:
41293-41357`). Cheap: `in_limits`; `on_support` (footprint inside the support
inset by the margin, bottom within ±5 mm of the top — Arena's
`on_relation_z_tolerance_m`); `no_overlap` with MuJoCo as the checker — write
the candidate to `qpos`, `mj_forward`, fail on any contact between placed
bodies or with a non-support geom (tray walls): Arena's pair table
(`collision_handling.rst:5158-5178`) on the simulator's own geometry.
Expensive, gated like `run_after_inexpensive_checks`
(`placement_validators.py:41454-41457`): `reachable` — our chained `arm_ik`
to the grasp pose at `grasp_axis` (`aloha2.py:447-453`, `:559`), the
`RequiresReachability` idea (`relations.py:44644`) with our IK instead of
cuRobo; `settled` — hold `NEUTRAL_CTRL`, step N physics steps, require
free-body speeds ≤ 0.1 m/s and 0.1 rad/s (`physics_settle_params.py:
40678-40686`), and take the settled state as the trial's initial state. On a
required failure resample with `attempt+1` up to `max_attempts`; on
exhaustion **raise** — Arena's default `allow_best_loss_fallbacks=True`
(`object_placer_params.py:39779`) has no place under a certificate; Arena's
own debugging advice turns it off (`collision_handling.rst:5258-5259`).

**A4. Precompute every trial's initial state before any episode; stamp the
verdicts.** Arena builds its pool at compile time
(`concept_object_and_robot_placement.rst:7529-7531`); ours is the tuple of
`trials` states computed in `score_policies` before the loop — the census
gate applied to placement: a protocol that cannot place trial 3 fails before
an episode is spent. Record `{seed, attempt, draws, verdicts}` per trial as
`kitting-demos.py` already records draws (`:168-175`); log candidate boxes
and per-check verdicts to Rerun as Arena does (`validation.rst:6014-6026`).

**A5. Later: variants as a seeded identity draw.** Choose a part variant per
trial with `placement_seed + set_idx` (`bounding_box_helpers.py:38515`) and
feed that variant's AABB to A2/A3 — the one-line form of `RigidObjectSet`.

### SKIP

**S1. The differentiable Adam solver.** Torch on the sim side breaks "this
side stays torch-free" (`kitting-demos.py:25`); with 1–3 boxes on a
rectangular table and disjoint bands the feasible region is a rectangle, so
seeded sampling + A3 finds a valid layout in one attempt almost always. Arena
needs gradients for `NextTo`/`NotNextTo` chains among many objects. Revisit
when a task declares a chain.

**S2. `MESH` mode (Warp meshes, SDF spheres).** MuJoCo has the collision
geometry already; `mj_forward` contacts are exact for boxes and free.

**S3. Per-environment pools, reset events, `PosePerEnv`.** Isaac Lab
parallel-env plumbing (`placement_events.py:40880-40903` exists to survive
Isaac Lab's deepcopy). Our trials are sequential; A4's tuple is the pool.

**S4. Robot embodiment placement and pose modifiers.** ALOHA 2 is bolted to
the table; the keyframe is its placement. `RandomAroundSolution` is what
`perturb` already is.

---

## 4. Open questions

1. **Yaw.** Arena never optimises yaw, samples it only on request. The
   kitting close is IK-nullspace-random about the closing plane
   (`docs/31-aloha2-e2e.md:122`): should the protocol sample part yaw
   (paired, seeded), and does `reachable` then depend on it?
2. **Which state is "initial" — written or settled?** Arena writes pose +
   zero velocity and lets gravity act; its settle audit reports, never
   gates. With soft contacts (`solimp` in `_add_free_box`) a part sinks by
   microns. Paired trials need one convention, hash-stamped.
3. **Protocol hash scope.** Arena's refill seeding depends on
   `max_placement_attempts` (`pooled_object_placer.py:42727-42728`): a solver
   knob changes layouts. Which of `max_attempts`, tolerances, settle steps
   enter our `recipe@hash`?
4. **Relations as predicates.** Arena has no `Inside`; our in-slot success is
   hand-written (`aloha2.py:496-507`). One geometry function serving spawn
   validation and the referee — Arena's `next_to_violations` pattern — would
   remove a class of spawn/judge disagreements. Worth it now?
5. **Reachability as gate or report.** Filtering unreachable starts lifts
   every policy equally but also defines the task. Required or optional —
   Arena allows either (`validation.rst:6067-6073`)?

## Postscript (2026-08-27) — A1, A3 and A4 built

`protocol.Placement(body, support, x, y, tolerance_m)` is the schema
(positions only, one relation: on a support inside bands — smaller
than A1's three, because no task needed `NextTo`). `physics/placement.py`
is the validation layer: `in_limits`, `on_support` and `no_overlap`, all
required, on `geom_aabb` and `mj_forward`'s contacts; `require_start`
raises with the trial, body and check named. The harness precomputes
every trial's start and validates all of them before the first episode
(A4); the env does the same at `reset`; the verdicts are on every
record. Measured while building it: a box resting on the table at
exactly its half-height produces **no contact** in MuJoCo at margin 0,
so "on support" had to be geometric (bottom within ±5 mm of the top, the
footprint inside), not read from contacts. Skipped, as planned: the
solver, the seeded sampler (A2), the optional expensive checks, the
Rerun boxes.

