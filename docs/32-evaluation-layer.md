# The evaluation layer, minimum lines: standard on the outside, ours on the inside

*Written 2026-08-26, the evening T5's loop closed. The operator's question:
"how can we create the most standard, ecosystem-fit, production-grade,
minimum-lines evaluation for ourselves — what can be used from open
source, what needs to be built — adhering to our coding standards?" This
is the answer, built on three primary-source reads made the same day:
LeRobot's evaluation contract from the installed package
([e2e-research/45](e2e-research/45-lerobot-eval-contract.md)), six
ecosystem interfaces from their repositories
([46](e2e-research/46-ecosystem-eval-interfaces.md)), and a line-by-line
audit of our own surface (numbers below are from it), on top of the
Arena sweep ([30 §7](30-the-full-loop.md)). It is a plan, not a build;
nothing here is implemented yet.*

Terms, defined once. An **environment** (env) is the simulated task as
code: holds state, takes an action, returns an observation. **Gymnasium**
is the Python interface the ecosystem speaks (`reset` → observation,
`step` → observation, reward, terminated, truncated, info). **LeRobot** is
Hugging Face's robot-learning library — trainer, policies, datasets, and
an evaluator (`lerobot-eval`). A **policy** maps observation to action. A
**trial** is one episode of the harness; trials are **paired** when trial
*k* of every policy starts from the identical state. The **certificate**
is our ranked-policies report with exact intervals and a gate on the
lower bound. A **record** is one JSON line per trial. `name@hash` is our
stamp: nothing is nameable without the hash of its content.

## 1. The answer in one paragraph

Split the layer at the boundary the ecosystem already draws. **Outside:
a gymnasium env.** Our tasks become one `gym.Env` class (~150 lines)
plus a LeRobot `EnvConfig` (~40) so that `lerobot-eval` runs the
rollouts, `lerobot-train` evaluates in-loop through the same env, every
LeRobot policy family loads through LeRobot's own processors, and GR00T /
openpi / SimplerEnv-style clients can drive the same env through an
adapter. **Inside: our judge, unchanged.** The stats package, the
certificate, the census gate, the paired protocol and the stamp stay
exactly as they are — nothing in the ecosystem provides any of them (46
§4: no interval anywhere in six systems). **Between them: the record.**
One JSON line per trial, in LeRobot's `eval_info.json` shape widened
with our identity fields, is what the certificate folds over. Net: the
library's evaluation path shrinks from 4,022 lines to about 1,750,
with the observation contract, the stepping loop, the keyframe ritual
and the control rate each defined once instead of three or four times.

## 2. What the ecosystem gives us (adopt, do not build)

| Need | Open-source piece | Fact that makes it safe |
|---|---|---|
| The env interface | gymnasium 1.3.0 (`gym.Env`, `SyncVectorEnv`/`AsyncVectorEnv`, `AutoresetMode`) | LeRobot 0.6.1 pins `gymnasium>=1.1.1,<2`; every system in 46 speaks it |
| Rollout runner, batching, video, per-episode JSON | `lerobot-eval` (*lerobot/scripts/lerobot_eval.py*) | reads four things from an env: `pixels`/`agent_pos` observations, `info["is_success"]`, `_max_episode_steps`, `task_description` (45 §1.2–1.4); seeds episode *i* with `seed + i` (45 §1.5) |
| In-loop eval while training, on OUR task | `lerobot-train` calls the same `make_env` and `eval_policy_all` at `env_eval_freq` (verified: *lerobot/scripts/lerobot_train.py* L84, L629, L720) | the "watch training in the sim" the operator asked for, through the ecosystem's own path |
| Policy loading, pre/post processors, action-chunk queues | LeRobot's `make_policy` + `make_pre_post_processors` — ACT, SmolVLA, π0/π0.5 (LeRobot format), XVLA | replaces our `lerobot_checkpoint_policy` adapter and its hand-rolled uint8→CHW conversion (45 §2) |
| Plugin registration without forking | `register_third_party_plugins()` imports any installed distribution named `lerobot_env_*` (*lerobot/utils/import_utils.py* L223–236); or `--env.discover_packages_path=<module>` on the CLI (*lerobot/configs/parser.py* L43, L126–128) | both verified in the installed package |
| Sharing the env with others | the EnvHub door: a repo with *env.py* exposing `make_env(n_envs, use_async_envs, cfg)` (46 §1.5) | Arena publishes this way; the contract is the code above |
| Recording rollouts as data | `--eval.recording=true` writes a LeRobotDataset with `next.success` per frame (45 §1.7) | the DAgger/correction loop's raw material, free |
| Tracking | wandb API surface (docs/30 §6, decided) | unchanged |

## 3. What nothing upstream provides (keep, and it is already written)

From the audit's "must stay" table, with line counts as measured:

- `stats/` — Clopper–Pearson, Fisher-z rank interval, exact permutation
  p, `top_pick_probability`, pooling with Cochran's Q: **647 lines,
  stdlib only.** Every system in 46 aggregates with a plain mean.
- `evaluate/certificate.py` — gate on the lower bound, stamps, n, exact
  p: **171.** No counterpart anywhere.
- `evaluate/armnetbench.py` — the real side of Gate A: **120.**
- `robot/model_checks.py` — the census gate (actuators, sensors, geoms,
  cameras alive before an episode is spent): **46.** Gymnasium's env
  checker validates spaces, not that a robot exists; LeRobot disables
  even that by default.
- `bundles/hashing.py` — `name@hash` and the refuse-unstamped rule: **43.**
  `eval_info.json` carries no identity of what was evaluated.
- `EpisodeProtocol` — trials, steps, control interval, `perturb(trial)`,
  the success predicate over the hold window, the keyframe home: **~35.**
  Gymnasium resets take an RNG seed; pairing by trial index is our
  contract on top of it.
- The paired fold and `join_with_real`: **~60.**

Total essential ≈ 1,470 lines, and none of it changes shape.

## 4. What to build (about 300 lines, three files)

**4.1 The env — `rq_pipeline/envs/lerobot` (~220 code lines).** One
class per rig, not per task: `RobotiqEnv(task, ...)` where `task` is
the existing `build_transfer_cube()` / `build_kitting()` result.

- `metadata = {"render_modes": ["rgb_array"], "render_fps": 50}` —
  the control rate, computed from the model timestep and
  `control_interval`, defined once (today it is spelled three ways).
- `observation_space = Dict({"pixels": Dict({cam: Box(0, 255, (H, W, 3), uint8)}), "agent_pos": Box((state_width,), float32)})`;
  `action_space = Box(model.actuator_ctrlrange)`. These are LeRobot's
  raw names (45 §1.3) and the ecosystem's common denominator (46 §3).
- `reset(seed, options)`: `trial = seed % protocol.trials`; initial state
  = `protocol.perturb(trial, home)`; **seed *k* is start *k* for every
  policy** — pairing survives `lerobot-eval` unchanged because it seeds
  episode *i* with `seed + i` (45 §3).
- `step(action)`: advance `control_interval` physics steps with the
  same-instant rule (`mj_forward` after `mj_step`, measured one-tick
  sensor lag — the R7 rule, one place); keep the physics-rate
  `(steps, nstate)` history because every referee reads the last
  `_HOLD_STEPS`; on the last tick `ok = protocol.success(states, sensors)`;
  return `truncated=True`, `terminated=False`, `reward=float(ok)`, and
  `info={"is_success": ok, "success": ok, "task": name}` — both names,
  because LeRobot reads one and GR00T/SimplerEnv/robomimic the other
  (46 §3.3); never `done = success` (LIBERO's conflation, which
  gymnasium's own text forbids).
- `render()` → the first camera's frame (what `lerobot-eval` stacks into
  mp4s); `task_description` and `_max_episode_steps` as attributes.
- The census gate runs in `__init__`; an unstamped `source` is refused
  there too — the same rules `score_policies` enforces today, moved to
  where the env is born.
- `RobotiqEnvConfig(EnvConfig)` registered as `robotiq_aloha2`:
  `features`, `features_map` (`agent_pos → observation.state`,
  `pixels/top → observation.images.top`), `create_envs` building
  `SyncVectorEnv`/`AsyncVectorEnv` over `partial(RobotiqEnv, ...)`, and
  `get_env_processors()` carrying the measured gripper mapping
  (closed = 0.0078 m) as an env post-processor, so the action space
  stays honest in metres.

**4.2 The Stepper (~40 lines moved, none duplicated).** `_closed_loop`
runs a whole episode and its docstring forbids forking the stepping
discipline; the env needs one control tick at a time. Extract a
`Stepper` (seat state; advance *k* substeps; rolling states/sensors
tails) that `_closed_loop`, the env, and the kitting choreographer's
`advance` all consume. This is the one real refactor; it also ends the
stepping loop existing three times (audit §2.1 item 2).

**4.3 The record — `evaluate/records` (~60 lines).** A frozen
`EpisodeRecord` appended as each trial finishes: `task, policy, trial,
seed, success, steps, source (name@hash), protocol_hash, perturb_hash,
robot_bundle, scene_bundle, physics_backend (with version), timestamp,
events` (milestones, 30 §7). `SimScore(successes, trials)` becomes a
~25-line fold over records that refuses unseeded rows and unequal trial
sets; `certify()` is unchanged. Written by a ~30-line `gym.Wrapper` (or
the env's own `step`), not by `lerobot-eval`, whose `per_episode` row
keeps only `episode_ix, sum_reward, max_reward, success, seed` (45 §1.7);
a collector folds `eval_info.json` rows into the same `SimScore` by seed.

**4.4 Milestones (~40 lines, after 4.3).** `EpisodeProtocol.milestones`
as an ordered chain of predicates evaluated offline over the arrays the
env already keeps; the funnel and the `success ≠ all_complete` flag from
Arena (30 §7 row 39). `object_moved` is milestone zero — the fact the T1,
T2 and T5 rows of zeros hid.

## 5. What to delete (~430 lines, plus ~110 optional)

| Item | Lines | Replaced by |
|---|---:|---|
| `MuJoCoBackend._closed_loop`, `closed_loop_rollout`, `closed_loop_vision_rollout` | 139 | `RobotiqEnv.reset/step` over the Stepper |
| `PhysicsBackend` Protocol (one implementation ever; keep `ModelCounts`) | ~77 | `gym.Env` is the backend-agnostic interface |
| `VisionPolicy`, `evaluate_vision_policies`, `lerobot_checkpoint_policy` | 115 | `lerobot-eval` + LeRobot's processors |
| `SimPolicy`, `evaluate_policies` | 23 | a ~15-line local runner over the same env with `environment_state` exposed (scripted experts are not `PreTrainedPolicy`s) |
| gym-aloha gripper/state adapters + `act_sim_vision_policy` | 56 | `get_env_processors()`; the measured 0.0078 m stays as data |
| `train-watch`'s `_Backend` shim and `_episode` loop | 64 | the env + `lerobot-train`'s own in-loop eval |
| Optional: `bootstrap_rank_ci` (diagnostic, no caller), `wilson` (test oracle) | 66 | keep the lesson in docs/22 and one regression test |

Four copies of the observation contract, three of the stepping loop,
three spellings of the control rate, and the `keyframe_state` shim
collapse to one each.

## 6. Fix while we are there (found by the audit)

- **"Counts, never rates" is broken at the join**: `join_with_real`
  collapses `SimScore` to a rate and the certificate carries no sim
  trial count. `PolicyOutcome` gains `sim_successes, sim_trials`; the
  certificate prints both n's.
- **`physics_backend` is a free string** while the certificate's own
  docstring says a different backend version is a different instrument:
  stamp `mujoco@<mujoco.__version__>` from the env.
- **`state_width` is caller-supplied** though the task knows it; the two
  task dataclasses (`SO101Task`, `ALOHA2Task`) become one
  `Task(name, spec, protocol, cameras, state_width, instruction)`.
- **The instruction moves into the protocol** (Arena verdict, 30 §7 row
  43): hash-stamped with the trials, exposed by the env as
  `task_description`.

## 7. The standards, applied

| Rule (as written in the repo) | How the plan keeps it |
|---|---|
| stats/bundles/certificate stdlib-only, so a certificate recomputes on any Python ≥ 3.10 | untouched; the env package depends on gymnasium and lerobot and lives behind the `train` extra — the certificate never imports it |
| Counts, never rates | `SimScore` folds records; sim n reaches the certificate (§6) |
| `name@hash` or refused; census before any episode | moved into `RobotiqEnv.__init__`, tested there |
| One truth per fact; pin both ends where a copy must exist | observation contract, stepping loop, control rate, keyframe home: one definition each; a test pins our `features_map` against LeRobot's `preprocess_observation` output — the one copy we cannot avoid |
| Magic numbers get names (PLR2004 on) | `render_fps` derived, hold window and thresholds already named |
| Paired trials by trial index, no RNG | `reset(seed)` maps seed → trial, never draws |
| No silent default where a wrong value is possible | `state_width` from the task; `gate_threshold` still required |
| Heavy imports lazy behind extras | gymnasium/lerobot imported inside the env module only |
| Fail loudly | `info["is_success"]` computed, never defaulted; unseeded or unequal trial sets refused by the fold |
| Every decision into docs + the log; `check-docs` gates it | this document; the log entry the same day |
| Both viewers for every rig session | `train-watch` keeps its viewers, loses its loop |
| Apache-2.0 tree | gymnasium MIT, LeRobot Apache-2.0 |

## 8. Order of work, with the test that proves each step

1. **Stepper extraction** — `_closed_loop` rewritten over it; the
   existing sim tests (Gate A pendulum, ALOHA ladders) must pass
   unchanged. No new behaviour.
2. **`RobotiqEnv` + `RobotiqEnvConfig`** — test: gymnasium's
   `check_env`; `preprocess_observation(env.reset()[0])` yields exactly
   `observation.state` (14,) and `observation.images.top` (3, 480, 640);
   two policies at `seed=1000..1003` start from identical states
   (pairing); `is_success` reaches `final_info` under `SAME_STEP`.
3. **`lerobot-eval` end to end** on the T5 checkpoint
   (`--env.type=robotiq_aloha2 --env.discover_packages_path=rq_pipeline.envs.lerobot --seed=1000 --eval.n_episodes=4`)
   — must reproduce this afternoon's 0/4, and write `eval_info.json`.
4. **Records + fold** — `score_policies` becomes the fold; `certify()`
   unchanged; the Gate A dry run passes on records; the §6 fixes land
   with their pins.
5. **Delete** the §5 rows; `train-watch` plays checkpoints through the
   env; `kitting-demos` uses the Stepper.
6. **Milestones + funnel**; then the variation schema and the
   main-effects table (30 §7 order), each a separate change.

Steps 1–4 are one working day on either machine; the WSL card is only
needed for step 3's checkpoint.

## 9. Progress (2026-08-26, evening — steps 1–3 done)

- **Step 1, the Stepper**: `physics/mujoco_backend.py::Stepper` — seat,
  `advance(control, substeps)`, the R7 rule, the rows. `_closed_loop` is
  now four lines over it; the kitting choreographer's `advance` lost its
  hand-rolled inner loop. A pin: the Stepper reproduces
  `closed_loop_rollout` bit for bit on the pendulum, including the short
  last tick. Existing sim tests unchanged and green.
- **Step 2, the env**: `rq_pipeline/envs/robotiq.py` (`RobotiqEnv`,
  `TASKS`, `make_env`, `gym.register("robotiq/<task>-v0")`) and
  `rq_pipeline/envs/lerobot_plugin.py` (`RobotiqEnvConfig`, registered
  as `robotiq`). gymnasium joined the `sim` extra. Pinned: gymnasium's
  own `check_env`; the raw observation shape; seed = trial pairing;
  truncation with the verdict under both names; the unstamped source
  refused; and, in the train venv, LeRobot's `make_env` +
  `preprocess_observation` producing `observation.state` (1, 14) and
  `observation.images.top` (1, 3, 480, 640).
- **Step 3, the acceptance run**: `lerobot-eval --env.type=robotiq
  --env.task=kitting --env.discover_packages_path=rq_pipeline.envs
  --policy.path=<T5 checkpoint> --seed=1000 --eval.n_episodes=4` →
  **0/4, the harness's verdict reproduced**, 78 s per episode (the
  harness took 73), an mp4 per episode, `eval_info.json` written.
- Three facts learned on the way, now in the code's docstrings:
  `discover_packages_path` takes a *package* (it walks `__path__`), so
  the flag is `rq_pipeline.envs`; LeRobot 0.6.1's `eval_info.json`
  carries `per_task[].metrics.successes` as a list in episode order and
  **no per-episode seed** — the fold (step 4) derives `seed + i` from
  the order and writes our own record; and the WSL box's D3D12 renderer
  is not bit-exact (±1 LSB in ~20 pixels, once ±2), so pixel pairing is
  asserted to within that noise while physics pairing stays exact.
- **Step 4, the record**: `evaluate/records.py` — `EpisodeRecord`
  (source, policy, trial, success, steps, instrument, protocol, seed,
  events), `append_records`/`read_records` (JSONL, NaN refused),
  `fold` (records → `SimScore`s; a duplicate trial or unpaired policy
  sets are refused with the culprit named), and `from_eval_info` for
  LeRobot's file (seed and trial recovered from episode order).
  `SimScore` moved here — it IS the fold. `score_policies` now builds a
  record per trial, appends to `record_to` as trials finish, and
  returns the fold. The two audit fixes: `PolicyOutcome`/`PolicyResult`
  carry `sim_successes, sim_trials` (validated on construction; the
  certificate's JSON now states its sim n), and `MuJoCoBackend.instrument`
  is `"mujoco-<version>"`, stamped into every record and exposed by the
  env. 192 tests green.
- **Step 5, the deletions**: `_closed_loop`, `closed_loop_rollout`,
  `closed_loop_vision_rollout` (the env and `run_sensor_episode` over
  the Stepper replace them); the `PhysicsBackend` Protocol (one
  implementation ever — `backend.py` keeps `ModelCounts`); `VisionPolicy`,
  `evaluate_vision_policies`, `lerobot_checkpoint_policy` (`lerobot-eval`
  through the env; `vision.py` is camera data now); `act_sim_vision_policy`
  (the two gym-aloha mapping functions stay as data adapters for demo
  replay); `train-watch`'s `_Backend` shim and hand-rolled `_episode`
  (the tool drives `RobotiqEnv`, binds its viewer to the env's persistent
  `MjData`, and loads checkpoints through LeRobot's own processors and
  `preprocess_observation`; `--action-space {act_sim,bundle}` says what
  the checkpoint speaks). `SO101Task` gained `cameras` so the ArmnetBench
  rig renders through the env. About 420 lines out; the evaluation path
  outside `tasks/` is 2,275 lines including the env (401) and the record
  (169), with the stepping loop, the observation contract, the keyframe
  ritual and the control rate defined once each. 190 sim tests + 2
  plugin tests green; the Stepper is now pinned against MuJoCo's own
  batched `rollout` bit for bit.
- **Step 6, milestones**: `EpisodeProtocol.milestones` — an ordered
  chain of `(name, predicate(states, sensors, step))`, unique names
  enforced — and `events_for` (Arena's tracker rule, offline: only the
  current milestone is evaluated, at most one advance per step, the
  first firing step recorded). `records.funnel` counts trials reaching
  each stage per policy; `records.disagreements` flags verdict ≠ chain.
  Transfer cube: `cube_moved → cube_lifted → cube_at_left`; kitting,
  order-free by construction: `part_moved → part_lifted → one_in_slot →
  both_in_slot`, all from the referees' own constants (`MOVED_M` = 1 cm
  is the one new number). The env writes the full row itself when given
  `record_to`/`policy_name` (`--env.record_to=… --env.policy_name=…` on
  `lerobot-eval`), so a runner that keeps only a success list still
  leaves events, seed and stamps behind. Pinned: the kitting expert
  walks all four; a limp policy leaves an empty chain. 196 sim tests.
- **The `Task` unification (§6)**: one frozen `Task(name, spec,
  protocol, cameras, state_width, instruction, target)` in
  `rq_pipeline/tasks/task.py`, validated on construction (a positive
  state width, a non-empty instruction, at least one camera) replaces
  `SO101Task` and `ALOHA2Task`; every builder states its rig's jointpos
  block and its sentence, and `RobotiqEnv(task, source=…)` reads them
  instead of taking them as keyword arguments. `KITTING_INSTRUCTION` is
  one string read by the exporter and the task. All six tasks — the
  four SO-101 and the two ALOHA 2 — register as `robotiq/<task>-v0` on
  their own bundles' stamps. 199 sim tests.
- Not yet: the variation schema with `draw(trial)` and the main-effects
  table; `train-watch` re-verified on screen after its rewrite.

## 10. Open questions (carried from 45 §4 and 46 §5)

1. Async vector envs under WSL: LIBERO defers simulator creation to the
   first `reset` inside the worker to dodge stale EGL contexts under
   `forkserver`; our `mujoco.Renderer` needs the same deferral before
   `batch_size > 1` is trusted on this box.
2. Executed action horizon: inside the env (GR00T's `MultiStepWrapper`
   shape, visible to every client) or in `EpisodeProtocol` (report 43)?
   Decide before the record schema freezes — it is a field either way.
3. Hub publishing: does a pinned `@<commit>` satisfy the `source@hash`
   rule, or do we hash the tree ourselves? Proposal: both, the tree hash
   is what the certificate cites.
4. `reward`: `float(success)` on the last tick (robomimic's sparse
   convention) is the plan; a shaped reward would be a different task.
