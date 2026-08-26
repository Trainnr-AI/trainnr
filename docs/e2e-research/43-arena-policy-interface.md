# Isaac Lab Arena: the policy interface, and how it evaluates real foundation-model policies

*Sprint pass, 2026-08-26. One agent, one field, one primary source: a repomix
bundle of the whole `isaaclab_arena` repository (*arena.md*, 132,303 lines).
Every claim is cited as `path:Lnnnn` — the repository path, and the line number
of that line INSIDE THE BUNDLE — with a short quote, checkable by
`sed -n 'nnnnp' arena.md`. Nothing comes from memory of Arena; arithmetic on
cited facts is marked "derived". Mapped against our harness
(`rq_pipeline/evaluate/{harness,vision}.py`, `rq_pipeline/physics/mujoco_backend.py`)
and docs/30–31.*

Terms, defined once. A **policy** is the thing judged: observation in, motor
command out. A **foundation-model policy** (VLA, vision-language-action model:
GR00T, π0/π0.5 via "openpi", DreamZero, Cosmos) takes camera images + joint
state + a sentence and predicts an **action chunk** — a block of future actions,
shape `(horizon, action_dim)`, from one forward pass. The **action horizon** is
how many steps it predicts; the **open-loop horizon** (Arena also says
`action_chunk_length`) is how many the evaluator executes before asking again;
executing fewer than predicted is **receding-horizon** replanning. A **control
tick** is one policy call; a **physics step** one simulator integration;
**decimation** is physics steps per tick.

## 1. What Arena does

### 1.1 The policy contract: one method, batched over environments

- `PolicyBase(ABC, Generic[PolicyCfgT])`, built from a typed dataclass config, has
  one abstract method: "def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:"
  (`isaaclab_arena/policy/policy_base.py:L36706`). Hooks: `reset(env_ids=None)` (L36719),
  `close()` (L36725), `set_task_description` (L36729, "Set the task description of the task being evaluated"),
  `has_length()/length()` for replay policies (L36734–36740), `is_remote` (L36743).
- The docs: "You implement one method — ``get_action(env, obs)`` — and the policy
  plugs into both the single-job runner and the Experiment Runner without any
  changes to either" (`docs/pages/concepts/policy/index.rst:L6092`). Policies are
  `@register_policy`-registered with a `name`; the runner derives CLI flags from
  the config type (L6162).
- Output is **one row per parallel environment**: "Per-step action tensor of shape
  ``(num_envs, action_dim)``" (`isaaclab_arena/policy/action_scheduling/action_scheduler.py:L36589`).
- The rollout loop: `obs, _ = env.reset(); policy.reset(); policy.set_task_description(env.unwrapped.get_language_instruction())`
  (`isaaclab_arena/evaluation/policy_runner.py:L34426–34428`), then
  `actions = policy.get_action(env, obs); obs, _, terminated, truncated, _ = env.step(actions)`
  under `torch.inference_mode()` (L34440–34442); on termination only those envs
  reset: "policy.reset(env_ids=env_ids)" (L34451). The instruction belongs to the
  ENVIRONMENT: the experiment YAML puts `language_instruction:` under `environment_builder:`
  (`isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml:L98706–98707`).

### 1.2 Observation contract: a nested gym dict, images as uint8 tensors

- Two groups by convention, `"policy"` (proprioception) and `"camera_obs"` (one
  `(N, H, W, 3)` tensor per camera): "arena_camera_obs_group  - set by
  isaaclab_arena.utils.cameras.make_camera_observation_cfg" / "arena_policy_obs_group
  - standard Isaac Lab ObservationsCfg field name"
  (`isaaclab_arena_openpi/policy/droid_adapter.py:L125695–125698`).
- Proprioception keys vary per embodiment: DROID `joint_pos`, `gripper_pos`
  (L125709–125710); GR00T reads `observation["policy"]["robot_joint_pos"]`,
  "(N, num_joints) array in sim joint order" (`isaaclab_arena_gr00t/policy/gr00t_core.py:L121461`);
  DreamZero slices the first 7 as arm, the next as gripper
  (`isaaclab_arena_dreamzero/policy/droid_adapter.py:L97553–97554`).
- Images must be uint8, asserted with a fix-your-scene message (L97601–97603,
  "expected uint8. If Isaac Sim is configured with float images, convert to uint8 in your scene config").

### 1.3 Action contract: absolute joint positions, dimension fixed by the checkpoint

- "Fixed by upstream pi0 droid checkpoints: 7 panda joints + 1 gripper command.
  action_dim = 8" (`…openpi/policy/droid_adapter.py:L125680–125681`); Cosmos and
  DreamZero declare the same (`…cosmos/policy/droid_adapter.py:L94521`, `…dreamzero/policy/droid_adapter.py:L97519`).
- The environment is picked to match: "GR00T N1.6-DROID uses absolute joint
  positions. The YAML therefore selects ``droid_abs_joint_pos``"
  (`docs/pages/quickstart/running_a_real_policy/gr00t.rst:L16349–16350`).
- GR00T remaps joint ORDER by name through three YAML joint-order files —
  "(policy_joints_config, robot_action_joints_config, robot_state_joints_config)"
  (`gr00t_core.py:L121384–121385`) — producing an "(N, horizon, action_dim) float64 numpy array" (L121628).

### 1.4 Action-chunk scheduling: three concrete algorithms

Arena separates WHEN to call the model from WHAT to send each tick.

- **`ActionScheduler`** (abstract): "The policy calls ``get_action(fetch_action_fn)``
  at every environment step. The scheduler controls when to query the action and
  how to derive a single action from one or more action outputs" (`…/action_scheduler.py:L36569–36571`).
- **`ActionChunkScheduler`** — per-env receding horizon. State: buffer
  `(num_envs, action_horizon, action_dim)`, per-env index, per-env
  `env_requires_new_chunk` mask (`…/action_chunk_scheduler.py:L36477–36485`). Per
  tick: if ANY env needs a chunk, call the model once, copy only masked rows
  ("self.current_action_chunk[mask] = new_chunk[mask]", L36511); emit
  `chunk[env, index]`; when `index == action_chunk_length` flag a refetch
  (L36533–36537). It counts wasted batch rows: "how many envs actually needed the
  fetch vs. total (wasted compute detection)" (L36488). The config asserts
  `action_chunk_length <= action_horizon` (`isaaclab_arena_gr00t/policy/config/gr00t_closedloop_policy_config.py:L121140–121142`),
  documented as "Number of actions to execute per inference rollout (can be less than action_horizon)" (L121134).
- **`SyncedBatchActionScheduler`** — "waits until ALL envs need a new chunk before
  calling inference. Envs that exhaust their chunk early hold their current robot
  state until every env is ready" (`…/synced_batch_action_scheduler.py:L36612–36614`);
  trade-off stated: "Action tensor batch is always full (N envs, never wasted)"
  versus holding "for up to (action_chunk_length - 1) steps" (L36618–36620). The
  GR00T wrapper builds the hold action from "their current sim joint positions
  copied into the action slots that share a joint name with the state config"
  (`…gr00t/policy/gr00t_remote_closedloop_policy.py:L121852–121853`); chosen by
  `scheduler: ActionSchedulerType = ActionSchedulerType.CHUNK` (L121759).
- **`RemoteChunkReplayPolicy`** (openpi, Cosmos) — no scheduler object: "fetch one
  ``(chunk, action_dim)`` prediction per env, keep the first ``open_loop_horizon``
  actions, and yield them in order before refetching"
  (`isaaclab_arena/policy/remote_policy_base.py:L36793–36795`); one observation per
  request, so it loops over envs (L36835–36837), asserts the server returned
  enough (L36901–36903) and truncates: "return chunk[: self._open_loop_horizon]" (L36904).
  A pluggable scheduler is a TODO: "add an action_scheduler_cls so action_chunk ->
  action is configurable (today it is a row-by-row replay)"
  (`isaaclab_arena_openpi/policy/pi0_remote_policy.py:L125804–125805`).

Executed horizons Arena ships, in control ticks: π0.5 = 15, π0 = 10
("open_loop_horizon_by_variant = {'pi05': 15, 'pi0': 10}", `…openpi/policy/droid_adapter.py:L125684–125687`;
docs table `openpi.rst:L16727,L16732`); Cosmos = 16, by eye: "COSMOS accepts both
32 and 16 step chunks. I empirically observed better behavior with 16 steps"
(`…cosmos/policy/cosmos_remote_config.py:L94422–94423`); DreamZero = 24
(`…dreamzero/policy/dreamzero_remote_policy.py:L96972`); GR00T-DROID predicts 32,
executes 32 (`…gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml:L121006,L121017`);
GR00T-GR1 fine-tune 16/16 (`docs/pages/example_workflows/static_manipulation/step_5_evaluation.rst:L15281,L15289`).

### 1.5 Inference frequency versus control frequency

- Default control rate 15 Hz: "Control rate: sim.dt (1/120 s) x decimation (8) = 15 Hz"
  (`isaaclab_arena/environments/isaaclab_arena_manager_based_env_cfg.py:L32509`;
  `dt=1 / 120` L32511, `decimation: int = 8` L32524). A helper restores 50 Hz:
  "Set 50 Hz control (sim dt 1/200, decimation 4), Arena's pre-15 Hz default rate"
  (L32529), used where the policy was trained at 50 Hz ("the rate the RL policies
  for this task were trained at", `isaaclab_arena_environments/dexsuite_lift_environment.py:L105470`).
- The model is queried only when a chunk is exhausted, so **inference rate =
  control rate / executed horizon** (derived): π0.5 → one call per second at
  15 Hz; GR00T-DROID at 32 → one call per ~2.1 s.
- Inference is **synchronous and blocking**: `get_action` then `env.step`
  (L34441–34442); the remote call is `self._websocket_client.infer(server_request)`
  (L36915). The sim waits; model latency never reaches the robot. No latency
  injection, no asynchronous overlap, no interpolation — "interpolation,
  smoothing, chunk-overlap blending" is future work (`…openpi/policy/pi0_remote_policy.py:L125800`).

### 1.6 The four wrappers, what each does per fetch

- **GR00T** (`Gr00tRemoteClosedloopPolicy`, `gr00t_remote_closedloop`): GR00T's own
  ZMQ client ("from gr00t.policy.server_client import PolicyClient", L121813).
  Per fetch: torch→numpy once (L121875); build the "language" / "video" / "state"
  dict whose keys come from GR00T's **modality config** (L121570–121595; video
  reshaped to `(N, 1, H, W, C)`, L121586–121588); `self._client.get_action(...)`
  (L121887); remap joints to sim order (L121890). Images are letterboxed — pad to
  square, then resize (`isaaclab_arena_gr00t/utils/image_conversion.py:L124726–124744`);
  DROID renders 720×1280, sends 180×320 (`…droid_manip_gr00t_closedloop_config.yaml:L121022–121023`).
  "Core logic is modular and numpy-only" (`gr00t_core.py:L121321`) so it tests without the sim.
- **openpi / π0, π0.5** (`Pi0RemotePolicy`, `pi0_remote`): ~35 lines over
  `RemoteChunkReplayPolicy` plus an **embodiment adapter** with two methods —
  `extract(observation, env_id)` pulls one env's tensors into a frozen dataclass,
  `pack_request(extracted, language_instruction)` builds the wire dict
  (`remote_policy_base.py:L36781–36787`). DROID sends
  `observation/exterior_image_1_left`, `observation/wrist_image_left`,
  `observation/joint_position`, `observation/gripper_position`, `prompt`, images
  resized-with-pad to 224×224 (`…openpi/policy/droid_adapter.py:L125689,L125715–125725`).
  Transport is openpi's websocket+msgpack client, subclassed only for keepalive
  pings (`isaaclab_arena/policy/websocket_client.py:L37216–37217`). New embodiment =
  "subclass ``Pi0EmbodimentAdapter`` … and register the adapter" (`openpi.rst:L16734–16736`).
- **Cosmos** (`CosmosRemotePolicy`, `cosmos_remote`): same protocol, "differs only in
  the response key (the server returns the chunk under the singular 'action') and
  the DROID wire format" (`…cosmos/policy/cosmos_remote_policy.py:L94455–94457`);
  its adapter tiles ONE image — wrist on top, two exterior views half-size below
  (`…cosmos/policy/droid_adapter.py:L94569–94577`).
- **DreamZero** (`DreamZeroRemotePolicy`, `dreamzero_remote`; a world-action model,
  H100-only per `dreamzero.rst:L7849`): the server is **stateful** — "it maintains a
  temporal observation history per session UUID. Each parallel environment gets
  its own UUID so their histories do not mix" (L97145–97146); requests carry
  `session_id` and `endpoint: "infer"` (L97308–97309). The adapter pads a
  7-column response to 8 with a zero gripper (L97582–97583), sends
  `observation/cartesian_position` as zeros (L97564), and has a `cam2_source`
  knob (`right | black | duplicate`) for rigs lacking a second exterior camera (L97488–97494).

### 1.7 Reliability rules baked in

Reconnect up to 3 times and on reconnect **flush every env's cached chunk** so
the next tick re-queries "rather than replay cached actions"
(`remote_policy_base.py:L36935–36941`); `num_envs` may not change mid-rollout
(L36887–36890); the chunk must be 2-D with the adapter's `action_dim`
(L36898–36900); `reset(env_ids)` drops that env's chunk and counter (L36865–36871),
GR00T also resets the server client (L121907); an empty instruction is a hard
error in every VLA wrapper (L36843–36846; L121832–121836).

## 2. What we already have that is equivalent

- **The one-method contract.** `VisionPolicy(name, act: (step, observation) -> controls, reset)`
  (`pipeline/rq_pipeline/evaluate/vision.py:53–62`) is `get_action` + `reset` for
  one environment, observation already LeRobot-shaped
  (`{"observation.images.<key>": uint8 (H,W,3), "observation.state": float32}`,
  `mujoco_backend.py:272–273`) — Arena's `"camera_obs"` / `"policy"` under other keys.
- **Control tick vs physics step.** `EpisodeProtocol.control_interval` (`harness.py:64`)
  IS Arena's `decimation`; `_closed_loop` calls the policy every `control_interval`
  steps and holds `data.ctrl` between (`mujoco_backend.py:211–219`). Same
  synchronous, blocking loop.
- **Chunk replay, hidden inside the policy.** `lerobot_checkpoint_policy` calls
  LeRobot's `select_action`, which pops the checkpoint's own action queue
  (`vision.py:162`); `reset` runs before every episode because "checkpoint policies
  carry action-chunk queues that must not leak between trials" (`vision.py:56–57`).
  That is `RemoteChunkReplayPolicy` semantics with the horizon not ours to set.
- **Boundary processing.** Our adapter runs the checkpoint's pre/post-processors
  (`vision.py:142–146, 162–163`); Arena's equivalents are `pack_request`
  (resize-with-pad, key renaming) and GR00T's joint remap.
- **Camera census before episodes** (`score_policies(gate_cameras=True)`,
  `harness.py:125–132`) — no Arena counterpart; its nearest is a per-fetch assert
  that a named camera exists (L97597–97599).
- **Paired trials, deterministic perturbation, counts not rates** (`harness.py:46–90`)
  — nothing in Arena's policy layer; its runner reports a `success_rate` over
  episodes (gr00t.rst:L16433–16451).

## 3. What to adopt, what to skip

**ADOPT**

1. **An explicit `ActionScheduler` in `VisionPolicy`, executed horizon as a protocol
   field.** Today the executed horizon is whatever the checkpoint's config says
   (`n_action_steps` inside LeRobot), invisible to the certificate. Port
   `ActionChunkScheduler` for N=1 (~40 lines, numpy): the policy exposes
   `predict(observation) -> (horizon, nu)`, the harness owns
   `executed_horizon <= horizon` (assert as L121140). A recipe walk can then vary
   π0.5 at 10/15/25 executed steps under the same paired trials — Arena picked
   Cosmos's 16 by eye (L94422–94423); we pick with an interval. Hash the horizon
   into `EpisodeProtocol` so two runs differing only in replan rate are different protocols.
2. **The embodiment-adapter split** (`extract` / `pack_request` / `parse_actions`,
   L36781–36787, L97099). `lerobot_checkpoint_policy` fuses model loading, key
   renaming and tensor conversion in one closure. Split into `RigAdapter`
   (bundle-specific: which sensor slice is state, which camera feeds which key,
   joint order) and `ModelAdapter` (checkpoint-specific: LeRobot local, openpi
   websocket, GR00T ZMQ). The rig adapter becomes bundle content (`adapter@hash`,
   docs/30 §3.6) — the agent-written leaf onboarding must emit. DreamZero's
   `cam2_source` (L97488) belongs here: ALOHA 2 has six cameras, ArmnetBench
   three; the adapter names the mapping.
3. **One remote path: a `RemoteChunkReplayPolicy` clone over openpi's websocket
   protocol.** Arena reuses it for π0, π0.5 and Cosmos (L94454–94455) with a
   keepalive subclass (L37216). π0/π0.5 in openpi format are exactly what our
   adapter cannot load ("pi0/pi0.5 in openpi format need their own adapter",
   `vision.py:113`); ~120 lines (`remote_policy_base.py:L36790–36942`) with the
   reconnect-flush rule gives π0.5 on the WSL card while the server runs in its
   own venv or on the cloud GPU (docs/31 T5 decision).
4. **Task description set by the protocol, not the policy** (L34428, YAML L98707).
   Move `task_instruction` from `lerobot_checkpoint_policy(...)` into
   `EpisodeProtocol`, so the wording is hash-stamped with the trials.
5. **Per-fetch assertions**: 2-D chunk, `action_dim == nu`,
   `chunk.shape[0] >= executed_horizon` (L36898–36903). Each turns a silent zero
   into a refused run — the census-gate principle applied to model output.

**SKIP**

- **`SyncedBatchActionScheduler` and hold-actions.** They keep a GPU batch full
  across N Isaac envs; ours are N=1 per process, and "hold current joint position
  for up to horizon−1 ticks" (L36619–36620) is a physics-visible artifact a
  certificate would have to explain. Revisit only on MJX-Warp (docs/30 §6).
- **Hydra/typed-config registration and CLI generation** (L6162–6174). Recipes are
  hash-stamped dataclasses already; a second config system is churn.
- **GR00T's three-YAML joint remap** (L121384). Our bundles fix joint order in the
  sensor wrapper (docs/31 §1); a by-name remap belongs in the rig adapter (item 2)
  when policy and sim disagree, not in a GR00T-specific core.
- **Cosmos's tiled image, DreamZero's session-UUID state.** Model-specific wire
  quirks; adopt the adapter SHAPE, write these only when evaluating those models.
- **Arena's implicit "no latency".** Same synchronous loop as ours, but our sysid
  path fits sensor delays (`mujoco_backend.py:225–228`); latency, when modelled,
  is a protocol parameter, never left implicit.

## 4. Open questions

1. **Executed horizon at 50 Hz.** Arena's 10–32 were chosen at 15 Hz control; the
   ACT sim and our kitting data run 50 fps (docs/31 T5). The same wall-clock
   replan (~1 s for π0.5) is ~50 ticks. Nothing in the bundle says whether horizon
   scales with control rate — a paired-trial experiment, not a lookup.
2. **Horizon as recipe axis or protocol constant.** Per-policy (Arena's choice,
   `recipe@hash`) or fixed per protocol? Which keeps pairing honest when two
   families prefer different horizons?
3. **Chunk boundaries and the referee.** Arena resets a policy on termination
   mid-chunk (L34451), dropping the remainder. Should our fixed-`steps` episodes
   end only at chunk boundaries so every policy is judged after complete replans?
4. **Match cosmetics before or after the model's letterbox?** Every Arena adapter
   pads-then-resizes to the checkpoint's training size (224², 180×320, 512²); our
   camera matching (`vision.py:17–20`) works at dataset resolution. Unanswered in
   the bundle; it bears on the cosmetics-vs-dynamics confound docs/31 T1 hit.
5. **`object_moved_rate`** (gr00t.rst:L16435): a partial-credit companion metric.
   Worth a standard "object displaced" referee sensor so zero rows (docs/31 T2,
   T5) still carry information?
