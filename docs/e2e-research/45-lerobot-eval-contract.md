# LeRobot's evaluation contract: what `lerobot-eval` expects from an env, and what it hands back

*Sixth pass, 2026-08-26. One agent, one field — the environment contract of
LeRobot's evaluator, read from the INSTALLED package (`lerobot` 0.6.1 in
`trainnr/.venv-train`; `gymnasium` 1.3.0, the wheel pins
`gymnasium<2.0.0,>=1.1.1`). Citations are `lerobot/<path>:L<line>` inside the
venv's site-packages; gymnasium likewise. Secondary source: the NVIDIA/HF blog
on the Environment Hub, by URL. Mapped against our harness
(`trainnr/trainnr/evaluate/`), backend and the ALOHA 2 tasks
(`trainnr/trainnr/tasks/aloha2/kitting.py`). Nothing below is from memory.*

---

## 1. The contract

### 1.1 Creation — an `EnvConfig` that returns `{suite: {task_id: VectorEnv}}`

A **gymnasium env** is the standard RL object: `reset(seed) -> (obs, info)`,
`step(action) -> (obs, reward, terminated, truncated, info)`. A **VectorEnv**
is N copies stepped together with batched arrays. LeRobot describes both with
an **`EnvConfig`** dataclass (`lerobot/envs/configs.py:L55-62`):

```
task: str | None = None; fps: int = 30
features: dict[str, PolicyFeature]   # env-side key -> (type, shape)
features_map: dict[str, str]         # env-side key -> policy key ("observation.state", ...)
max_parallel_tasks: int = 1; disable_env_checker: bool = True; gym_kwargs -> dict (abstract, L78-81)
```

`type` is the registered name (`L64-66`); `package_name` defaults to
`f"gym_{type}"` and `gym_id` to `f"{package_name}/{task}"` (`L68-76`). The
default `create_envs(n_envs, use_async_envs)` (`L83-124`) imports the package
if the id is unregistered (`L95-108`), builds copies with `gym.make(gym_id,
disable_env_checker=..., **gym_kwargs)` (`L110-111`), uses `AsyncVectorEnv`
(one subprocess per copy, `context="forkserver"`, `L113-115`) when
`use_async_envs and n_envs > 1` else `SyncVectorEnv`, with
`autoreset_mode=AutoresetMode.SAME_STEP` (`L119-121`), and returns
`{self.type: {0: vec}}` (`L124`) — the shape `make_env` documents
(`lerobot/envs/factory.py:L82-86`, returned via `cfg.create_envs`, `L122`).
Benchmarks override `create_envs` and never call `gym.make`: LIBERO builds
`LiberoEnv(...)` factories itself (`lerobot/envs/libero.py:L419-439`), RoboTwin
likewise (`configs.py:L826-843`). The dict is the contract; `gym.make` is a
convenience.

### 1.2 What the rollout calls on the env

`rollout()` (`lerobot/scripts/lerobot_eval.py:L167-411`), in order:

| Call | Line | The env must provide |
|---|---|---|
| `policy.reset()` | `L220` | — |
| `env.reset(seed=seeds, options={"lerobot_new_rollout": True})` | `L223`; option name `lerobot/envs/utils.py:L183` | `reset(seed=, options=)`, one seed per copy |
| `env.call("_max_episode_steps")[0]` | `L263` | attribute `_max_episode_steps` (RoboTwin: "`# lerobot_eval.rollout reads this`", `lerobot/envs/robotwin.py:L382`) |
| `env.call("task_description")`, else `env.call("task")`, else `""` | `L278-286` | string attributes; missing both only warns (`utils.py:L318-327`) |
| `env.step(action_numpy)`, `ndim == 2` asserted | `L303-307` | `(batch, action_dim)` floats |
| `env.envs[i].render()` / `env.call("render")` | `L488-493` | `render() -> (H, W, 3)` |
| `env.unwrapped.metadata["render_fps"]` | `L608`, `L232` | class `metadata = {"render_modes": ["rgb_array"], "render_fps": N}` |

Loop: `while not np.all(done) and step < max_steps` (`L272`); `done` latches
(`L365`) and is forced True for every copy at `step + 1 == max_steps`
(`L366-367`).

### 1.3 Observations — `pixels` and `agent_pos`

`preprocess_observation` (`utils.py:L68-149`) is hard-coded (its own TODO,
`L69`): `"pixels"` as a dict `{name: img}` → `observation.images.<name>`, a
bare array → `observation.image` (`L78-82`); images must be **uint8,
channel-last** (asserted `L93-97`), converted to float32 CHW in [0, 1]
(`L100-102`); `"agent_pos"` → `observation.state` float32 (`L113-117`);
`"environment_state"` → `observation.environment_state` (`L106-111`); any
other ndarray key passes through as `observation.<key>` (`L129-147`). The
constants: `OBS_STATE = "observation.state"`, `OBS_IMAGES =
"observation.images"`, `ACTION = "action"` (`lerobot/utils/constants.py:L23-33`).
`features`/`features_map` tell the **policy** those shapes:
`env_to_policy_features` flips visual shapes channel-first and renames through
`features_map` (`utils.py:L152-169`); `make_policy(cfg, env_cfg, rename_map)`
sets `cfg.output_features` from the ACTION entry and `cfg.input_features` only
if the checkpoint left them empty (`lerobot/policies/factory.py:L302-306`).
Concrete space (RoboTwin): `Dict({"pixels": Dict(image_spaces), "agent_pos":
Box((14,), float32)})` (`robotwin.py:L400-403`), returned as `{"pixels":
images, "agent_pos": joint_state}` (`L451`).

### 1.4 Language, success, episode end

Task text is read each step and put at `observation["task"]`, one string per
copy (`L281`). **Success** is the env's `info["is_success"]` (`L311-339`):
under SAME_STEP autoreset gymnasium resets a finished copy inside the same
`step` and stashes its info under `info["final_info"]`
(`gymnasium/vector/sync_vector_env.py:L277-292`), which the rollout reads
(`L313-321`); otherwise `info["is_success"]` directly (`L331-337`). Reward is
whatever `step` returns. Both reference envs derive success from a checker:
LIBERO `is_success = self._env.check_success(); terminated = done or
is_success` (`libero.py:L376-377`), info `{task, task_id, done, is_success}`
(`L378-385`), `truncated = False` (`L390`); RoboTwin `reward =
float(is_success); terminated = is_success; truncated = step_count >=
episode_length` (`robotwin.py:L515-517`). Both return `{"is_success": False,
...}` from `reset` (`libero.py:L363`, `robotwin.py:L492`).

### 1.5 Action chunks, seeds, reset

One `select_action` per env step (`L293`), inside four **processor
pipelines** — LeRobot's tensor-transform lists (normalisation, device, renames):
env-pre, policy-pre, policy-post, env-post (`L289-300`). Chunking is the
policy's: ACT keeps `deque(maxlen=n_action_steps)`
(`lerobot/policies/act/modeling_act.py:L98`), refills with
`predict_action_chunk(batch)[:, :n_action_steps]` when empty (`L117-122`),
pops one (`L123`); `n_action_steps: int = 100`
(`lerobot/policies/act/configuration_act.py:L86`). `policy.reset()` is "to be
called whenever the environment is reset" (`lerobot/policies/pretrained.py:L245-250`)
— once per batched rollout (`L220`), not per copy. **Seeds**: `range(start_seed
+ batch_ix*num_envs, start_seed + (batch_ix+1)*num_envs)` (`L528-533`),
`start_seed = cfg.seed` (`L801`), default `seed: int | None = 1000`
(`lerobot/configs/eval.py:L39`). Whether a seed fixes the start is the env's
choice: LIBERO ignores it and cycles stored init states by `episode_index +
n_envs` (`libero.py:L173-175`, `L344-346`); RoboTwin uses `actual_seed =
self.episode_index if seed is None else seed` (`robotwin.py:L470`).
`n_batches = ceil(n_episodes / num_envs)` (`L472`), surplus discarded.

### 1.6 Registering a new env — three doors, no fork

1. `@EnvConfig.register_subclass("name")` on a dataclass (`configs.py:L320-322`);
   `EnvConfig` is a `draccus.ChoiceRegistry` (`L56`) — draccus is the
   CLI/dataclass parser — so `--env.type=name` resolves through
   `EnvConfig.get_choice_class` (`factory.py:L26-34`) and every field is an
   `--env.<field>` flag.
2. **Plugin discovery**: `register_third_party_plugins()` imports every
   installed distribution named `lerobot_robot_*`, `lerobot_camera_*`,
   `lerobot_teleoperator_*`, `lerobot_policy_*` or `lerobot_env_*`
   (`lerobot/utils/import_utils.py:L231-237`, via
   `importlib.metadata.distributions()`, `L250-255`); `lerobot-eval`'s `main`
   calls it first (`lerobot_eval.py:L1108-1111`); an import error is logged,
   not raised (`L246-248`). A package `lerobot_env_robotiq` whose init runs
   the decorator is found with no LeRobot change.
3. **Hub env**: `HubEnvConfig.hub_path` (`configs.py:L131-144`); the repo's
   file (default *env.py*, `utils.py:L387`) "must expose
   `make_env(n_envs=int, use_async_envs=bool)`" (`L460-462`), called with
   `cfg=` when an `EnvConfig` is given (`L466-469`); it may return the dict, a
   `VectorEnv` or one `gym.Env` (`L472-498`); refused without
   `trust_remote_code=True` (`L410-416`). The blog ("Generalist Robot Policy
   Evaluation in Simulation with NVIDIA Isaac Lab-Arena and LeRobot", NVIDIA
   + HF, 2026-01-05,
   <https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot>)
   shows exactly this door — `--env.type=isaaclab_arena
   --env.hub_path=nvidia/isaaclab-arena-envs --trust_remote_code=True
   --rename_map='{"observation.images.robot_pov_cam_rgb": ...}'` — and states
   no contract beyond it; the contract is the code above.

### 1.7 What it writes

`eval_main` (`L740-822`): `make_env(cfg.env, n_envs=cfg.eval.batch_size,
use_async_envs=cfg.eval.use_async_envs)` (`L753-758`); `EvalConfig` defaults
`n_episodes=50`, `batch_size=0` = auto `min(0.7*cores, n_episodes, 64)`,
`use_async_envs=True` (`lerobot/configs/default.py:L94-101`, `L118-126`);
`max_episodes_rendered = 0 if cfg.eval.recording else 10` (`L786`); output
`outputs/eval/<date>/<time>_<job_name>` with `job_name =
f"{env.type}_{policy.type}"` (`eval.py:L65-74`). It dumps `eval_info.json`
(`L819-820`):

```
per_task:  [{task_group, task_id, metrics: {sum_rewards[], max_rewards[], successes[],
             video_paths[], predicted_video_paths[]}}]                   L1043, L878-884
per_group: {group: {avg_sum_reward, avg_max_reward, pc_success, n_episodes,
             video_paths, predicted_video_paths}}                        L1080-1087
overall:   {same + eval_s, eval_ep_s}                                    L1090-1099
```

Per episode inside `eval_policy` (`L652-670`): `episode_ix, sum_reward,
max_reward, success, seed` — reward summed/maxed up to the first done
(`L553-564`), `success = any(success_t)` over that mask (`L565`). Videos:
`videos/<group>_<task_id>/eval_episode_<n>.mp4` (`L599-612`, `L912-915`). With
`--eval.recording=true` the rollout is saved as a LeRobotDataset with
`next.reward`, `next.success`, `next.done`, `task` per frame (`L108-120`,
`L159-163`). **Statistics**: every aggregate is `np.nanmean` (`L672-674`,
`L1071-1075`) — no interval, no paired statistic, no p-value; a NaN episode
silently leaves the denominator (the Arena pattern, docs/40 §1.4).

### 1.8 Cost of a compliant env

`LiberoEnv` is `libero.py:L107-401`, 295 lines, of which the nested space is
`L195-252` and LIBERO init-state logic `L60-104`, `L339-362`; `RoboTwinEnv` is
`robotwin.py:L318-540`, ~220. The reusable skeleton — `metadata`, spaces,
`reset`, `step`, `render`, `close`, the three attributes — is under 100 lines.

## 2. What we already have

| LeRobot expects | Ours | Where |
|---|---|---|
| `{"pixels": {cam: uint8 HWC}, "agent_pos": float32}` per step | `closed_loop_vision_rollout` builds `{"observation.state": first state_width sensors, "observation.images.<key>": renderer.render()}` per control tick — the post-`preprocess_observation` shape, one rename away | `trainnr/trainnr/physics/mujoco_backend.py:L293-303` |
| `select_action` behind processors, `observation["task"]` | `lerobot_checkpoint_policy` does the same (`HWC→CHW/255`, `{"task": [instruction]}`, pre → `select_action` → post) | `trainnr/trainnr/evaluate/vision.py:L148-164` |
| `policy.reset()` per rollout | `VisionPolicy.reset` per trial — stricter | `vision.py:L52-62`, `L82-84` |
| `render_fps`, `_max_episode_steps` | `steps=4000, control_interval=10` → 400 ticks; the bundle sets no `<option timestep>` (`robots/aloha2-nominal/aloha.xml:L4`), so MuJoCo's 0.002 s → **50 Hz** — the rate docs/31 T5 exported at and LeRobot's own `AlohaEnv` declares (`fps=50, episode_length=400`, `configs.py:L150-152`) | `trainnr/trainnr/tasks/aloha2/kitting.py:L99-101` |
| `reset(seed)` decides the start | `perturb(trial, home)` — deterministic corners by `trial % 4` | `aloha2.py:L313-320`, `L338-347`; `trainnr/trainnr/evaluate/harness.py:L45-67` |
| `info["is_success"]` per step | `success(states, sensors)` over the last `_HOLD_STEPS=250` physics steps (lifted AND held) — a tail predicate | `aloha2.py:L349-356`, `L102` |
| `pc_success = nanmean` | `SimScore(successes, trials)` counts; `clopper_pearson`, `wilson`; `certify`, `fisher_rank_ci`, `top_pick_probability` | `harness.py:L79-90`; `trainnr/trainnr/stats/intervals.py:L78`, `L107`; `trainnr/trainnr/evaluate/certificate.py:L99`; `trainnr/trainnr/stats/ranking.py:L101`, `L228` |
| `job_name = env_policy` | `name@hash` stamp refused if absent; census gate first | `harness.py:L117-132` |
| `AsyncVectorEnv` batching; mp4 per episode; recorded rollouts | none — serial trials; Rerun + MuJoCo viewers | `harness.py:L133-144` |

## 3. Recommendation: expose the task as a LeRobot env; keep our harness as the judge

**Do both.** A ~150-line package `lerobot_env_robotiq` (named so
`register_third_party_plugins` imports it, §1.6) wrapping ONE control tick of
our backend as `gym.Env.step`, plus an `EnvConfig`; the harness stays the only
thing that turns outcomes into a certificate. The env is an adapter; physics
and referee stay where they are.

```python
class TransferCubeEnv(gym.Env):                      # field names are the contract, verbatim
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}                    # L608, L232
    def __init__(self, task: ALOHA2Task, episode_index: int = 0, n_envs: int = 1):
        self.task = task.name; self.task_description = "transfer the cube"          # L278-286
        self._max_episode_steps = task.protocol.steps // task.protocol.control_interval  # L263 → 400
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({c.key: spaces.Box(0, 255, (c.height, c.width, 3), np.uint8)
                                   for c in task.cameras}),                         # utils.py:L78-97
            "agent_pos": spaces.Box(-np.inf, np.inf, (14,), np.float32)})           # utils.py:L113-117
        self.action_space = spaces.Box(-np.inf, np.inf, (14,), np.float32)
        self._trial, self._stride = episode_index, n_envs                            # libero.py:L173-175
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        trial = self._trial if seed is None else seed                                # robotwin.py:L470
        self._trial += self._stride
        self._stepper = backend.stepper(task.protocol.perturb(trial, home)); self._tick = 0
        return self._obs(), {"is_success": False, "task": self.task}                 # libero.py:L363
    def step(self, action):
        self._stepper.advance(action, task.protocol.control_interval)                # R7 rule inside
        self._tick += 1; done = self._tick >= self._max_episode_steps
        ok = done and task.protocol.success(self._stepper.states_tail, self._stepper.sensors_tail)
        return self._obs(), float(ok), False, done, {"is_success": ok, "task": self.task}
    def render(self): return self._obs()["pixels"]["top"]

@EnvConfig.register_subclass("trainnr_aloha2")
@dataclass
class RobotiqAloha2Env(EnvConfig):
    task: str = "transfer_cube"; fps: int = 50; episode_length: int = 400
    features = {ACTION: PolicyFeature(FeatureType.ACTION, (14,)),
                "agent_pos": PolicyFeature(FeatureType.STATE, (14,)),
                "pixels/top": PolicyFeature(FeatureType.VISUAL, (480, 640, 3))}    # configs.py:L782-795
    features_map = {ACTION: ACTION, "agent_pos": OBS_STATE, "pixels/top": f"{OBS_IMAGES}.top"}
    @property
    def gym_kwargs(self): return {}
    def create_envs(self, n_envs, use_async_envs=False):
        fns = [partial(TransferCubeEnv, build_transfer_cube(), episode_index=i, n_envs=n_envs)
               for i in range(n_envs)]
        return {self.type: {0: _make_vec_env_cls(use_async_envs, n_envs)(fns)}}     # L48-52, L124
```

`truncated=True` with `is_success` only on the last tick, because our predicate
is a hold over the final 250 physics steps (`aloha2.py:L349-356`) while LeRobot
credits `any` step (`L565`); under SAME_STEP the flag reaches the rollout via
`final_info` (`sync_vector_env.py:L286-290`, read at `L313-321`).

**Pairing survives**: `--seed=1000 --eval.n_episodes=4 --eval.batch_size=4`
resets every policy's copies with seeds 1000..1003 (`L528-533`) →
`perturb(seed % 4)` → the same four corners for every policy. Pairing becomes
a property of the seed, recorded per episode (`L659`); a collector joins
policies by `seed`, folds `per_episode[].success` into `SimScore(successes,
trials)` → `certify()`. Nothing in `stats/` changes.

**Free from `lerobot-eval`** (§1.7): process-parallel copies; every policy
family LeRobot ships (π0/π0.5, SmolVLA, XVLA with its own processor override,
`factory.py:L48-53`) without a per-family adapter of ours; `--rename_map`;
mp4s; rollouts recorded as a LeRobotDataset with `next.success` (a DAgger
loop's seed); the Hub door for others to evaluate on our bundles. **Not free**:
intervals, rank statistics, the census gate, the `name@hash` stamp, more than
one policy per invocation, and tail-vs-any-step success semantics — the
harness's job, unchanged.

**The one real cost**: `_closed_loop` runs a whole episode
(`mujoco_backend.py:L173-234`) and its docstring forbids forking the stepping
discipline (`L180-187`). Extract a `Stepper` (seat state `L203-207`; advance k
substeps with the R7 `mj_forward` `L220-233`; rolling `states`/`sensors`
tails) that both `_closed_loop` and the env consume. ~40 lines moved, none
duplicated.

## 4. Open questions

1. **Does `lerobot-train`'s periodic eval use the same `make_env`?** Not read
   here; if so the env doubles as the trainer's in-loop eval for free.
2. **Async + EGL on WSL.** LIBERO defers simulator creation to the first
   `reset` inside the worker to dodge stale EGL contexts under forkserver
   (`libero.py:L258-277`); our `mujoco.Renderer` needs the same deferral —
   confirm against the WSL box's Mesa gotchas before trusting `batch_size > 1`.
3. **Where the gripper normalisation lives.** ALOHA checkpoints emit a [0, 1]
   gripper channel we map to `ctrlrange` (`aloha2.py:L75-84`): in `step`, or
   as an `env_postprocessor` via `get_env_processors` (`configs.py:L126-128`)?
   The processor keeps the env's action space honest (metres).
4. **Contract stability.** `preprocess_observation` carries a refactor TODO
   (`utils.py:L69`, `L153`); pin `lerobot==0.6.1` in the plugin and test the
   four keys (`pixels`, `agent_pos`, `is_success`, `_max_episode_steps`).
5. **Multi-task shape.** `{suite: {task_id: vec}}` yields `per_task` rows per
   id; kitting and transfer_cube as ids 0/1 of one suite, or two suites,
   decides how the collector feeds `pool_rank_correlations` (docs/40 §4.3).
