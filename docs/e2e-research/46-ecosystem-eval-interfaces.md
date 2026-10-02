# The ecosystem's evaluation interfaces: what a task must expose, what an eval writes

*Sprint pass, 2026-08-26. One agent, six fields, primary sources only: the
current `main`/`master` code of each repository fetched raw from GitHub, plus two
Hugging Face pages. Every claim is cited as `path:Lnn` — path inside that
repository, line number as the fetch reported it (approximate, ±5) — with a
short quote. Where a fetch failed or a grep found nothing, that is stated.
Nothing from memory. Mapped against `trainnr/trainnr/evaluate/harness.py`,
`trainnr/trainnr/evaluate/vision.py` and docs/30 §7 (Arena is covered
there and in e2e-research/39–44; not repeated).*

Terms, defined once. An **environment** (env) is the simulated task as code: it
holds state, accepts an **action** (motor command), returns an **observation**
(what the policy may see). A **policy** maps observation to action. An
**episode** is one attempt from reset to end; a **rollout** is running one.
**Success** is the task's boolean verdict on an episode. **Gymnasium** is the
Python interface (`gym.Env`) most of the ecosystem speaks; a **vector env** runs
N copies in lockstep, arrays in and out; **autoreset** is a vector env resetting
a finished copy by itself. An **action chunk** is a block of future actions from
one model call. A **hub** is a git-hosted repository on huggingface.co.

## 1. Per system: interface required, artifact written

### 1.1 Gymnasium — the substrate everyone else assumes

- **Version.** `gymnasium/__init__.py:L1-40` on main: `__version__ = "1.4.0"`; released tags
  newest-first v1.3.0 (22 Apr), v1.2.3, v1.2.2, v1.2.1, v1.2.0 (27 Jun), v1.1.1, v1.1.0 (26 Feb),
  v1.0.0 (08 Oct) (github.com/Farama-Foundation/Gymnasium/releases).
- **Env contract.** `def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
  (`gymnasium/core.py:L91-93`); terminated = "Whether the agent reaches the terminal state (as
  defined under the MDP of the task)", truncated = "Whether the truncation condition outside the
  scope of the MDP is satisfied" (L101-116); `reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]`
  (L125-131); `action_space` L69, `observation_space` L70, `metadata` L65, `spec` L66.
- **Registration.** `register(id, entry_point, reward_threshold, nondeterministic, max_episode_steps, order_enforce, disable_env_checker, additional_wrappers, vector_entry_point, kwargs)`
  (`gymnasium/envs/registration.py:L557`); `make_vec(id, num_envs, vectorization_mode, vector_kwargs, wrappers, **kwargs)`
  (L532-542), `VectorizeMode` = `ASYNC | SYNC | VECTOR_ENTRY_POINT` (L213-216).
- **Vector contract.** `step(self, actions) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]`
  (`gymnasium/vector/vector_env.py:L128-131`); `AutoresetMode` = `NEXT_STEP | SAME_STEP | DISABLED` (L29-32).
- **Breaking since 1.0.** v1.1.0: "in v1.1, we have added support to the implemented vector
  environments (`SyncVectorEnv` and `AsyncVectorEnv`) and wrappers for all three possible modes:
  next-step, same-step and disabled", declared in "`metadata["autoreset_mode"]`" (releases/tag/v1.1.0).
  v1.2.0: "moved the MuJoCo v2 and v3 to the Gymnasium-Robotics project", Python 3.8/3.9 dropped
  (`requires-python = ">= 3.10"`). v1.3.0: no Env/VectorEnv API change listed.
- **Success signal: none** — fetch of `gymnasium/core.py:L1-300` for "success": "no occurrence"; the only
  documented `info` key is the deprecated `"TimeLimit.truncated"`. **Artifact: none.**

### 1.2 openpi (π0 / π0.5) — a websocket policy, a LIBERO client, a log line

- **Policy contract.** `class BasePolicy(abc.ABC)`: `def infer(self, obs: Dict) -> Dict: """Infer actions from observations."""`
  plus `reset(self) -> None` (`packages/openpi-client/src/openpi_client/base_policy.py:L1-12`, whole
  file). Client `infer`: msgpack-pack → `ws.send` → `ws.recv`; a string reply is an error,
  `raise RuntimeError(f"Error in inference server:\n{response}")`
  (`packages/openpi-client/src/openpi_client/websocket_client_policy.py:L35-42`); `reset` is `pass` (L44-45).
  Server: "Currently only implements the `load` and `infer` methods"
  (`src/openpi/serving/websocket_policy_server.py:L9-11`); sends `metadata` first (L36),
  `action = self._policy.infer(obs)` (L44), adds `"server_timing": {"infer_ms", "prev_total_ms"}` (L45-51);
  no reset message on the wire. `EnvMode` = `ALOHA | ALOHA_SIM | DROID | LIBERO` (`scripts/serve_policy.py:L11-16`).
- **What the task exposes.** Wire keys `"observation/image"`, `"observation/wrist_image"`,
  `"observation/state"`, `"prompt"`; state = `np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(...), obs["robot0_gripper_qpos"]))`
  (`examples/libero/main.py:L115-128`); images 224², flipped `[::-1, ::-1]`. Chunk replay:
  `action_chunk = client.infer(element)["actions"]; action_plan.extend(action_chunk[: args.replan_steps])`
  (L129-135), `replan_steps: int = 5`, `num_trials_per_task: int = 50`, `seed: int = 7` (L16-34).
  Starts: `initial_states = task_suite.get_task_init_states(task_id)`; `obs = env.set_init_state(initial_states[episode_idx])`
  (L100-108); `max_steps` per suite 220/280/300/520/400.
- **Success.** `obs, reward, done, info = env.step(action.tolist()); if done: task_successes += 1`
  (L137-138) — success IS `done` (LIBERO's convention, §1.4).
- **Artifact.** `imageio.mimwrite(... / f"rollout_{task_segment}_{suffix}.mp4", fps=10)` (L157-161);
  `logging.info(f"Total success rate: ...")`, `logging.info(f"Total episodes: ...")`. Fetch verdict on
  result files: "None found. Logging only; no JSON or CSV output to disk." The *README.md* table gives
  π0.5 98.8 / 98.2 / 98.0 / 92.4 (avg 96.85) with no trial count or interval.

### 1.3 NVIDIA Isaac-GR00T — flat `video./state./annotation.` dicts, ZeroMQ, a printed mean

- **Layout (the brief's paths moved).** *scripts/eval_policy.py* → HTTP 404 on main; `scripts/eval/` holds
  only *check_sim_eval_ready.py*. `gr00t/eval/` = *_horizon_contract.py*, *open_loop_eval.py*, *rollout_policy.py*,
  *run_gr00t_server.py*, `sim/` (LIBERO, SimplerEnv, robocasa, robocasa365, robocasa-gr1-tabletop-tasks, `wrapper/`), `real_robot/SO100/`.
- **Policy contract.** `get_action(self, observation: dict[str, Any], options=None) -> tuple[dict[str, Any], dict[str, Any]]`
  (`gr00t/policy/policy.py:L54-72`); `reset(options) -> dict` (L80-86). Transport ZeroMQ REQ/REP,
  msgpack with `allow_pickle=False` (`gr00t/policy/server_client.py:L23-100`); endpoints `ping`, `reset`,
  `get_modality_config`, `kill` (L141-144). **Modality config** = `ModalityConfig(delta_indices: list[int], modality_keys: list[str])`,
  "Delta indices to sample relative to the current index" (`gr00t/data/types.py:L69-104`): which keys,
  which time offsets, per modality.
- **What the task exposes.** Validator: "Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]",
  "'state.state_name': np.ndarray[np.float32, (B, T, D)]", "Language keys: tuple[str] or list[str] with
  shape (B,)"; output "'action.action_name': np.ndarray[np.float32, (B, T, D)]"
  (`gr00t/policy/gr00t_policy.py:L514-705`). The LIBERO adapter is a `gym.Env` with `observation_space` keys
  `"video.image"`, `"video.wrist_image"`, `"state.x"…`"state.yaw"`, `"state.gripper"`,
  `"annotation.human.action.task_description"`; `action_space` `"action.x"…`"action.gripper"`; step returns
  `(observation, reward, done, truncated, info)` with `info["success"]`; `register(id=f"libero_sim/{task_name}", ...)`
  (`gr00t/eval/sim/LIBERO/libero_env.py:L67-165`). `MultiStepWrapper(env, contract, max_episode_steps, reward_agg_method=MAX, terminate_on_success=False)`
  runs `n_action_steps` env steps per call, stacks `video`/`state` by delta indices, emits info keys
  `"states","rewards","model","actions","dones","n_env_steps","intermediate_signals","success"`;
  `if self.terminate_on_success and any(info["success"]): done = True` (`gr00t/eval/sim/wrapper/multistep_wrapper.py:L20-140`).
  `PolicyHorizonSpec` — "Single source of truth for the policy-coupled horizons that the sim-eval wrapper
  and real-robot clients must agree on": `n_action_steps, action_horizon, video_delta_indices, state_delta_indices`
  (`gr00t/eval/_horizon_contract.py:L62-100`); `--action-horizon` deprecated for `--execution-horizon` (L29-49).
- **Runner.** `RolloutConfig`: `n_episodes: 50`, `n_envs: 8`, `max_episode_steps` 720, optional `seed`
  (`gr00t/eval/rollout_policy.py:L463-503`); `reset_seeds = [int(seed) + i for i in range(n_envs)]` (L293-298);
  `actions, _ = policy.get_action(observations); next_obs, rewards, terminations, truncations, env_infos = env.step(actions)`
  (L304-307); success = OR of `env_infos["success"][env_idx]` and `env_infos["final_info"][env_idx]["success"]`,
  other dtypes → `raise ValueError(f"Unknown success dtype ...")` (L309-350).
- **Artifact.** Returns `episode_successes, episode_lengths, episode_rewards, episode_infos` (L378-390);
  `print("results: ", results); print("success rate: ", np.mean(results[1]))` (L543-545); verdict "does not
  write JSON or other files". Open-loop: `mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)`
  (`gr00t/eval/open_loop_eval.py:L234`), a JPEG plot; "No success rate metric is computed".

### 1.4 LIBERO and SimplerEnv — success as `done`, success as a filename

**LIBERO.** `ControlEnv` wraps robosuite: `def step(self, action): return self.env.step(action)`
(`libero/libero/envs/env_wrapper.py:L100-101`), `check_success` L106, `get_sim_state` L111, `set_init_state` L123,
cameras `['agentview', 'robot0_eye_in_hand']` L30-31. Success IS done:
`obs, reward, done, info = super().step(action); done = self._check_success()`
(`libero/libero/envs/bddl_base_domain.py:L647-654`), `horizon=1000` L70; `_check_success` is a conjunction over
`self.parsed_problem["goal_state"]` via `_eval_predicate` (`libero/libero/envs/problems/libero_tabletop_manipulation.py:L98-121`).
Eval: `env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1` (`libero/lifelong/metric.py:L66`),
`SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) ...])` (L60-72),
`indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]; obs = env.set_init_state(init_states_)`
(L91-98: fixed start states, cycled), `dones[k] = dones[k] or done[k]` (L114), `while steps < cfg.eval.max_steps` (L117),
`num_success += int(dones[k])` (L126), `success_rate = num_success / cfg.eval.n_eval` (L129), `return success_rate` (L133).
Defaults `n_eval: 20`, `max_steps: 600`, `num_procs: 20` (`libero/configs/eval/default.yaml`), `seed: 10000`
(`libero/configs/config.yaml`); obs `agentview_rgb`, `eye_in_hand_rgb`, `gripper_states`, `joint_states`, 128×128
(`libero/configs/data/default.yaml`). Artifact: `torch.save(eval_stats, save_folder)` with
`{"loss": test_loss, "success_rate": success_rate}` to `..._{algo}_{policy}_{seed}_load{load_task}_on{task_id}.stats`
(`libero/lifelong/evaluate.py:L177-180`); a `_videos` folder. Two scalars per task; no per-episode row; no interval.

**SimplerEnv.** `obs, reward, done, truncated, info = env.step(...)` (`simpler_env/evaluation/maniskill2_evaluator.py:L102`);
underneath, `terminated = self.get_done(obs=obs, info=info); return obs, reward, terminated, False, info` and
`def get_done(self, info, **kwargs): return bool(info["success"])` (`mani_skill2_real2sim/envs/sapien_env.py:L473-487`,
repo simpler-env/ManiSkill2_real2sim); `evaluate()` returns `dict(is_grasped, consecutive_grasp, lifted_object,
lifted_object_significantly, success, episode_stats)` (`mani_skill2_real2sim/envs/custom_scenes/grasp_single_in_scene.py:L496-520`);
`get_language_instruction` → `f"pick {obj_name}"` (L582-584); `is_final_subtask` (`mani_skill2_real2sim/envs/custom_scenes/base_env.py:L345-347`).
Starts: `env.reset(options=env_reset_options)` with `robot_init_options` and `obj_init_options` (`init_xy` or `episode_id`)
(`maniskill2_evaluator.py:L55-68`); the grid is CLI: `--obj-init-x-range [-0.35,-0.12,5]`, `--obj-init-y-range [-0.02,0.42,5]`,
`--obj-variation-mode {xy,episode}`, `--obj-episode-range [0,60]`, `--max-episode-steps 80`, `--control-freq 3`,
`--sim-freq 513`, `--logging-dir ./results` (`simpler_env/evaluation/argparse.py:L9-70`). Success:
`success = "success" if done else "failure"` (L104). Artifact: **the video filename**
`f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{...}/{video_name}"`
(L124-133), verdict "does not write JSON or CSV files";
`success_arr = maniskill2_evaluator(model, args); print(" " * 10, "Average success", np.mean(success_arr))`
(`simpler_env/main_inference.py:L57-59`). Metrics re-parse filenames: `glob.glob(dir_name + "/**/*.mp4", recursive=True)`,
`if succ_fail_pattern[0] in fname: results.append(1)` (`simpler_env/utils/metrics.py:L108-129`);
`mean_maximum_rank_violation` (L68-84) and `pearson_correlation` (L53-65) run over `REAL_PERF`/`SIMPLER_PERF`
dicts of point estimates (`"rt-2-x": 0.907`). No interval anywhere.

### 1.5 LeRobot EnvHub — *env.py* + `make_env`, `info["is_success"]`, eval_info.json

- **Blog** ("Generalist Robot Policy Evaluation in Simulation with NVIDIA Isaac Lab-Arena and LeRobot",
  5 Jan 2026): EnvHub "enables developers to share simulation environments, and easily load them for
  training, evaluation, or teleoperation, directly from the LeRobot framework"; the eval line is
  `lerobot-eval --env.type=isaaclab_arena --env.hub_path=nvidia/isaaclab-arena-envs ...`. **Spec** (huggingface.co/docs/lerobot/envhub): "The only requirement is that the package contains an
  *env.py* file"; "you must provide a `make_env(n_envs: int = 1, use_async_envs: bool = False)` or
  `make_env(n_envs: int = 1, use_async_envs: bool = False, cfg: EnvConfig)` function"; it returns "A
  `gym.vector.VectorEnv` (most common)", "A single `gym.Env` (will be automatically wrapped)", or "A dict
  mapping `{suite_name: {task_id: VectorEnv}}`"; "your environment must implement the standard
  `gym.vector.VectorEnv` interface"; loading needs `trust_remote_code=True`; pin as `user/repo@rev:path`.
  Loader: `_normalize_hub_result` wraps a `gym.Env` in `SyncVectorEnv([lambda: result])`
  (`src/lerobot/envs/utils.py:L318-345`); "The hub module must expose `make_env(n_envs=int, use_async_envs=bool)`" (L300-315);
  `make_env` returns "dict[str, dict[int, gym.vector.VectorEnv]]" (`src/lerobot/envs/factory.py:L63-88`);
  `HubEnvConfig(EnvConfig): hub_path` (`src/lerobot/envs/configs.py:L162-175`); `EnvConfig` fields `task, fps=30, features, features_map, disable_env_checker: bool = True` (L54-65).
- **Obs contract.** `preprocess_observation`: `"pixels"` → `observation.images.*` (dict) or `observation.image`,
  `"agent_pos"` → `observation.state`, `"environment_state"` → `observation.environment_state`, uint8 HWC →
  float32 CHW /255 (`src/lerobot/envs/utils.py:L75-135`); instruction added as `observation["task"]`
  (`src/lerobot/scripts/lerobot_eval.py:L227`); `--rename_map` re-keys at the CLI. Arena's hub page fixes
  per-env knobs `state_keys`, `camera_keys`, `state_dim=54`, `action_dim=36`, `episode_length=300`, `seed=42`.
- **Success.** `observation, reward, terminated, truncated, info = env.step(action_numpy)` (L287);
  `if "final_info" in info: ... is_success = final_info.get("is_success", [False] * env.num_envs)`,
  `elif "is_success" in info`, else all False (L291-310). `EvalConfig`: `n_episodes: int = 50`, `batch_size: int = 0`
  (auto), `use_async_envs: bool = True` (`src/lerobot/configs/default.py:L102-122`); `EvalPipelineConfig`:
  `seed: int | None = 1000`, `rename_map`, `trust_remote_code: bool = False` (`src/lerobot/configs/eval.py:L28-36`);
  seeds `range(start_seed + batch_ix * env.num_envs, ...)`.
- **Artifact (the only written per-episode record of the six).** Per episode
  `{"episode_ix": i, "sum_reward": ..., "max_reward": ..., "success": ..., "seed": ...}` (L412); aggregated
  `{"avg_sum_reward": nanmean, "avg_max_reward": nanmean, "pc_success": nanmean*100, "eval_s", "eval_ep_s"}` (L417-422);
  `info = {"per_episode": [...], "aggregated": {...}, "video_paths": [...], "predicted_video_paths": [...]}` →
  `json.dump(info, f, indent=2)` as eval_info.json (L662-716) under `outputs/eval/{now:%Y-%m-%d}/{now:%H-%M-%S}_{job_name}`
  (`src/lerobot/configs/eval.py:L56-58`); videos `eval_episode_{n}.mp4` at `env.unwrapped.metadata["render_fps"]`.

### 1.6 robomimic — `is_success()["task"]`, per-rollout dicts, a mean per env

- **Env contract** (`robomimic/envs/env_base.py:L51-147`): `step` → "observation (dict) / reward (float) /
  done (bool): whether the task is done / info (dict)" (L51-60); `reset_to(state)` "Reset to a specific
  simulator state" (L70-79); `get_state` "compatible with @reset_to" (L104-106); `is_done` "Check if the
  task is done (not necessarily successful)" (L138-141); `is_success` "Should return a dictionary
  { str: bool } with at least a 'task' key for the overall task success, and additional optional keys
  corresponding to other task criteria" (L142-147). Done and success are **separate** here.
- **Rollout** (`robomimic/utils/train_utils.py:L240-375`): `run_rollout(policy, env, horizon, use_goals, render, video_writer, video_skip, terminate_on_success)`;
  `policy.start_episode()`; `ac = policy(ob=policy_ob, goal=goal_dict)`; `cur_success_metrics = env.is_success()`;
  `results["Return"] = total_reward; results["Horizon"] = end_step + 1; results["Success_Rate"] = float(success["task"])`,
  plus `"{}_Success_Rate".format(k)` per extra key and `"time"` (L240-280). `rollout_with_stats(..., num_episodes, video_dir, epoch, ...)`:
  `rollout_logs.append(rollout_info)`, `dict((k, np.mean(v)) for k, v in rollout_logs.items())`,
  `rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60.`, `return all_rollout_logs, video_paths` (L320-375);
  `should_save_from_rollout_logs` keeps `best_return` / `best_success_rate` — rollouts, not loss, pick checkpoints.
- **Standalone** (`robomimic/scripts/run_trained_agent.py:L68-207`): `--n_rollouts` default 27, `--seed`;
  `success = env.is_success()["task"]` (L68); per-rollout `Return, Horizon, Success_Rate=float(success)` (L78-80);
  `avg_rollout_stats` = means + `"Num_Success" = np.sum(rollout_stats["Success_Rate"])` (L197-199); stdout via
  `json.dumps(avg_rollout_stats, indent=4)` (L204-207; no `--json_path` in this file); optional HDF5 `demo_{i}`
  groups `actions, states, rewards, dones, obs/...` (L182-189).

## 2. Comparison

| System | Env API | Obs keys | Action semantics | Success signal | Per-episode record | Aggregate | Interval? |
|---|---|---|---|---|---|---|---|
| Gymnasium 1.x | `reset(seed, options)→(obs, info)`, `step→(obs, r, term, trunc, info)` | any `Space` | any `Space` | none defined | none | none | no |
| openpi | LIBERO `step→(obs, r, done, info)`; policy `infer(dict)→dict`, websocket+msgpack | `observation/image`, `observation/wrist_image`, `observation/state`, `prompt` | `["actions"]` chunk, first `replan_steps`=5 executed | `done` | none (mp4 + log) | log `total_successes/total_episodes` | no |
| GR00T | `gym.Env` + `MultiStepWrapper`; policy `get_action(dict)→(dict, dict)`, ZeroMQ | `video.<cam>` (B,T,H,W,C) u8, `state.<n>` (B,T,D), `annotation.human.action.task_description` | `action.<n>` (B,T,D), `n_action_steps` executed | `info["success"]` ∨ `final_info[i]["success"]` | memory only | `print(np.mean(...))` | no |
| LIBERO | robosuite `step→(obs, r, done, info)`, `set_init_state` | `agentview_rgb`, `eye_in_hand_rgb`, `gripper_states`, `joint_states` | 7-D OSC delta | `done = _check_success()` | none | `.stats` `{loss, success_rate}` | no |
| SimplerEnv | Gymnasium 5-tuple, `reset(options={obj/robot_init_options})`, `get_language_instruction()` | image by camera name | `world_vector, rot_axangle, gripper` | `terminated = info["success"]` | video filename `success_/failure_` | `np.mean(success_arr)`; MMRV/Pearson vs real | no |
| LeRobot EnvHub | *env.py* `make_env(n_envs, use_async_envs[, cfg])`→VectorEnv; Gymnasium 5-tuple | `pixels`→`observation.images.*`, `agent_pos`→`observation.state`, `task` | policy's `action` | `info["is_success"]` / `final_info` | eval_info.json `per_episode[{episode_ix, sum_reward, max_reward, success, seed}]` | `pc_success, avg_sum_reward, avg_max_reward` | no |
| robomimic | `EnvBase.step→(obs, r, done, info)`, `reset_to(state)`, `is_success()→{"task": bool,…}` | robosuite dict | env's | `is_success()["task"]` | dict `{Return, Horizon, Success_Rate, time}` (memory; HDF5 optional) | means + `Num_Success` | no |

## 3. The common denominator

The smallest task interface all six can drive, as exact names:

1. **A `gym.Env` subclass, Gymnasium ≥1.1 semantics**: `reset(*, seed=None, options=None) -> (obs, info)`,
   `step(action) -> (obs, reward, terminated, truncated, info)`, `observation_space: spaces.Dict`,
   `action_space: spaces.Box`, `metadata = {"render_fps": <control Hz>, "autoreset_mode": AutoresetMode.NEXT_STEP}`;
   registered `gym.register(id="trainnr/<task>-v0", entry_point=..., max_episode_steps=protocol.steps)`.
2. **One observation dict, three things, renamed at the adapter**:
   `{"pixels": {"<cam>": uint8 (H,W,3)}, "agent_pos": float32 (state_width,), "task": str}` — LeRobot's raw form
   (`preprocess_observation` maps it to `observation.images.<cam>` / `observation.state`); GR00T's
   `video.<cam>` / `state.<n>` / `annotation.human.action.task_description` and openpi's `observation/image` /
   `observation/state` / `prompt` are the same three fields under prefixes, applied by the rig adapter
   (report 43 §3 item 2). Plus `get_language_instruction() -> str` (SimplerEnv, Arena).
3. **Success on every step, under both names**: `info["success"] = bool` (GR00T, SimplerEnv, robomimic's
   `"task"`) and `info["is_success"] = bool` (LeRobot), same value; `terminated` stays the MDP end and
   `truncated` the step cap — never `done = success` (LIBERO/openpi conflate them; Gymnasium's text forbids it).
4. **Paired starts through the standard door**: `reset(seed=trial)` calls `protocol.perturb(trial, home)`
   (`trainnr/trainnr/evaluate/harness.py:65`), so seed *k* is start *k* for every policy; plus
   `get_state()` / `reset_to(state)` (robomimic; LIBERO's `set_init_state`) for exact replay.
5. **Hub form**: *env.py* with `make_env(n_envs=1, use_async_envs=False, cfg=None)` returning
   `gym.vector.SyncVectorEnv` of item 1 (the spec's "most common" return); `requirements.txt`; pinned `@<commit>`.
6. **Artifact in LeRobot's shape, widened**: eval_info.json with `per_episode[i] = {episode_ix, seed, success,
   sum_reward, max_reward, episode_length, policy, task, perturb_hash, bundle, protocol}` and
   `aggregated = {pc_success, successes, trials, ...}` — a superset of LeRobot's keys, robomimic's `Horizon` /
   `Num_Success`, GR00T's `episode_lengths`, and docs/30 §7's `EpisodeRecord`. `SimScore(successes, trials)`
   (`trainnr/trainnr/evaluate/harness.py:80`) becomes a fold over it.

## 4. What none of them provides

- **No interval, anywhere.** Every aggregate is a mean: LeRobot `float(np.nanmean(all_successes) * 100)`,
  LIBERO `num_success / cfg.eval.n_eval`, SimplerEnv `np.mean(success_arr)`, GR00T `np.mean(results[1])`,
  robomimic `np.mean(v)`, openpi `total_successes / total_episodes`. Fetches of the six eval files surfaced no
  "confidence", "interval", "binom", "bootstrap" or "Clopper". SimplerEnv's real-vs-sim MMRV/Pearson run on
  point estimates in a dict. The certificate's Clopper–Pearson and paired test have no counterpart.
- **No paired trials as a statistic.** Pairing exists by construction (LIBERO's fixed init states,
  SimplerEnv's grid, GR00T/LeRobot `seed + i`) but nothing compares policy A to B on matched starts;
  LeRobot's per-episode `seed` is the only hook.
- **No written per-episode record except LeRobot's.** openpi and GR00T print; LIBERO saves two scalars;
  SimplerEnv encodes the verdict in a filename and re-globs it; robomimic keeps dicts in memory.
- **No trial count in the artifact** except by list length (LeRobot) or `Num_Success` (robomimic).
- **No stamp of what was evaluated**: versions live in the gym id suffix (`-v0`) and the hub revision; no
  artifact hashes dynamics, protocol or adapter. `perturb_hash` / `bundle` / `protocol` in item 6 are ours alone.
- **No census gate**: LeRobot sets `disable_env_checker: bool = True`; nothing checks the model loaded with
  its actuators, sensors and cameras before episodes are spent (`trainnr/trainnr/evaluate/harness.py:125-132` does).
- **Success conflated with termination** in LIBERO and openpi; robomimic and Gymnasium keep them apart.

## 5. Open questions

1. **`info["success"]` under autoreset.** GR00T and LeRobot both read `final_info` for the last step of an
   auto-reset env. Which mode should our vector env declare so a single-step `success` is never lost —
   `NEXT_STEP` (default) or `DISABLED` with our own loop?
2. **Chunk execution inside or outside the env.** GR00T's `MultiStepWrapper` executes `n_action_steps` per
   `step`; openpi replays `replan_steps` in the client; LeRobot's policy holds the queue. Report 43 put the
   executed horizon in `EpisodeProtocol`; a GR00T-style wrapper would make it visible to every client. Which layer owns it?
3. **Instruction: observation key or method?** GR00T wants it in the obs dict, SimplerEnv/Arena as a method,
   LeRobot as config `task`. Provide all three, or one and let adapters call it?
4. **LeRobot's `sum_reward` / `max_reward`** presume a per-step reward; ours is a boolean referee. Emit
   `reward = float(success)` (robomimic's sparse convention; LIBERO's `reward = 1.0 if success`) or 0?
5. **Hub publishing of a MuJoCo bundle**: the spec allows only `requirements.txt`; assets and the adapter
   hash must ride in the repo. Does `@<commit>` suffice as the `source@hash` stamp `score_policies` demands
   (`trainnr/trainnr/evaluate/harness.py:120-124`), or do we hash the tree ourselves?
