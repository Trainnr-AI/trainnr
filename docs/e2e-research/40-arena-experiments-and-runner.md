# Arena experiments and the evaluation runner: what to take, what to leave

*Source: the `isaaclab_arena` repository, read 2026-08-26 from a repomix
bundle made in session (132,303 lines; commit not recorded). The bundle is
not shipped; every `L<number>` below is a line of that bundle. Our-side line
numbers are from the 2026-08-26 tree: `EpisodeProtocol` is now
`trainnr/trainnr/protocol.py`, the aloha2 task module is now the package `trainnr/trainnr/tasks/aloha2/`, `bundles/` is
`trainnr/trainnr/bundles/`.*

*Fifth pass, 2026-08-26. One agent, one field — how NVIDIA Isaac Lab Arena
specifies an "experiment", runs it locally and across nodes, records each
episode, and aggregates — read from the ACTUAL code (repomix bundle of
`isaaclab_arena`, 132,303 lines). Citations are `<repo path>:L<bundle line>`;
the number is the bundle's line. Mapped against our harness
(`trainnr/trainnr/evaluate/`), certificate (`stats/`) and bundles
(`trainnr/trainnr/bundles/`). Nothing below comes from memory of Isaac Lab.*

---

## 1. What Arena does

### 1.1 The experiment definition — `shared` + named `runs`

An **Experiment** is a YAML file; a **Run** is one (environment × policy ×
rollout settings) unit whose name is its mapping key. "The YAML has a required
``runs`` mapping. A ``shared`` mapping is optional."
(`docs/pages/concepts/concept_arena_experiments.rst:L7332`); "The only
top-level fields are ``shared`` and ``runs``" (`:L7451`). Shared values merge
under each run — `OmegaConf.merge(shared_run_defaults, raw_run_config)`
(`isaaclab_arena/hydra/typed_experiment_loader.py:L35301`). Precedence, low to
high: "typed configuration defaults → shared values, including shared.* CLI
overrides → values written in an individual Run → runs.<name>.* CLI overrides"
(`concept_arena_experiments.rst:L7440-7446`). Overrides "cannot add Runs"
(`typed_experiment_loader.py:L35157`); names match `[A-Za-z_][A-Za-z0-9_-]*`
(`:L35207`) and must be safe path components
(`evaluation/arena_experiment_result.py:L33242-33248`). The typed shape
(*isaaclab_arena/evaluation/arena_run.py*):

```
ArenaExperimentCfg.runs: dict[str, ArenaRunCfg]   # "keyed by the names used for overrides, execution, and results" L33378-33379
ArenaRunCfg: name, environment: ArenaEnvironmentCfg, policy: PolicyCfg,
             environment_builder: ArenaEnvBuilderCfg, rollout_limit: RolloutLimitCfg,
             num_rebuilds: int = 1, variations: dict[str, Any]                  L33440-33459
RolloutLimitCfg: num_steps | num_episodes — "mutually exclusive"                L33431
ArenaEnvBuilderCfg: num_envs=1, seed=42, placement_seed=None, resolve_on_reset,
             device, language_instruction        environments/arena_env_builder_cfg.py:L31734-31744
```

`environment.type` / `policy.type` select registered dataclasses (policies may
also be a dotted class path, `arena_experiment_config_loader.py:L33128-33139`);
remaining keys are validated against that class by Hydra, a config-composition
library (`typed_experiment_loader.py:L35439-35486`). Worked file: `shared:`
carries environment, `policy: type: zero_action`, `rollout_limit: num_episodes:
1`; then `runs: baseline: {}`, `swap_objects: environment: pick_up_object: ...`,
`parallel_envs: environment_builder: num_envs: 64`
(`isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml:L99025-99054`).
Variations are dotted keys in a run — `light.hdr_image.enabled: true` with
`num_rebuilds: 5`, "Number of times the environment is rebuilt with different
variation values." (`droid_pnp_variations_experiment.yaml:L99002-99008`). The
task text sits on the environment builder, `environment_builder:
language_instruction: Pick up the Rubik's cube...`
(`droid_pnp_srl_gr00t_experiment.yaml:L98764-98765`), and is pushed into the
policy at rollout start:
`policy.set_task_description(env.unwrapped.get_language_instruction())`
(`evaluation/policy_runner.py:L34428`). A legacy JSON form `{"jobs": [...]}`
(`legacy_eval_config.py:L33921`, required `("name", "arena_env_args",
"policy_type")` `:L33938`) converts to the same `ArenaRunCfg` and is slated for
deletion (`:L33895`).

### 1.2 Local execution — one process, runs in order, fresh env per rebuild

*experiment_runner.py* loads the file, prints a run table, executes, logs
metrics, writes the result JSON, builds an HTML report
(`evaluation/experiment_runner.py:L33784-33819`). Runs execute "in order"; a
raising run becomes `ArenaRunResult(run_name=..., status=RunStatus.FAILED)` and
without `--continue_on_error` the exception re-raises
(`run_execution.py:L34732-34751`). Inside a run (`build_and_run`,
`:L34757-34804`) the episode budget is split evenly across **rebuilds** — fresh
environment constructions (`:L34889-34902`), each offsetting the seed:
`cfg.environment_builder.seed += rebuild_index` (`:L34810`) — the recorder is
pointed at `episode_results_rebuild{rebuild_index}.jsonl` and stamped with the
run name (`:L34783-34785`), resources close in `finally` (`:L34798`);
`num_episodes >= num_rebuilds` is asserted (`arena_run.py:L33465`). The rollout
loop (`policy_runner.py:L34439-34470`) steps all parallel **envs** (scene copies
inside one run) together, resets the policy only for finished env ids —
`policy.reset(env_ids=env_ids)` (`:L34451`) — and stops at
`num_episodes_completed >= num_episodes` (`:L34463`). A policy may declare its
own length (`run_execution.py:L34873-34886`).

Flags (*experiment_runner_cli.py*): `--output_base_dir` (reverse-dated
`<base>/<timestamp>`) vs `--experiment_output_directory` (exact, "must be
missing or empty", `:L33574-33593`); `--record_camera_video` "one mp4 per (env,
camera, episode)" (`:L33566`); `--chunk_size` — "Each restart lets the OS
reclaim accumulated memory" (`:L33618`); `assert not args_cli.distributed,
"Distributed evaluation is not supported yet"` (`:L33636`). The older
*policy_runner.py* shards one run over `torchrun` ranks: `args_cli.seed +=
local_rank` (`:L34546`), `episode_results_rank{local_rank}.jsonl` (`:L34564`),
report by rank 0 only (`:L34609`).

### 1.3 The per-episode record

An `EpisodeRecorderManager` "Records per-episode data, described by terms.
Written out as JSONL on request." (`recording/episode_recorder_manager.py:L38134`;
JSONL = one JSON object per line). The env's reset hook fires it once per
finishing episode — `self.episode_recorder_manager.record_pre_reset(env_ids)`
(`environments/isaaclab_arena_manager_based_env.py:L32644`); each **term** is a
function `func(env, env_id) -> dict`, fields merge, collisions are asserted
(`:L38196-38199`), JSON-serializability is checked per term "rather than
surfacing a cryptic error later at write() time" (`:L38232-38234`), the line is
appended (`:L38209-38210`). Schema (`arena_experiment_result.py:L33171-33182`):

```
job_name: str          # the Run name, stamped by the manager (L38190-38192)
env_id: int            # which parallel copy
episode_in_env: int    # per-env episode counter (isaaclab_arena_manager_based_env.py:L32627-32635)
seed: int | None       # env.cfg.seed — the RUN seed, not per-episode (recording/common_terms.py:L38062)
success: bool | None   # None when no "success" termination term exists (L38056-38058)
episode_length: int;  language_instruction: str | None
timestamp: str         # datetime.now().isoformat() (L38066)
variations: NotRequired[dict]  # value drawn for this (env, episode) (L38070-38082; variations/variation_recorder.py:L90985-90994)
progress: NotRequired[dict]    # partial credit, below
```

`progress` (`recording/progress_terms.py:L38272-38298`): `overall_score`,
`all_complete`, per-objective `{score, is_complete, completed_groups,
total_groups, active_predicates}`, and `events` — "Per-episode predicate
transitions, in the order they fired", each `{step, objective, group,
predicate_index, predicate_name, score_delta}`.

### 1.4 Aggregation and outputs — three layers, none statistical

1. **Metrics.** `MetricTermCfg` pairs a recorder term with
   `compute_metric_func(recorded_metric_data: list[np.ndarray], **params)`
   (`metrics/metric_term_cfg.py:L35841-35853`); success rate is `np.mean` of
   the flags (`metrics/success_rate.py:L36396-36399`). Across rebuilds
   `aggregate_metrics` "concatenates the recorded data across runs and
   recomputes each metric value from the combined data"
   (`metrics/aggregate_metrics.py:L35686-35688`). `MetricsLogger` prints a
   table; "save_metrics_to_file() is unused" (`legacy_experiment_runner.py:L34133`).
2. **Result JSON** `arena_experiment_result.json` (`:L33168`): `{"runs": {name:
   {environment: {name, definition}, policy_variant, status:
   "completed"|"failed", rebuilds: [{index, episodes: [...]}]}}}`
   (`:L33186-33202`), `allow_nan=False` (`:L33308`). "It does not compute
   aggregate metrics or collect videos." (`:L33238`). `policy_variant` is the
   configured variant string else the registered policy name (`:L33341-33349`).
3. **Report and plots.** `build_report` writes "an overview of success rate by
   task and policy, a page per task ..., and a page per Run holding the episode
   videos" (`visualization/report.py:L93898-93900`). Task and policy are
   *inferred from run-name strings* — an explicit `_<policy>` suffix, else
   "repeated final tokens" (`report_data.py:L92920-92968`). Success rate =
   `num_successes / num_scored_episodes`; an episode whose `success` is not a
   bool silently leaves the denominator (`:L92372-92374`, `:L92485-92487`);
   *plot_success_rates.py* does the same (`:L92145-92146`) and draws bars with
   no interval (`:L92194-92227`). A bundle-wide grep for Wilson,
   Clopper-Pearson, binomial intervals or p-values finds nothing; the only
   statistics past a mean is the sensitivity package, which "Fits a neural
   posterior over all factors, conditioned on all outcomes" via `sbi` NPE/MNPE
   (`analysis/sensitivity/analyzer.py:L20936-20941`). Result files are found by
   regex `episode_results(?:_rebuild(\d+))?(?:_rank(\d+))?\.jsonl`
   (`visualization/episode_results_files.py:L91794-91796`).

### 1.5 Multi-node — one run per scheduled group, one collector

Motivation: "20 tasks, 2 policies, 100 episodes per task, for a total of 4000
rollouts" (`docs/.../multi_node_evaluation.rst:L11762`). Arena submits to OSMO,
NVIDIA's cluster workflow scheduler. `ArenaExperimentWorkflow`: "Run every
Arena Experiment Run in its own OSMO group, co-scheduling each Run's server."
(`osmo/workflows/arena_experiment_workflow.py:L127616`). Each run is
snapshotted as a single-run experiment — `ArenaExperimentCfg(runs={run_name:
deepcopy(run_config)})` (`:L127665`) — serialised to YAML inside the task
(`osmo/tasks/experiment_runner_task.py:L127018-127021`); a remote-client policy
gets a server task in the same group with `remote_host`/`remote_port` rewritten
to it (`:L127670-127682`), hence "executing everything in parallel requires ``2
× number of Runs`` GPUs" (`multi_node_evaluation.rst:L11815`). The task's shell
wrapper writes `experiment_runner_result.json` = `{"execution_status",
"process_exit_code", "runs": <metadata>}` then `exit 0`, so a policy crash is
data, not a scheduler failure (`experiment_runner_task.py:L127049-127067`). A
CPU-only collector (`"gpu": 0`, `:L127721`) runs
*osmo/scripts/build_experiment_output.py*: completed run directories copy to
`<experiment-output>/<run-name>`, "failed results are preserved without
partial Run artifacts" (`:L126197-126200`), status and exit code are
cross-checked (`:L126235-126237`), and the same `ArenaExperimentResult` +
`build_report` produce what a local run would (`:L126363-126364`). Submission
composes `ArenaExperimentSubmissionCfg(experiment_cfg, osmo: WorkflowCfg,
experiment_runner: ExperimentRunnerTaskCfg)`
(`osmo/submit_arena_experiment.py:L128572-128582`); `WorkflowCfg` defaults
`cpus=15, gpus=1, memory="120Gi", storage="200Gi", exec_timeout="1d"`, output
URL with a `{{workflow_id}}` token (`osmo/workflows/workflow.py:L128145-128180`);
`--dry_run` prints the rendered workflow, `--list_overrides` the composed
config, "every leaf is a valid Hydra KEY=VALUE override"
(`submit_arena_experiment.py:L128684-128691`). Camera video defaults ON on the
cluster (`record_camera_video: bool = True`, `experiment_runner_task.py:L126982`).

## 2. What we already have

| Arena | Ours | Where |
|---|---|---|
| Run = env × policy × rollout limit | `EpisodeProtocol(trials, steps, control_interval, perturb, success, home)`, one protocol for every policy; `SimPolicy` / `VisionPolicy` | `evaluate/harness.py:L45-67`, `evaluate/vision.py:L52-62` |
| Seed offset per rebuild; `placement_seed` "same positions across runs" (`cli/isaaclab_arena_cli.py:L26810`) | `perturb(trial_index, home)` — deterministic, so "policy A's trial 7 starts exactly where policy B's trial 7 starts" (`harness.py:L5-7`). Arena never ties policy A's episode k to policy B's; pairing there is implicit through equal seeds at best | `harness.py:L48-51` |
| `compute_success_rate = np.mean` | `SimScore(successes, trials)`, counts "never just a rate" (`harness.py:L81-82`); `clopper_pearson`, `wilson` | `stats/intervals.py:L78-132` |
| Success by task × policy from run-name suffixes | `certify()` → `Certificate`: per-policy intervals, `fisher_rank_ci`, `exact_spearman_p`, `top_pick_probability`; `pool_rank_correlations` with Cochran's Q across tasks | `evaluate/certificate.py:L99-171`, `stats/ranking.py`, `stats/pooling.py:L90-160` |
| `policy_variant`, `environment.definition` strings | `name@hash` stamps, refused if absent in both `score_policies` and `certify()` | `bundles/hashing.py:L39-43`, `harness.py:L120-124`, `certificate.py:L113-118` |
| Run `status: failed`, listed and skipped | `join_with_real` refuses a sim/real mismatch — a hole is an error, not a row | `harness.py:L176-183` |
| `_assert_camera_support_enabled` (`experiment_runner.py:L33691-33697`) | `assert_model_alive(actuators, sensors, geoms, cameras)` before any episode | `harness.py:L125-132` |
| mp4 per (env, camera, episode) | Rerun + MuJoCo viewers every session; filmstrips in docs/31 | — |

Missing on our side: any per-episode record (docs/31 §3 reports T1, T2, T5 as
bare "0/4"), an experiment file, a run-per-directory layout, a collector, and
any way to run policies on more than one machine.

## 3. Adopt / skip

*Status (2026-10-03): items 1–2, the per-episode record and its exit
status, are `trainnr/trainnr/evaluate/records.py`; the rest below is as
written on 2026-08-26.*

### ADOPT

1. **The per-episode JSONL record, plus the pairing key Arena lacks.** New
   `evaluate/records`: frozen `EpisodeRecord` appended by `score_policies`
   as each trial finishes (a crash at trial 9 leaves 8 lines — Arena's
   append-on-finish, `:L38209`): `run, policy, task, trial, success: bool,
   steps, home, perturb_hash` (sha of the initial state — what Arena's
   run-level `seed` cannot reconstruct), `protocol, robot_bundle, scene_bundle,
   physics_backend, timestamp, events: [{step, predicate}]`. `SimScore` becomes
   a fold over records. Write with `allow_nan=False` (`:L33308`); assert no
   field collision (`:L38196-38199`). `task` and `policy` are explicit fields —
   Arena's `_pi0` suffix inference (`:L92933-92968`) is the failure to design out.
2. **The objective funnel.** Arena's `progress.events` (`:L38286-38297`) is
   what turns three "0/4" rows into "reach 4/4, grasp 0/4". The referee already
   evaluates predicates; record `(step, predicate_name)` transitions per trial
   and count `num_reached` per stage (`report_data.py:L92404-92416`). Cheap, and
   it is the diagnostic the certificate deliberately withholds.
3. **Experiment definition as stdlib dataclasses.** `ExperimentCfg(protocol,
   robot_bundle, scene_bundle, runs: dict[str, RunCfg])`, `RunCfg(policy, task,
   variations)`. One deliberate difference: the protocol is experiment-level —
   that IS the pairing guarantee; a run wanting its own protocol is another
   experiment. Run name = key, `[A-Za-z_][A-Za-z0-9_-]*` and path-safe
   (`:L35207`, `:L33242-33248`); shared-then-run merge; overrides edit, never
   add, runs (`:L35157`). YAML front-end optional; the dataclass is the contract.
4. **Run-per-directory output, `continue_on_error`, a collector.**
   `<experiment>/<run>/episodes.jsonl` + `experiment_result.json` `{"runs":
   {name: {policy, status, episodes}}}` (`:L33193-33202`). A raising run is
   `status: failed` and the rest continue (`:L34745-34751`); `certify()` then
   refuses on the hole — correct, now with a named culprit. Collector =
   `build_experiment_output`'s shape: copy completed dirs, keep failed statuses
   "without partial Run artifacts" (`:L126197-126200`), rebuild the JSON. This
   makes multi-node trivial — each cloud job is the local runner on a
   single-run experiment (`:L127665`); a rented GPU per job (docs/34-cloud-gpu.md)
   stands in for OSMO.
   Keep the wrapper discipline: write `{execution_status, process_exit_code}`
   and exit 0, so no scheduler retry hides a policy crash (`:L127054-127067`).
5. **Aggregate by recomputing from raw records, never by averaging rates**
   (`:L35686-35688`) — already our habit; make it the collector's rule.
6. **`--dry_run` / `--list_overrides`** on the submit path (`:L128684-128691`):
   print the composed experiment before spending GPU-hours.

### SKIP

- **Hydra/OmegaConf composition** — the stats/bundles layer is stdlib-only so
  a signed report recomputes anywhere (`bundles/profile.py:L9-12`); same rule.
- **OSMO** — NVIDIA-internal pools (`pool="isaac-dev-l40s-04"`, `:L128148`);
  the platform rents one GPU per run instead (docs/34-cloud-gpu.md). Take the
  shape, not the tool.
- **`num_rebuilds` + seed offsets** — exist because build-time variation draws
  apply to every episode (`variation_recorder.py:L90978-90983`); ours are
  per-trial functions. **`--chunk_size`** — an Isaac Sim memory-leak
  workaround (`:L33618`). **torchrun sharding** (`:L34505-34546`) — our sim is
  CPU MuJoCo; the GPU path is MJX-Warp ([36](36-newton-status.md), [49](49-gpu-path-mjxwarp.md)), a different batching shape.
- **`success: None` dropped from the denominator** (`:L92485-92487`) — an
  unscored episode is a protocol bug for us; refuse it.
- **HTML report + mp4 per episode** — Rerun is the viewer; a funnel table
  beside the certificate suffices. **sbi sensitivity** (`:L20936`) — heavy
  dependency, no intervals; revisit only for factor screening on large sweeps.

## 4. Open questions

1. **Remote policy servers.** Arena's client/server split with a co-scheduled
   server per run (`:L127670-127682`) is how π0.5-class models are evaluated
   outside the sim process; ours is in-process (`vision.py:L103-170`). When the
   first cloud-GPU checkpoint ([docs/31](../31-aloha2-e2e.md) T5) needs judging, do we add a
   `remote_host/port` PolicySpec — and does server latency count in the protocol?
2. **Where the language instruction lives.** Arena: environment builder
   (`:L98764`), pushed at reset (`:L34428`). Ours: a policy constructor argument.
   If the referee judges it, it belongs in `EpisodeProtocol` and in the record.
3. **Run = policy, or policy × task?** With protocol experiment-level, a
   multi-task certificate (`pool_rank_correlations`) spans experiments; the
   collector must merge by `task`. Decide before the record schema freezes.
4. **Real-side records.** Arena has no real-robot half. `join_with_real` takes
   bare `(successes, trials)`; real trials deserve the same `EpisodeRecord`
   (operator, takeover timestamp; [38](38-fleet-data-planes.md)) so both halves share one shape.
