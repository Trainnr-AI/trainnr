# Isaac Lab Arena: metrics, progress tracking, predicates and composite tasks

*Research pass, 2026-08-26. One agent, one field, one primary source: a
repomix bundle of the whole `isaaclab_arena` repository (132,303 lines).
Every claim cites `<repo path>:<line>`, the line in that bundle
(*scratchpad/arena.md*), with the bundle's text quoted; nothing is filled
from memory of Isaac Lab in general. The field: how Arena decides an
episode succeeded, scores partial progress, records episodes, and judges
multi-stage tasks — mapped against our harness (`rq_pipeline/evaluate/`),
task referees (`rq_pipeline/tasks/aloha2.py`) and the certificate
(`rq_pipeline/stats/`, `evaluate/certificate.py`).*

---

## 1. What Arena does

Arena's motivation is coverage: "the evaluation set needed to test
generalization grows combinatorially with tasks, scenes, embodiments,
objects, and environment variations" (`docs/pages/motivation/motivation.rst:16214`).
A task therefore defines success, failure, reset and metrics once, and
every composed environment inherits them.

### 1.1 Success is a termination term built from predicates

A task "specifies four things": "**Success condition** — when has the
robot completed the task?", "**Failure condition** — when should the
episode end early?", "**Reset action**", "**Metrics**"
(`docs/pages/concepts/task/index.rst:7018-7021`).

A **predicate** is "a callable that receives the manager-based
environment ... and returns one Boolean per parallel environment"
(`docs/pages/concepts/task/concept_progress_tracking_design.rst:6719-6720`).
Predicates read privileged simulator state: `get_root_pos_w` returns
`wp.to_torch(get_rigid_object(env, name).data.root_pos_w)`
(`isaaclab_arena/tasks/predicates/predicate_utils.py:46348`). The
placement predicate `object_on_destination` needs a contact sensor plus a
velocity check — `condition_met = torch.logical_and(force_above_threshold,
velocity_below_threshold)`, defaults `force_threshold = 1.0`,
`velocity_threshold = 0.5` (`isaaclab_arena/tasks/predicates/spatial.py:46472-46499`).

**Success** is one termination term composing predicates:
`check_success(env, predicates, mode, k)` stacks
`predicate.func(env, **predicate.params)` and combines by `SuccessMode`
— `ALL`, `ANY`, or `CHOOSE` (`results.sum(dim=0) >= k`)
(`isaaclab_arena/tasks/terminations.py:49682-49738`). Pick-and-place
passes `{"mode": SuccessMode.ALL, "predicates": [...]}` with
`object_on_destination` and an optional `objects_in_proximity` box
(`isaaclab_arena/tasks/pick_and_place_task.py:48287-48321`); its failure
term is `object_dropped = TerminationTermCfg(func=mdp_isaac_lab.root_height_below_minimum, ...)`
(`:48322-48328`). In evaluation success ENDS the episode: "set
`rl_training_mode=False` so episodes end on success"
(`concept_rl_tasks_design.rst:6999`) — success means "true at some
step", not "true at the end".

### 1.2 Metrics: record per step, compute per run

"A metric does two things: **Recording** ... **Computing**"
(`concept_metrics_design.rst:6664-6667`). Data structures:

- `MetricBase`: `name`, `recorder_term_name`, `get_recorder_term_cfg()`
  (a per-step recorder term) and `get_metric_term_cfg()`
  (`isaaclab_arena/metrics/metric_base.py:35756-35774`).
- `MetricTermCfg.compute_metric_func` — "Signature:
  `compute_metric_func(recorded_metric_data: list[np.ndarray], **params) -> Any`"
  (`metrics/metric_term_cfg.py:35841-35844`).
- `MetricData(term_name, term_cfg, recorded_data, metric_value)` with
  `recorded_data` "one array per simulated episode";
  `MetricsDataCollection(num_episodes, metric_data_entries)`
  (`metrics/metric_data.py:35792-35817`). `MetricsManager.compute()`
  reads the arrays back from HDF5 and applies the function
  (`metrics/metrics_manager.py:35994-36012`).

| Metric | Records per step | Computes per run |
|---|---|---|
| `SuccessRateMetric` | `record_pre_reset`: `success_results \|= self._env.termination_manager.get_term("success")[env_ids]`, skipping the first reset (`metrics/success_rate.py:36360-36374`) | `success_rate = np.mean(all_demos_success_flags)` (`:36399`) |
| `ObjectMovedRateMetric` | object linear velocity (`metrics/object_moved.py:36115`) | `object_moved = np.any(magnitude > object_velocity_threshold)`, default `0.5` m/s, then the mean (`:36144-36147, 36160`) |
| `RevoluteJointMovedRateMetric` | openness fraction (`metrics/revolute_joint_moved_rate.py:36250-36252`) | moved if openness ever left `reset ± 0.05` (`:36285-36288, 36302`) |

Cross-run aggregation never averages rates: `aggregate_metrics`
"concatenates the recorded data across runs and recomputes each metric
value from the combined data" (`metrics/aggregate_metrics.py:35686-35688`).
Output is a flat JSON dict per job (`metrics/metrics_logger.py:35899-35913`);
a documented run prints `{'success_rate': 0.75, 'object_moved_rate_subtask_0': 1.0,
'revolute_joint_moved_rate_subtask_1': 1.0, 'subtask_success_rate': [0.75, 0.75], 'num_episodes': 4}`
(`docs/.../sequential_static_manipulation/step_5_evaluation.rst:14114`).
Nothing in the twelve files of `isaaclab_arena/metrics/` computes an
interval, a paired comparison or a rank statistic; `num_episodes` is the
only sample-size field that survives.

### 1.3 Progress tracking: ordered predicate chains, scored, never terminating

"Progress tracking does not replace task success and does not terminate
an episode. A policy can receive partial progress even when the task
ultimately fails" (`concept_progress_tracking_design.rst:6711-6712`).

- `ProgressObjective(name, predicate_groups, score=1.0, logical=ALL, K=None, description=None)`
  (`isaaclab_arena/progress_tracking/progress_objective.py:37361-37366`);
  `score` is the objective's weight, asserted in `[0, 1]` (`:37375`).
  `predicate_groups` is a callable, an ordered list (one chain), or a
  dict of named chains; canonical form `{group: [(func, score), ...]}`
  (`progress_tracking_utils.py:37928-37936`). Unweighted chains get
  `equal = 1.0 / len(value)` (`:38000`), every chain is normalised to
  sum to 1 (`:38023`). Completion is ALL groups, ANY, or CHOOSE ≥ `K`
  (`progress_objective.py:37325-37335`).
- `ProgressObjectiveRunner` keeps per-env `current_predicate_index`,
  `group_score`, `group_complete` (`progress_tracker.py:37501-37508`)
  and evaluates ONLY the predicate each env currently sits at:
  `at_position = (self.current_predicate_index[group_name] == chain_idx) & ~advanced & gating_mask`
  (`:37579`); at most one advance per env per step; score
  `+= advance_mask.float() * float(score_weight)` (`:37602`); emits
  `PredicateEvent(env_idx, step, progress_objective, group, predicate_index, predicate_name, score_delta)`
  (`:37427-37449`).
- Objective score = `torch.topk(stacked, self._num_required_groups(), dim=1).values.mean(dim=1)`,
  "reaches 1.0 exactly when the objective completes" (`:37653-37657`);
  task-level `ProgressState(progress_objectives, overall_score, all_complete)`,
  `overall_score = clamp(weighted_score / total_objective_weight)`
  (`:37472-37483, 37745-37749`).
- Pick-and-place's chain is settle → lift → place: `objects_settled`
  (thresholds `1e-2` m/s, `5e-2` rad/s; "records each object's rest
  pose on first settle", `predicates/object_settling.py:46289-46314`),
  `object_is_above_height(use_settled_state=True)` —
  `result = has_settled & (object_z > (settled_pos[:, 2] + distance))`
  (`predicates/spatial.py:46413`) — then `object_on_destination`
  (`pick_and_place_task.py:48355-48378`).

### 1.4 The per-episode record

Each finished episode is one JSON line. Core fields: `env_id`,
`episode_in_env`, `seed`, `success` (the `"success"` termination flag,
`None` if absent), `episode_length`, `language_instruction`, `timestamp`
(`isaaclab_arena/recording/common_terms.py:38054-38067`); the sampled
variation values (`:38070-38082`); and
`"progress": {"overall_score", "all_complete", "objectives": {...}, "events": [{"step", "objective", "group", "predicate_index", "predicate_name", "score_delta"}]}`
(`recording/progress_terms.py:38264-38299`). The builder always adds
`core`, `variations`, `progress` (`environments/arena_env_builder.py:31946-31950`).

The report layer derives a **funnel** — episode instances that reached
each predicate index (`visualization/report_data.py:92499-92528`) — a
`mean_progress` (`:92490-92492`), and one honesty check:
`outcome_disagrees_with_progress` is true when `success` and
`all_complete` are both present and differ (`:92391-92393`).

### 1.5 Composite and sequential tasks

"`CompositeTaskBase` creates an **order-independent** task ...
`SequentialTaskBase` creates an **ordered** task"
(`concept_composite_tasks_design.rst:6557-6559`).

- Composite success is "ever succeeded, all of them": per-env
  `env._subtask_ever_succeeded` latches True when a subtask's success
  function fires; success is `all(env_successes)`
  (`tasks/composite_task_base.py:47199-47209, 47230`).
- `desired_subtask_success_state: list[bool | None]` adds a final-state
  requirement — each non-None entry must satisfy
  `ever_succeeded and currently_matches` (`:47215-47228`);
  put-away-then-close-door uses `[True, True]`
  (`tasks/sequential_composite_tasks/franka_put_and_close_door_task.py:46653`).
- Sequential adds per-env `env._current_subtask_idx`; only the current
  subtask's success is evaluated and the index advances on success
  (`tasks/sequential_task_base.py:49270-49289`).
- Metrics: subtask metrics renamed `f"{metric.name}_subtask_{subtask_idx}"`,
  duplicate `success_rate` collapsed, and `SubtaskSuccessRateMetric`
  appended, returning a LIST: per subtask `np.any(recorded_metric_data[ep], axis=0)`
  then the mean (`composite_task_base.py:47042-47063, 47347-47361`).
- Progress objectives are namespaced `f"subtask_{i}/{progress_objective.name}"`
  with `parent_subtask_idx=i` (`:47376-47377`); in sequential tasks the
  gating mask is `current_idx_tensor == int(self.progress_objective.parent_subtask_idx)`
  (`progress_tracker.py:37532`), so a later chain cannot advance early.

## 2. What we already have that is equivalent

| Arena | Ours | Where |
|---|---|---|
| Success term over privileged state | `EpisodeProtocol.success(states, sensors) -> bool`; "success reads privileged STATE — no sensor carries the cube's pose"; referee `framepos` sensors on the finger pads | `evaluate/harness.py:45-67`; `tasks/aloha2.py:9-10, 286-310, 349-356` |
| `check_success(mode=ALL)` | hand-written `and`: transfer cube = `lifted and held`; kitting = every part in its slot | `aloha2.py:349-356, 496-507` |
| `SuccessRecorder` + `num_episodes` | `SimScore(name, successes, trials)` — "as counts — never just a rate" | `harness.py:79-90` |
| `desired_subtask_success_state=[True, True]` + `objects_settled` | the tail window: both parts within `_PART_IN_SLOT_XY_M=0.035`, `_PART_IN_SLOT_Z_M=0.045` for the last `_HOLD_STEPS=250` steps | `aloha2.py:102, 420-421, 496-507` |
| per-episode `seed` + `variations` | paired starts by construction: `perturb(trial, home)` cycles the spawn box's corners | `harness.py:49-52`; `aloha2.py:313-320` |
| `aggregate_metrics` (recompute from raw) | counts → `PolicyOutcome` → `clopper_pearson`; `pool_rank_correlations` (Fisher-z, Cochran's Q) | `evaluate/certificate.py:28-35, 151`; `stats/pooling.py:1-20` |
| — (absent in Arena) | the certificate: per-policy Clopper–Pearson, Fisher rank-CI gated on the LOWER bound, exact permutation p, `top_pick_probability` | `certificate.py:50-76, 99-171`; `stats/ranking.py:228-262` |

Missing on our side: any partial-progress signal for a policy episode
(`KittingStats(retries, steps_used_before_hold, truncated)`,
`aloha2.py:538-545`, is the scripted choreographer's self-report, not a
judge); a per-episode record (the harness returns only counts,
`harness.py:135-143`); a "did the object move" metric; per-part success
for kitting; any funnel or success-vs-progress check.

The definitional difference to write down: Arena's success is
**ever-true** (the episode ends when the term fires); ours is **true over
the final 250-step window** — stricter (lifted-then-dropped fails; in
Arena it passes unless `desired_subtask_success_state` says otherwise)
and it keeps every trial the same length, which paired trials need.

## 3. What to ADOPT and what to SKIP

### Adopt

**A1. Milestone chains on the protocol, computed offline over the
recorded rollout.** Arena ticks its tracker live because the env
discards state; our `run_episode` returns the full `(states, sensors)`
(`harness.py:106, 139`), so the same semantics is a pure function
afterwards:

```python
Milestone = tuple[str, Callable[[Any, Any, int], bool]]   # (name, pred(states, sensors, t))
EpisodeProtocol.milestones: tuple[Milestone, ...] = ()     # ordered chain = Arena's single group
MilestoneEvent(index: int, name: str, step: int)           # Arena's PredicateEvent minus env/group
EpisodeOutcome(success: bool, progress: float, events: tuple[MilestoneEvent, ...], steps: int)
```

Rule, from `progress_tracker.py:37579-37622`: walk `t`, evaluate only
the chain's current predicate, advance at most one per step, record the
first step each fires; `progress = reached / len(chain)` (Arena's
equal-weight default, `:38000`). Transfer cube: `cube_moved` →
`cube_lifted` → `at_right_pads` → `at_left_pads`. Kitting: one chain per
part (Arena's dict of groups, `logical=ALL`). ~40 lines, no dependency.

**A2. A per-episode JSONL row beside `SimScore`.** Arena's row
(`common_terms.py:38059-38067`, `progress_terms.py:38273-38298`) maps to
`{policy, trial, success, progress, events, steps, robot_bundle, scene, protocol_hash}`.
`score_policies` keeps returning counts (the certificate's input is
unchanged) and additionally writes rows — the audit trail `recipe@hash`
(docs/30 §3.2) needs, and the input to A3/A4.

**A3. `object_moved` as milestone zero, and the funnel.** The rows of
zeros in docs/31 (T1, T2, T5 all 0/4) hid the one useful fact until a
filmstrip showed it: the cube never moved. Arena's rule
(`object_moved.py:36144`) is one line over our states,
`np.any(speed > threshold)`. With A1, the funnel "reached stage k of n"
per policy (`report_data.py:92519-92527`) separates "did nothing" from
"grasped then dropped" from "handed over then let go" — what the viewer
does by eye today.

**A4. The success-versus-progress disagreement flag**
(`report_data.py:92391-92393`). `success=True` with the chain incomplete
means the referee is looser than the milestones (a referee bug
detector); `success=False` with the chain complete means the hold
window rejected a late drop. One line each in the run log.

**A5. Per-subtask success as a list, for kitting.** Arena's
`subtask_success_rate: [0.75, 0.75]` (`step_5_evaluation.rst:14114`)
becomes `EpisodeOutcome.subtask_success: tuple[bool, ...]` =
`(right_in_slot, left_in_slot)` under the same tail rule, so a bimanual
failure is attributed to an arm. The verdict stays the AND.

### Skip

- **Progress as a ranking statistic for the certificate.** Arena: it
  "does not replace task success" (`:6711`). Our real side is binary
  counts with Clopper–Pearson; a mean-progress real measure has no
  interval in `stats/` and no real-robot referee. Diagnostic only.
- **The live tracker, GPU tensors, `RecorderTerm`/`configclass`/HDF5
  plumbing.** It exists to score thousands of parallel envs without
  keeping state; we keep the trajectory.
- **`SequentialTaskBase`'s state machine and gating.** An ordered
  chain (A1) already says "right picks before left holds".
- **`objects_settled`'s rest-pose recorder.** Episodes start from a
  named keyframe plus deterministic `perturb`; rest height is a
  protocol constant (`_TRANSFERRED_HEIGHT_M = 0.06`, `aloha2.py:103`).
- **Success terminating the episode early.** Fixed-length trials keep
  A's trial 7 and B's trial 7 identical in length as well as start;
  record `steps` (A2) so time-to-success survives.
- **Contact-sensor placement predicates.** Ours read geometry; a
  contact query adds a sensor the real rig does not have.

## 4. Open questions

1. **Are milestone thresholds protocol content?** Arena hard-codes
   `distance=1e-2`, `velocity_threshold=0.5`, `force_threshold=1.0`.
   Since milestones never touch `success`, proposal: a separate hash,
   recorded in the row, so re-tuning a milestone does not invalidate a
   certificate.
2. **Real-side milestones.** Sim milestones read privileged state; on
   the rig the only referee is the operator's label. Paper-2
   sub-question: does sim `progress` rank policies the way real success
   does — a cheaper early signal?
3. **Progress inside the recipe engine.** `mean_progress` could order
   candidates before any policy succeeds (the zero-row problem) — a
   search heuristic, admissible only if it never reaches a customer
   number.
4. **Joining Arena output later.** Its `success_rate` is `np.mean` with
   no interval (`success_rate.py:36399`), but its JSONL rows carry
   `success` per episode, so counts can be recovered and intervalled by
   us. The join point would be the row, not the summary dict.
