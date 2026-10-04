# Glossary

The words this repository uses with a fixed meaning. Each entry names the
document that defines it.

## Artifacts and identity

| Word | Meaning | Defined in |
|---|---|---|
| bundle | A robot or actuator as a directory with its model, its assets and a record of where it came from; the unit everything else cites | [76 §1](76-the-loop.md) |
| `name@hash`, stamp | An artifact's identity: its name and the hash of its content; nothing is nameable without it | [76 §1](76-the-loop.md) |
| record | The JSON file an evaluation, a gate or a drift check leaves behind; every number in the paper resolves to one | [findings/](findings/README.md) |
| datasheet | The provenance sheet beside a generated dataset: what produced it, under which draws, kept by which criterion | [e2e-research/60](e2e-research/60-the-data-press.md) |
| manifest | The deployment's description read from the built environment: joint and actuator orders, gains, home pose, action scale, observations, control rate | [77 §6](77-the-unitree-loop.md) |

## Measuring

| Word | Meaning | Defined in |
|---|---|---|
| identification, fit | Fitting a robot's dynamics from its own telemetry; the fit record carries each parameter with a confidence interval | [28](28-quickstart-identify.md) |
| fit record | The file an identification writes beside the robot (`robots/<name>/fits/`): each parameter's estimate, interval and verdict, the recording it came from, and the method | [28](28-quickstart-identify.md) |
| CP95 | The exact 95 % confidence interval on a success rate (Clopper-Pearson), printed as `CP95 [low, high]`; gates read the low end | [32 §3](32-evaluation-layer.md) |
| survived, tracked | A walk episode survived when the robot did not fall; it tracked when it also followed the commanded velocity within the task's error bound. Success is tracked | [77 §5](77-the-unitree-loop.md) |
| learnability check | What `check_task` runs on a walk task: the trainer for two iterations on two environments, which proves the task builds and trains before hours are spent on it | [77 §3](77-the-unitree-loop.md) |
| g3 | The full walk-training recipe in trainnr-mjlab (`train_walk(agent="g3")`), hours on a GPU; `smoke` is the minutes-long check | [trainnr-mjlab](../trainnr-mjlab/README.md) |
| pinned / NOT PINNED | A fitted parameter whose interval is narrow enough to use (half-width within a tenth of its allowed range), or not | [26](26-sts3215-synthetic-identifiability.md) |
| basis | Where a randomization range comes from: `identified` (the fit's interval), `declared` (a stated range around nominal constants), `pinned` (a point) | [paper §4](paper/manuscript.md) |
| world | The simulator a policy is judged in: the `identified` (fit) world or the `declared` world | [77 §8](77-the-unitree-loop.md) |
| certificate, evaluation | Paired, seed-matched trials with an exact confidence interval and a milestone funnel; the verdict gates on the lower bound | [32 §3](32-evaluation-layer.md) |
| funnel, milestones | The ordered stages an episode passes before success (reached, grasped, lifted); each counted | [32 §3](32-evaluation-layer.md) |
| referee, success criterion | The task's judge of an episode, run on simulator state, never on pixels | [76 §7](76-the-loop.md) |
| cliff | The demonstrator's edge: the hardest truth condition the scripted expert still passes; studies are run at it | [paper §5](paper/manuscript.md) |
| truth | The test condition a policy is judged under, as opposed to the range it trained on | [paper §5](paper/manuscript.md) |
| SENSITIVE / INSENSITIVE / UNRESOLVED | A parameter's verdict in a sensitivity or drift check: it matters, it does not, or the data cannot say | [77 §8](77-the-unitree-loop.md) |

## Deploying

| Word | Meaning | Defined in |
|---|---|---|
| gate | The sim-to-sim check: the exported policy driven through its manifest alone by a registered runtime, judged the evaluation's way, passed within a tolerance of the cited certificate | [77 §6](77-the-unitree-loop.md) |
| runtime | What drives the exported policy at the gate: plain MuJoCo, or Unitree's own simulator over DDS | [77 §7](77-the-unitree-loop.md) |
| attribution | When a gate fails, which difference between the two runtimes caused it, latency first | [77 §9](77-the-unitree-loop.md) |
| pre-flight | The checks before the first tick on a robot: widths, joint order, gains, a dry rollout against ranges, compute per tick, the ramp-in and the stops measured | [77 §10](77-the-unitree-loop.md) |
| drift | A fresh recording identified against the robot's fitted intervals, naming what left | [76 §8](76-the-loop.md) |
| door, tool | An MCP tool of the server; "door" is the older word in the record | [76 §6](76-the-loop.md) |

## Runs and campaigns

| Label | Meaning | Defined in |
|---|---|---|
| walk-c1 | The microduck walk campaign: fifteen policies across randomization spans, judged at the fit | [e2e-research/63](e2e-research/63-the-flagship.md) |
| c1, c2, c3-fit, c4-fit-lag | The Go2 training recipes: declared constants (c1, c2), trained in the fit world (c3-fit), with latency randomization (c4-fit-lag) | [77 §8](77-the-unitree-loop.md) |
| point, narrow, wide, identified#n, point-refit | The lift and walk study arms: a point estimate, ±10 %, ±30 %, a bootstrap set, the refit point | [artifacts/](artifacts/README.md) |
| T0 to T6 | The ALOHA 2 stage ladder of the first end-to-end run | [31 §3](31-aloha2-e2e.md) |
| E0 to E5 | The scene loop's experiments | [78 §4](78-the-scene-loop.md) |
| A0 to A8 | The loop's build order: project layer, artifact kinds, robot adapters, identification methods, tasks, generic train and eval, deployment, drift, headless control | [76 §2](76-the-loop.md) |
