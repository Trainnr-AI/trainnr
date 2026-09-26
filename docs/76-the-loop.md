# The loop: a RoboticOps harness for the agentic era

*2026-09-08. The design for what this repo is becoming: an MCP app whose
tools let an agent take a real robot from "just plugged in" to "trained,
certified and deployed", and keep it improving. Written before the build,
because the concepts have to be right before hardware is involved. The
build sequence and its gates live beside this doc; what follows is the
design those phases implement.*

## 0. The question

A user has a robot. Maybe a ROS 2 arm, maybe a Unitree quadruped, maybe
something they built. They want a policy that works on it, and they want
it fast, without waiting on months of real-world data collection. Today
that means assembling a stack by hand: a simulator, a model of their
robot, a task definition, a data pipeline, a training run, an evaluation
they have to design themselves, and a deployment path they write from
scratch. Every step is a place to be quietly wrong, and nothing records
what was actually done.

The answer this repo proposes: **the agent does the assembly, and every
step leaves a record that a stranger can recompute.** Plug in the robot;
the agent measures it, builds the scene, presses the data, trains, judges
with intervals, exports a deployable artifact, and watches for drift.
The human reads a window, or an evaluation, or nothing at all.

## 1. The one contract

The harness is not a pile of tools. It is one rule, applied everywhere:

> **Every stage consumes stamped artifacts and emits stamped artifacts,
> and every tool returns a record naming what it consumed, what it
> produced, and what the next legal move is.**

A *stamp* is `name@hash`: twelve hex characters of SHA-256 over a
directory's file bytes and their relative paths, sorted
(`pipeline/rq_pipeline/bundles/hashing.py`). For things that are data
rather than files — a task specification, a protocol — the same shape is
computed over the fields instead (`content_stamp`). Nothing in this
system is nameable without its hash; `require_stamp` is the single place
that rule is enforced, and six modules used to each spell their own
version of it.

Three consequences fall out, and they are the whole reason for the rule:

- **The agent can always answer "where am I?"** It reads the artifacts
  that exist and their kinds, and the state machine in §3 says what is
  legal next. No hidden workflow state, no session that can be lost.
- **A stranger can recompute any claim.** The evaluation names the
  robot, the task, the policy, the instrument and the protocol by hash;
  re-running with those inputs is a defined operation, not an
  archaeology project.
- **The cloud, when it comes, stores rather than decides.** A registry
  keyed by the same stamps inherits the honesty layer instead of
  replacing it (docs/64 §1).

## 2. Artifact kinds

*Vocabulary (2026-09-09): the app and the tools speak the field's words —
Isaac Sim and Isaac Lab for assets and environments, MuJoCo for the model,
RL and system identification for training and fitting, LeRobot for
datasets, MLOps for versions. Nothing new is introduced unless the field
has no word. Code paths and frozen file names keep their internal kind
names (`certificate`, `batch`); what a user reads says **evaluation**,
**dataset**, **version**, **success criterion**, **scripted policy**,
**system identification**.*

The pipeline already stamps some things as wholes and merely *cites*
stamps in others. That gap is the first thing the harness closes: every
artifact gets a kind, and every kind gets exactly one stamping door, so
nothing downstream ever invents a name.

| kind | what it is | stamped today |
|---|---|---|
| robot | a bundle directory: model, assets, profile, fits | yes |
| actuator | a certified actuator bundle: fit parameters, provenance, checks, uncertainty | yes, content-derived |
| task | a specification, stamped over its fields | yes |
| recording | telemetry from a real robot, with its source and clock | yes (2026-09-09): `robots/ingest` |
| fit | an identification record: estimates, intervals, pinned verdict, anchor | yes |
| scene | a captured environment: the splat the cameras see, the collision proxy the solver touches, the gap between them measured, physics by basis | yes — built 2026-09-22 (docs/78 §3, E1) |
| batch | pressed episodes with a datasheet and per-episode manifests | cites, not stamped |
| dataset | a training dataset with its provenance sidecar | cites, not stamped |
| run | a training run with its manifest | cites, not stamped |
| policy | a checkpoint directory or file | sometimes |
| certificate | an evaluation: trials, confidence interval, funnel, every input version | yes for walk |
| deploy | a manifest making a policy runnable on the robot | no — new |
| drift | fresh telemetry judged against the identified interval | no — new |
| finding | a tracked claim with its commit, argv and instrument | yes |

The rule for detecting a kind is the file that must be at its root — a
dataset has a datasheet, a run has an identity file, an evaluation has its
JSON beside its records. An unrecognised shape is refused by name rather
than guessed at.

### 2.1 Policies, evaluations and findings as artifacts (2026-09-09)

Prakhar: "I want to see data and proper visually appealing data and
sections that are meaningful and delightful to users in policies,
evaluations, findings, deployments, monitoring." Filled from what exists
first (his call): the flagship study's fifteen trained arms and the
findings ledger. `pipeline/rq_pipeline/project/importer.py`:

- `import_experiment(project, arm_dir)` — an rq_mjlab run (`train/`
  with `identity.json`, `model_*.pt`, `verdict/`) becomes three kinds:
  a **run** (the identity, the log when kept), a **policy** (the
  checkpoint plus `policy.json`, schema `trainnr-policy/1`: checkpoint,
  format, iterations, and the run, robot, actuator model, domain
  randomization basis and seed it cites — no `identity.json` inside,
  since that file marks a run to the kind detector), and one
  **evaluation** per verdict file (`certificate.json`, schema
  `trainnr-evaluation/1`: the verdict's successes, trials, exact
  interval, funnel and protocol, citing the policy, robot, environment
  and run; the trial records copied beside it).
- `import_finding(project, id)` — a ledger record into `findings/`.
- Doors: `import_experiment`, `import_finding`, `list_ledger_findings`.
- The run's console log comes along: its own `train.log`, else the
  segment of a study log whose `log_dir:` banner names the run (a study
  launches several runs into one log; a restarted run leaves two
  segments and the longer wins). `rq_pipeline/envs/rsl_rl_log.py` reads
  it once into `training.json` (schema `trainnr-training/1`: trainer,
  iterations, parallel environments, device, wall time, the final and
  best mean reward, and the curve sampled to 400 rows), so the index
  never re-reads a 330,000-line log. Three of the fifteen arms (the
  first replicates of point, narrow and wide) have no log in the repo
  and say so.

What the Studio shows for them: an experiment card is its reward curve
(the picture every RL platform shows for a run) with iterations and the
final reward under it; its drawer holds the training facts and the
sampled curve as a table. A policy card is the robot it drives
(rendered from the project's bundle), subtitled by iterations,
randomization and format; its drawer lists its facts and every
evaluation in the project that judged it. An evaluation card draws the
rate large, the exact interval as a bar on 0–1 with the point on it,
and the funnel as bars; its drawer holds the facts, the funnel and the
trial table. A finding card draws a bar per condition when its outcome
has conditions with successes over trials, else the claim; its drawer
holds the claim, the record's facts (date, commit, simulator build,
protocol, command), the outcome by condition, inputs, artifacts,
sources and caveats. Measured on duck-walk: 15 policies, 15 runs, 297
evaluations, 10 findings, and the loop's "policy trained" and "policy
evaluated" states proved by them.

Two rules of the doors learned by the scripted walk (2026-09-09, a
fresh project, 37 steps): a capture asked right after a move waits
300 ms for the frame to settle, since egui's fades run about 100 ms and
the first capture of the table modal showed it half-drawn; and the
import doors answer a refusal record with the reason (a name already in
the project, a directory with no `identity.json`, an unknown ledger id)
the way the Studio doors do, instead of raising.

Two rules of the grid learned by capture: the drawer opens directly
under the row that holds the selected card (one grid per row), never
below hundreds of cards; and a page opened by name — the agent's door or
the sidebar — starts at the top, while opening an artifact scrolls its
drawer into view in the same frame, so the agent's capture is already
looking at it.

Monitoring fills from A7 (built 2026-09-13, §9.1): a drift check's card,
its drawer with every parameter's fresh interval beside the reference,
its shift bars in the viewer. Deployments fill from A6
(built 2026-09-11, docs/77 §6): the manifest with its ONNX and scene,
and the sim-to-sim gate's verdict beside it.

## 3. The state machine

Each state is *proved by an artifact*, never by a flag. The agent's job
is to move right; the tools refuse moves whose inputs are missing.

```
   telemetry ingested ──▶ robot known ──▶ dynamics identified ──▶ task declared ──▶ data pressed
        │                                        │                │
        │                                        ▼                ▼
        │                                  (acceptance)      policy trained
        │                                                         │
        ▼                                                         ▼
   loop closed ◀── drift watched ◀── deployed ◀── certified ◀─────┘
```

| state | proved by | refusal when it is skipped |
|---|---|---|
| telemetry ingested | a recording: named channels with units and rates, from a robot adapter | optional when the robot entered as a model; the identifier refuses a channel whose unit it cannot place |
| robot known | a robot bundle stamp and a capability census | a task cannot name a robot that is not stamped |
| dynamics identified | fit records with intervals and a pinned verdict, or a certified actuator bundle with an uncertainty section | randomization over an unmeasured span must be *declared* as such; a bundle with no uncertainty refuses interval sampling |
| task declared | a task stamp plus an acceptance verdict | a task the scripted expert cannot pass is rejected with its funnel, before any policy is trained on it |
| data pressed | a batch whose datasheet names the dynamics basis | a dataset mixing two dynamics bases is refused at export |
| policy trained | a run manifest quoting the dataset's stamps | — |
| evaluated | an evaluation: successes over trials, the exact confidence interval, the funnel, every input version | an evaluation cannot be built from records whose trial sets do not pair |
| deployed | a deployment manifest plus a passing sim-to-sim check | a manifest whose policy version does not match the evaluation's is refused |
| loop closed | a drift record, or a clean check | — |

The two nulls this repo has already measured are why the refusals are
there rather than being warnings: a basis effect needs both far truth and
a competent recipe before it shows up at all, and a fit the bench rates
better can train a *worse* gait when the bench does not identify the
parameter the task depends on.

## 4. The tool families

One MCP server, three families, all over the same seams the pipeline
itself uses — never around them.

- **DESCRIBE** (built, 14 tools): bundles, actuators, certified bundles,
  tasks, engines, runs, evaluation records, a batch's datasheet, an
  actuator's friction curves. Read-only windows.
- **ACT** (built, 23 tools plus the 18 project-and-loop doors of §3):
  press demonstrations three ways, multiply, run the whole chain, train
  and certify the walk, onboard a robot, ingest or capture telemetry
  (the three capture doors were written 2026-09-09 and registered
  2026-09-24), identify, create and accept tasks, export and gate a
  deployment, check drift, capture and stage a scene, and the job
  handles (`job_status`, `cancel_job`, `list_jobs`) that make long work
  pollable.
- **STUDIO** (built, 14 tools): launch, quit, open, show, compare, time,
  panels, simulate, the simulator's controls, screenshot, events.
- **CLOUD** (8 tools, off by default): custody by stamp and managed jobs;
  offline is a named state, not an error. Built on the branch
  `cloud-seam-2026-09-08`, NOT merged as of 2026-09-24; this list said
  "built" without saying where until the market read named it.

Counted from the registration list in `pipeline/rq_pipeline/mcp_server.py`
on 2026-09-24: 69 tools at noon, 75 by the evening (ACT gained
`attribute_deployment`, `preflight_deployment`, `stop_deployment`,
`ingest_public_log`; DESCRIBE gained `list_public_logs` and
`list_capture_sources`).

What the harness adds, by stage: a workspace description (what artifacts
exist, by kind, with their lineage), robot ingest and identification
doors, task authoring and acceptance doors, the four chain doors that
were named in docs/64 §3 but never built (dataset export, policy
training, policy evaluation, generic certification), a tool to point the
Studio at an artifact, deployment export and its gate, and the drift
check.

**The result envelope.** Every tool answers with the same shape: a state
word, the record it produced, the artifacts it touched by stamp, and the
moves that are now legal. The cloud family already proves the pattern —
its results carry `ok`, `offline`, `refused` (with the scope and the plan
that carries it) or `error`, so an agent reads a sentence rather than a
traceback.

## 5. Ingest: the robot seam

This is the hole that blocks everything else. Today a robot enters only
as a MuJoCo model directory, and telemetry enters only as this repo's own
wire format or a thin CSV reader. The generic identification path is
shaped like a two-wheel drivetrain, and the robot profile schema refuses
fields it does not know.

The fix is a third registry seam, shaped exactly like the two that
already work — plugin tasks and rented-GPU providers, both a Protocol
plus an entry-point group. A robot adapter answers three questions: what
this robot is (joints, actuators, sensors, rates, limits), what it can
report and accept, and how to read a recording of it into timestamped
signals.

Adapters, in build order, chosen so that hardware is never on the
critical path:

1. **replay** — a recorded fixture. This is the one that makes every
   later phase testable with no robot in the room; the repo already has
   twenty-five real recordings.
2. **wire** — this repo's own rig, wrapping the existing reader.
3. **mcap** — the ROS 2 path. Since the Iron release (2023-05-23),
   `ros2 bag record` writes MCAP by default, and an MCAP file carries the
   message schemas inside it, so an offline reader needs no ROS
   installation at all. Joint position, velocity and effort arrive in one
   standard message; everything else a controller exposes — current,
   temperature — arrives in a second one. Reading a bag is therefore a
   file operation, which is why offline comes before live.
4. **lerobot** — a recorded dataset directory from the LeRobot
   ecosystem, the community standard for arm teleoperation.
5. **unitree** — the vendor SDK's low-level state. Gated on a research
   pass that did not complete; rates and licence terms get verified
   before the adapter is written.

Two honesty requirements ride along. The robot profile becomes a census
plus per-robot extras instead of a fixed rig schema. And onboarding
accepts URDF — the ROS description format — through MuJoCo's model
editing API *with a loud advisory*, because MuJoCo's URDF path takes only
compiler, option and size settings from its extension element: **no
actuators, no sensors and no contact pairs come across**. The wrapper
that adds them is authored work. This repo's own ALOHA 2 bundle is the
honest pattern: include the upstream scene unmodified, add only what is
missing, and say so in the bundle's README.

### 5.1 Data collection: where recordings come from

A recording says how it was collected, because a policy trained on human
demonstrations and one trained on a scripted expert are different claims.
The manifest carries one of six words, set by the adapter and overridable
by the caller:

| collection | what it means | source today |
|---|---|---|
| teleop | a human drove a leader device; the follower's joints and cameras were recorded | a LeRobot dataset (`lerobot` adapter). LeRobot's leader-arm flow is the community standard; the pipeline ingests its recordings rather than rebuilding teleop |
| robot-operation | the robot ran, under an operator, a script or a policy, and its telemetry was captured | this repo's rig over UDP, captured live (`robots/capture`: listen, append, ingest, with the Studio watching); a ROS 2 graph via `ros2 bag record` into MCAP, ingested afterwards |
| mocap | optical or inertial motion capture of a human | a marker or joint CSV, or a BVH skeleton (`mocap` adapter): positions in metres, rotations in radians, the capture rate and marker set in the census. **Not retargeted to any robot** — that mapping, with the sim-replay honesty check the Unitree review named (docs/e2e-research/65 §4), is the retargeting step, designed and not built |
| wearable | IMUs, gloves, suits worn by a human | **designed only.** A wearable produces the same channels an IMU on a robot does (`imu.angular_velocity`, `imu.linear_acceleration`, `imu.orientation`) plus per-segment poses; an adapter for a device is one module and one entry point, written when a device is on hand |
| scripted | a scripted expert pressed demonstrations in simulation | every batch the press writes; a LeRobot dataset this repo exported carries an expert stamp and is marked scripted, not teleop |
| unknown | the adapter could not say | never invented into one of the others |

Live capture is a small state machine with its state on disk (idle,
listening with datagrams so far, ingested with the stamp, failed with the
reason), so the Studio shows it and a tool can poll it, the way jobs
work. There is one live listener today, this repo's rig over UDP; a live
ROS 2 listener would need a ROS installation, which is what the seam
avoids, and a Unitree listener waits on its SDK research.

### 5.2 Showing an artifact as itself

The Studio can index a project and launch the viewer; `present`
(`pipeline/rq_pipeline/project/present.py`, the door docs/64 §3 named `show_in_studio`) points
the viewer at one artifact: a robot as its meshes in a 3D view posed at
its keyframe, a recording's channels as time series on its own clock, an
experiment's curves from its log, a batch's kept frames beside its
datasheet, a dataset's video, a task's scene with its spawn bands drawn as
boxes, an evaluation's funnel as bars beside its interval. Every artifact
is its own Rerun recording named by its stamp, so the viewer's recording
list is the project's artifact list. The Studio asks by writing an intent
file the presenter watches — the same file contract as the index, so a
headless caller uses the function directly.

## 6. Identify

The fitter exists and is good: parameters with confidence intervals, a
pinned-or-not verdict at ten percent of the estimate, records that refuse
to be written without an anchor statement and the recording's hash, and a
cross-run spread check whose verdict on this repo's own drivetrain reads
"trust the spread" for three parameters of five.

What is missing is the door, and the generality. The door becomes an MCP
tool over the existing fitter, taking a recording from any adapter. Two
protocol rules stay enforced because both were learned the hard way:
anchor at least one parameter from outside the data, and verify the
timestamp convention against a known-truth rollout before trusting a fit.

What "identified" means is set by the actuator work: **an interval, not a
point.** The bootstrap refit of a published servo fit produced tight
intervals on the three parameters the bench can see and search-range-wide
intervals on the three it cannot — and the parameters it cannot identify
are named on the bundle rather than hidden. A robot whose dynamics are
guessed is allowed, but the guess is labelled a declared span everywhere
downstream, in the datasheet and on the evaluation.

**The door, built 2026-09-09.** `pipeline/rq_pipeline/robot/methods.py` is
the third registry seam, the adapters' shape: an `IdentificationMethod`
answers whether it can fit this bundle from this recording (and why not,
in a sentence) and runs the fit; `@method(name)`, the built-ins, and the
`rq_pipeline.identification_methods` entry-point group. Two built-ins
as of 2026-09-24: `drivetrain-ratio`, the rig's ratio-form fit from a
`.wire` sweep, and `legged-joints` (`robot/legged_fit.py`, a quadruped's
per-joint armature, damping and Coulomb friction with bootstrap
intervals, docs/77 §8); the first — and because a fit reads the original bytes, ingest now keeps
the raw source file inside the recording artifact (`raw/`). Three MCP
doors: `list_identification_methods`, `identify_system(robot,
recording, method?)` — both by version; writes the fit record into the
bundle's `fits/`, the spread past two records, re-indexes so "system
identified" is proved by the record, and returns the parameters with
intervals and identified-or-not, the anchor verbatim, and the robot's
NEW version (the record lives inside the bundle, so an identified robot
is a different artifact from an unidentified one) — and
`describe_identification(robot)`. Refused by name: an unknown version,
a robot no method can fit from this recording, with every method's
reason. What the door does not do: define the parameters for a new
robot family — that is a method, authored once per family, and the
seam is where it goes.

## 7. Scene and task

Task authoring becomes conversational: declare a specification over an
existing family, get it stamped by content, and submit it to the
acceptance critic, which runs the scripted expert and a do-nothing floor
over paired trials. The expert must pass every trial and the floor none;
a rejection returns the funnel showing where it failed, so the fix is
visible. This is the gate that keeps a policy from being trained on a
task that was never doable.

**Built 2026-09-09 (A4).** Three doors. `describe_task_families` lists
every registered task whose builder takes a spec — kitting and the lift
study today — with each spec field's type and default, so the agent
writes only what it changes. `create_task(task_id, name, overlay)`
builds the family with the overlay replaced into its spec
(`rq_pipeline/tasks/overlay.py`: an unknown field is refused naming the
real ones; a task that composes a fixed scene is refused naming the
families; JSON lists become the tuples the spec keeps), stamps it by
content (`Task.stamp`, so two agents writing the same numbers get the
same version) and writes it into the project as a task reference of kind
`declared` under `name`, spec and all. `accept_task(name)` runs the
critic as a job (`tools/accept-task.py --project … --name …`): the
scripted policy must succeed on every paired trial and the floor policy
on none; the verdict, counts, funnel and reasons land beside the task as
`acceptance.json` (schema `trainnr-acceptance/1`), the tool reindexes,
and the task's card says accepted, rejected or unreviewed with the whole
verdict in its drawer. The readers that rebuild a task — the drawer, the
scene card, the presenter — rebuild the variant, not the family default
(`build_from_reference`).

Two identity rules came with it. A task's version in the index is its
spec's content hash, not a hash of its folder, so a review written
beside it later does not rename it; its name is the folder's. And a cite
names a version: when the name half differs (a certificate cites
`kitting@7d4f…`, the project holds it as `tray-far@7d4f…`) the hash
decides, in the index (`_link_cited_by`) and in the Studio's lookup.

The gate, run on the aloha-kitting project by the doors alone
(2026-09-09): `tray-far` (the tray at y = 0.9, two trials) was rejected
in 5 s — "IK failed: right arm to (0.09, 0.9, 0.14) — the choreography
must not pretend a reach happened", the expert 0/2, the floor 0/2 — and
`kitting-default` was accepted in 10 s, the expert 4/4 with the funnel
full at every milestone, the floor 0/4. Both read back in the Studio.
The scripted experts live in `rq_pipeline/tasks/experts.py`; a family
without one is refused by name, since acceptance is the expert's
verdict.

**Environment capture was designed before it was built, and the reason is
recorded** (built 2026-09-22 onward: docs/78, the scene loop; the paragraph
below is the verdict that shaped it). This repo re-checked the field on 2026-08-25 and amended its
own verdict: Gaussian splats still do not carry contact, every shipped
system keeps a mesh or particle proxy as the physics carrier, and one
2026 result measured 65 to 80 percent geometric degradation when
converting appearance-grade geometry into physics-grade topology. The
recommendation on record is to standardize a scanner's output as a splat
plus a watertight collision proxy, built from permissively licensed
pieces, and to benchmark against the closest existing implementation
before writing one. So the harness reserves the scene kind and the seam,
and refuses to pretend the capture is solved.

## 8. Deploy, and the gate that needs no robot

Written before the deployment stage existed (built 2026-09-11, A6 in §2.1;
the MuJoCo gate 20/20 and Unitree's DDS gate 20/20 by 2026-09-13, docs/33).
The shape it named is the shape that was built, from prior art
(docs/e2e-research/65):
a manifest that travels *with* the policy carrying the joint-order map,
the gains, the action scale and offset, and the ordered observation list
whose order in the file is the concatenation order — with normalization
baked into the exported model so every runtime scale is 1.0.

Ours adds what that prior art lacks: the stamps. The manifest names the
policy, the robot bundle it was trained against, the task and the
evaluation, so a robot can answer "what physics trained this, and what
was it measured at" — a question a policy cannot answer anywhere else
today.

The gate is the part that matters for building without hardware: **the
exported policy, driven only through its manifest, is re-certified
against the simulator and must reproduce the original evaluation's
interval within a stated tolerance.** If the manifest is wrong — a
transposed joint order, a missing scale — the numbers move and the gate
fails. That is a full test of the deployment contract with no robot
involved. It was the last thing built before hardware enters until
2026-09-24, when the gate learned to say why (attribution, docs/77 §9)
and the runtime gained a pre-flight, a ramp-in and a soft stop (docs/77
§10); since the evening of that day the gate's trials are drawn per
trial index, so a gate, its attribution and its pre-flight run the same
commands.

The safety doctrine binds it and does not move: limits are enforced
*below* the policy, and a model output is never wired to a safety
function.

## 9. The loop

Fresh telemetry from a deployed robot, read through the same adapter,
compared against the identified interval. A parameter that has left its
interval produces a drift record naming it and recommending
re-identification. Injecting synthetic drift into a recorded fixture
tests the whole path with no robot.

This is the smallest honest version of the fleet data plane designed in
docs/30 §3.3 — telemetry always, interventions on event, heavy logs
batched — and it is the piece that turns a one-shot pipeline into a loop
that keeps improving.

### 9.1 The design (2026-09-13, before building)

**The question a drift check answers.** "Is the robot I have still the
robot the evaluation was judged on?" The evaluation was judged at the
bundle's identified dynamics; each parameter there carries an interval
from system identification. A drift check re-identifies from a fresh
recording and asks, parameter by parameter, whether the fresh interval
still overlaps the identified one.

**The reference interval is the union of the bundle's PINNED fit
intervals**, per parameter: the lowest lower bound to the highest upper
bound across every record that pinned it (amended 2026-09-13 by review:
a record that never pinned a parameter carries an unbounded interval,
and one of those in the union would make every later check `within`). That is the spread rule the repo already
trusts (`robot/fit_record.cross_run_spread`: when runs disagree beyond
their own intervals, trust the spread, never one run). The rig's own
records show why it must be the union: its right gear differs between
the b and c sweeps by more than either interval, so a check against one
record alone would call a second healthy recording drift. One record is
a legal reference; the record says how many it rests on.

**Three verdicts per parameter, no fourth.** *within*: the fresh
interval overlaps the reference. *left*: the fresh interval is pinned
(the identifier's own criterion, half-width within 10 % of the allowed
range) and does not overlap the reference. *unresolved*: the fresh
interval is not pinned, so this recording cannot say either way. An
anchored parameter (the rig's damping, fixed as the scale reference) is
reported and never judged: it was not measured. A record is *drifted*
when any parameter left; its recommendation names them and says
re-identify, then re-evaluate.

**What it reuses, and what is new.** The recording enters through the
robot seam (A2, any adapter). The fresh fit is the registered
identification method (A3, `robot/methods`) run without writing, so a
drift check never adds a fit record to the bundle — a check is a
question, identification is a decision. New: the comparison, the drift
record (kind `drift`, schema `trainnr-drift/1`, one folder per check
under the project's `monitoring/`, citing the robot, the recording and
the fit records it compared against), the door `check_drift(robot,
recording)`, and the Studio's Monitoring page reading it (card: the
verdict; drawer: every parameter's two intervals; viewer: the shift of
each parameter against its reference width).

**Proved without a robot.** The rig's committed sweep recordings are the
fixture. Synthetic drift is one wheel's encoder ticks scaled in a copy of
the recording — a worn or swapped gear, seen by the encoder — ingested
like any telemetry. The check must name that wheel's gear as left and
the other wheel's as within; the unmodified recording, checked against
the bundle it identified, must come back within on every judged
parameter. Two real sweeps as the reference, so the union rule is what
the test exercises.

**What it does not claim.** A drift check is only as good as the method
behind it: the rig has one; the Go2 has no telemetry and no method, so on
that robot the stage stays empty and the strip says so. The fleet tiers
(always-on telemetry, events, batches) are a service; this is one record
per check, run by an agent when it decides to.

### 9.2 Built (2026-09-13, the Mac)

`rq_pipeline/fleet/drift.py` (the rule, the record, `judge`), the kind's
marker imported from there by `project/kinds.py`, the project's
`monitoring/` folder, the door `check_drift(robot, recording, method,
name)` in `rq_pipeline/mcp_server.py`, the index's summary, the drawer,
the tile and the viewer presentation, and `anchored` on the
identification method's Protocol (`rq_pipeline/robot/methods.py`: the
drivetrain ratio fit anchors its damping) so an anchor is reported and
never judged. Tests in `pipeline/tests/test_drift.py`: the rule on
constructed records, then the whole path on the rig's committed sweeps.

The proof the design asked for, on a fresh project (`projects/rig-drift`,
not tracked): two real sweeps identified (the reference), then a copy of
the first with the LEFT wheel's encoder ticks scaled by 1.4 — the check
named `left_gear_per_damp` as left and the right wheel's gear as within;
then an untouched copy of the first sweep — every judged parameter
within, nothing left, nothing unresolved. The worn check named exactly
one parameter: the left gear's fresh estimate sat 5.2 reference
half-widths above the reference's centre (8.4e-5 against [5.8e-5,
6.6e-5]); the left wheel's friction, the right wheel's gear and friction
all stayed within, and the anchored damping was reported at its fixed
value and not judged. The Studio: the loop strip at Monitoring, the
card's word, the drawer's two intervals per parameter, the viewer's
shift bars.

Two rules learned building it. A check is a question: it runs the
identifier with `write=False` and the bundle's version does not move,
where identification (a decision) writes a record and moves it. A check
is an artifact: a second check of the same recording under the same name
is refused, like a deployment.

## 10. The window, in three modes

The Studio (docs/35) is the human's live view and comes first. But the
harness must be equally usable with no window at all, because a run on a
rented machine or in continuous integration has none.

- **Native**: the app, streamed into by every stage, driven by the
  agent in real time through the control surface (§10.1), with the
  simulator's own page (§10.2) — both built 2026-09-09.
- **Headless** (built 2026-09-13, §10.5): every feed that narrates an
  artifact also writes its stream into that artifact, a tool reads the
  file back with no window, and Show in viewer replays it. The parity
  requirement holds: a headless run reports the same facts a window
  shows — the datasheet, the evaluation, the findings record — and now
  keeps the same picture.
- **Web**: serving a saved recording to a browser. Designed, gated on a
  research pass that did not complete. Nothing else depends on it.

### 10.1 The control surface (2026-09-09)

Prakhar's requirement, verbatim: "we have to make sure everything the
users claude agent must be able to completely control on the studio in
real time." Three decisions, taken the same day: the channel is a
**command log on disk** (not a socket), what the human does **flows back
as an event log**, and the agent **may launch and quit** the window.

The one law holds: the Studio reads files and never talks to the MCP
server. So control is three records under `<project>/.index/`, all
mirrored between `pipeline/rq_pipeline/project/control.py` and
`crates/studio-shell/src/control.rs`:

| record | direction | shape | cadence |
|---|---|---|---|
| `commands/<id>.json` | agent → Studio | `{"schema": "trainnr-command/1", "id", "verb", …args}`; the id is the send time in nanoseconds plus the verb, so a directory listing is the session in order | the Studio reads the directory at 20 Hz and answers each with `<id>.ack.json`: `done`, `refused` (with the reason) or `failed`; an unparseable file is refused, never dropped; the last 200 are kept |
| `studio-state.json` | Studio → agent | `{"schema": "trainnr-studio-state/1", "pid", "heartbeat", "project", "project_name", "section", "selected", "live": {"recording", "timeline", "seconds" or "sequence"}, "presenter_running", "jobs_running"}`; after a project switch the old project's file is a pointer, `{…, "moved_to": "<new root>"}` | on every change, and at least once a second; a heartbeat older than 3 s, or a dead pid, is a dead Studio; a pointer is followed to the live one |
| `events.jsonl` | Studio → agent | one line per human action: `open` (a page, a project), `select` / `deselect` (a card), `show` (the viewer button), `time` (a scrub, reported once the cursor rests 250 ms and only when no command of ours moved it); each carries `t` in epoch nanoseconds and `by: user`, `by: agent` (a door) or `by: studio` (the window's own move) | appended, never rewritten |

The verbs, and the MCP door over each (`pipeline/rq_pipeline/mcp_server.py`):

| verb | door | what it does in the window | refused when |
|---|---|---|---|
| — | `describe_studio` | reads the state file, adds `alive` and the presenter's last status | never; a missing Studio is reported, not raised |
| — | `launch_studio` | starts the built binary (`$TRAINNR_STUDIO`, else `crates/studio-shell/target/release/studio-shell`) on the project and waits for its first heartbeat | one already runs; no binary |
| `quit` | `quit_studio` | closes the window; past 5 s, terminates the pid | never |
| `open` | `open_in_studio` | a page by its rail name; an artifact by version (its page opens with the drawer); a project by root; a `table` of the selected artifact by title (Joints, Actuators, Episodes…) in the exploration modal, an empty string closes it | no such page, artifact, project or table; a table with nothing selected |
| `show` | `show_in_studio` | the presenter streams the artifact as itself; the Live view opens | no such artifact |
| `compare` | `compare_in_studio` | two artifacts side by side in one recording, `a` left and `b` right, each presenter under its own entity root (`a/robot`, `b/robot`) | either is missing |
| `time` | `set_studio_time` | the viewer's own time control: active timeline, cursor in `seconds` or `sequence` (refused if the timeline counts the other way), play or pause, speed, a `start`..`end` selection, follow, step | nothing is streaming |
| `panels` | `set_studio_panels` | expand or toggle the blueprint (left) and selection (right) panels | the time panel, which has no command in this viewer build |
| `screenshot` | `screenshot_studio` | the whole window as a PNG under `.index/screenshots/`, scaled to a stated width (default 1600, never upscaled); with a page or an artifact named, it navigates first, lets the page draw, then captures — the agent reads the file and sees what the human sees | a capture is already in flight; no frame arrives within 3 s (a hidden or minimized window) |
| — | `read_studio_events` | the event log after a time | never |

The tables (2026-09-09, Prakhar: "when a user clicks on the table, or
click on the small expand button on table, a new modal popup opened
with proper table visible for exploration"): a section's table shows
its first eight rows in the card, column names on top, nothing
scrolling, each column sized to its longest value; the table itself,
the expand button beside its title and the "Explore all N rows" footer
open a modal over the page with the whole table — a fixed header, sort
on a column click (ascending, descending, off; numbers sort as numbers,
a range by its low end), a filter box that keeps matching rows, resizable
columns, the full value on hover, closed by ×, Escape or a click outside.
The state file reports the open table; the agent opens one by title.

How the agent sees the window (asked 2026-09-09, "how will the agent see
the window?"): the state file is a description, the screenshot is the
picture, and the detail tables and recordings stay the source of
numbers — a picture is for layout and sanity, never for reading a
value off. Measured: navigate-and-capture round trip 515 ms for a
1600-wide PNG on this machine, 100 ms for the capture alone.

Two things the design refuses. A door never waits on a Studio that
is not alive: every act tool checks the heartbeat first and answers
"launch_studio first" in a word. And the state is not the truth: it is
what the window shows, rebuilt every second; the project's truth stays
in the artifacts and the index (docs/76 §1).

What this does not do yet, said plainly: a time selection on a
sequence timeline is passed through as raw steps; the viewer's own
selection (which entity is highlighted) is neither reported nor
settable; the presenter's layouts are fixed per kind and `compare` is
the only composition an agent can ask for. Each is a verb away, and
each waits for the loop stage that needs it.

### 10.2 The Simulator page (2026-09-09)

Prakhar's call after the research pass (docs/e2e-research/74): a
**SIMULATION** group with one page, **Simulator**, absorbing Live view;
and after the measurements (findings `studio-viewport-pipe` and
`studio-viewport-two-process`), the engine stays the pipeline's own
MuJoCo in a subprocess — now two: physics and render, on a
shared-memory state ring.

**The page** (redesigned the same evening after Prakhar's verdict on the
first cut — "there is no user based thinking in terms of delight and how
easy and simple things are for user to see and interact" — around what
a person does in a simulator, in order of how often: watch; pause,
step, reset, change speed; poke the robot; flip an overlay; look up a
fact). The picture is the page: the MuJoCo viewport fills the strip and
nothing permanent sits beside it. Under it a **transport bar**, the
video-player shape and the shape of Rerun's timeline right below: the
scene picker (a named menu, the empty state's only door), play or
pause, one step, ten steps, reset, the keyframes as a menu, a compact
speed control, the **mode** said plainly as a toggle ("runs itself" /
"you drive"), three chips that never move (sim time, real-time factor,
frames per second), and an *agent · pause* tag for 1.8 s whenever the
agent presses something. Space, → and R do what they do in `simulate`.
In the picture's corner an **overlay toolbar**: contacts, forces,
joints, inertia, transparent, shadows as one-click lit toggles, and a
camera menu with Front, Side, Top, Reset view. An **Inspect** button
opens a drawer over the right of the picture with four tabs: Control
(a row per actuator, grouped by the name's prefix — an arm, a leg),
Joints (a row per hinge or slide joint; free and ball joints have none,
simulate's rule), Physics (timestep, integrator, solver, gravity,
counts, keyframes), Visuals (every MuJoCo visualization and rendering
flag, and the seven group masks, 2026-09-12), and in walk scenes
Commands (the twist for the followed world, bounded by the task's
ranges, on mjlab's joystick override). While the scene runs itself the rows are read-only
bars showing what the policy does; in drive mode they are sliders with
a typed value. Ctrl+drag on the picture still shoves a body.

**The wire** (`tools/studio-render-stream.py`, `crates/studio-shell/src/viewport.rs`):
the same tagged stdin the camera uses, ten new tags — RUN, STEP, RESET,
SPEED, MANUAL, CTRL, QPOS, VIS, RND, VIEW (a named camera view). RUN, STEP, RESET, SPEED, MANUAL,
CTRL and QPOS cross into the physics process through the ring (a
mailbox for commands, seqlocked arrays for the sliders); VIS and RND
stay with the renderer. Back over stdout, beside the frame token, a
status message every 100 ms: sim time, real-time factor, paused,
manual, speed, qpos, ctrl, shadows, render time, the flags, the camera pose (azimuth, elevation, distance, lookat; `simulator.camera` in the state file) — and once,
the model: joints with qpos addresses and ranges, actuators with control
ranges, keyframes, the physics facts, the flag tables in MuJoCo's names.

**Taking control.** A slider, or Step, hands the scene to the human:
the physics process leaves the scene's own loop (the scripted policy,
the parade) and runs simulate's loop — ctrl from the sliders, qpos edits
with a forward pass, Run on or off, Step n while paused. Manual off, or
Reset, returns the scene to its own motion from its start; Reset to a
keyframe stays manual. Speed is a real-time factor the physics paces
to (0.01–100), reported back as the factor achieved.

**The agent's doors.** `simulate_in_studio(task)` starts or stops a
scene; `control_simulator(run, step, reset, keyframe, speed, manual)`;
`set_simulator_input(value, actuator= | joint= | command=)` — an actuator or joint slider, or in a walk scene one twist axis (vx, vy, wz) for the followed world on mjlab's own joystick override, `command="own"` handing back; `set_simulator_view(flag, on | group=, kind=, on | camera= | inspect= | fullscreen=)` — a MuJoCo flag, one group-mask bit (geom, site, joint, tendon, actuator, flex, skin; 0-5), a named camera view (front, side, top, reset), the Inspect drawer's tab (control, joints, physics, visuals, close), or the viewport alone on the page.
Names are the model's own and a wrong one is refused naming the
choices; nothing runs → refused with "simulate_in_studio first". The
state file carries `simulator: {time, rtf, paused, manual, speed,
render_ms}` beside `viewport_task` and `viewport_fps`.

**Are the overlays true?** (Prakhar, 2026-09-09: "is the contact
forces joints section all correct and showing correct info?") The
renderer draws MuJoCo's own visualization (`mjv_updateScene` with the
flag set) on a forward pass over the state the ring carries — qpos,
mocap, ctrl, and since that question qvel and act as well. Measured on
the kitting scene, ten states, random controls: with velocity carried,
the renderer's contact forces equal the physics' at the published pose
to 0.000 % with identical contact counts; without velocity they were
within 0.5 %. (A first comparison read 13 % — against `MjData` after
`mj_step`, whose contact list belongs to the pose BEFORE the integration
while `qpos` is already advanced; that stale list is also what MuJoCo's
native window draws. Ours draws the contacts of the pose on screen.)
The flag indices follow `mjtVisFlag` and `mjtRndFlag` exactly, and the
status now reports every flag's value as rendered, so the toolbar
shows MuJoCo's state rather than a default. Arrow lengths are MuJoCo's
own `visual.map.force` scale — metre-long arrows under a collapsed arm
are the model's, not an error.

**Not built, and said so:** simulate's history scrubber (a state buffer
to rewind through), Reload (re-read the model file), the Watch field,
Align, the profiler and sensor overlays, and loading a scene that is
not a registered task or the duck preview. Each is a tag away; each
waits for the loop stage that needs it.

### 10.3 Many worlds (2026-09-09)

Prakhar: "how would this look with tens of simultaneous simulations?" —
then "ok walk scene." Built on the RL view, the one many-worlds scene
that exists: nine policy-driven worlds in a batched mjlab env. The
shape follows Isaac Lab's own answer (`ViewerCfg.origin_type = env`,
`env_index`) and the flock's lesson (twenty full-mesh robots are a
slideshow on this GL path in every viewer):

- **Physics, one process, N worlds.** `rq_mjlab.walk_view` is the physics
  side of the two-process stream: it builds the env from the
  checkpoint's own identity (the basis string's wording changed on
  2026-09-06, so the gate compares the basis by what it means — the
  same span, or both a point fit — and says so on stderr), creates the
  state ring for a mirror model (one copy of the robot per world,
  `wNN/` prefixed, built by recipe in the stream so the render process
  can build the same one without mjlab), spawns the renderer on the
  Studio's own pipes, and each control step publishes every world's
  qpos plus per-world reward and done. A shove on the picture routes
  back into that world's batched `xfrc_applied`.
- **The picture, one world at a time, by rule.** The bar shows a dot
  per world (green running, red ended, the followed one ringed — click
  one to follow it) and a follow menu: none, worst reward, a failing
  world, cycle (four seconds each). The renderer keeps the camera on
  the followed world's root and closes in to 0.9 m; unfollow returns to
  the rig's distance. On the wire: `TAG_FOLLOW`; in the status: `follow`
  and `worlds: [{reward, done}]`; in the model: `nworld`.
- **The overview, in the Rerun view.** The physics side logs one marker
  per world at its root, coloured by done, and a reward series per
  world — the picture that scales past tens without a thought, with the
  timeline scrubbing every world together.
- **The drawer.** The mirror's joints and actuators are named
  `wNN/...`, so the Control and Joints tabs group by world for free;
  manual control is refused in the RL view (the policy drives every
  world) and says so on stderr.
- **The door.** `control_simulator(follow="worst" | "failing" | "cycle" |
  "none" | "w4")`.

Measured on this Mac (Apple M1 Pro, warp on CPU): nine worlds step at
a real-time factor of 0.27–0.34 (70 ms per 20 ms control step; the GPU
box runs this at rate), the mirror renders in 49–61 ms a frame
(16–21 fps on screen — nine copies of the 431,750-face duck, the
asset-weight problem again), and the first frame arrives about 17 s
after launch (env build, checkpoint load). Not done: a per-world scope
on the sliders, the "worlds per wall second" number, and manual drive
of one world while the policy runs the rest.


### 10.4 The window from first principles (proposed 2026-09-09, not built)

Prakhar, after the data pages filled: "we have to look at everything
from first principles and users current window and view and interaction
and delight and amazing easy access to data in visuals." The pass was
made against the captures of duck-walk (15 runs, 15 policies, 297
evaluations, 10 findings), page by page, asking one question of each:
what does the person at this window need to learn here, and how many
steps does it take today?

What the window does well today: every artifact has a real picture
(the robot it drives, its reward curve, its rate and interval, its
conditions), the loop strip says where the project stands and what is
next, and every number in a drawer traces to a file the agent wrote.

Where it fails the question, ranked by what the user learns:

1. **Evaluations are a comparison, shown as a pile.** The study asks
   "which policy, under which condition" — and the page answers with
   297 look-alike cards in 75 rows. A comparison wants a matrix: rows
   the policies, columns the conditions (`judged at`), each cell the
   rate with its interval, coloured by the rate; sortable by any column;
   a click opens the evaluation. Every page then gets a view switch,
   Cards / Table, the table being the sortable, filterable view the
   modal already draws — the cards stay for browsing pictures, the
   table for finding a number.
2. **Lineage is written but not walkable.** A drawer prints its cites
   (run, robot, policy, environment) as text. Each should be a link that
   opens that artifact, with a back step; and every drawer should show
   both directions — what this came from and what was made from it
   ("Evaluations of this policy" is the first instance; a run's
   policies, a robot's runs, a recording's fits follow the same shape).
3. **The Overview does not answer "what happened, what is best, what is
   the agent doing".** "Latest" is meaningless at 297 evaluations: show
   the best policy per condition and the study's spread instead. "Recent
   activity" says "No jobs yet" while the command log and the event log
   on disk hold everything the agent did — read them back into the
   panel (open, show, simulate, screenshot; who asked, when, the
   outcome).
4. **A page with one artifact makes the user click.** Robots and
   Environments hold one card on an empty page; when a page has one
   artifact its drawer is the page.
5. **Some kinds still have no picture.** The environment card is an
   icon though the presenter already builds the scene; datasets and
   deployments will need theirs when they arrive.
6. **A finding card is a wall of text.** A finding is a headline and a
   chart: title (the record's human title, not its id), the chart, the
   id and date in the footer; the claim belongs in the drawer.
7. **The Simulator's empty state is stale.** It still says "Live view"
   and "Nothing is streaming" and points at a port; it should offer the
   scenes the project can run, in one click.
8. **Nothing is ordered by time.** Prakhar, same day: "things are not
   structured based on time like when it was added or updated." The
   index records no time (`Artifact` has kind, stamp, path, cites,
   summary, preview, detail) and every page sorts by name, so the newest
   evaluation sits wherever the alphabet puts it. Every artifact gets
   `created` and `updated`: the record's own date where it has one (a
   verdict's `judged`, a finding's `date`, a run's log start), else the
   directory's newest file; pages sort newest first by default, group
   the cards by day ("today", "yesterday", then dates), show the time on
   the card and in the drawer, and the Overview's activity reads in the
   same order.
9. **No way to jump or to move.** A command palette (⌘K: any artifact
   by name, any page), arrow keys between cards, Esc closing the drawer,
   and a card-size control the page remembers.

Proposed order: 1–3 and 8 first (they change what the user can learn),
4–7 as one polish pass, 9 after. Each lands with captures through the
agent's own door, the standing rule.

**Built 2026-09-09 (Prakhar: "ok continue"), items 1, 2, 3 and 8:**

- *Time.* The index writes `created` and `updated` on every artifact
  (`index.py`, `_times`): the record's own date where it keeps one (a
  finding's `date`), else the oldest file under it; the newest file
  under it. Never invented — an artifact with no files has none. The
  index also writes `cited_by`, the reverse of `cites`, in the same pass
  (`_link_cited_by`). Every page lists newest first, grouped by the day
  the artifact last changed (Today, Yesterday, then dates; UTC days,
  since the index writes UTC), and a card's footer says how long ago.
  The Studio parses the ISO instants without a calendar crate
  (`model.rs`, `epoch_of`, Hinnant's days-from-civil).
- *Views.* `crates/studio-shell/src/listing.rs`: every page has a
  Cards / Table / Matrix switch (remembered per page for the session,
  and set by the agent: `open_in_studio(view=...)`). The table's columns
  are the name, the summary keys the artifacts share (first six), and
  `updated`; a click on a header sorts (numbers by value, so `19 / 40`
  sorts as 19), a filter box narrows, a click on a row opens the drawer
  under the table. The matrix appears on Evaluations when they were
  judged under two or more named conditions: rows the policies, columns
  the conditions, each cell the newest evaluation of that policy under
  that condition, its rate as text on a one-hue fill whose strength is
  the rate; wide matrices scroll sideways inside their own box. On
  duck-walk that is 15 policies by 27 conditions — the study's question
  on one screen.
- *Lineage.* In a drawer, every cite that names an artifact in the
  project is a link that opens it; "Used by" lists what was made from
  this one, by kind; a back button returns to the page and selection
  the link left (the sidebar clears the trail).
- *Overview.* "Best policy by condition" (one row, with a link that
  opens the matrix when there are more conditions than fit) sits under
  the counts; "Recent activity" now merges the job table with the
  window's event log — who did what, how long ago — and comes before
  "Latest", which is now truly the latest (sorted by `updated`).

**Built 2026-09-09, items 4–7 (the polish pass):** a page with one
artifact opens its drawer on arrival (a click still closes it; the flag
is set only when the page is entered, so the user is never fought);
an environment's card is its scene — the registered task built and
compiled, one offscreen frame from a camera facing the table
(`previews.py`, `_render_task`, sharing `_render_model` with the robot
card); a task that does not build simply has no picture, as duck-walk's
walk task (registered nowhere) shows. A finding with no numbers to draw
shows its headline — the claim's first sentence, large, under the
record's id — and keeps the rest for the drawer. The Simulator's empty
page names the Scene strip and the agent's door instead of a port. And
"Used by" is one wrapped row per kind (a grid cannot size a wrapped
cell: 297 evaluation links overlapped the rows beneath).

**Built 2026-09-09, item 9:** a command palette
(`crates/studio-shell/src/palette.rs`) on ⌘K (Ctrl+K elsewhere) over
every page and every artifact by name — pages first, then artifacts
whose name starts with the query before those that merely hold it,
newest first — Enter opens the first hit; the agent can open it with a
query typed (`open_in_studio(search=...)`) to point the user at
something. Esc closes what is open (the palette, then the table, then
the drawer); ← and → walk the page's artifacts in the order it lists
them. Text boxes keep their keys. A card-size control was not built:
the table view already serves the reader who wants density.

Found on the way, by measuring rather than guessing: the Overview ran
past the window's right edge whenever the window was narrower than the
content cap plus margins. Two causes. The page column kept its cap but
not its margin in a narrow window (now the column gives way, never the
edge); and the pipeline strip's chips are frames in a wrapped row, and
a frame never wraps itself — only a label does — so the eighth chip ran
past the card and widened every block after it (egui grows a
container's `max_rect` to include an overflowing child). A chip is now
measured before placement and the row broken by the distance from the
cursor to the right edge, since `available_width` in a wrapped row is
the whole row. The state file now records the window's size and pixel
ratio, so a capture's pixels read back as layout.

**"Show in viewer" (2026-09-09, Prakhar: "the show in viewer button is
not working").** It worked for robots, recordings, runs, datasets, tasks
and evaluations; for a policy or a finding the presenter had nothing to
show and said so only in `present-status.json`, which the Studio never
read — so the click looked dead. Three fixes: presenters for the missing
kinds (a policy is the robot it drives in 3D, its facts, and every
evaluation of it as bars; a finding is its outcome by condition as bars
and its record as a document; a reinforcement-learning run is every
curve of its training record on the iteration timeline); the Studio now
reads the presenter's answer and, when it is a failure for the artifact
last asked for, says so in a banner over the page with the reason,
dismissable; and the `show_in_studio` door waits for that answer
(`control.wait_presented`) and returns shown, failed with the reason, or
pending after 20 s, instead of "done" the moment the Studio took the
request. A deployment presents as the trained scene, the gate's error
ratio per trial, and a reading (A6, 2026-09-11); a drift check as each
parameter's shift against its reference and a reading (A7, 2026-09-13).
Only a fit record, which rides inside its bundle, has no presentation of
its own (tested by name in `tests/test_present.py`).

*2026-09-27, after the end-to-end review:* (1) a Studio that switches
project leaves a pointer in the old project's `studio-state.json`
(`moved_to`, the new root; never a live heartbeat), and `control.state`
follows it: a door called under the old project sees `alive: true`,
`elsewhere: true`, `asked` (the project it was called under) and
`project` (where the window is). Page, time, screenshot, focus and quit
commands are routed there; `show`, `compare`, `simulate` and an `open` of
an artifact are refused with the remedy (`open_in_studio(project=…)`),
and `launch_studio` under the old project moves the window instead of
refusing on the port. (2) The presenter claims `present.json` by renaming
it to `present.json.busy` before it streams, so a show asked for during
a long presentation is a new file, served next, not deleted with the one
before; it writes `{"presenting": stamp}` when it takes a request up, and
`show_in_studio` answers `presenting` (taken up, still streaming at the
20 s timeout; `describe_studio` reports `presenter.shown` when it lands,
with `presenter.age_s`) as distinct from `pending` (nobody answered).
(3) A show or compare by a door is logged `by: agent`; `by: studio` marks
the window's own moves (the turn to Live when a recording arrives).
(4) `set_studio_time` refuses a timeline the recording has not got, naming
the ones it has; `focus_studio_recording` answers `not shown` when the
live recording after the command is not the one asked for. (5) The
recording the presenter lands is brought to the front (`ActivateApp`),
so a second show is seen, not streamed behind the first. (6) Application
ids are folded to rerun's entry-name alphabet (`viz.entry_name`: a
stamp's `@` becomes `-`), which ends the "requires migration" toast.

Tried and dropped the same day: a `window` verb to resize the Studio so
a capture could show a whole page. On macOS a programmatic resize left
the render surface at the old size, so every later capture came out
clipped; not worth a broken door for a convenience.

### 10.5 Headless (designed 2026-09-13, before building)

**What "headless" means here.** Not a second window and not a web page:
a run with no window at all — the box overnight, a rented pod, CI —
must leave behind the same two things a windowed run has. The facts:
every artifact's JSON, which it already leaves, because every stage
writes records before it paints anything. And the picture: the stream
every feed sends into the Studio, which today exists only while the
process runs and a viewer listens. A8 keeps the picture.

**The rule: every feed that narrates an artifact also writes its stream
to a file inside that artifact.** Rerun's file sink beside its viewer
sink, set together, because saving alone replaces the viewer connection
and the window goes dark while the file fills (measured 2026-08-28, the
training watcher). One seam opens every stream (`rq_pipeline.viz`), so
the rule is one function, not a habit. The file is `.viewer/<name>.rrd`
under the artifact's folder: hidden, because the artifact's version is a
hash over its visible files and the index walks only those — a picture
never moves a version, and a reindex never reads a picture. A training
run leaves `runs/<run>/.viewer/train.rrd`; an evaluation its
`.viewer/verdict-<checkpoint>.rrd` in the run it judged; a data batch
`.viewer/press.rrd`; a deployment `.viewer/gate-<runtime>.rrd` per gate;
a reward preview `.viewer/preview-<controller>.rrd` in its task. A
session that narrates no artifact — the simulator, a checkpoint played,
a presentation — writes nothing; there is nothing to leave it in. One
knob, `TRAINNR_VIEWER_FILE=0`, turns the file off where disk matters.

**Reading it back with no window.** Two doors into the file. Rerun's own
command line ships with the SDK and answers what the file holds: every
entity path, every timeline, every component, chunk counts and size —
enough for an agent to know the picture exists and what is in it. The
values themselves need Rerun's local catalog, which needs DataFusion
(98 MB): a separate extra, `viz-query`, so the core stays light; with
it the door reads the columns — rows per clock, and for every scalar
series on each clock it was logged on: its count, its width (components
per row), minimum and maximum over every component, and the last row. Without it the door
says so by name and gives the inventory. The door is
`describe_viewer_recording(artifact)`: the files inside the artifact,
each described.

**The window follows the file.** "Show in viewer" on an artifact that
carries viewer files sends them into the Studio first — each lands
under its own recording id with the layout it was saved with, exactly
as the live stream looked — then the derived presentation. So a run
that happened on the box opens on the Mac as it ran.

**Parity, stated as a test.** A gate run with the Studio quit leaves its
file; the door reads from that file the same number of ticks the gate's
record says it stepped, and the same trials the record lists; the Studio
launched afterwards shows the gate's picture from the file. That is the
gate for A8: a headless run reports the same facts a window shows, and
keeps the same picture.

**What this does not do.** No web viewer (§10, gated on research). No
service, no upload, no retention rule: a file per run, where the run
lives. The cloud feed that follows a remote log stays as it is; the
remote's own feeds now leave their files beside the remote's artifacts,
which is what a pull brings home.

**Built (2026-09-13, the Mac).** The seam: `rq_pipeline/viz.py`
(`open_stream`, `sinks`, `viewer_file`, `viewer_files`,
`studio_listening`, the `TRAINNR_VIEWER_FILE` knob). Through it: the
mjlab recorder (a training run's `.viewer/train.rrd`, the reward
preview's `.viewer/preview-<controller>.rrd`), the walk verdict
(`.viewer/verdict-<checkpoint>.rrd` in the run), the data-generation
feed (`.viewer/press.rrd`), the gate's mirror
(`.viewer/gate-<runtime>.rrd`), and the training watcher's own file
flag. The reader: `rq_pipeline/project/viewer.py` — Rerun's command
line for the inventory, Rerun's local catalog (the `viz-query` extra,
DataFusion, 98 MB) for the values. The door `describe_viewer_recording`
and the replay in `project/present.py`. Tests in
`pipeline/tests/test_viewer_stream.py`.

The parity test, run: the Studio quit, the gate door on a laptop
deployment — the job done in 6 s, one line on its stderr saying no
Studio listens and the stream is saved only, a 12.3 MB file inside the
deployment; the door read it back in 0.8 s at 0.2 GB: five scalar
series of 2000 rows on the gate's own clock, the commanded forward speed
spanning −0.594 to 0.208 with −0.594 last — the gate record's four
commands were 0.043, 0.208, −0.058 and −0.594, in that order. The Studio
launched afterwards: Show in viewer on the deployment replayed the file
under its own recording id, the gate's picture as it ran.

Two things the parity test found, both in Rerun 0.36.2 and both now
handled in the seam rather than worked around per feed. First, **a feed
toward a viewer that never answers does not end**: the SDK's server sink
keeps every chunk in a bounded queue; once it is full, the flush the SDK
runs at exit waits for an acknowledgement that never comes, and the
process sits in a channel receive forever (the gate under the job
runner: verdict printed, process alive for ten minutes; reproduced with
one chunk per row, five thousand rows, no viewer). A small queue times
out in seven seconds and exits; a full one never does. So the seam asks
one TCP question before choosing sinks: no Studio and a file to save
to means the file alone. Second, **the catalog reads every column
unless told otherwise**: the first reader materialized a gate file's
meshes per tick across every timeline and was killed at the machine's
memory on a 12 MB file; the reader now selects the scalar entities and
one clock. Sixty seconds of a gate is 12 MB because the mirror carries
the scene's meshes; the `sim` clock has 2000 rows for four trials of a
thousand steps, the mirror's own sampling.

## 11. What this refuses to claim

- Simulation is not reality. An evaluation is sim-only at a named fit on
  a named instrument, and says so.
- Success is the referee's definition, not the buyer's operational one.
- The dynamics basis is stated verbatim: an identified interval and a
  caller-declared span are different claims and are never printed the
  same way.
- Rigid objects on characterized robots. Deformables and sub-millimetre
  insertion are refused loudly rather than attempted quietly.
- The instrument is part of the result: the same engine at a different
  version, or on a different processor architecture, is a different
  instrument and the stamp says so.
