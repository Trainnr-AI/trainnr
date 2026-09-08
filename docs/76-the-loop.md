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
The human reads a window, or a certificate, or nothing at all.

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
- **A stranger can recompute any claim.** The certificate names the
  robot, the task, the policy, the instrument and the protocol by hash;
  re-running with those inputs is a defined operation, not an
  archaeology project.
- **The cloud, when it comes, stores rather than decides.** A registry
  keyed by the same stamps inherits the honesty layer instead of
  replacing it (docs/64 §1).

## 2. Artifact kinds

The pipeline already stamps some things as wholes and merely *cites*
stamps in others. That gap is the first thing the harness closes: every
artifact gets a kind, and every kind gets exactly one stamping door, so
nothing downstream ever invents a name.

| kind | what it is | stamped today |
|---|---|---|
| robot | a bundle directory: model, assets, profile, fits | yes |
| actuator | a certified actuator bundle: fit parameters, provenance, checks, uncertainty | yes, content-derived |
| task | a specification, stamped over its fields | yes |
| recording | telemetry from a real robot, with its source and clock | no — new |
| fit | an identification record: estimates, intervals, pinned verdict, anchor | yes |
| scene | a captured or composed environment | no — reserved, see §7 |
| batch | pressed episodes with a datasheet and per-episode manifests | cites, not stamped |
| dataset | a training dataset with its provenance sidecar | cites, not stamped |
| run | a training run with its manifest | cites, not stamped |
| policy | a checkpoint directory or file | sometimes |
| certificate | trials, interval, funnel, every input stamp | yes for walk |
| deploy | a manifest making a policy runnable on the robot | no — new |
| drift | fresh telemetry judged against the identified interval | no — new |
| finding | a tracked claim with its commit, argv and instrument | yes |

The rule for detecting a kind is the file that must be at its root — a
batch has a datasheet, a run has an identity file, a certificate has its
JSON beside its records. An unrecognised shape is refused by name rather
than guessed at.

## 3. The state machine

Each state is *proved by an artifact*, never by a flag. The agent's job
is to move right; the tools refuse moves whose inputs are missing.

```
   robot known ──▶ dynamics identified ──▶ task declared ──▶ data pressed
        │                                        │                │
        │                                        ▼                ▼
        │                                  (acceptance)      policy trained
        │                                                         │
        ▼                                                         ▼
   loop closed ◀── drift watched ◀── deployed ◀── certified ◀─────┘
```

| state | proved by | refusal when it is skipped |
|---|---|---|
| robot known | a robot bundle stamp and a capability census | a task cannot name a robot that is not stamped |
| dynamics identified | fit records with intervals and a pinned verdict, or a certified actuator bundle with an uncertainty section | randomization over an unmeasured span must be *declared* as such; a bundle with no uncertainty refuses interval sampling |
| task declared | a task stamp plus an acceptance verdict | a task the scripted expert cannot pass is rejected with its funnel, before any policy is trained on it |
| data pressed | a batch whose datasheet names the dynamics basis | a dataset mixing two dynamics bases is refused at export |
| policy trained | a run manifest quoting the dataset's stamps | — |
| certified | a certificate: successes over trials, the exact interval, the funnel, every input stamp | a certificate cannot be built from records whose trial sets do not pair |
| deployed | a deploy manifest plus a passing simulation-to-simulation gate | a manifest whose policy stamp does not match the certificate's is refused |
| loop closed | a drift record, or a clean check | — |

The two nulls this repo has already measured are why the refusals are
there rather than being warnings: a basis effect needs both far truth and
a competent recipe before it shows up at all, and a fit the bench rates
better can train a *worse* gait when the bench does not identify the
parameter the task depends on.

## 4. The tool families

One MCP server, three families, all over the same seams the pipeline
itself uses — never around them.

- **DESCRIBE** (built, 15 tools): bundles, actuators, certified bundles,
  tasks, engines, runs, evaluation records, a batch's datasheet, an
  actuator's friction curves. Read-only windows.
- **ACT** (built, 12 tools): press demonstrations three ways, multiply,
  run the whole chain, train and certify the walk, onboard a robot, open
  the Studio, and the job handles (`job_status`, `cancel_job`,
  `list_jobs`) that make long work pollable.
- **CLOUD** (built, 8 tools, off by default): custody by stamp and
  managed jobs; offline is a named state, not an error.

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
downstream, in the datasheet and on the certificate.

## 7. Scene and task

Task authoring becomes conversational: declare a specification over an
existing family, get it stamped by content, and submit it to the
acceptance critic, which runs the scripted expert and a do-nothing floor
over paired trials. The expert must pass every trial and the floor none;
a rejection returns the funnel showing where it failed, so the fix is
visible. This is the gate that keeps a policy from being trained on a
task that was never doable.

**Environment capture stays designed, not built, and the reason is
recorded.** This repo re-checked the field on 2026-08-25 and amended its
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

Nothing in this repo currently makes a policy runnable on a robot. The
shape to build is well established by prior art (docs/e2e-research/65):
a manifest that travels *with* the policy carrying the joint-order map,
the gains, the action scale and offset, and the ordered observation list
whose order in the file is the concatenation order — with normalization
baked into the exported model so every runtime scale is 1.0.

Ours adds what that prior art lacks: the stamps. The manifest names the
policy, the robot bundle it was trained against, the task and the
certificate, so a robot can answer "what physics trained this, and what
was it measured at" — a question a policy cannot answer anywhere else
today.

The gate is the part that matters for building without hardware: **the
exported policy, driven only through its manifest, is re-certified
against the simulator and must reproduce the original certificate's
interval within a stated tolerance.** If the manifest is wrong — a
transposed joint order, a missing scale — the numbers move and the gate
fails. That is a full test of the deployment contract with no robot
involved, and it is the last thing built before hardware enters.

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

## 10. The window, in three modes

The Studio (docs/35) is the human's live view and comes first. But the
harness must be equally usable with no window at all, because a run on a
rented machine or in continuous integration has none.

- **Native**: the app, streamed into by every stage. Missing piece: a
  tool that points it at a named artifact rather than merely launching
  it.
- **Headless**: every feed also writes a recording file beside its live
  stream, so a windowless run leaves something to open later, and a tool
  reads that file back. The parity requirement: a headless run reports
  the same facts a window shows — the datasheet, the certificate, the
  findings record.
- **Web**: serving a saved recording to a browser. Designed, gated on a
  research pass that did not complete. Nothing else depends on it.

## 11. What this refuses to claim

- Simulation is not reality. A certificate is sim-only at a named fit on
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
