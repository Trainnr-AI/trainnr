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
| scene | a captured or composed environment | no — reserved, see §7 |
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
evaluation, so a robot can answer "what physics trained this, and what
was it measured at" — a question a policy cannot answer anywhere else
today.

The gate is the part that matters for building without hardware: **the
exported policy, driven only through its manifest, is re-certified
against the simulator and must reproduce the original evaluation's
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

- **Native**: the app, streamed into by every stage, driven by the
  agent in real time through the control surface (§10.1), with the
  simulator's own page (§10.2) — both built 2026-09-09.
- **Headless**: every feed also writes a recording file beside its live
  stream, so a windowless run leaves something to open later, and a tool
  reads that file back. The parity requirement: a headless run reports
  the same facts a window shows — the datasheet, the evaluation, the
  findings record.
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
| `studio-state.json` | Studio → agent | `{"schema": "trainnr-studio-state/1", "pid", "heartbeat", "project", "project_name", "section", "selected", "live": {"recording", "timeline", "seconds" or "sequence"}, "presenter_running", "jobs_running"}` | on every change, and at least once a second; a heartbeat older than 3 s, or a dead pid, is a dead Studio |
| `events.jsonl` | Studio → agent | one line per human action: `open` (a page, a project), `select` / `deselect` (a card), `show` (the viewer button), `time` (a scrub, reported once the cursor rests 250 ms and only when no command of ours moved it); each carries `t` in epoch nanoseconds and `by: user` or `by: agent` | appended, never rewritten |

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
opens a drawer over the right of the picture with three tabs: Control
(a row per actuator, grouped by the name's prefix — an arm, a leg),
Joints (a row per hinge or slide joint; free and ball joints have none,
simulate's rule), Physics (timestep, integrator, solver, gravity,
counts, keyframes). While the scene runs itself the rows are read-only
bars showing what the policy does; in drive mode they are sliders with
a typed value. Ctrl+drag on the picture still shoves a body.

**The wire** (`tools/studio-render-stream.py`, `crates/studio-shell/src/viewport.rs`):
the same tagged stdin the camera uses, ten new tags — RUN, STEP, RESET,
SPEED, MANUAL, CTRL, QPOS, VIS, RND, VIEW (a named camera view). RUN, STEP, RESET, SPEED, MANUAL,
CTRL and QPOS cross into the physics process through the ring (a
mailbox for commands, seqlocked arrays for the sliders); VIS and RND
stay with the renderer. Back over stdout, beside the frame token, a
status message every 100 ms: sim time, real-time factor, paused,
manual, speed, qpos, ctrl, shadows, render time, the flags — and once,
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
`set_simulator_input(value, actuator= | joint=)`; `set_simulator_view(flag, on | camera= | inspect=)` — a MuJoCo flag, a named camera view (front, side, top, reset), or the Inspect drawer's tab (control, joints, physics, close).
Names are the model's own and a wrong one is refused naming the
choices; nothing runs → refused with "simulate_in_studio first". The
state file carries `simulator: {time, rtf, paused, manual, speed,
render_ms}` beside `viewport_task` and `viewport_fps`.

**Not built, and said so:** simulate's history scrubber (a state buffer
to rewind through), Reload (re-read the model file), the Watch field,
Align, the profiler and sensor overlays, and loading a scene that is
not a registered task or the duck preview. Each is a tag away; each
waits for the loop stage that needs it.

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
