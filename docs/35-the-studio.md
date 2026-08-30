# The Studio: a GPU-first, agentic, sim2real robot platform

*2026-08-30. Synthesis of five research passes this week
(docs/e2e-research/47-54, plus the pre-existing 23/34 corpus on USD
and Gaussian splatting) into one architecture and phased plan. Every
claim below traces to a dated, sourced finding already on record;
this doc composes them, it does not re-derive them. The brief: a
GPU-first, agentic-coding, sim2real robot training platform —
bring your robot, bring your environment, simulate, generate data,
train, evaluate, deploy — built as one native application, not a
loose collection of tools someone runs by hand.*

## 0. What "the Studio" actually is

A single native desktop application. Not Zed with a panel bolted on,
not Rerun with an editor bolted on — a new app, built on **GPUI**
(Zed's own UI framework, deliberately Apache-2.0 and reusable, 136
real third-party apps already prove it), because embedding one
finished GPU application inside another turned out to be unsupported
anywhere in the Rust windowing stack, in *either* direction. GPUI
gives the app native GPU acceleration on every platform for free
(Metal on macOS, Vulkan on Linux, DirectX 12 on Windows) — it is
GPU-first at the UI layer by construction, before a single robotics
decision is made.

Inside that shell: an agent panel (ACP-native — Claude Code already
has an official adapter, zero server code needed), a 3D view fed by
MuJoCo's offscreen renderer (a texture upload, the same pattern GPUI
already ships for video frames — no new rendering engine, no
unproven dependency), telemetry/plot panels, and the whole
sim-generate-train-evaluate-deploy pipeline this repo already builds,
now reachable from inside the app instead of a terminal.

## 1. The agentic interface — ACP + MCP, not a Zed fork

**What**: the app's coding/reasoning surface speaks two open,
editor-agnostic protocols rather than embedding anyone's editor.

- **ACP (Agent Client Protocol)** — JSON-RPC, modeled on LSP,
  genuinely multi-editor (JetBrains, Neovim, Emacs, ~40 agents
  besides Zed's own). Claude Code already ships an **official**
  adapter (`@zed-industries/claude-code-acp`) — the Studio's agent
  panel talks ACP and gets Claude Code (or any ACP agent) for free.
- **MCP (Model Context Protocol)** — the Studio ships **a robotiq
  MCP server**: the bundle registry, task registry, engine registry,
  evaluation records, and run status as tools/resources an agent can
  query and act on. This is not a Zed-specific integration — it is
  the actual product surface, usable from Claude Code today, from
  the Studio's own panel tomorrow, from anyone else's MCP client
  besides. Precedent this pattern already leans on: Runpod ships
  exactly this shape (a hosted, OAuth MCP server as a Claude Code
  plugin) for their own cloud-GPU product (docs/34 §4) — this is a
  proven, not speculative, integration pattern in this exact space.

**Already have**: nothing to build for ACP (Claude Code's adapter
exists); the MCP server is new but small — it's a thin read/query
layer over registries (`tasks/registry.py`, `physics/registry.py`,
the bundle store, `evaluate/records.py`) that already exist and are
already typed.

**New work**: the MCP server itself (§9 has it first in the
roadmap — it's useful standalone, before any native app exists).

**Built (2026-08-30)**: all of the above is now running in
`crates/studio-shell` — the ACP session against Claude Code's adapter
(the package name moved to `@agentclientprotocol/claude-agent-acp`;
the `@zed-industries` name in the paragraph above is its dead earlier
org), the MCP server handed to it at session creation, and a
first-class panel on top: tool calls as live cards keyed by
`ToolCallId` (spinner → ✓/✗ mutating in place, expandable to output
and real ± diffs), agent replies as rendered markdown, a Stop button
wired to ACP `session/cancel`, prompt queueing mid-turn, and the
seven pipeline specialists (`.claude/agents/`) as stage chips that
route prompts via Claude Code's own subagent dispatch. Every
specialist's definition carries the Studio streaming contract: render
into the embedded viewer on :9876, `flush(timeout_sec=10.0)` for
one-shots, one recording per clock, ~10 Hz narration — so agents'
evidence lands in the running window, not in terminal scrollback.

## 2. The native app shell — GPUI, not a fork of Zed

**What**: Zed's *editor* (the crates worth reusing for text editing)
is GPL-3.0-or-later, unpublished, and carries no external-use
contract — not a viable base to fork the way Cursor forked VS Code
(MIT, which is why Cursor's model works at all). GPUI is the one
deliberately-carved-out, Apache-2.0, genuinely reusable layer. The
Studio is a new GPUI application: panels for the agent, the 3D
view, telemetry/plots, the bundle/task browser, and run management.

**Already have**: nothing — this is genuinely new, and it's the one
piece of real native-application engineering in this whole plan that
isn't "wire an existing pipeline into a UI." Budget for it
accordingly; it's the long pole.

**GPU-first angle**: automatic. GPUI's whole reason to exist is
GPU-native rendering; there is no CPU-first version of this decision
to make.

## 3. Physics and rendering — MuJoCo, in two modes, one model

**What**: the same compiled MJCF model serves two completely
different jobs, and the Studio's 3D panel is agnostic to which one
is live:

- **Sim mode** — `mj_step`, full dynamics, the contact solver
  engaged. This is training, evaluation, and data generation. The
  existing dual-instrument doctrine holds unchanged: CPU MuJoCo is
  the metrology instrument (identification, certificates — pinned
  to `mujoco~=3.11.0`, docs/47-52), MJX-Warp is the throughput
  instrument for anything batched.
- **Real mode** — a *kinematic twin*. Real sensor readings go
  straight into `qpos`/`qvel`, then `mj_forward` alone (never
  `mj_step`) recomputes everything derived from that pose. This is
  **already working code** — `tools/rig-rerun.py`'s own comment
  says it plainly: *"the twin, posed kinematically from the wire."*
  Generalizing it to Franka, Unitree, or anything else is the same
  per-robot adapter work as onboarding their bundle (§4) — map their
  telemetry API's joint state into `qpos`, the same job
  `parse_status` already does for our own rig.

**Rendering**: MuJoCo's `Renderer` already runs offscreen and
already produces plain RGB buffers — every camera capture and the
whole vision harness in this repo runs through it today. GPUI
already has the receiving end proven: `Img`/`Image`/`RenderImage`
for byte-buffer-backed display, and a `Surface` element that already
does exactly this pattern (externally-rendered video frames fed into
a panel, shipped today for `CVPixelBuffer` on macOS). This replaces
the far riskier alternative considered and rejected — depending on
Rerun's `re_renderer`, where the Studio would be the first-ever
external adopter of a pre-1.0, zero-reverse-dependency crate. MuJoCo
is a tool this repo already runs in production; this is a real
de-risking, not a compromise.

**Two honest exceptions, not new problems**: non-joint sensors
(cameras, force/tactile) are separate panels, not fused into the 3D
view — they never needed to be. And our own cheap arm specifically
has no feedback sensor at all — no software fixes that; it's the
already-known, already-deferred hardware gap (feedback-capable
servos, or vision-based pose estimation as its own project).

**Already have**: `mujoco.Renderer`, the kinematic-twin pattern, the
whole engine registry and instrument-stamp doctrine.

**New work**: the GPUI-side texture-upload plumbing (small, proven
pattern); per-robot telemetry adapters as each new robot is onboarded.

## 4. Bringing a robot — the bundle system, already built, already this general

**What**: "bring your robot" is the bundle system this repo has run
all session — hash-stamped MJCF + sensors + keyframes + identified
dynamics, entering through one census-gated door
(`assert_model_alive`). Menagerie already ships **Franka (3
variants)** and the **full Unitree line** (Go1/Go2/A1/H1/G1/Z1) plus
dozens of arms, quadrupeds, humanoids, and mobile manipulators, in
the exact MJCF shape our bundles already wrap twice (SO-101, ALOHA
2). Onboarding a new named robot is a wrapping exercise, not new
architecture.

**Actuator dynamics for a new robot**: the BAM library
(`robots/actuators/`) already carries 8 servos' identified friction
models (M1-M6), provenance-gated, extensible via
`tools/sync-bam-actuators.py`. A robot with no identified fit yet
gets a nominal (MuJoCo-native, M1-equivalent) starting condition —
the same nominal-vs-identified split Paper 2 already runs on.

**Already have**: everything except the per-robot telemetry adapter
(§3) and, per-robot, deciding which Menagerie MJCF to wrap.

**New work**: generalizing the bundle-builder tooling so wrapping a
new Menagerie robot is a short, repeatable recipe rather than
hand-written each time (it's been done twice by hand; a third and
fourth time is where this should become a tool).

## 5. Bringing an environment — USD for composition, splats for capture

**What**: two different jobs, already researched separately, that
compose cleanly:

- **Authored environments (USD)** — variant sets, payloads, and
  sublayers give exactly the "same scene, several configurations" /
  "base + per-site + per-calibration override" composition our
  bundle store already wants (docs/23). USD is **never** the robot
  format (MJCF→USD conversion loses all sensors and all
  cameras/lights — disqualifying for us) — its job is environments
  and asset libraries only. A documented, real failure mode applies
  if the Studio ever ingests external USD robots: a well-formed USD
  robot can import into MuJoCo with **zero actuators and no error**
  (`UsdPhysicsDriveAPI` has no MJCF-side check) — the same
  census-gate discipline that already guards MJCF imports must guard
  this door too, never assumed.
- **Captured environments (Gaussian splatting)** — a user scans
  their real kitchen or warehouse; the output is a **splat
  (appearance) + watertight collision-proxy mesh (physics)** pair,
  the now-industry-standard shape (Isaac Sim 6.0 ships exactly this).
  The mesh enters the same `MjSpec` scene composition as any other
  environment; the splat becomes an *optional* higher-fidelity render
  pass for photoreal camera observations, layered on top of — never
  replacing — MuJoCo's physics. A permissive pipeline is already
  scoped (docs/34): `gsplat` (Apache-2.0) → Open3D (MIT) TSDF fusion
  → marching cubes, explicitly avoiding the INRIA-non-commercial
  tools (MILo, SuGaR, 2DGS, GOF). DISCOVERSE (MIT, MuJoCo + 3DGS,
  650 FPS) is architecturally this exact idea, already built, and
  the standing recommendation is to benchmark against it before
  building our own.

**Sequencing, stated as plainly as the prior research already put
it**: splat-based capture is *"a phase-7 visual-gap optimisation,
after you [already have physics]"* — a later enrichment layer, not
a launch requirement. USD composition for authored environments is
more foundational and can come earlier.

**Already have**: `MjSpec` scene composition, the census-gate
doctrine to extend.

**New work**: a USD-environment ingestion path (with the actuator-
and sensor-count check applied at the door); the splat-capture
pipeline, explicitly deferred past the initial launch.

## 6. Generating sim training data — GPU-first is a real design decision here

**What the user asked to make sure was covered explicitly.** This
repo's demo generation today (`tools/kitting-demos.py`) is a CPU
loop: one episode at a time, scripted expert + chained IK, ±30% DR
sampled per attempt, kept only if the task's own referee scores it a
success. It works, and it is not GPU-first — it runs one world on
one CPU core.

**The GPU-first version, concretely, using capability already
proven this session**: MJX-Warp batches per-world *model* parameters
natively (docs/49, measured — the exact fields `physics/variations.py`
scales), not just per-world state. That means the DR sampling that
today happens one attempt at a time can become **one batched call
across hundreds of worlds**, each with its own sampled damping/gain/
friction draw, stepped in lockstep on the GPU, with the referee
filter applied as a single vectorized pass over the batch rather
than a discard-and-retry loop. This is the concrete, load-bearing
meaning of "GPU-first data generation" — not "run the same CPU loop
on a GPU machine," but restructuring generation as one batched
program.

**What stays CPU**: the *expert* choreography that requires
per-step, sequential closed-loop IK correction (today's
`scripted_kitting_episode`) is a harder batching target than open-
loop DR sampling — IK's damped-least-squares solve is not (yet)
written to vectorize across worlds. A staged approach is honest:
batch the DR sampling and rollout-and-filter step now (real,
available throughput win), keep the expert's IK on CPU per-world
until a batched IK solve is worth building, and measure before
promising a number.

**Already have**: the referee/DR/export machinery
(`kitting_export.py`, `physics/variations.py`), the measured MJX-Warp
throughput (17,666 steps/s at 64 worlds on the RTX, docs/e2e-research/52-annotated
numbers) as a real baseline to scale from.

**New work**: a batched generation path — the genuinely new
engineering this pillar needs, distinct from the training-side
batching that already exists (T6's PPO run).

## 7. Training and RL — cloud-GPU by default, not by hand

**What**: the LeRobot plugin (`envs/lerobot_plugin.py`) and the MJX
batched RL stepper already exist. "GPU-first" here means a UX
decision as much as an architecture one: today, running training on
a rented GPU is a person typing `tools/cloud-gpu.py` commands by
hand (docs/34) — real, working, but manual. The Studio's job is to
make **"train this policy"** or **"generate 10k demos"**, issued
from the agent panel, route to a rented GPU pod through the existing
seam by default, with local CPU/Mac execution as the fallback for
small smoke runs (exactly the `e2e-smoke.py` role today) — not the
other way around.

**Already have**: the whole cloud-GPU seam, Runpod as the first
provider, the measured cost/throughput numbers from the first real
run (docs/34, docs/07 2026-08-27).

**New work**: wiring that seam into the agent-driven workflow (an
MCP tool: "launch a training run," "check its status," "pull its
checkpoint") rather than a person running the CLI.

## 8. Evaluation — the honesty layer, and the actual moat

**What**: this doesn't change. The statistics module (dependency-
free, exact intervals), the certificate/harness architecture, the
per-trial record + funnel + fold pipeline, the paired-trial protocol
— this is the thing that already distinguishes this platform from
Isaac Lab Arena (coverage machinery, no honesty layer, per the prior
audit) and from every leaderboard in the field that reports a number
with no interval. The Studio surfaces it — a panel showing a run's
certificate, its funnel, its confidence interval — it does not
replace or simplify it. If anything, a GUI is where this pays off
most: right now these numbers live in JSONL files and progress-log
prose; a panel that renders a certificate is a genuinely better
product for a number that already exists.

**Already have**: all of it.

**New work**: a rendering layer for records/certificates in the
Studio's UI — a display problem, not a design problem.

## 9. Deploy to real — the Rust side, generalized

**What**: the firmware/wire protocol/HIL-replay discipline this
whole venture is built on for the little rig. Generalizing "deploy
to real" to Franka/Unitree/etc. does not mean reimplementing their
control stacks — it means the Studio's role there is thin: push a
trained policy's checkpoint to whatever the target robot's own
deployment path is (libfranka, Unitree's SDK, ROS2, whatever the
vendor ships), and pull telemetry back through the same per-robot
adapter §3 already needs for the kinematic twin. Our own rig's
firmware discipline (the wire protocol, HIL, the pin-both-ends
tests) is the reference architecture for what "trustworthy real
deployment" looks like, not something every robot re-implements.

**Already have**: the whole Rust deployment discipline, as a proof
that this approach works end to end on real hardware.

**New work**: the adapter layer per vendor SDK, done once per robot,
same shape as the telemetry adapter.

## 10. Telemetry and plots — what fills Rerun's role without embedding Rerun

**What**: loss curves, success-rate funnels, GPU utilization —
things MuJoCo has no concept of and the 3D panel was never going to
show. Two honest options, not mutually exclusive: (a) a companion
Rerun window/tab (the pattern this repo already runs by hand every
session, and Rerun's own web-embed via `@rerun-io/web-viewer` makes
this a real browser-tab option, not just a separate native window),
or (b) native GPUI plot panels for the metrics that matter most,
avoiding a runtime dependency on Rerun's still-shifting API surface
(`rerun.dataframe` already renamed to `rerun.catalog` once) for
anything load-bearing. Rerun's own multi-user/live-sharing story is
paid-Hub-only with no confirmed self-serve GA — not a foundation to
build a collaborative feature on for free.

**Already have**: `train-watch.py`'s existing Rerun dashboard
pattern, directly reusable as the companion-window option.

**New work**: none required for launch if (a) is chosen; native
plot panels are a nice-to-have, not a blocker.

## 11. GPU-first, stated as one cross-cutting principle

Collecting the thread that runs through every layer above:

- The **UI is GPU-native by construction** (GPUI).
- The **rendering path is GPU-accelerated by construction**
  (MuJoCo's offscreen renderer, typically EGL/OpenGL-backed).
- **Sim training data generation defaults to batched MJX-Warp**,
  not a CPU loop scaled up (§6) — this is a real architecture choice
  this plan makes explicitly, not an assumption.
- **Training and RL default to a rented GPU pod** through the
  existing cloud-GPU seam, with local CPU execution as the
  smoke-scale fallback, never the primary path (§7).
- **CPU MuJoCo is not deprioritized** — it stays the metrology
  instrument for identification and certificates, where precision
  and reproducibility matter more than throughput (the same
  dual-instrument doctrine docs/47-52 already established). GPU-
  first describes the *training and data-generation* path, not a
  wholesale replacement of the instrument that produces trustworthy
  numbers.

## 12. The ledger — already built vs genuinely new

| Already built, reused as-is | Genuinely new work |
|---|---|
| Bundle system, census gates, engine registry | The GPUI native app shell (the long pole) |
| BAM actuator library (8 servos, M1-M6) | The robotiq MCP server |
| MuJoCo dual-mode (sim + kinematic twin), proven on our rig | Per-robot telemetry + deploy adapters (Franka, Unitree, ...) |
| LeRobot plugin, MJX batched RL stepper | Batched (GPU-first) sim data generation |
| Cloud-GPU seam (Runpod), measured costs | Wiring the cloud-GPU seam into agent-driven workflows |
| Statistics/certificate/evaluation layer | UI panels for certificates, telemetry, run status |
| Firmware/wire/HIL deployment discipline | USD environment ingestion (with census-gate extension) |
| ACP support (Claude Code's official adapter) | GPUI texture-upload plumbing for MuJoCo frames |
| — | Gaussian-splat capture pipeline (explicitly deferred, §5) |

The honest read of this table: most of the *pipeline* already
exists and is reused unchanged; almost all the *new* work is in
making it reachable as one application and one agent-driven
workflow, plus the specific GPU-first restructuring of data
generation (§6) that doesn't exist in any form yet.

## 13. Phased plan, gated in order

1. **The robotiq MCP server.** Useful standalone, before any native
   app exists — works with Claude Code today. Lowest risk, fastest
   to real value, and it's the thing every later phase's agent
   integration depends on.
2. **Generalize the bundle system past SO-101/ALOHA** — pick one
   Menagerie robot (Franka is the obvious first: well-documented,
   heavily used in the literature, real joint-state telemetry via
   libfranka) and wrap it end to end: bundle, nominal actuator model,
   one task. This proves the "bring your robot" claim on a robot
   that isn't ours.
3. **The kinematic-twin adapter for that robot** — closes the real-
   telemetry gap for a genuine second robot, proving §3's pattern
   generalizes rather than being a one-off for our own rig.
4. **Batched (GPU-first) data generation** — the concrete new
   engineering §6 named, measured against the existing CPU path so
   the throughput claim is a number, not an assumption.
5. **The GPUI app shell**, MVP: agent panel (ACP) + 3D panel
   (MuJoCo-into-GPUI) + a run/bundle browser talking to the MCP
   server from step 1. This is the first point at which "the Studio"
   exists as a thing rather than a plan.
6. **Wire the cloud-GPU seam into the agent panel** — "train this
   policy" becomes an agent action, not a CLI command.
7. **Certificate/telemetry panels**, and the Rerun-companion-window
   or native-plot decision (§10) — made once real usage shows which
   metrics actually get looked at.
8. **USD environment ingestion**, with the actuator/sensor census
   check applied at the door from day one, not bolted on after a
   silent-failure incident.
9. **Gaussian-splat environment capture** — deliberately last,
   per the field's own "phase-7" framing, and only after benchmarking
   against DISCOVERSE rather than building from scratch.

Not scheduled, and stated as a decision rather than an oversight:
forking or rebranding Zed; depending on Rerun Hub before a self-
serve GA exists; the NVIDIA proprietary Omniverse core (RTX
rendering, `ovrtx`/`ovstage`) — the genuinely Apache-2.0 slice
(`usd-exchange`, `mujoco-usd-converter`) stays the ceiling for how
far into that ecosystem this plan reaches.
