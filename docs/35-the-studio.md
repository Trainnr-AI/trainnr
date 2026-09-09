# The Studio: the window on the loop

*Rewritten 2026-09-08. The 2026-08-30 version of this page specified a
GPUI application with an embedded chat panel and argued against embedding
Rerun. The app that got built does none of those three things, and the
page was never corrected — S1's own DONE criterion (docs/64 §6) was "the
app builds, docs/35 updated", and only the first half happened. This
rewrite states what the Studio IS, why each superseded decision was
reversed and when, and keeps the design content that survived. Every
reversal is dated and reasoned; nothing is quietly dropped.*


*2026-09-09: the agent drives the window in real time through files — commands in, state and events out; the contract and the nine MCP doors are in docs/76 §10.1.*


*2026-09-09, measured (finding `studio-viewport-pipe-2026-09-09`): the viewport pipe delivers 32.6 fps on the kitting scene; the offscreen render is 26 ms/frame at any size, 17 ms of it shadows; shadows off gives 41 fps; on macOS the render is inline with physics, and a CGL context renders from a background thread at 12 ms, so the two fixes are known. The native window was not measured.*


*2026-09-09, later: the stream is two processes (physics and render, a shared-memory state ring between them) because MuJoCo's Python render holds the GIL; kitting reaches 66 fps on screen at real-time factor 1.00 (finding `studio-viewport-two-process-2026-09-09`). The duck preview is four ducks: each microduck is 431,750 faces and this GL path draws ~7 ms per duck; MuJoCo's own viewer crawled on twenty too.*


*2026-09-09, the Simulator page: Live view became the SIMULATION group's Simulator, with MuJoCo simulate's own sections as a control panel over the stream's status — run, step, reset, keyframes, speed, joint and control sliders, visualization and rendering flags; the agent drives the same through three doors (docs/76 §10.2).*

## 0. What the Studio is

One native desktop application, `crates/studio-shell`, about 1,200 lines
of Rust in three files. It is an [eframe](https://github.com/emilk/egui)
app (eframe = the application harness for the egui immediate-mode GUI
library) that does two things:

1. **It embeds the real Rerun viewer.** Not a lookalike, not a
   reimplementation — `re_viewer::App`, the whole thing, with its
   blueprint panel, selection panel, time scrubber and all eleven view
   types. The app binds Rerun's gRPC ingest server on port 9876 and
   anything speaking the Rerun SDK streams into it.
2. **It runs a live MuJoCo viewport.** A subprocess renders frames into
   a shared-memory ring; the shell displays them and sends camera,
   selection and perturbation input back over a tagged stdin protocol.
   Drag orbits, scroll zooms, Ctrl-drag shoves the robot and the policy
   recovers.

Around those: a brand bar, and an optional file-tree-and-editor panel
behind a `code` button. That is the whole app. There is **no chat panel**
— see §2.

The single ingest address has one definition, `STUDIO_ADDRESS` in
`pipeline/rq_pipeline/viz.py`, mirrored against the Rust bind string by a
test (`pipeline/tests/test_studio_mirrors.py`) so the two halves cannot
drift apart silently.

## 1. Superseded: GPUI as the shell (decided 2026-08-30, reversed the same day)

The original plan was to build on GPUI, Zed's Apache-2.0 UI framework,
because "embedding one finished GPU application inside another turned out
to be unsupported anywhere in the Rust windowing stack, in either
direction."

**What actually happened:** that premise was wrong in one specific case,
and the exception is the whole architecture. Rerun's viewer *is* an egui
application, so an egui host can embed it through Rerun's own
`extend_viewer_ui` seam — no windowing tricks, no second event loop. The
same trick does not work for a text editor (Lapce runs on Floem with its
own event loop, checked 2026-08-31), which is why the code panel is a
small egui editor rather than an embedded IDE.

The MuJoCo half was decided by a licence-and-version wall rather than a
UI one: the `mujoco-rs` binding needed a patched personal-fork `glutin`
on macOS and MuJoCo 3.9.0, against this repo's deliberate `~=3.11.0` pin
(the engine version is part of the identified artifact — docs/e2e-research/49).
So MuJoCo renders in a subprocess on the Python side, where the pin already
holds, and the frames cross by shared memory.

Consequence worth keeping: the Studio is GPU-accelerated because egui and
Rerun render through wgpu, not because of a framework choice.

## 2. Superseded: the embedded agent panel (built 2026-08-30, deleted 2026-09-02)

The app briefly had a full ACP (Agent Client Protocol) chat panel: live
tool-call cards, markdown replies, a Stop button, prompt queueing, and
the seven pipeline specialists as stage chips. It worked, and it was
removed in S1 along with about a thousand lines and the ACP dependency.

**Why (docs/64 §1, operator's call):** developers already have an agent
they trust and pay for. Embedding one makes us a worse IDE instead of a
better instrument. The agent lives in the developer's own tool — Claude
Code in a terminal, Cursor, VS Code, Claude Desktop — and reaches this
repo through the MCP server (`pipeline/rq_pipeline/mcp_server.py`). The
Studio is what that agent *opens* to show a human something.

The ACP work stays in git history deliberately; it taught the protocol,
and the protocol may matter again if the app ever hosts an agent for a
non-developer audience.

## 3. Superseded: "what fills Rerun's role without embedding Rerun"

The old §10 offered two options: a companion Rerun window, or native
plot panels to avoid depending on Rerun's shifting API. An evening was
spent building bespoke egui plot panels; the operator's verdict —
*"use exactly what Rerun is doing, don't reinvent the wheel"* (2026-08-30)
— replaced them with the embed.

The rule that came out of it, and still holds
(docs/e2e-research/55): **native panels only for glanceable state; the
real Rerun viewer for anything with a scrubber, a query or a camera; the
Studio never re-implements those.** Rerun's own time-series view is
`egui_plot` on the exact egui version the shell pins, so a hand-rolled
panel would have been the same code with fewer features.

## 4. Physics and rendering: two modes, one model

Durable, unchanged from the original §3. The same compiled MJCF model
serves two jobs and the 3D view is agnostic to which is live:

- **Sim mode** — `mj_step`, full dynamics, contact solver engaged. This
  is training, evaluation and data generation. The dual-instrument
  doctrine holds: CPU MuJoCo is the metrology instrument (identification
  and certificates, pinned engine version), MJX-Warp is the throughput
  instrument for anything batched.
- **Real mode** — a kinematic twin. Real sensor readings go into
  `qpos`/`qvel` and `mj_forward` alone (never `mj_step`) recomputes
  everything derived from that pose. Working code: `tools/rig-rerun.py`
  poses the twin from the wire telemetry. Generalizing it to another
  robot is the same per-robot adapter work as onboarding its bundle —
  map that robot's joint state into `qpos`, the job `parse_status`
  already does for our own rig.

## 5. What the Studio shows today

- **Idle**: a strip offering four scene previews — kitting, lift, duck,
  and the newest trained walk checkpoint rolled live.
- **Active**: a resizable MuJoCo image, orbit and zoom and shove, with
  contact-force arrows; a red banner if the render stream dies.
- **Everything else**: whatever streams in on 9876. In practice that
  means the press feed (kept and wanted counts, the keep-rate bound, a
  scrubbable strip of kept-episode frames per camera), training telemetry
  (loss, throughput, GPU, progress), the RL feed (reward terms, episode
  length, nine worlds mirrored into one model), verdict funnels, and the
  cloud feed — a rented pod's log tailed over SSH into live cards, which
  has watched three pods at once.

## 6. Performance lessons, all measured

These are the ones that cost real time and are worth not rediscovering:

- **Frames cross by shared memory, not stdout.** Raw RGB over a pipe was
  about 6 MB per frame at a 30 Hz cap and it lagged; the ring plus a
  one-byte wake-up token per frame took the read to roughly a
  millisecond and the cap to 60 Hz (2026-09-02).
- **Stdout is the token channel.** A banner printed there corrupts the
  frame count. Banners go to stderr.
- **The renderer needs its GL environment.** A spawn without it landed on
  llvmpipe at about 3 fps and 299 % CPU; the Linux launch now carries the
  EGL and driver variables explicitly (2026-09-01).
- **Anything streaming in budgets its entity count, not just its rate.**
  Twenty ducks' worth of series plus seven hundred mesh transforms per
  tick wedged the log channel — the Rerun SDK blocks under backpressure.
- **The viewport counts pixels, not points.** Sending egui points to a
  renderer that wants pixels rendered every second frame at half
  resolution; passing the physical size made it Retina-sharp.
- **Kill the whole process tree.** A surviving Python grandchild once
  flooded the ingest as a zombie; killing the Studio by signal skips
  Rust's `Drop`, so the viewport child exits on stdin EOF instead.
- **WSLg needs the Wayland variable unset** and Rerun's client-drawn
  decorations turned off, or the window has no chrome (2026-08-31).

## 7. Where the Studio sits in the loop

The Studio is the window, not the workflow. The workflow is the MCP
surface (docs/64 §3) and the loop it walks (docs/76). The app's jobs:

- **Launched by a tool.** `open_studio` starts it; anything speaking the
  Rerun SDK then streams in.
- **Pointed at an artifact.** This does not exist yet: a tool to open a
  named recording or run in the window is designed but unbuilt, and it is
  the app's next piece of work.
- **The human's live view.** Headless runs must report the same facts
  without it — the datasheet, the certificate, the findings record. That
  parity is a design requirement, not an accident: a run on a rented
  machine or in CI has no window, and must still be readable.

## 8. Deliberately not scheduled

- Forking or rebranding Zed (GPL-3.0-or-later editor crates, no
  external-use contract; only GPUI was carved out permissively, and GPUI
  is no longer the shell anyway).
- Depending on Rerun Hub before a self-serve general availability exists.
- The NVIDIA proprietary Omniverse core; the permissive slice stays the
  ceiling for how far into that ecosystem this reaches.
- A chat panel, per §2.

## 9. Standing sources when this app grows

For agentic-AI interface *ideas*, read Zed — never its code: its agent
and editor crates are GPL-3.0-or-later and hard-wired to GPUI. For
data-visualisation widgets, reuse Rerun's own crates (`re_ui`,
`re_renderer`), which are dual MIT and Apache-2.0 and meant for reuse.
