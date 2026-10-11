# 85 — The desktop app

*Decided 2026-10-11 (Prakhar): trainnr becomes a desktop app people
download and use on their own, which their coding agent can also drive
over MCP, the way Paper and Spline work. This page holds the evidence,
the architecture and the phases. Branch: `feat/desktop-app`.*

## 1. The decision

Until now the Studio was a window the agent opened: every action was an
MCP tool call, and the Studio showed the results. From here the app comes
first. A person downloads it, opens it, and walks the loop with its own
buttons; an agent connected over MCP does the same work through the same
tools, and its results appear in the open window.

The line the app now opens with (Welcome page, 2026-10-11):

> The robotics platform on your desktop: real-to-sim, train, sim-to-real,
> and back. Work here, or let your coding agent drive it.

The README and the site change only once a fresh machine can do this end
to end (phase 5).

## 2. What Paper and Spline do (primary sources, read 2026-10-11)

| | Paper (paper.design) | Spline (spline.design) |
|---|---|---|
| Download | Desktop v0.5.15 (2026-10-01): Windows, macOS ARM, Linux AppImage/deb/rpm; also a web app | macOS and Windows; the web editor "works fully" without it |
| Updates | an in-app update prompt; About, Check for updates | "The app updates itself"; a toast reloads tabs |
| Account | sign-in required | required to create or publish |
| MCP server | opening the app installs a CLI (`~/.paper/bin/paper`); the agent runs `paper mcp` over stdio, which talks to the running app; tools are listed even when the app is closed, and act only while it is open | built into the app, stdio, `127.0.0.1` only with an origin allowlist; the app must be open |
| Connecting an agent | an in-app "Connect your agent" panel with one-click "Add to {agent}"; a Claude Code plugin; commands for Codex and Cursor | at launch the app writes its entry into Claude Code, Claude Desktop, Cursor, VS Code and others; a Settings, MCP panel lists each client with its status |
| Agent edits | presence and animations in the file; results appear in the open file | the frame being edited lights up; several sessions at once |
| Limits | Free: 100 MCP tool calls a week; Pro $20: 1M a week | MCP call limits rise with the plan (Max) |
| Headline | "design incredible": "the connected canvas for teams shipping with agents" | "Make anything 3D": "Create 3D with AI or build with direct control" |

Sources: paper.design/downloads, /docs/mcp, /build-log, /pricing,
github.com/paper-design/agent-plugins; docs.spline.design/generate/spline-mcp-server,
/basics/download-spline-for-desktop, spline.design/pricing, updates.spline.design.

Paper moved from a local HTTP endpoint (`127.0.0.1:29979/mcp`) to a stdio
CLI that bridges to the running app, and now asks users to remove the
HTTP entries. trainnr already has that shape: `trainnr mcp` is a stdio
server, and it drives a running Studio through the control files
(`project/control.py`), whichever of the two started first.

## 3. Where trainnr stands (2026-10-11)

- **Tools:** 77 MCP tools, each a plain Python function. Create, onboard,
  ingest, identify and drift run in-process; check, train, evaluate,
  export and the gates start jobs (`mcp_jobs`) the Studio's Running panel
  already lists and stops.
- **Buttons today:** Open a sample, the simulator's scenes, Show, Stop a
  run. Every other step is a prompt to copy to the agent.
- **Python:** the Studio runs `uv run …` from a checkout (`spawn.rs`), and
  jobs run scripts from the checkout's `tools/`. A Studio downloaded from
  a release runs no Python on its own: it finds no checkout. It works
  today only because the plugin install is a full checkout.
- **Agent link:** attaching to a running app already works (state file,
  heartbeat, `moved_to` pointers, the Welcome root). What is missing is
  the app asking Python to do work, and the app saying whether an agent
  is connected (PR #32, under review).

## 4. The architecture

**One tool surface, two front-ends.** The app's buttons and the agent call
the same tools. `trainnr call <tool> --args '<json>'` (since 2026-10-11)
runs one tool by name through the MCP server's own registry: the same
argument checks, refusals and plugin tools. It prints the result as JSON,
exits 3 on a refusal and 1 on a failure. The Studio runs it for its
buttons, so nothing a person does in the app differs from what an agent
does.

**Forms from schemas.** A tool's input schema (`tool_api`) already says
its arguments, their types and what they mean. The app's forms for
onboarding, ingesting, training and the rest are drawn from those
schemas, so a new tool, or a plugin's, gets its form without app code.

**The engine, downloaded as it is needed.** Rule (Prakhar, 2026-10-11):
*we download things for the user when they are needed; lean from the
start, for adoption.* The app is the only thing a person downloads. The
Python environment that does the work (the "engine") is fetched in
pieces, each the first time something needs it, with its size shown on
the button that triggers it and a progress line while it lands. Sizes
measured on 2026-10-11 (installed on disk, Linux x86-64; the downloads are compressed and smaller, not yet measured):

| Piece | Fetched when | What it holds | Size |
|---|---|---|---|
| The app | the download | the Studio, the Rerun viewer, the Welcome page, browsing any project | 63 to 71 MB archive |
| Core | the first action (New project, Add robot, Identify) | `uv`, Python, `trainnr`, the MCP server, MuJoCo, numpy, scipy | about 380 MB: packages 254, Python 85, `uv` 39 |
| Viewer stream | the first Show or simulator scene | `rerun-sdk` (it carries its own viewer binary) | 253 MB |
| Deploy | the first export or gate | `onnx`, `onnxruntime` | 89 MB |
| Local training | the first training run on this machine | the trainer: mjlab, PyTorch, Warp (CUDA on Linux and Windows) | 1.8 GB on a Mac, 6.5 GB on Linux with CUDA |
| Imitation, USD, scenes | the first use of each | LeRobot; Newton's USD reader; Open3D and CoACD | per extra |

A user who trains on a rented GPU (`tools/cloud-gpu.py`) never fetches
the local trainer. The Python side owns the map from a tool to the
pieces it needs, so the app's buttons and the agent's tools fetch the
same way: an agent that calls `export_deployment` on a machine without
the deploy piece gets it fetched, not an import error.

**A Python the user already has is reused, never changed.** `uv` looks
for a compatible interpreter on the machine before it downloads one, so
a user with a fitting Python saves its 85 MB, and a user with `uv` saves
its 39 MB. Fitting is strict: the tools need Python 3.11 to 3.13 and the
trainer exactly 3.12. A default `python3` is often too old (3.10 on the
WSL box, 3.9 on macOS) or a Store shortcut on Windows, and then a
managed Python is fetched. Packages always go into the engine's own
environment, never into the user's Python: that would change their other
projects, and many systems refuse it. A user who already works with `uv`
gets packages they have cached linked from that cache, not downloaded
again.

**The sample needs no engine.** The sample's archive ships with its
index already built, and the app downloads, checks (size and SHA-256)
and unpacks it itself. A new user goes from the download to a finished
project on screen with nothing else fetched.

**MCP stays stdio.** The agent starts `trainnr mcp`; it acts on the
project files and drives the open app through the control files. Agent
results appear in the window because the window watches the files. No
local HTTP port, no tokens to manage.

## 5. The phases

Each phase ends with a check on a fresh machine, not with code merged.

1. **Actions in the app.** Done: New project and Open sample; forms over
   twelve tools in the page headers (Add robot, Identify dynamics, Add
   recording, Add public log, Define environment, Check environment,
   Train, Evaluate, Export, Run gate, Pre-flight, Check for drift), each
   drawn from the tool's schema with the hints in `trainnr/forms.py`;
   choices a registry holds (task families, public logs, runtimes) are
   read when the forms are described, so plugins' entries appear. *Check,
   2026-10-11:* the Go2 quickstart through `trainnr call` alone, as the
   buttons run it, on a fresh home with no agent: create, onboard Unitree's
   Go2, ingest the IIT chirp (3 s), identify (31 of 36 parameters in
   2.7 s), define and check the walk (accepted, 27 s), a smoke training
   run (29 s), ingest the leg-odometry log and check drift: five of nine
   stages. Train, evaluate, export and the gates wait on a full training
   run, which is the user's to start.

2. **The engine, on demand and without a checkout.** The scripts jobs
   run move into the package (`python -m trainnr.…`); `trainnr-mjlab`
   comes from PyPI; `onnx` and `onnxruntime` leave the base dependencies
   for the deploy piece; the tool-to-piece map lands in the package; the
   app fetches each piece the first time it is needed, and the sample
   opens with none. *Check:* on a fresh Mac and a fresh Windows machine,
   the release's Studio alone opens the sample with no engine, then
   creates a project, then trains a smoke run, fetching only what each
   step needs, each size shown before it downloads.
3. **Connect your agent.** A panel with one click for Claude Code (the
   plugin), and the entry to copy for Cursor, Codex and others; the
   connection status (PR #32 once fixed). It asks before writing another
   app's configuration. *Check:* a fresh machine connects Claude Code in
   one click and the agent's first call shows in the window.
4. **Distribution.** Installers built by `release.yml`: a signed and
   notarised macOS DMG, a signed Windows installer, a Linux AppImage; an
   update check; a download page on trainnr.ai. *Check:* download, install
   and first run take under five minutes, with no terminal.
5. **Messaging.** The README, the site and the plugin descriptions lead
   with the app.

## 6. What we do not copy

- **Accounts.** trainnr runs locally with no sign-in; nothing leaves the
  machine unless the user rents a GPU or downloads a sample.
- **Metered MCP calls.** Paper's free plan allows 100 tool calls a week.
  trainnr's tools run on the user's machine and are not counted.
- **Writing other apps' configuration unasked.** Spline writes its MCP
  entry into every client it finds at launch. trainnr will ask first,
  per client: phase 3, not built yet, so not a claim until it ships.

## 7. Open questions for Prakhar

- **Signing:** an Apple Developer account (99 USD a year) for notarised
  macOS builds, and a Windows code-signing certificate. Without them the
  first launch shows the operating system's warning.
- **The site:** when trainnr.ai gets its download page (phase 4).
