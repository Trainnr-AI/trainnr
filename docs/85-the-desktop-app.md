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

**The engine.** The app owns a Python environment of its own (the
"engine"): a pinned `uv` and `trainnr[sim,mcp]` at the app's version,
installed on first run into the user's trainnr folder, with a progress
page. The Studio and the jobs run from there, never from a checkout. The
agent's MCP entry points at the engine's `trainnr mcp`, a stable path as
Paper's `~/.paper/bin/paper` is.

**MCP stays stdio.** The agent starts `trainnr mcp`; it acts on the
project files and drives the open app through the control files. Agent
results appear in the window because the window watches the files. No
local HTTP port, no tokens to manage.

## 5. The phases

Each phase ends with a check on a fresh machine, not with code merged.

1. **Actions in the app.** New project and Open sample (done on this
   branch); then Add robot (a file picker over `onboard_robot`), Ingest a
   recording, Identify, Define and check a task, Train (smoke, then full),
   Evaluate, Export and gate, Check for drift, each a form from its schema
   and a job in Running now. *Check:* a person walks the Go2 quickstart in
   the app with no agent.
2. **The engine without a checkout.** The scripts jobs run move into the
   package (`python -m trainnr.…`), `trainnr-mjlab` comes from PyPI, and
   the app installs and finds its engine. *Check:* the release's Studio,
   alone on a fresh Mac and a fresh Windows machine, opens the sample and
   trains a smoke run.
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
  entry into every client it finds at launch. trainnr asks first, per
  client.

## 7. Open questions for Prakhar

- **Signing:** an Apple Developer account (99 USD a year) for notarised
  macOS builds, and a Windows code-signing certificate. Without them the
  first launch shows the operating system's warning.
- **The engine:** install on first run (a smaller download, needs the
  network once) or ship it inside the installer (about 1 GB, works
  offline). First run is the plan.
- **The site:** when trainnr.ai gets its download page (phase 4).
