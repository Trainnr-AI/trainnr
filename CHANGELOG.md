# Changelog

All notable changes to trainnr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

First public push, 2026-10-02: `github.com/Trainnr-AI/trainnr`.

### Changed

- The project is now **trainnr** (packages `trainnr` and `trainnr-mjlab`,
  the Studio `trainnr-studio` (crate, binary, `trainnr studio`, app id
  `ai.trainnr.studio`; the app's persisted window state resets once),
  the `trainnr` command, environment
  variables `TRAINNR_*`, task ids `trainnr/<task>`). The old import names
  and the old task namespace are gone; nothing outside this repository
  ever used them.

### Removed

- The 2025–26 rig (14 Rust crates, the Pico firmware, their tools and
  gates) moved to its own archive repository, `Trainnr-AI/rig`, with its
  history. `recordings/` and the drivetrain bundle stay as evidence.

### Fixed

- From a fresh clone of the public edition, walked end to end (2026-10-03):
  `evaluate_walk` hands its job the resolved checkpoint path (a bare
  `run/model_N.pt` died in the judge); the artifact drawer's header
  picture (a red triangle after the picture decoder moved off the UI
  thread); the README's tool names and the address of Unitree's `go2.xml`.

### Added

- A light theme for the Studio, designed (a white page, warm paper
  panels, soft blue tints): a switch in the title bar (dark, light or the
  system's), kept across launches; the embedded viewer, the pages and the
  card pictures follow; `set_studio_theme` sets it from an agent.
- The project switcher sits at the head of the sidebar; the Simulator's
  empty state is one card with the scene picker.
- The app's heartbeat reports frame time, frame rate and repaint causes.
- The chrome after Zed's: a title bar with the brand and a project › page
  crumb; a status bar along the bottom with what runs, the presenter's
  state, the viewer's panel toggles, the theme switch and the frame time.
- Card pictures are decoded off the UI thread at the card's size; the
  window no longer polls for agent commands (a watcher thread wakes it).
  Under WSLg the app now takes the Wayland window path (`TRAINNR_X11=1`
  for the old one), which presents a frame ten times faster there.
- `trainnr.mcp_tools`: an entry-point group through which an installed
  package registers tools on the same MCP server.
- `trainnr mcp` serves the MCP server over stdio; `trainnr studio`
  launches the app; `trainnr version`.
- The repository installs as a Claude Code plugin and marketplace
  (`.claude-plugin/`), with the MCP server, the skills and the agents.
- `tools/check-layers.py` pins the package layers.
- Governance files: `CONTRIBUTING.md` (DCO), `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`, `CITATION.cff`.
