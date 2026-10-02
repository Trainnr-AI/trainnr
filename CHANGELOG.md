# Changelog

All notable changes to trainnr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

First public push, 2026-10-02: `github.com/Trainnr-AI/trainnr`.

### Changed

- The project is now **trainnr** (packages `trainnr` and `trainnr-mjlab`,
  the desktop app `trainnr-desktop`, the `trainnr` command, environment
  variables `TRAINNR_*`, task ids `trainnr/<task>`). The old import names
  and the old task namespace are gone; nothing outside this repository
  ever used them.

### Removed

- The 2025–26 rig (14 Rust crates, the Pico firmware, their tools and
  gates) moved to its own archive repository, `Trainnr-AI/rig`, with its
  history. `recordings/` and the drivetrain bundle stay as evidence.

### Added

- A light theme for the desktop app, designed (a white page, warm paper
  panels, soft blue tints): a switch in the title bar (dark, light or the
  system's), kept across launches; the embedded viewer, the pages and the
  card pictures follow; `set_studio_theme` sets it from an agent.
- The project switcher sits at the head of the sidebar; the Simulator's
  empty state is one card with the scene picker.
- The app's heartbeat reports frame time, frame rate and repaint causes.
  Under WSLg the app now takes the Wayland window path (`TRAINNR_X11=1`
  for the old one), which presents a frame ten times faster there.
- `trainnr.mcp_tools`: an entry-point group through which an installed
  package registers tools on the same MCP server (the seam the cloud
  package extends, docs/83).
- `trainnr mcp` serves the MCP server over stdio; `trainnr desktop`
  launches the app; `trainnr version`.
- The repository installs as a Claude Code plugin and marketplace
  (`.claude-plugin/`), with the MCP server, the skills and the agents.
- `tools/check-layers.py` pins the package layers.
- Governance files: `CONTRIBUTING.md` (DCO), `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`, `CITATION.cff`.
