# Changelog

All notable changes to trainnr. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

- `trainnr mcp` serves the MCP server over stdio; `trainnr desktop`
  launches the app; `trainnr version`.
- The repository installs as a Claude Code plugin and marketplace
  (`.claude-plugin/`), with the MCP server, the skills and the agents.
- `tools/check-layers.py` pins the package layers.
- Governance files: `CONTRIBUTING.md` (DCO), `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`, `CITATION.cff`.
