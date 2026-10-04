# AGENTS.md

Instructions for coding agents working in this repository (Claude Code
reads them through `CLAUDE.md`). Humans: the same rules are in
[`CONTRIBUTING.md`](CONTRIBUTING.md), with the reasons.

## What this is

trainnr is a physical AI platform for robot learning, driven by an AI
agent through one MCP server: telemetry and identification (real-to-sim),
simulation, synthetic data, reinforcement and imitation learning,
evaluation with exact intervals, sim-to-real deployment through a gate,
and drift monitoring. Three parts, in layers that never import upward
(`tools/check-layers.py`):

| Path | What | Language |
|---|---|---|
| `trainnr/` | the core package, the MCP server (`trainnr mcp`), the `trainnr` command | Python 3.11–3.13 |
| `trainnr-mjlab/` | the trainer on mjlab (GPU); imports `trainnr`, never the reverse | Python 3.12 |
| `crates/trainnr-studio/` | the desktop app; talks to `trainnr` through files and processes only | Rust |

Also: `robots/` (actuator library, nominal robot bundles), `tools/` (gates
and scripts, listed in `tools/README.md`), `docs/` (decisions, research,
the findings records and the paper; index in `docs/README.md`, words in
`docs/GLOSSARY.md`), `.claude/` and `.claude-plugin/` (the plugin).

## Set up

```sh
cd trainnr && uv sync --extra sim --extra mcp --extra deploy --extra viz && cd ..
cd crates/trainnr-studio && cargo build && cd ../..     # only for Studio work
cd trainnr-mjlab && uv sync --extra viz && cd ..         # only for trainer work (6 GB, CUDA)
```

## Before you say a change is done

Run the gates for what you touched, and say which you ran:

```sh
cd trainnr && uv run python -m unittest discover -s tests && cd ..   # Python tests
uvx ruff check trainnr tools && uvx ruff format --check trainnr tools
(cd trainnr && uv run --extra sim --extra mcp --extra deploy --extra viz --with mypy==2.3.1 mypy trainnr)
python3 tools/check-docs.py && python3 tools/check-layers.py && python3 tools/check-numbers.py
python3 tools/release.py check                                       # one version everywhere
(cd crates/trainnr-studio && cargo fmt --check && cargo clippy -q --all-targets && cargo test -q)
```

`tools/verify.sh` runs all of it. If you changed an MCP tool's name,
arguments or description, also run `python3 tools/api-snapshot.py`
(regenerate with `--write` and show the diff) and the agent test,
`python3 tools/agent-e2e.py --go2 <go2.xml>`, and paste its output.

## Rules

- **Commits:** sign off every commit (`git commit -s`, the DCO; CI rejects
  unsigned ones). Conventional titles: `feat(studio): …`, `fix(mcp): …`.
  One change per pull request. Never force-push `main`.
- **Docs move with code.** A changed contract changes its document in the
  same pull request; a user-visible change gets a line under *Unreleased*
  in `CHANGELOG.md`.
- **Numbers come from records.** A measured result is a JSON record under
  `docs/findings/` (rendered by `tools/findings.py`); never type a success
  ratio into prose. `tools/check-numbers.py` refuses one no record carries.
- **The MCP tools are public API** (GOVERNANCE.md): names are verb first,
  `list_*` returns many, `describe_*` one; an existing artifact argument
  is named by its kind (`robot`, `task`, `experiment`, `evaluation`,
  `deployment`), `name` only names something new. Tool errors are refusals
  with a reason, never bare exceptions. Mark read-only and destructive
  tools with their annotations.
- **Identity:** every artifact is `name@hash`; a bundle's hash covers its
  files, so editing a bundle changes its stamp and the records citing it.
  Licence texts go in a bundle's `LICENSES/`, which the hash leaves out.
- **Paths:** no machine-specific paths, no `/home/...`, no WSL-only code
  outside the WSL branch (`control.on_wsl()`); user data goes under
  `$TRAINNR_HOME` (default `~/trainnr`), never inside the checkout or an
  installed package.
- **Words:** use the field's vocabulary (evaluation, dataset, experiment,
  deployment, gate). Not: door, press, referee, the box, viewport.
- **Never** read, print or commit a `.env` file or any key; never add a
  network call that runs without the user asking (SECURITY.md lists the
  ones that exist); never change the MuJoCo, mjlab or Warp pins without a
  re-measured record (they are the instrument).

## Using trainnr while you work

The repository is also a Claude Code plugin; from a checkout, the server
is `uv run --directory trainnr --extra sim --extra mcp trainnr mcp`. With
no project open, start with `create_project`; `use_project` switches.
`describe_project` shows the loop's state; `launch_studio` opens the
desktop app on it.
