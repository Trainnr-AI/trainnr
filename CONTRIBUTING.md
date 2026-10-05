# Contributing to trainnr

Coding agents: start with [`AGENTS.md`](AGENTS.md), the same rules in
short, which Claude Code also reads through `CLAUDE.md`.

Thank you for considering it. This page is short on purpose; the long
version of every rule below is in `docs/`.

## Asking, reporting, proposing

- **Questions and ideas** go to [Discussions](https://github.com/Trainnr-AI/trainnr/discussions).
- **Bugs, feature requests and robot support** go through the issue forms
  (New issue); each asks for what a maintainer needs to act.
- **Security problems** never go in an issue: use "Report a vulnerability"
  under the Security tab (`SECURITY.md`).

## Before you write code

- **Open an issue first** for anything beyond a typo or a one-line fix,
  so the change can be shaped before it is built. Bugs: what you ran, what
  you expected, what happened, the commit. Features: the user and the
  workflow, not the implementation.
- **Read the decision documents** that touch your area:
  `docs/80-trainnr-names-and-repos.md` (names, repositories, layers),
  `docs/22-pipeline-architecture.md` (the codebase map and its contracts),
  `docs/76-the-loop.md` (the contract, the state machine and the tool
  families), and `CHANGELOG.md` for recent history. A decision that is not in `docs/` has not been made; if your
  change makes one, write it there in the same pull request.

## Setting up

```sh
git clone https://github.com/Trainnr-AI/trainnr && cd trainnr
cd trainnr && uv sync --extra sim --extra mcp --extra viz && cd ..   # the Python package
cd trainnr-mjlab && uv sync --extra viz && cd ..                      # the mjlab trainer
cd crates/trainnr-studio && cargo build --release && cd ../..       # the Studio
tools/setup-hooks.sh                                                  # the pre-commit gates
# or the standard tool: pip install pre-commit && pre-commit install
```

You need [uv](https://docs.astral.sh/uv/), git and
[rustup](https://rustup.rs). The Studio's Rust toolchain is pinned in
`crates/trainnr-studio/rust-toolchain.toml`, and rustup installs it on the
first `cargo` command there; on Linux the build needs the system libraries
the README lists ([The Studio](README.md#the-studio)), and a cold build
takes about 3.5 GB of disk and 4 to 6 minutes. trainnr-mjlab's environment
is about 6 GB, and training in it needs an NVIDIA GPU with CUDA.
The first test run fetches the microduck's meshes (about 22 MB, once) from
Pollen Robotics' repository, since they are not redistributed here
(`robots/microduck/LICENSES/MESHES.md`); offline, fetch them ahead with
`python -m trainnr.bundles.fetch robots/microduck` from `trainnr/`.

The layers (docs/80 §4): `trainnr` never imports `trainnr-mjlab` or
the Studio; `trainnr-mjlab` imports `trainnr`; the Studio talks to the
packages through files and processes only. `tools/check-layers.py` enforces
it.

## Making a change

- One pull request, one change. Keep refactors and behaviour changes apart.
- **Sign off every commit** (`git commit -s`), which adds
  `Signed-off-by: Your Name <you@example.com>` and certifies the
  [Developer Certificate of Origin 1.1](DCO) (the text is in `DCO`, from
  https://developercertificate.org/): that you wrote the change or have
  the right to submit it under the Apache License 2.0 (see *Licence*
  below). Unsigned commits
  are not merged; a squash merge keeps every commit's sign-off. The
  history before the public launch is the maintainer's own work and
  predates this rule.
- **Pull request titles** follow Conventional Commits
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`,
  `chore:`), with a scope where it helps (`feat(studio): ...`,
  `fix(mjlab): ...`) and a `!` before the colon for a breaking change; a
  squash merge makes the title the commit's title on main, and a check
  reads it. The body says why, what the alternative was, and how you
  verified it.
- **One changelog line** under *Unreleased* in `CHANGELOG.md` for anything a
  user would notice.
- **Docs in the same commit.** A changed contract changes its document; a
  gotcha you hit goes into the pull request and, if it will bite again,
  into the document it belongs to; a new result is a record under
  `docs/findings/`, never a number typed into prose.
- **Names** follow the industry's vocabulary (evaluation, dataset,
  experiment, deployment, gate), not house words. Nothing is named
  `robotiq` except the vendor's gripper and the frozen bundle schema string
  (docs/80 §6).

## Before you push

Run the gates for what you touched (`AGENTS.md` says which);
`tools/verify.sh` runs every one of them and is the one list, the
pre-commit hook runs the fast subset, and CI runs them all on your pull
request. trainnr-mjlab's suite needs its 6 GB CUDA environment; CI runs
it when `trainnr-mjlab/` changes. What they check:

- `ruff format --check` and `ruff check` for `trainnr` and `tools`, `ruff
  check` for `trainnr-mjlab`; `mypy` for both packages (zero errors is the
  baseline).
- `python -m unittest discover -s tests` in each package. Tests that need a
  GPU, a trained checkpoint or a project skip with a reason and say so.
- `cargo fmt --check`, `cargo clippy` and `cargo test` for the Studio.
- `tools/check-docs.py` (docs describe real code), `tools/check-layers.py`,
  `tools/check-numbers.py` (every success ratio the README quotes
  resolves to a finding record): in CI, in the pre-commit hook and in
  `tools/verify.sh`.
- `tools/release.py check` (one version everywhere), zizmor over the
  workflows, `tools/supply-chain.py --policy` (the Studio's crates
  against `deny.toml`) and `tools/check-package.py` (both wheels carry
  LICENSE and NOTICE).

Before a release, and in any pull request that changes a tool's name,
arguments or description, run the agent test: a real Claude Code session
that sees only the trainnr server is given the laptop half of the
Quickstart in plain words, with no tool names, and must finish it.

```sh
python3 tools/agent-e2e.py --go2 <unitree_rl_mjlab>/src/assets/robots/unitree_go2/xmls/go2.xml
```

It prints the tools the agent called, the errors it hit, its turns, time
and cost, and PASS or FAIL. Paste that block into the pull request. CI
cannot run it (it needs a logged-in Claude account and costs about $0.20
to $0.40 a run).

A screen you changed is verified by looking at it: launch the Studio,
read every label as a user would, and attach the screenshot to the pull
request.

## What happens to your pull request

- CI runs on every pull request: the sign-off check, the Rust and Python
  gates and the test suite with line coverage (the coverage table is in the
  run's summary). A first-time contributor's run starts once a maintainer
  approves it; that is GitHub's guard against a stranger's code running
  with this repository's token, not a judgement of the change.
- `main` accepts changes only through a pull request with green checks and
  a code owner's approval (`.github/CODEOWNERS`), and it is never
  force-pushed. This repository is where trainnr is developed; the
  maintainers' own work arrives through pull requests like anyone's.
  Only maintainers merge, by squash or rebase.
- Dependabot opens weekly update pull requests; they go through the same
  gates. The required supply-chain job checks the Studio's crates against
  their licence, source and ban policy; a separate `advisories` job,
  which is not required, audits every locked dependency set and the
  crates for known vulnerabilities (`tools/supply-chain.py`), so an
  advisory published today does not block your pull request: a
  maintainer answers it with an update or a written exception. The
  installed Python packages are checked against a licence allow-list
  with written exceptions.

## Licence

trainnr is distributed under FSL-1.1-ALv2 (`LICENSE`). Contributions come
in under the Apache License 2.0 (`LICENSE-APACHE`): by contributing you
license your contribution to Trainnr AI under Apache-2.0, which lets
Trainnr AI distribute it as part of trainnr under FSL-1.1-ALv2 and, as the
licence grants, under Apache-2.0 two years after each release. Third-party
material you bring in is listed in `NOTICE` with its licence.
