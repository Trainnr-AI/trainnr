# Contributing to trainnr

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

The layers (docs/80 §4): `trainnr` never imports `trainnr-mjlab` or
the Studio; `trainnr-mjlab` imports `trainnr`; the Studio talks to the
packages through files and processes only. `tools/check-layers.py` enforces
it.

## Making a change

- One pull request, one change. Keep refactors and behaviour changes apart.
- **Sign off every commit** (`git commit -s`), which adds
  `Signed-off-by: Your Name <you@example.com>` and certifies the
  [Developer Certificate of Origin](https://developercertificate.org/):
  that you wrote the change or have the right to submit it under the
  Apache License 2.0. Unsigned commits are not merged. The history before
  the public launch is the maintainer's own work and predates this rule.
- **Pull request titles** follow Conventional Commits
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`), with a scope
  where it helps (`feat(studio): ...`, `fix(mjlab): ...`). The body says
  why, what the alternative was, and how you verified it.
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

`tools/verify.sh` is the whole gate; the pre-commit hook runs the fast
subset. Both must be green:

- `ruff format --check` and `ruff check` for `trainnr` and `tools`, `ruff
  check` for `trainnr-mjlab`; `mypy` for both packages (zero errors is the
  baseline).
- `python -m unittest discover -s tests` in each package. Tests that need a
  GPU, a trained checkpoint or a project skip with a reason and say so.
- `cargo fmt --check`, `cargo clippy` and `cargo test` for the Studio.
- `tools/check-docs.py` (docs describe real code), `tools/check-layers.py`
  and `tools/check-numbers.py` (every number in the docs resolves to a
  record): in CI, in the pre-commit hook and in `tools/verify.sh`.

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
  gates. A supply-chain job audits every locked dependency set for known
  vulnerabilities and the Studio's crates for advisories and licences
  (`tools/supply-chain.py`); a new advisory fails the build unless its
  exception is written down with a reason.

## Licence

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (`LICENSE-APACHE`) and that third-party material you bring in
is listed in `NOTICE` with its licence.
