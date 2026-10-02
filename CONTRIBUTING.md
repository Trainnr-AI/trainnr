# Contributing to trainnr

Thank you for considering it. This page is short on purpose; the long
version of every rule below is in `docs/`.

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
cd crates/trainnr-desktop && cargo build --release && cd ../..       # the desktop app
tools/setup-hooks.sh                                                  # the pre-commit gates
```

The layers (docs/80 §4): `trainnr` never imports `trainnr-mjlab` or the
desktop; `trainnr-mjlab` imports `trainnr`; the desktop talks to the
packages through files and processes only. `tools/check-layers.py` enforces
it.

## Making a change

- One pull request, one change. Keep refactors and behaviour changes apart.
- **Sign off every commit** (`git commit -s`), which adds
  `Signed-off-by: Your Name <you@example.com>` and certifies the
  [Developer Certificate of Origin](https://developercertificate.org/):
  that you wrote the change or have the right to submit it under the
  Apache License 2.0. Unsigned commits are not merged.
- **Pull request titles** follow Conventional Commits
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`), with a scope
  where it helps (`feat(desktop): ...`, `fix(mjlab): ...`). The body says
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

- `ruff format --check` and `ruff check` for `trainnr`, `trainnr-mjlab` and
  `tools`; `mypy` for both packages (zero errors is the baseline).
- `python -m unittest discover -s tests` in each package. Tests that need a
  GPU, a trained checkpoint or a project skip with a reason and say so.
- `cargo fmt --check`, `cargo clippy` and `cargo test` for the desktop app.
- `tools/check-docs.py` (docs describe real code), `tools/check-layers.py`,
  `tools/check-numbers.py` (every number in the docs resolves to a record).

A screen you changed is verified by looking at it: launch the desktop app,
read every label as a user would, and attach the screenshot to the pull
request.

## Licence

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (`LICENSE-APACHE`) and that third-party material you bring in
is listed in `NOTICE` with its licence.
