# Governance

trainnr is developed in the open by Trainnr AI (https://trainnr.ai) under
the Functional Source License (FSL-1.1-ALv2).

## Roles

- **Maintainers** merge changes, cut releases and decide the roadmap.
  Today there is one: Prakhar Aggarwal (@aggprakhar). Maintainers are
  listed in `.github/CODEOWNERS`; a contributor becomes one by sustained,
  reviewed contributions and the agreement of the existing maintainers.
- **Contributors** are everyone who opens an issue or a pull request, under
  the terms in `CONTRIBUTING.md` (Developer Certificate of Origin sign-off).

## How decisions are made

- Technical decisions are written down before they are acted on, with the
  alternative that was rejected and why, in `docs/` (the dated research
  notes under `docs/e2e-research/` and the numbered decision documents).
  The progress log `docs/07-progress-log.md` records what happened, with
  the gotchas, one entry per session. A decision that is not in `docs/`
  has not been made.
- Claims about the product's results resolve to records in `docs/findings/`
  and are checked by `tools/check-numbers.py`; the docs are checked against
  the code by `tools/check-docs.py`. Both run in CI and in `tools/verify.sh`.
- Disagreements are settled by the maintainers after discussion in the
  issue or pull request; the reasoning is recorded with the decision.

## What is open and what is not

The product repository (`trainnr`: the Python packages, the desktop app,
the CLI, the agent plugin, the robot bundles, the docs and the paper's
records) is open source. The hosted control plane described in
`docs/64` (`trainnr-cloud`) is closed by decision (docs/70 §7); nothing in
the open packages depends on it.

## Security and conduct

Vulnerabilities go through `SECURITY.md`; conduct through
`CODE_OF_CONDUCT.md`. Until Trainnr AI has public mailboxes, both reach
the maintainers through GitHub (private vulnerability reporting, and a
direct message or private issue to @aggprakhar).
