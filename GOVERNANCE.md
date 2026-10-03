# Governance

trainnr is developed in the open by Trainnr AI (https://trainnr.ai) under
the Functional Source License (FSL-1.1-ALv2).

## Roles

- **Maintainers** merge changes, cut releases and decide the roadmap. Only
  maintainers hold write access; `main` is protected by a ruleset that
  requires a pull request, green checks and a code owner's approval, and
  only the organisation's admins may bypass it (`tools/github-setup.sh`
  applies these settings and lists who can merge).
  Today there is one: Prakhar Aggarwal (@aggprakhar). Maintainers are
  listed in `.github/CODEOWNERS`; a contributor becomes one by sustained,
  reviewed contributions and the agreement of the existing maintainers.
- **Contributors** are everyone who opens an issue or a pull request, under
  the terms in `CONTRIBUTING.md` (Developer Certificate of Origin sign-off).

## How decisions are made

- Technical decisions are written down before they are acted on, with the
  alternative that was rejected and why, in `docs/` (the dated research
  notes under `docs/e2e-research/` and the numbered decision documents).
  `CHANGELOG.md` records what a user would notice, release by release. A
  decision that is not in `docs/` has not been made.
- Claims about the product's results resolve to records in `docs/findings/`
  and are checked by `tools/check-numbers.py`; the docs are checked against
  the code by `tools/check-docs.py`. Both run in CI and in `tools/verify.sh`.
- Disagreements are settled by the maintainers after discussion in the
  issue or pull request; the reasoning is recorded with the decision.

## What is open and what is not

The product repository (`trainnr`: the Python packages, the Studio,
the CLI, the agent plugin, the robot bundles, the docs and the paper's
records) is open source. Anything built on top of these packages that is
not in this repository extends them through the same entry points any
other package would (`trainnr.mcp_tools`, the task and robot registries);
nothing in the open packages depends on it.

## Security and conduct

Vulnerabilities go through `SECURITY.md` (GitHub's private vulnerability
reporting on this repository). Conduct goes through `CODE_OF_CONDUCT.md`:
a GitHub issue labelled `conduct`, or the contact on the maintainer's
GitHub profile (@aggprakhar). A private address at trainnr.ai will be
named here once it exists.

## Versions and the public API

One version covers the whole product (the Python packages, the Studio,
the plugin, the MCP Registry entry, the citation); `tools/release.py`
writes it everywhere and CI fails when any copy disagrees. Versions follow
Semantic Versioning. Before 1.0, a minor release (0.2.0) may break the
public API and a patch release (0.1.1) never does.

The public API is what other people's code and agents depend on:

- the MCP tools' names, arguments and reply fields;
- the `trainnr` command line;
- the record schemas (`trainnr-*/1`: project index, commands, Studio
  state, findings, certificates, deployments);
- the Python import paths of `trainnr` and `trainnr_mjlab`.

A change to any of them is listed under *Breaking* in `CHANGELOG.md`.
A renamed or removed tool keeps working, with a deprecation warning in
its reply, for one minor release before it goes.

A release is a tag `vX.Y.Z` pushed by a maintainer: it builds the Studio
for each platform and attaches the downloads, each with its SHA-256.
