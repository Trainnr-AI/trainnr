# Governance

trainnr is developed in the open by Trainnr AI under
the Functional Source License (FSL-1.1-ALv2).

## Roles

- **Maintainers** merge changes, cut releases and decide the roadmap. Only
  maintainers hold write access; `main` is protected by a ruleset that
  requires a pull request, green checks and a code owner's approval, and
  only the organisation's admins may bypass it, and only through a pull
  request (`tools/github-setup.sh` applies these settings, reads them
  back and lists who can merge).
  Today there is one, Prakhar Aggarwal (@aggprakhar), so the project's
  bus factor is one; a second maintainer is wanted, and until there is
  one the maintainer's own pull requests merge through the admin bypass.
  Maintainers are
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

## Open source and the hosted service

Everything in this repository (the Python packages, the Studio, the
CLI, the agent plugin, the robot bundles, the docs and the findings
records) is FSL-1.1-ALv2, and each version becomes Apache-2.0 two years
after it is made available. A hosted service may be
built on top of it; if it is, it extends these packages through the same
public entry points any other package can use (`trainnr.mcp_tools`, the
task and robot registries), which are part of the public API below.
Nothing in this repository depends on a hosted service, and nothing here
requires an account.

The open packages send no telemetry and never phone home. Every network
connection they make, when it happens and how to turn it off, is listed
in [SECURITY.md](SECURITY.md#network).

The Studio is built without the Rerun viewer's analytics feature
(`rerun`'s default features are off in `crates/trainnr-studio/Cargo.toml`).

## Security and conduct

Vulnerabilities go through `SECURITY.md` (GitHub's private vulnerability
reporting on this repository). Conduct reports go privately to the
contact named under *Enforcement* in `CODE_OF_CONDUCT.md`, read only by
the maintainers; never through a public issue, discussion or pull
request.

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

A change to any of them is listed under *Changed (breaking)* in
`CHANGELOG.md`. From v0.1.0 on, a renamed or removed tool keeps working,
with a deprecation warning in its reply, for one minor release before it
goes. The renames before v0.1.0, while the tool API was being settled for
the first release, had no aliases; the CHANGELOG lists them.

## Releasing

A release is a signed tag `vX.Y.Z` that a maintainer pushes on a commit
of `main`. The release workflow (`.github/workflows/studio-release.yml`)
checks that the tag names the tree's version, builds the Studio for
Linux, macOS and Windows, and attaches the archives, each with its
SHA-256 and a build-provenance attestation, to a draft release; a
maintainer checks the draft and publishes it. Step by step:

1. **The release pull request.** On a branch from an up-to-date `main`:

   ```sh
   python3 tools/release.py bump X.Y.Z   # every version copy, the lockfiles' own entries,
                                          # CHANGELOG's Unreleased dated as [X.Y.Z], CITATION.cff's date-released
   (cd crates/trainnr-studio && cargo build)   # the Studio builds at the new version (Cargo.lock refreshed)
   (cd trainnr && uv lock --check) && (cd trainnr-mjlab && uv lock --check)
   python3 tools/release.py check
   git commit -s -am "chore(release): X.Y.Z"
   ```

   Read the dated CHANGELOG section: it becomes the release notes. Run
   the agent test (`tools/agent-e2e.py`, CONTRIBUTING.md) and paste its
   result into the pull request. Merge when every check is green.
2. **Rehearse with a release candidate** (for a first release, or after a
   change to the release workflow or the installer). Tag the merged
   commit `vX.Y.Z-rc.N` exactly as in step 3; the workflow builds a draft
   pre-release of the same version. Check it as in step 5, publish it,
   and install it from a fresh clone as a user would:

   ```sh
   TRAINNR_STUDIO_RELEASE=vX.Y.Z-rc.N uv run --directory trainnr trainnr studio --install
   ```

3. **Tag and push**, with the merged release commit's SHA from `main`:

   ```sh
   git fetch origin
   git tag -s vX.Y.Z -m "trainnr X.Y.Z" <sha>
   git push origin vX.Y.Z
   ```

   `-s` signs the tag with the maintainer's key. Only the organisation's
   admins can create a `v*` tag (the tag ruleset), and the workflow's
   first job refuses a tag that does not name the version in the tree.
4. **Wait for the draft.** The draft release appears only if all three
   platforms build; when one fails, there is no release. Re-running the
   failed job replaces the draft's own uploads; a published release is
   never touched again.
5. **Check the draft** before publishing:
   - the archives: one per platform
     (`trainnr-studio-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz`,
     `-aarch64-apple-darwin.tar.gz`, `-x86_64-pc-windows-msvc.zip`), each
     holding the binary, LICENSE, NOTICE and THIRD_PARTY_LICENSES.md;
   - the notes: the CHANGELOG's `[X.Y.Z]` section;
   - the checksums: `gh release download vX.Y.Z -R Trainnr-AI/trainnr -D rel`,
     then `sha256sum -c *.sha256` inside `rel`;
   - the attestation, for each archive:
     `gh attestation verify <archive> -R Trainnr-AI/trainnr`.
6. **Publish** the draft (the release page, or
   `gh release edit vX.Y.Z --draft=false -R Trainnr-AI/trainnr`). A
   published release is final: immutable releases lock its tag and
   assets, and a mistake is fixed by a new patch version, never by
   replacing an archive.
7. **Install as a user** from a fresh clone: `uv run --directory trainnr
   trainnr studio --install` downloads the new archive and checks its
   SHA-256; then, in Claude Code, `claude plugin marketplace add
   Trainnr-AI/trainnr` and `claude plugin install trainnr@trainnr`, and a
   new session, which fetches the Studio in the background.
