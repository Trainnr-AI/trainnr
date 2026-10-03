<!-- Title: Conventional Commits, e.g. "fix(studio): ..." or "feat(identify): ...". -->

## What and why

<!-- What changes for a user, and why. Link the issue: "Closes #123". -->

## How it was verified

<!-- The commands you ran and what they printed. For a Studio change, a screenshot. -->

## Checklist

- [ ] Every commit is signed off (`git commit -s`); CI checks it.
- [ ] `tools/verify.sh` (or the pre-commit gates) passes locally.
- [ ] Tests added or changed for the behaviour.
- [ ] Docs in `docs/` changed with the contract, and a line under *Unreleased* in `CHANGELOG.md` if a user would notice.
- [ ] A new result is a record under `docs/findings/`, never a number typed into prose.
