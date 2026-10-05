<!-- Title: Conventional Commits, e.g. "fix(studio): ..." or "feat(identify): ...". -->

## What and why

<!-- What changes for a user, and why. Link the issue: "Closes #123". -->

## How it was verified

<!-- The commands you ran and what they printed. For a Studio change, a screenshot. -->

## Checklist

- [ ] Every commit is signed off (`git commit -s`); CI checks it.
- [ ] The gates for what you touched (AGENTS.md) pass locally; the commands are under "How it was verified".
- [ ] Tests added or changed for the behaviour.
- [ ] Docs in `docs/` changed with the contract, and a line under *Unreleased* in `CHANGELOG.md` if a user would notice.
- [ ] A new result is a record under `docs/findings/`, never a number typed into prose.
- [ ] If a tool's name, arguments or description changed: the agent test passed (`python3 tools/agent-e2e.py --go2 <go2.xml>`), and its block is pasted under "How it was verified" (CONTRIBUTING.md).
