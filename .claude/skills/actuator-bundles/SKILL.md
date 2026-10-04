---
name: actuator-bundles
description: Wrap, verify and interpret certified actuator bundles — BAM fits with provenance, checks and honesty advisories. Use when working with servo dynamics, identified parameters, or preparing dynamics for simulation/DR.
allowed-tools: Bash(uv run:*), Read, Grep
---

# Certified actuator bundles

The interchange artifact of docs/e2e-research/58 §1: BAM's fitted
params VERBATIM inside a stamped envelope of provenance, checks and
advisories. The committed store is `robots/actuator-bundles/` (49
bundles, 8 motors × m1..m6).

## Commands (from `trainnr/`)

- Wrap every vendored fit: `uv run python ../tools/actuator-bundle.py wrap --all`
- Verify any bundle: `uv run python ../tools/actuator-bundle.py verify <file.bundle.json>`
- MCP (read-only): `list_actuators`, `describe_actuator(actuator, tier)`

## Interpreting output

- `check near_search_bound: alpha` — the fitted value sits on BAM's
  observed optimizer rail (e.g. alpha ≈ 10). A certification signal:
  the fit may be constrained by its search box, not by the data.
- `check at_floor: ...` — values ~1e-13: the optimizer's floor, not a
  measurement.
- `advice no 'uncertainty' section` — point estimates only. Sampling
  dynamics from such a bundle REFUSES unless the caller declares a
  span, and the declared basis is recorded on every episode.

## Refusal discipline

Never invent uncertainty, vin/kp context, or metrics a bundle does not
carry — absences are advisories by design. Never edit a bundle's inner
`params` (the stamp will catch it; that is the point). A stamp mismatch
means the artifact was edited after stamping: stop and say so.
