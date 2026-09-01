---
name: data-press
description: Generate referee-gated demonstration batches with stamped provenance, and read their datasheets. Use for synthetic training data, demo generation, or auditing an existing batch.
allowed-tools: Bash(uv run:*), Read, Grep
---

# The data press

The success-gated generator (docs/e2e-research/60): every kept episode
passed its task's own referee, and its sidecar records the task stamp,
engine instrument stamp, expert stamp, and the NAMED dynamics values it
ran under, with their basis (identified interval vs caller-declared
span). Every run writes `datasheet.md` beside the batch.

## MCP doors (any agent, docs/64 stage 1)

With the rq MCP server connected, the same work is tool calls:
`generate_demos(episodes, seed, out)` → job handle;
`multiply_demos(seeds_dir, out, ...)` (GPU box);
`run_chain(name, scale, ...)` — the whole press→train→certify chain;
`job_status(job_id)` polls, artifacts land under `runs/` as always.

## Commands (from `pipeline/`)

- Kitting batch: `uv run --extra sim python ../tools/kitting-demos.py N <out> --seed S`
  (episodes and out are POSITIONAL, in that order)
- Datasheet for an existing batch (one import, no CLI needed):
  `uv run python -c "from rq_pipeline.collect.datasheet import write_datasheet; print(write_datasheet('<demos_dir>'))"`
- MCP (read-only): `describe_datasheet(demos_dir)`

## Interpreting a datasheet

- "keep rate ≤ X%" is a BOUND, not a rate — attempts after the last
  keep are unrecorded; never quote it as an exact rate.
- The `Basis` line says whether dynamics were sampled from an
  identified region or a declared span. Mixed bases, mixed stamps and
  unrecorded provenance appear under Warnings — surface them, never
  smooth them over.

## Refusal discipline

"N demos" means N successful demos; a batch that gave up is reported
with its attempts, not padded. Never mix batches whose stamps disagree
in one dataset. Never present declared-span randomisation as
identified — the basis string exists so that cannot happen silently.
