# Certified actuator bundles

The interchange artifacts of docs/e2e-research/58 §1: every vendored
BAM fit under `../actuators/`, wrapped — the published params dict
VERBATIM inside (any BAM loader still reads it), provenance, sanity
checks and a content-derived `name@hash` stamp around it. Regenerate
with `tools/actuator-bundle.py wrap --all`; the wrap is deterministic,
so unchanged fits re-wrap to a zero diff.

Honesty is in the file: these are BAM point estimates, so `metrics`,
`uncertainty` and `context` are ABSENT and `verify` says so as
advisories. Of the 48: 6 carry a `near_search_bound` flag (a fitted
value on BAM's observed optimizer rail — e.g. xl330 m4's
alpha = 9.999999997) and 9 carry `at_floor` flags (values at the
optimizer's floor, ~1e-13). Those flags are the reason the envelope
exists: BAM ships these numbers with nothing saying so.

The files carry `"schema": "robotiq-actuator-bundle/1"`. The id is part of
each bundle's content hash, so it stays as written: renaming it would
change every actuator stamp the findings cite. It names this project's
format under the project's earlier name; it is not Robotiq Inc.'s, and
the actuators here are not Robotiq products. Readers also accept
`trainnr-actuator-bundle/1` as the same format.

