# Actuator models — identified friction, not just an inherited default

Every joint in this repo's task bundles runs on MuJoCo's native
Coulomb-Viscous friction (`dof_frictionloss`/`dof_damping`) unless a
scene explicitly loads a richer model from here. This directory is
that library: one subdirectory per actuator, one JSON per friction
model tier (`m1.json` through `m6.json`, BAM's M1→M6 hierarchy — see
[docs/e2e-research/53-bam-actuator-identification.md](../../docs/e2e-research/53-bam-actuator-identification.md)
for the equations), and a **required** `PROVENANCE.json` naming where
the numbers came from.

## The rule

**No `PROVENANCE.json`, no load.** `trainnr.robot.actuator_library`
refuses a directory that doesn't declare its `source` (`"bam"` /
`"own-bench"` / `"datasheet"`), citation, and license — an actuator
model with unknown provenance is worse than no model, because it reads
as identified when it might be a guess. This mirrors the
nominal-vs-identified split every robot bundle in this repo already
makes.

## What's here

Eight actuators vendored from [Rhoban/bam](https://github.com/Rhoban/bam)
(ICRA 2025, Apache-2.0) on 2026-08-28: Dynamixel MX-64/MX-106/XL-320/
XL-330, eRob80:50/eRob80:100, Feetech STS3215 (7.4V), Waveshare
ST3025. **None of these are fits of a robot in this repo** — they're
BAM's own pendulum-bench identifications, kept as a literature
reference and a starting point for comparison, not as ground truth for
our SO-101 or ALOHA arms.

## Adding another actuator

Three ways, same directory shape, different `PROVENANCE.json.source`:

- **From a newer BAM release**: `tools/sync-bam-actuators.py` vendors
  new or changed `params/*` from a local BAM checkout or repomix pack,
  refusing a changed file with no version bump.
- **From our own bench** (once real hardware exists): run BAM's own
  acquisition + `bam.fit` pipeline (it's an offline tool that produces
  a JSON in this exact schema — using it to *identify* costs nothing
  at runtime, unlike depending on its MuJoCo/mjlab controller classes)
  and drop the output here with `source: "own-bench"`.
- **From a datasheet only** (no bench access yet): a nominal-only
  entry, `source: "datasheet"` — lower confidence, explicitly labeled
  as such, useful only until a real fit replaces it.
