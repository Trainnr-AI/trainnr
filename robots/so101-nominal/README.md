# so101-nominal — the arm the field downloads, exactly as downloaded

This bundle is **Paper 2's nominal condition**: Menagerie's
`trs_so_arm100` model, fetched byte-identical from
`google-deepmind/mujoco_menagerie` (branch `main`, 2026-08-23,
Apache-2.0 — upstream `LICENSE`, `CHANGELOG.md` and their README —
preserved here as `MENAGERIE-README.md` — travel with it). The
evaluation harness loads `so101.xml`, which includes the upstream file
unmodified and adds only a `<sensor>` block (jointpos + jointvel per
joint), because harness policies observe sensors by contract and the
upstream model ships none.

**Every dynamics number in here is nominal, and that is the point.**
The upstream file gives all six joints `frictionloss=0.1 armature=0.1`
and every actuator `kp=50 dampratio=1` — one value repeated across
mechanically different joints, the signature of a default rather than a
measurement (docs/e2e-research/30 §3.4). The wild now holds at least
three mutually contradictory nominal parameterisations of this same
arm: Menagerie's kp=50, Lightwheel `leisaac`'s kp=17.8/damping=0.60,
and `LW-BenchHub`'s stiffness=1000/damping=100. Paper 2 runs the sim
side against THIS file first precisely because it is what everyone
uses; Paper 1's measured bundle is the second condition, and the gap
between their Gate A results is the number the product charges for.

Known nonlinearities this model does NOT carry (measured on a real
STS3215-12V, docs/e2e-research/27 §5): ~0.87° backlash, a 10-count
firmware dead zone (~0.88°), and an overload governor that throttles to
~20% of rated torque at ~2/3 rated load. Their absence from the nominal
model is part of what Paper 2 measures.

No `profile.json` yet: `RobotProfile`'s current fields are
drivetrain-shaped (tick scale, duty, PWM pulse band) and do not
describe a bus-servo arm honestly. Generalising the profile schema is
queued work; inventing values to fill the wrong fields is not.

Addressed as `so101-nominal@hash` via `rq_pipeline.bundles.stamp`, like
every bundle.
