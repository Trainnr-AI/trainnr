# Quickstart: identify your robot

*The product's core workflow, start to finish, for someone who has never
seen this repo. Written 2026-08-25 after the practitioner review found
every piece existed but was scattered across four files and a progress
log. Rig owners have a shorter path — see the end.*

## 0. Install

One prerequisite: [uv](https://docs.astral.sh/uv/).

```sh
cd pipeline && uv sync --extra sim     # mujoco[sysid] and friends
```

(Rust and the Pico toolchain are only for this repo's own rig — you
don't need them to measure your robot.)

## 1. Make a bundle

A bundle is a directory: `robots/<your-robot>/` with a `profile.json`
and a `model.xml`. Copy
[`robots/rig-drivetrain/`](../robots/rig-drivetrain/) as the template.

- `profile.json`: the measured constants your fit consumes, each with a
  provenance string. Honest placeholders say so — the template's own
  `ticks_per_revolution` carries "hand-count pending" in its provenance.
  (The schema is currently rig-shaped — extra fields are refused, and a
  non-rig robot fills the servo/camera fields with its own facts or
  nominal values, stated as such. Schema generalisation is queued with
  the first arm bundle.)
- `model.xml`: a MuJoCo model of just the parts you are fitting. Three
  rules, learned the hard way and stated once:
  **sensor order = your measurement column order · actuator order =
  your control column order · timestep = your sample period.**

## 2. Get your data in

The fit consumes three row-aligned arrays
(`rq_pipeline.robot.identify.ExcitationData`): `times` (seconds,
strictly increasing), `controls` (your model's actuator units, one
column per actuator), `measurements` (your model's sensor outputs, one
column per sensor). The run must start at rest at the model's home
state — trim anything recorded mid-motion.

From any logger's CSV:

```python
from rq_pipeline.collect.csv_data import excitation_from_csv

data = excitation_from_csv(
    "run-1.csv", time="t", controls=["duty"], measurements=["angle_rad"]
)
```

Telemetry design matters more than telemetry rate: the synthetic servo
study ([docs/26](26-sts3215-synthetic-identifiability.md)) found
position+load recovers everything at 25 Hz while a firmware-derived
velocity register poisons the fit at any rate. Log the torque-side
signal if your bus has one, and never fit against a derived channel.

## 3. Declare what you're fitting

One `ParameterSpec` per parameter: bounds, and a named modifier that
writes the value into the MjSpec (see
[`rq_pipeline/robot/drivetrain_fit.py`](../pipeline/rq_pipeline/robot/drivetrain_fit.py)
for a worked, committed example). Two protocol rules that cost this
repo real sessions:

1. **Anchor one parameter from outside the data.** The overall torque
   scale is typically unobservable from command→angle data alone; fix
   one parameter from a datasheet, a kitchen scale (a weighed link
   turns gravity into a free torque anchor), or a bench measurement —
   and say so. The fit-record writer refuses a record without an
   anchor statement.
2. **Verify your timestamp convention against a known-truth rollout
   first.** A one-sample shift produces a biased fit with tight
   intervals (R13, and three repeats in docs/26 §5). Generate data
   from your model with known parameters, fit it, and demand exact
   recovery before touching real data.

## 4. Fit, record, repeat

```python
from rq_pipeline.bundles.hashing import stamp
from rq_pipeline.robot.identify import identify
from rq_pipeline.robot.fit_record import write_fit_record, write_spread_record

result = identify(model_xml, data, parameter_specs)
print(result.summary())                    # pinned / NOT PINNED, per parameter

write_fit_record(
    bundle_dir, result,
    robot="your-robot",
    recording=stamp("run-1", Path("run-1.csv")),
    anchor="scale anchored from <source>, because <reason>",
    units={"your_param": "N*m*s/rad (joint damping)", ...},
)
```

Run **at least three** excitations (rest between; alternate directions)
and then:

```python
write_spread_record(bundle_dir)            # fits/SPREAD.json
```

When the cross-run spread exceeds the per-run intervals, the intervals
are lying and **the spread is the number to report** — that verdict is
written into the bundle as an artifact, not left as prose.

## 5. Read the report

```sh
uv run --extra sim python ../tools/fit-report.py ../robots/<your-robot>
```

Every parameter with estimate, interval, units and verdict; the anchor
statement verbatim; the cross-run verdict. That report — including its
NOT PINNED rows — is the deliverable. A measurement that cannot admit
what it doesn't know is a guess with confidence theater.

## Rig owners' shortcut

For this repo's own drivetrain the whole chain is one command:

```sh
uv run --extra sim python ../tools/fit-report.py ../robots/rig-drivetrain \
    --fit ../recordings/<your-sweep>.wire
```
