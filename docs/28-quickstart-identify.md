# Quickstart: identify your robot

*The identification workflow through the Python API, start to finish,
for someone who has never seen this repo. Written 2026-08-25 after an
earlier review found every piece existed but was scattered across four
files.*

**The tool route first.** From an agent, the same workflow is three MCP
tools: `onboard_robot` (an MJCF or USD file becomes a stamped bundle),
`ingest_recording` (or `ingest_public_log`) for the telemetry, and
`identify_system` for the fit, with its intervals and pinned / NOT PINNED
verdicts written into the bundle. For a legged robot the `legged-joints`
method (since 2026-09-24) fits per-joint armature, damping and friction
from the robot's own joint telemetry; the README's Quickstart walks it on a
Go2 (reference updated 2026-10-04).
This page is the Python-API path underneath those tools, for a robot
whose telemetry arrives as a CSV.

## 0. Install

One prerequisite: [uv](https://docs.astral.sh/uv/).

```sh
cd trainnr && uv sync --extra sim     # mujoco[sysid] and friends
```

(Rust is only for the Studio; the Pico toolchain belongs to the original
rig, now [rigrs](https://github.com/Trainnr-AI/rigrs) — you need neither to
measure your robot.)

## 1. Make a bundle

A bundle is a directory: `robots/<your-robot>/` with a `profile.json`
and a `model.xml`. Copy
[`robots/rig-drivetrain/`](../robots/rig-drivetrain/) as the template.

- `profile.json`: the measured constants your fit consumes, each with a
  provenance string. Honest placeholders say so — the template's own
  `ticks_per_revolution` carries "hand-count pending" in its provenance.
  (`profile.json` is the measured-constants file the drivetrain template
  carries. A bundle made by `onboard_robot` carries `bundle.json` instead:
  the model file compiled, the source, the census; only
  `robots/rig-drivetrain` still has a `profile.json`.)
- `model.xml`: a MuJoCo model of just the parts you are fitting. Three
  rules, learned the hard way and stated once:
  **sensor order = your measurement column order · actuator order =
  your control column order · timestep = your sample period.**

## 2. Get your data in

The fit consumes three row-aligned arrays
(`trainnr.robot.identify.ExcitationData`): `times` (seconds,
strictly increasing), `controls` (your model's actuator units, one
column per actuator), `measurements` (your model's sensor outputs, one
column per sensor). The run must start at rest at the model's home
state — trim anything recorded mid-motion.

From any logger's CSV:

```python
from trainnr.collect.csv_data import excitation_from_csv

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
[`trainnr/robot/drivetrain_fit.py`](../trainnr/trainnr/robot/drivetrain_fit.py)
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
   intervals (a review finding, and three repeats in docs/26 §5). Generate data
   from your model with known parameters, fit it, and demand exact
   recovery before touching real data.

## 4. Fit, record, repeat

```python
from trainnr.bundles.hashing import stamp
from trainnr.robot.identify import identify
from trainnr.robot.fit_record import write_fit_record, write_spread_record

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

## The drivetrain shortcut

For the 2025–26 rig's drivetrain (its bundle is `robots/rig-drivetrain`;
the firmware and the recorder are in [rigrs](https://github.com/Trainnr-AI/rigrs), and its
recordings are under `recordings/`) the whole chain is one command over
one of those recordings:

```sh
uv run --extra sim python ../tools/fit-report.py ../robots/rig-drivetrain \
    --fit ../recordings/<your-sweep>.wire
```
