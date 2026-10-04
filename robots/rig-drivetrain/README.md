# rig-drivetrain — the first robot bundle, before its first measurement

The rig's two-motor drivetrain as a fittable MuJoCo model. **Every dynamics
number in `model.xml` is a placeholder**, and that is the point: this bundle
exists so Paper 0 (the maintainers' research agenda (private)) has a parameter vector to fit
into. After the fit, this directory carries the parameters *with confidence
intervals* and becomes the catalogue's entry #0.

Per wheel, the fittable vector: `gear` (torque per %-duty), `damping`
(viscous), `frictionloss` (Coulomb), `armature` (rotor inertia). The model
describes **free-spinning wheels on the bench stand** — the configuration
the firmware's calibration sweep runs in — not ground contact.

Provenance rules of the house apply: the bundle is addressed as
`rig-drivetrain@hash` via `rq_pipeline.bundles.stamp`, and any fit recorded
here must name the exact recording (`name@hash`) it was fitted from.

## Session 2026-08-24: the first real fits are in `fits/`

Three sweeps (b, c, d), ratio-form fits — see the anchor note inside
each record. Findings, all live in the maintainers' progress log (private) and the paper draft queue:
the torque scale is unobservable exactly as the rehearsal predicted
(the ~2 ms motor defeats the armature anchor at 50 Hz), the ratio
`gear/damping` pins at ~1.5% per run, and **cross-run spread (5.6%
left, 14.6% right) dominates every per-run interval** — per-wheel and
run-level, not explained by battery alone. Encoder decode errors are
speed-correlated (~0.7% of transitions at full speed).

## The data path (proven on hardware 2026-08-24)

1. Flash the default `pico-odom` build (the calibration sweep — not
   teleop, not chase) and record it:
   `cargo run -p hil-host -- --serial /dev/cu.usbmodemXX --record
   recordings/sweep-<date>.wire`. The sweep waits for the host to
   connect before it moves. **Wheels off the ground** — the model is
   the bench configuration, and the first live session proved a ground
   run is unusable (and drives the car off the desk).
2. `rq_pipeline.collect.excitation.drivetrain_excitation(recording,
   profile)` turns the recording into `ExcitationData` (times from the
   50 Hz status counter; both wheels' commanded duty; encoder ticks
   converted to radians). The `profile` is
   `rq_pipeline.bundles.load_profile(<this directory>)` — this bundle's
   `profile.json` is the one place the robot's numbers live.
3. `rq_pipeline.robot.identify` fits the eight parameters and reports
   which are pinned.
4. `rq_pipeline.robot.fit_record.write_fit_record(bundle_dir, result,
   robot=..., recording=..., anchor=...)` lands the fit in this
   directory's `fits/` as a committed artifact. The recording must be a
   `name@hash` stamp and the anchor statement is mandatory (see the
   torque-scale caveat below) — the writer refuses both omissions.
   Repeat the sweep, record each run, and read
   `fit_record.spread_summary(load_fit_records(bundle_dir))`: when the
   cross-run spread exceeds the per-run intervals, the spread is the
   number to report.

⚠️ Chase recordings cannot substitute for the sweep: the wire status
carries one scalar duty, so the per-wheel split during turns is
unobserved. The sweep drives both wheels with the same known duty, which
makes the two encoder streams two independent single-wheel experiments.

⚠️ `ticks_per_revolution` lives in `profile.json`, not in code, and is
never fitted — it is degenerate with `gear` (both scale the output), so
it must come from the encoder datasheet or a hand-count. The profile's
provenance entry proved its worth once already: the value shipped as an
UNVERIFIED nominal 960.0 until the R13 investigation surfaced the
firmware's own doc recording a **bench count of 4290** (pico-odom
main.rs, which also notes its `RobotSpec` constant 1024.0 is wrong the
same way). The profile now says 4290 with that source named — re-verify
by hand count at the Paper 0 session before trusting any fit.

⚠️ **The torque scale is structurally unobservable from duty→angle data
alone** — scaling `gear`, `damping`, `frictionloss` and `armature` by a
common factor leaves the trajectory identical (multiply the torque
balance through by k). One parameter must be ANCHORED from outside the
data. This bundle anchors `armature` (reflected rotor inertia through
the gearbox, physically ~1e-4–1e-3 for these geared motors), and the
anchor only bites if the motor time constant armature/damping is well
above the 20 ms sample period — which for these visibly-ramping motors
it is. Discovered by this repo's own rehearsal test, whose first run
converged to 2× truth with a tight interval because the anchor was too
small to see. Paper 0's protocol must state the anchor and its source.

⚠️ **The "quantization bias" finding was wrong, and the correction is a
better finding (R13, 2026-08-23).** The rehearsal's ~2%/~5% systematic
error survived unchanged when NOTHING was quantized — it was never
quantization. It was a one-sample timestamp convention mismatch in the
rehearsal's own synthesis: `mujoco.rollout`'s output row k belongs to
time (k+1)·dt, and stamping it k·dt biased every parameter. With the
wire-faithful convention (the firmware reports "ticks as of now, duty
in force now", which matches `mujoco.sysid`'s pairing — verified
empirically to recover truth to 0.00% on unquantized data), the true
single-run floor is **~0.6% gear / ~0.5% damping / ~1.2% friction — and
it is resolution-independent** (identical at 960 and 4096 ticks/rev),
so finer encoders buy accuracy nothing here; the residual is integer
duty and boundary effects. Protocol consequences stand in amended form:
**timestamp conventions are a first-class part of the instrument** —
state them, test them against a known-truth rollout, and only then
fit; and repeat runs still, because real hardware has noise no
rehearsal synthesizes.
