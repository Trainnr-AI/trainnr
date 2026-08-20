# rig-drivetrain — the first robot bundle, before its first measurement

The rig's two-motor drivetrain as a fittable MuJoCo model. **Every dynamics
number in `model.xml` is a placeholder**, and that is the point: this bundle
exists so Paper 0 (docs/23-research-agenda.md) has a parameter vector to fit
into. After the fit, this directory carries the parameters *with confidence
intervals* and becomes the catalogue's entry #0.

Per wheel, the fittable vector: `gear` (torque per %-duty), `damping`
(viscous), `frictionloss` (Coulomb), `armature` (rotor inertia). The model
describes **free-spinning wheels on the bench stand** — the configuration
the firmware's calibration sweep runs in — not ground contact.

Provenance rules of the house apply: the bundle is addressed as
`rig-drivetrain@hash` via `rq_pipeline.bundles.stamp`, and any fit recorded
here must name the exact recording (`name@hash`) it was fitted from.

## The data path (ready; waiting on one rig session)

1. Flash the default `pico-odom` build (the calibration sweep — not
   teleop, not chase) and record it: `rig_view /dev/tty.usbmodemXX
   --record recordings/sweep-<date>.wire`.
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
`provenance` entry says which; right now it says **UNVERIFIED**, and the
fit cannot be trusted until that word is replaced by a source.

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

⚠️ **Quantization bias is systematic, and now measured.** Integer tick
rounding of an integrated signal is *correlated* noise: the rehearsal
recovered gear to ~2% but damping only to ~5% from a single sweep, and
doubling the sweep from 12 s to 24 s barely moved it — the bias does not
average away, and the iid-assuming intervals cannot cover it. Protocol
consequences for Paper 0: expect ~5% single-run bias on damping-class
parameters at 960 ticks/rev and 50 Hz; repeat runs; and report the
spread across runs alongside the per-run intervals.
