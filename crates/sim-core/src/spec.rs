//! One definition of *what the robot is*, and *how it is tuned*.
//!
//! Before this module existed, `WHEEL_RADIUS` was written in four files and
//! the heading gains in three — and the gains had already drifted apart
//! (6.0/0.6 in the simulator and the firmware, 3.0/0.3 in the camera
//! chase) with nothing to say whether that was deliberate. It was. But you
//! could not tell that from the code, and a difference you cannot
//! distinguish from a bug is a bug waiting to be "fixed".
//!
//! So the rule here is:
//!
//! - **Geometry** ([`RobotSpec`]) is a property of the *machine*. It is the
//!   same number everywhere or the robot is lying to itself. One `const`.
//! - **Gains** ([`ControlGains`]) are a property of the *control problem*.
//!   Steering to a known map coordinate and steering onto a jittery camera
//!   bearing are genuinely different problems, so they get genuinely
//!   different named profiles — and the name carries the reason.
//!
//! What does *not* belong here: per-experiment dials like `BEARING_ALPHA`
//! or `AVOID_ENTER`. Those exist to be edited while watching the viewer.
//! DRY applies to definitions, not to knobs.

use crate::robot::{DiffDrive, WheelSpeeds};

/// The physical robot: the numbers firmware and simulator must agree on.
///
/// If these ever disagree between the chip and the host, odometry silently
/// integrates a robot that does not exist — the drift looks like sensor
/// noise and is nearly impossible to diagnose from a plot. Hence: one
/// definition, imported by both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobotSpec {
    /// Wheel radius, metres. Converts wheel *angle* to ground *distance*.
    pub wheel_radius: f64,
    /// Distance between the wheel contact patches, metres. Sets how much
    /// turning you get per unit of left/right wheel-speed difference.
    pub track_width: f64,
    /// Encoder counts per full wheel revolution (post-quadrature).
    pub ticks_per_rev: f64,
    /// Wheel speed at full motor command, rad/s. The duty-cycle scale.
    pub max_wheel_rad_s: f64,
}

impl RobotSpec {
    /// Build a spec from the units you actually measure in.
    ///
    /// Calipers give **millimetres**; motor datasheets give **RPM**. Doing
    /// the conversion here means the one place a human types numbers is
    /// the one place those numbers look like what they read off the part.
    ///
    /// ```
    /// # use sim_core::RobotSpec;
    /// // 60 mm wheels, 150 mm apart, 1024 counts/rev, 200 RPM motor
    /// let spec = RobotSpec::from_measurements(60.0, 150.0, 1024.0, 200.0);
    /// assert!((spec.wheel_radius - 0.030).abs() < 1e-12);
    /// ```
    pub const fn from_measurements(
        wheel_diameter_mm: f64,
        track_width_mm: f64,
        ticks_per_rev: f64,
        max_rpm: f64,
    ) -> RobotSpec {
        RobotSpec {
            wheel_radius: wheel_diameter_mm / 2000.0, // mm diameter -> m radius
            track_width: track_width_mm / 1000.0,
            ticks_per_rev,
            // rev/min -> rev/s -> rad/s
            max_wheel_rad_s: max_rpm / 60.0 * core::f64::consts::TAU,
        }
    }

    /// The robot the simulator has modelled since Stage 0.
    ///
    /// # Provenance — read before trusting these
    ///
    /// **These are plausible placeholders, not measurements.** They were
    /// chosen on 2026-07-27 when there was no hardware in view, and they
    /// are self-consistent, which is all the simulator needs.
    ///
    /// | field | where it came from |
    /// |---|---|
    /// | `wheel_radius` | 60 mm diameter — typical small 2WD chassis wheel |
    /// | `track_width` | plausible for an N20-class chassis |
    /// | `ticks_per_rev` | a round number |
    /// | `max_wheel_rad_s` | 30 rad/s ≈ 286 RPM |
    ///
    /// The ordered motor is **~200 RPM** (docs/09), i.e. ~20.9 rad/s — so
    /// `max_wheel_rad_s` here is ~43% optimistic. That is left alone
    /// deliberately: retuning the simulator to a motor that has not arrived
    /// would move the recorded Stage 0 baseline (22.5 s, 0.052 m) for no
    /// gain. Measure the real part, then use [`Self::from_measurements`].
    pub const SIM_BOT: RobotSpec = RobotSpec {
        wheel_radius: 0.03,
        track_width: 0.15,
        ticks_per_rev: 1024.0,
        max_wheel_rad_s: 30.0,
    };

    /// The physical robot. **Fill this in from the bench, not the datasheet.**
    ///
    /// When the parts arrive, measure and replace — this is the single
    /// place to edit, and everything (simulator, HIL host, firmware)
    /// follows from it.
    ///
    /// 1. **Wheel diameter** — calipers, and measure it *under load* with
    ///    the robot's weight on it. A squashy tyre has a smaller effective
    ///    radius than a free one, and this is the **largest single source
    ///    of odometry drift**. It is why the simulator models a deliberate
    ///    1% error.
    /// 2. **Track width** — centre-to-centre of the two *contact patches*,
    ///    not the axle length and not the outer edges.
    /// 3. **Ticks per rev** — do **not** compute it from
    ///    `PPR × 4 × gear_ratio`. Gear ratios advertised as "50:1" are
    ///    routinely 51.45:1. Spin the wheel exactly ten turns by hand,
    ///    read the counter, divide by ten.
    /// 4. **Max RPM** — measure it on *your* battery at the voltage the
    ///    robot actually runs at, not the datasheet's nominal 6 V.
    ///
    /// # ⚠️ NOT YET MEASURED
    ///
    /// This is currently `SIM_BOT` — placeholders, not measurements. The
    /// parts were ordered 2026-08-02 and have not arrived.
    ///
    /// **Do not fill this in with plausible-looking guesses.** It was
    /// briefly set to invented values while testing the mechanism, and the
    /// danger was instructive: every derived quantity came out physically
    /// sensible, [`Self::check`] passed, and nothing anywhere indicated the
    /// numbers were fiction. `pico-robot` and `hil-host` both read this
    /// constant, so a guess here is a rig quietly validating a robot that
    /// does not exist.
    ///
    /// Replace it in one line once bench measurements exist:
    ///
    /// ```ignore
    /// pub const REAL_BOT: RobotSpec =
    ///     RobotSpec::from_measurements(diameter_mm, track_mm, ticks, rpm);
    /// ```
    ///
    /// `changing_real_bot_is_a_single_edit` will then fail on purpose —
    /// that is the signal to record the new HIL baseline in docs/07 and
    /// delete the test.
    pub const REAL_BOT: RobotSpec = RobotSpec::SIM_BOT;

    /// The kinematic model implied by this geometry.
    pub fn drive(&self) -> DiffDrive {
        DiffDrive {
            wheel_radius: self.wheel_radius,
            track_width: self.track_width,
        }
    }

    /// Distance the robot travels per full wheel revolution, metres.
    pub fn wheel_circumference_m(&self) -> f64 {
        core::f64::consts::TAU * self.wheel_radius
    }

    /// Ground distance per encoder tick, metres. The resolution of
    /// odometry: no position change smaller than this is observable.
    pub fn metres_per_tick(&self) -> f64 {
        self.wheel_circumference_m() / self.ticks_per_rev
    }

    /// Top forward speed the motors can actually deliver, m/s.
    pub fn max_body_speed(&self) -> f64 {
        self.wheel_radius * self.max_wheel_rad_s
    }

    /// Motor speed in RPM — the unit the datasheet uses.
    pub fn max_rpm(&self) -> f64 {
        self.max_wheel_rad_s / core::f64::consts::TAU * 60.0
    }

    /// Does the forward axis alone fit? Split out of [`check`](Self::check)
    /// so a test can assert *"forward is fine"* independently of the turn
    /// axis — the distinction that hid a real problem for as long as
    /// `check` only looked at this half.
    pub fn v_max_is_reachable(&self, gains: &ControlGains) -> bool {
        gains.v_max <= self.max_body_speed()
    }

    /// Fastest the robot can spin, rad/s — the *other* limit, and the one
    /// that actually bites.
    ///
    /// A differential drive has two ceilings, not one. Drive both wheels
    /// flat out in opposite directions and the body rotates at
    /// `2·r·ω_max / L` with no forward motion at all:
    ///
    /// ```text
    ///     ω_L = −ω_max  ◀──(L)──▶  ω_R = +ω_max
    ///     v = r(ω_R + ω_L)/2 = 0        ← they cancel
    ///     w = r(ω_R − ω_L)/L = 2·r·ω_max/L
    /// ```
    ///
    /// For `SIM_BOT`: 2 × 0.03 × 30 / 0.15 = **12 rad/s**.
    pub fn max_turn_rate(&self) -> f64 {
        2.0 * self.wheel_radius * self.max_wheel_rad_s / self.track_width
    }

    /// The heading error, in radians, below which the steering P term is
    /// still *proportional* rather than pinned at the motors' limit.
    ///
    /// The P term asks for `heading_kp · e`. Past `max_turn_rate / kp` the
    /// wheels cannot deliver it, so every larger error produces the same
    /// command — the controller stops being proportional and becomes
    /// bang-bang:
    ///
    /// ```text
    ///   turn rate
    ///   commanded
    ///      ▲
    ///  12 ─┤        ╱▔▔▔▔▔▔▔▔▔▔▔▔  saturated: every error here
    ///      │      ╱                 gets the SAME command
    ///      │    ╱ ← slope = kp
    ///      │  ╱
    ///    0 └╱──────┬─────────────▶  |heading error e|
    ///      0       │
    ///        band = max_turn_rate/kp
    /// ```
    ///
    /// Saturation is not automatically a bug — see [`RobotSpec::check`]
    /// for the criterion that decides whether this band is wide enough.
    pub fn turn_proportional_band(&self, gains: &ControlGains) -> f64 {
        if gains.heading_kp <= 0.0 {
            return f64::INFINITY;
        }
        self.max_turn_rate() / gains.heading_kp
    }

    /// Is this control profile physically achievable on this robot?
    ///
    /// Catches the mismatch class that is otherwise diagnosed as "the
    /// controller is badly tuned": commanding a speed the motors cannot
    /// reach means they saturate, the robot moves slower than the model
    /// believes, and odometry blames the encoders.
    pub fn check(&self, gains: &ControlGains) -> Result<(), &'static str> {
        if self.wheel_radius <= 0.0 || self.track_width <= 0.0 {
            return Err("geometry must be positive");
        }
        if self.ticks_per_rev <= 0.0 {
            return Err("ticks_per_rev must be positive");
        }
        if gains.v_max > self.max_body_speed() {
            return Err("v_max exceeds what the motors can deliver — the \
                        controller will command speeds the robot cannot reach");
        }
        // ---- the turn axis ----
        //
        // This used to check only `v_max`, which is half the story: a
        // differential drive saturates in *rotation* long before it
        // saturates going forward, and on the shipped gains it does.
        // `heading_kp · π` = 18.85 rad/s against a 12 rad/s ceiling.
        //
        // That is deliberately NOT an error. Demanding a hard turn when
        // badly misaimed is correct, and `fit_wheels` scales the pair so
        // the arc survives. Rejecting it would reject a good tune.
        //
        // What matters is *where* saturation starts. The alignment
        // throttle already sets the natural boundary: at |e| ≥ π/2 the
        // forward speed is zero, so the robot is spinning in place and
        // bang-bang is exactly what you want. Below π/2 it is driving,
        // and steering must stay proportional or it will not track.
        //
        // So the criterion is: the proportional band must cover the whole
        // region where the robot is actually moving forward.
        if self.turn_proportional_band(gains) < core::f64::consts::FRAC_PI_2 {
            return Err("heading_kp saturates the wheels while the robot is \
                        still driving forward — steering goes bang-bang \
                        inside the alignment throttle's ±90° window");
        }
        Ok(())
    }

    /// Scale a wheel-speed pair down until neither exceeds the motor,
    /// **keeping their ratio** — and therefore the arc the robot drives.
    ///
    /// # Why not just clamp each wheel
    ///
    /// Clamping independently distorts the turn. If the controller asks
    /// for (40, 10) rad/s on a 30 rad/s motor, clamping gives (30, 10) —
    /// a difference of 20 where 30 was intended, so the robot turns more
    /// tightly than commanded and leaves the planned path. Scaling gives
    /// (30, 7.5): same curvature, just slower.
    ///
    /// "Slower along the right arc" is almost always what you want; "fast
    /// along the wrong one" is how a robot ends up somewhere surprising.
    ///
    /// Returns the speeds unchanged when nothing saturates, so this is
    /// free in the normal case.
    ///
    /// There used to be two of these — `fit_wheels(left, right)` and
    /// `fit_wheels_of(pair)` — which existed only because there was no
    /// type to pass. [`WheelSpeeds`] made the second one redundant.
    pub fn fit_wheels(&self, wheels: WheelSpeeds) -> WheelSpeeds {
        let peak = wheels.peak();
        if peak <= self.max_wheel_rad_s || peak == 0.0 {
            return wheels;
        }
        let scale = self.max_wheel_rad_s / peak;
        WheelSpeeds {
            left: wheels.left * scale,
            right: wheels.right * scale,
        }
    }

    /// Wheel speed (rad/s) → motor command in ±1000 duty units, saturated.
    ///
    /// Lives here because the conversion is only meaningful in terms of
    /// [`Self::max_wheel_rad_s`], and firmware was doing it inline.
    pub fn duty(&self, wheel_rad_s: f64) -> i32 {
        let full = f64::from(DUTY_FULL);
        ((wheel_rad_s / self.max_wheel_rad_s) * full).clamp(-full, full) as i32
    }
}

/// Full-scale motor command on the wire. `±DUTY_FULL` is "everything the
/// motor has".
///
/// # Why it lives here and not in `hil-protocol`
///
/// It is the protocol's number, but `RobotSpec::duty` is the only thing
/// that *produces* it, and `hil-protocol` already depends on this crate —
/// so putting it there would leave the encoder unable to name the limit it
/// must respect. It did: the literal `1000` was written three times across
/// two crates, and `hil_protocol::clamp_duty` documented itself as
/// *"Both ends call this"* while nobody called it, because `duty` had
/// reimplemented the clamp inline.
///
/// `hil-protocol` re-exports it, so the protocol still names it.
pub const DUTY_FULL: i32 = 1000;

/// A tuned control profile: gains plus the speed policy they were tuned
/// against.
///
/// Gains are meaningless without the plant they were tuned for, so the
/// speed limits travel with them rather than sitting in a separate const.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlGains {
    pub heading_kp: f64,
    pub heading_ki: f64,
    pub heading_kd: f64,
    /// Anti-windup clamp on the heading integral.
    pub heading_i_limit: f64,
    /// Anti-**kick** clamp on the heading derivative's contribution, rad/s.
    ///
    /// The mirror of `heading_i_limit`, for the opposite failure. `D` is
    /// `Kd · de/dt`, and at `dt = 0.02` that multiplies any error jump by
    /// 50 — while the error jumps whenever the *setpoint* moves, which our
    /// planner does every time it hops to the next lookahead node. The
    /// robot has not moved; the controller reacts as though it lurched.
    ///
    /// Measured before this clamp existed: **94 rad/s** of commanded turn
    /// rate, ~235 rad/s at the wheel, against motors that deliver 30.
    /// Sized to the fastest turn the robot can physically make.
    pub heading_d_limit: f64,
    /// Forward speed per metre of remaining distance, m/s per m.
    pub kp_dist: f64,
    /// Speed cap, m/s.
    pub v_max: f64,
    /// Close enough to call a waypoint reached, metres.
    pub arrive_radius: f64,
}

impl ControlGains {
    /// Driving to a **known coordinate** — Stage 0's waypoint follower and
    /// the Pico's tour.
    ///
    /// The target is exact and stationary, so the derivative term can be
    /// aggressive: there is no measurement noise for it to amplify.
    /// Tuned by eye in the Rerun viewer during M3.
    pub const WAYPOINT: ControlGains = ControlGains {
        heading_kp: 6.0,
        heading_ki: 0.0,
        heading_kd: 0.6,
        heading_i_limit: 1.0,
        // 12 rad/s = 2·r·max_wheel_rad_s / L for SIM_BOT — the fastest
        // this robot can spin. Asking the D term for more than the wheels
        // can deliver only produces a command that gets scaled away.
        heading_d_limit: 12.0,
        kp_dist: 0.8,
        v_max: 0.45,
        arrive_radius: 0.15,
    };

    /// Steering onto a **camera bearing** — visual servoing.
    ///
    /// Half the gain of [`Self::WAYPOINT`], **on purpose**. The setpoint
    /// here is a detector output that jitters box-to-box every frame, and
    /// D differentiates whatever jitter survives filtering. The plant is
    /// also slower: detection runs at ~20 fps against the sim's 50 Hz, so
    /// the loop has less authority per unit time and a stiff controller
    /// oscillates. See docs/11-perception-stack.md.
    pub const VISUAL_SERVO: ControlGains = ControlGains {
        heading_kp: 3.0,
        heading_ki: 0.0,
        heading_kd: 0.3,
        heading_i_limit: 1.0,
        heading_d_limit: 12.0,
        kp_dist: 0.8,
        v_max: 0.35,
        arrive_radius: 0.15,
    };

    /// [`WAYPOINT`](Self::WAYPOINT) with a wider arrival radius, for the
    /// chip driving a robot over a serial link.
    ///
    /// The round trip adds a tick of latency on top of the motor lag the
    /// pure simulator already models, so the robot overshoots a little
    /// further before it registers arrival.
    ///
    /// This lived inline in `firmware/pico-robot` and so could not be
    /// validated: `RobotSpec::check` never saw it, and neither did any
    /// test. Shipped profiles belong here, next to the others.
    pub const HIL: ControlGains = ControlGains {
        arrive_radius: 0.18,
        ..Self::WAYPOINT
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duty_saturates_symmetrically() {
        let s = RobotSpec::SIM_BOT;
        assert_eq!(s.duty(s.max_wheel_rad_s), 1000);
        assert_eq!(s.duty(-s.max_wheel_rad_s), -1000);
        assert_eq!(s.duty(s.max_wheel_rad_s * 10.0), 1000, "must clamp");
        assert_eq!(s.duty(0.0), 0);
    }

    #[test]
    fn measurements_convert_to_si() {
        // 60 mm wheels, 150 mm apart, 200 RPM — what you read off the parts.
        let s = RobotSpec::from_measurements(60.0, 150.0, 1024.0, 200.0);
        assert!(
            (s.wheel_radius - 0.030).abs() < 1e-12,
            "mm diameter -> m radius"
        );
        assert!((s.track_width - 0.150).abs() < 1e-12);
        assert!(
            (s.max_wheel_rad_s - 20.943_951).abs() < 1e-5,
            "200 RPM -> rad/s"
        );
    }

    #[test]
    fn rpm_round_trips() {
        let s = RobotSpec::from_measurements(60.0, 150.0, 1024.0, 200.0);
        assert!((s.max_rpm() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn derived_quantities_are_consistent() {
        let s = RobotSpec::SIM_BOT;
        // One revolution of a 30 mm-radius wheel covers 2*pi*r.
        assert!((s.wheel_circumference_m() - 0.188_495).abs() < 1e-5);
        // ...spread over ticks_per_rev counts.
        assert!((s.metres_per_tick() * s.ticks_per_rev - s.wheel_circumference_m()).abs() < 1e-12);
        // Sub-millimetre resolution at 1024 ticks: odometry cannot see
        // motion finer than this.
        assert!(
            s.metres_per_tick() < 0.001,
            "{} m/tick",
            s.metres_per_tick()
        );
    }

    #[test]
    fn the_simulator_profile_is_physically_achievable() {
        // The check that would have caught a v_max nobody can reach.
        assert!(RobotSpec::SIM_BOT.check(&ControlGains::WAYPOINT).is_ok());
        assert!(RobotSpec::SIM_BOT
            .check(&ControlGains::VISUAL_SERVO)
            .is_ok());
    }

    #[test]
    fn an_unreachable_v_max_is_rejected() {
        // A 200 RPM motor on 60 mm wheels tops out at ~0.63 m/s.
        let slow = RobotSpec::from_measurements(60.0, 150.0, 1024.0, 200.0);
        let greedy = ControlGains {
            v_max: 2.0,
            ..ControlGains::WAYPOINT
        };
        assert!(
            slow.check(&greedy).is_err(),
            "should reject an impossible v_max"
        );
        // The forward axis on the ordered motor is genuinely fine:
        // 0.45 m/s wanted, 0.628 m/s available.
        assert!(slow.v_max_is_reachable(&ControlGains::WAYPOINT));
    }

    /// ⚠️ **A finding, not a passing test.**
    ///
    /// This assertion used to read `slow.check(&WAYPOINT).is_ok()` with
    /// the comment *"the real profile fits comfortably on the ordered
    /// motor"*. That was true of forward speed and false of rotation —
    /// and the turn axis was the half `check` did not look at.
    ///
    /// The N20 in `docs/09` is 200 RPM against the 286 RPM `SIM_BOT`
    /// assumes. On the forward axis that is slack. On the turn axis:
    ///
    /// ```text
    ///   max_turn_rate   8.378 rad/s   (vs 12.0 simulated)
    ///   band = 8.378/6  1.396 rad = 80°     ← need ≥ 90°
    /// ```
    ///
    /// So between **80° and 90°** of heading error the robot would be
    /// driving forward with the steering pinned at the motor limit —
    /// bang-bang exactly where it still needs to track a path.
    ///
    /// **When the motor arrives, `heading_kp` must come down from 6.0 to
    /// at most 5.33**, and the mission must be re-run to see what that
    /// costs. Caught before the part shipped, by a check that had been
    /// looking at the wrong axis.
    #[test]
    fn the_ordered_motor_cannot_run_the_current_heading_gain() {
        let ordered = RobotSpec::from_measurements(60.0, 150.0, 1024.0, 200.0);
        assert!(
            ordered.check(&ControlGains::WAYPOINT).is_err(),
            "if this now passes, the motor spec or the gains changed — \
             re-derive the band before deleting this test"
        );

        let band = ordered.turn_proportional_band(&ControlGains::WAYPOINT);
        assert!(
            (band.to_degrees() - 80.0).abs() < 0.1,
            "band should be ~80°, got {:.1}°",
            band.to_degrees()
        );

        // The gain that *would* fit, quoted in the doc comment above.
        let retuned = ControlGains {
            heading_kp: 5.33,
            ..ControlGains::WAYPOINT
        };
        assert!(
            ordered.check(&retuned).is_ok(),
            "kp 5.33 should clear the bar on the ordered motor"
        );
    }

    #[test]
    fn nonsense_geometry_is_rejected() {
        let bad = RobotSpec::from_measurements(0.0, 150.0, 1024.0, 200.0);
        assert!(bad.check(&ControlGains::WAYPOINT).is_err());
        let no_encoder = RobotSpec::from_measurements(60.0, 150.0, 0.0, 200.0);
        assert!(no_encoder.check(&ControlGains::WAYPOINT).is_err());
    }

    /// The gap that made `check` worth revisiting: it validated `v_max`
    /// and ignored rotation, which is the axis that actually saturates.
    #[test]
    fn check_looks_at_the_turn_axis_not_just_forward_speed() {
        // Forward speed alone is comfortable — this is why the old check
        // passed a profile it should have had an opinion about.
        let roomy = ControlGains {
            heading_kp: 40.0, // 12/40 = 0.3 rad band, far inside ±90°
            ..ControlGains::WAYPOINT
        };
        assert!(
            roomy.v_max < RobotSpec::SIM_BOT.max_body_speed(),
            "the forward axis must be fine, or this proves nothing"
        );
        assert!(
            RobotSpec::SIM_BOT.check(&roomy).is_err(),
            "a kp that goes bang-bang while driving must be rejected"
        );
    }

    #[test]
    fn every_shipped_profile_is_physically_achievable() {
        // Including HIL, which used to be defined inline in the firmware
        // where nothing could check it.
        for (name, gains) in [
            ("WAYPOINT", ControlGains::WAYPOINT),
            ("VISUAL_SERVO", ControlGains::VISUAL_SERVO),
            ("HIL", ControlGains::HIL),
        ] {
            assert!(
                RobotSpec::SIM_BOT.check(&gains).is_ok(),
                "SIM_BOT rejects {name}: {:?}",
                RobotSpec::SIM_BOT.check(&gains)
            );
            assert!(
                RobotSpec::REAL_BOT.check(&gains).is_ok(),
                "REAL_BOT rejects {name}: {:?}",
                RobotSpec::REAL_BOT.check(&gains)
            );
        }
    }

    #[test]
    fn the_waypoint_band_covers_the_whole_driving_window() {
        // The margin the criterion actually passes on, spelled out so a
        // future gain change shows up as a number rather than a surprise.
        let band = RobotSpec::SIM_BOT.turn_proportional_band(&ControlGains::WAYPOINT);
        assert!(
            (band - 2.0).abs() < 1e-12,
            "12 rad/s / kp 6.0 = 2.0 rad, got {band}"
        );
        assert!(band > core::f64::consts::FRAC_PI_2, "2.0 rad > π/2 = 1.571");
    }

    /// `heading_d_limit` is written as the literal `12.0` with a comment
    /// saying it *is* `2·r·ω_max/L`. Now that the formula has a function,
    /// check the literal still matches it.
    #[test]
    fn the_d_limit_literal_equals_the_formula_it_claims_to_be() {
        let computed = RobotSpec::SIM_BOT.max_turn_rate();
        assert!(
            (computed - 12.0).abs() < 1e-12,
            "max_turn_rate = {computed}"
        );
        assert!((ControlGains::WAYPOINT.heading_d_limit - computed).abs() < 1e-12);
        assert!((ControlGains::VISUAL_SERVO.heading_d_limit - computed).abs() < 1e-12);
    }

    #[test]
    fn the_ordered_motor_is_slower_than_the_simulated_one() {
        // Documents the known mismatch rather than letting it be
        // rediscovered on hardware. SIM_BOT assumes ~286 RPM; docs/09
        // orders ~200 RPM.
        let ordered = RobotSpec::from_measurements(60.0, 150.0, 1024.0, 200.0);
        assert!(
            ordered.max_wheel_rad_s < RobotSpec::SIM_BOT.max_wheel_rad_s,
            "if these ever match, delete this test and the note on SIM_BOT"
        );
    }

    #[test]
    fn changing_real_bot_is_a_single_edit() {
        // `pico-robot` (the chip) and `hil-host` (the physics) both read
        // REAL_BOT, and `sim-run` reads SIM_BOT. This test documents the
        // split rather than enforcing it — its job is to make anyone
        // editing REAL_BOT aware that both halves of the HIL rig move
        // together, and the Stage 0 baseline deliberately does not.
        assert_eq!(
            RobotSpec::REAL_BOT,
            RobotSpec::SIM_BOT,
            "REAL_BOT has been measured — good. Expect the HIL numbers to \
             change, and check docs/07 records the new baseline. sim-run \
             stays on SIM_BOT so its regression test still means something."
        );
    }

    #[test]
    fn unsaturated_wheel_commands_pass_through_untouched() {
        let s = RobotSpec::SIM_BOT;
        let pass_through = |left, right| {
            assert_eq!(
                s.fit_wheels(WheelSpeeds::new(left, right)),
                WheelSpeeds::new(left, right)
            );
        };
        pass_through(10.0, -5.0);
        pass_through(0.0, 0.0);
        // Exactly at the limit is still fine.
        pass_through(s.max_wheel_rad_s, -s.max_wheel_rad_s);
    }

    #[test]
    fn saturated_commands_keep_their_ratio() {
        // THE point: (40, 10) must not become (30, 10). That would turn
        // twice as hard as asked.
        let s = RobotSpec::SIM_BOT; // 30 rad/s
        let fitted = s.fit_wheels(WheelSpeeds::new(40.0, 10.0));
        let (left, right) = (fitted.left, fitted.right);
        assert!((left - 30.0).abs() < 1e-12, "peak should sit at the limit");
        assert!((right - 7.5).abs() < 1e-12, "ratio should be preserved");
        assert!(
            ((left / right) - 4.0).abs() < 1e-9,
            "4:1 in must stay 4:1 out, got {left}:{right}"
        );
    }

    #[test]
    fn scaling_handles_the_negative_and_mixed_cases() {
        let s = RobotSpec::SIM_BOT;
        // Spin in place, over the limit both ways.
        let spun = s.fit_wheels(WheelSpeeds::new(-90.0, 90.0));
        assert!((spun.left + 30.0).abs() < 1e-12 && (spun.right - 30.0).abs() < 1e-12);
        // Only one wheel over: both still scale.
        let lopsided = s.fit_wheels(WheelSpeeds::new(60.0, -15.0));
        assert!((lopsided.left - 30.0).abs() < 1e-12);
        assert!(
            (lopsided.right + 7.5).abs() < 1e-12,
            "the in-range wheel scales too"
        );
    }

    #[test]
    fn nothing_ever_leaves_scaled_above_the_motor() {
        let s = RobotSpec::SIM_BOT;
        for (a, b) in [(200.0, 3.0), (-1.0, 400.0), (35.0, -35.0), (1e6, -1e6)] {
            let fitted = s.fit_wheels(WheelSpeeds::new(a, b));
            assert!(
                fitted.peak() <= s.max_wheel_rad_s + 1e-9,
                "{a},{b} -> {fitted:?}"
            );
        }
    }

    #[test]
    fn drive_matches_the_spec() {
        let d = RobotSpec::SIM_BOT.drive();
        assert_eq!(d.wheel_radius, RobotSpec::SIM_BOT.wheel_radius);
        assert_eq!(d.track_width, RobotSpec::SIM_BOT.track_width);
    }

    // Not a style assertion — this encodes the reason the two profiles
    // exist. If someone "unifies" them, this explains the cost.
    //
    // A `const` assertion rather than a `#[test]`: both sides are compile
    // -time constants, so this fails the BUILD rather than a test run.
    // You cannot merge a change that breaks it.
    const _: () =
        assert!(ControlGains::VISUAL_SERVO.heading_kp < ControlGains::WAYPOINT.heading_kp);
    const _: () =
        assert!(ControlGains::VISUAL_SERVO.heading_kd < ControlGains::WAYPOINT.heading_kd);
}
