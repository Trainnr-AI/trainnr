//! Failsafes: what the robot does when it stops hearing from whatever is
//! supposed to be telling it what to do.
//!
//! # Why this exists before the hardware does
//!
//! In the HIL rig the chip does not drive motors — it sends a duty over
//! UART and the *host* simulates the consequences. So if the link dies,
//! nothing moves, and the missing failsafe is invisible.
//!
//! At H4 the Pico drives a TB6612 directly and owns the motor outputs.
//! From that moment "the link died and the last command stands" means a
//! robot that keeps driving into whatever is in front of it, indefinitely,
//! with nobody able to stop it. That is the classic runaway failure, and
//! the time to fix it is before there is a motor attached.
//!
//! # Time is a parameter, not a dependency
//!
//! [`CommandWatchdog`] never reads a clock. The caller passes the current
//! time in, exactly like [`crate::Odometry`] takes tick deltas rather than
//! reading an encoder. That keeps it `no_std`, keeps it deterministic, and
//! makes every edge case below testable without waiting in real time.

/// Milliseconds since some fixed origin. `u64` at millisecond resolution
/// does not wrap for ~584 million years, so wrap-around is not modelled.
pub type Millis = u64;

/// Zeroes a command when its source goes quiet.
///
/// Feed it whenever a valid command arrives; ask it what to output every
/// control tick. While the source is fresh it passes commands through
/// unchanged; once `timeout_ms` elapses with no feed it outputs zero and
/// keeps doing so until fed again.
///
/// # Choosing a timeout
///
/// Long enough that normal jitter never trips it, short enough that a
/// runaway is caught before it travels far. At our 50 Hz control rate and
/// 0.45 m/s top speed, 200 ms is ten missed ticks and about 9 cm of
/// travel — comfortably beyond any real scheduling hiccup, comfortably
/// less than a table edge.
use crate::{BodyTwist, RobotSpec};

#[derive(Debug, Clone, Copy)]
pub struct CommandWatchdog {
    timeout_ms: Millis,
    last_fed_ms: Option<Millis>,
}

impl CommandWatchdog {
    pub const fn new(timeout_ms: Millis) -> Self {
        CommandWatchdog {
            timeout_ms,
            last_fed_ms: None,
        }
    }

    /// Record that a valid command arrived at `now_ms`.
    pub fn feed(&mut self, now_ms: Millis) {
        self.last_fed_ms = Some(now_ms);
    }

    /// Has the source gone quiet?
    ///
    /// **A watchdog that has never been fed is stale.** Starting in the
    /// safe state matters: a robot that powers up before its controller
    /// does must not treat the gap as "no news is good news" and act on
    /// whatever happens to be in its command registers.
    pub fn is_stale(&self, now_ms: Millis) -> bool {
        match self.last_fed_ms {
            None => true,
            // saturating_sub, not `-`: if a caller ever passes a `now`
            // earlier than the last feed (a clock correction, a test),
            // plain subtraction underflows and panics on this target.
            // Treating it as "no time has passed" keeps the robot moving
            // rather than making it stutter on a clock glitch.
            Some(last) => now_ms.saturating_sub(last) >= self.timeout_ms,
        }
    }

    /// Milliseconds since the last feed, or `None` if never fed.
    pub fn age_ms(&self, now_ms: Millis) -> Option<Millis> {
        self.last_fed_ms.map(|last| now_ms.saturating_sub(last))
    }

    /// The command to actually apply: `command` while fresh, all-zero once
    /// stale.
    ///
    /// Generic over the pair so it serves wheel duties (`i32`), body
    /// twists (`f64`), or anything else with a zero.
    pub fn gate<T: Default>(&self, now_ms: Millis, command: (T, T)) -> (T, T) {
        if self.is_stale(now_ms) {
            (T::default(), T::default())
        } else {
            command
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT: Millis = 200;

    fn fed_at(t: Millis) -> CommandWatchdog {
        let mut w = CommandWatchdog::new(TIMEOUT);
        w.feed(t);
        w
    }

    #[test]
    fn a_watchdog_that_has_never_been_fed_is_stale() {
        // Power-up before the controller is ready must NOT be read as
        // "no news is good news".
        let w = CommandWatchdog::new(TIMEOUT);
        assert!(w.is_stale(0));
        assert!(w.is_stale(1_000_000));
        assert_eq!(w.gate(0, (500i32, -500)), (0, 0));
    }

    #[test]
    fn a_freshly_fed_watchdog_passes_the_command_through() {
        let w = fed_at(1_000);
        assert!(!w.is_stale(1_000));
        assert_eq!(w.gate(1_050, (500i32, -500)), (500, -500));
    }

    #[test]
    fn the_command_survives_right_up_to_the_timeout() {
        let w = fed_at(1_000);
        assert!(!w.is_stale(1_000 + TIMEOUT - 1), "tripped one ms early");
        assert_eq!(w.gate(1_000 + TIMEOUT - 1, (500i32, 500)), (500, 500));
    }

    #[test]
    fn the_command_is_zeroed_exactly_at_the_timeout() {
        // Boundary pinned deliberately: >= not >. An off-by-one here is a
        // failsafe that fires a tick late, which is the tick that matters.
        let w = fed_at(1_000);
        assert!(w.is_stale(1_000 + TIMEOUT));
        assert_eq!(w.gate(1_000 + TIMEOUT, (500i32, 500)), (0, 0));
    }

    #[test]
    fn a_long_silence_stays_stopped() {
        let w = fed_at(1_000);
        for t in [1_500, 10_000, 1_000_000] {
            assert!(w.is_stale(t));
            assert_eq!(w.gate(t, (900i32, 900)), (0, 0));
        }
    }

    #[test]
    fn feeding_again_recovers() {
        // The link coming back must restore control, not latch the robot
        // off until a power cycle.
        let mut w = fed_at(1_000);
        assert!(w.is_stale(2_000));
        w.feed(2_000);
        assert!(!w.is_stale(2_050));
        assert_eq!(w.gate(2_050, (300i32, 300)), (300, 300));
    }

    #[test]
    fn a_backwards_clock_does_not_panic_or_trip() {
        // On this target a plain `now - last` underflows and panics when
        // `now < last`. saturating_sub reads it as "no time passed", which
        // keeps the robot running through a clock glitch rather than
        // stuttering.
        let w = fed_at(5_000);
        assert!(!w.is_stale(4_000));
        assert_eq!(w.age_ms(4_000), Some(0));
        assert_eq!(w.gate(4_000, (100i32, 100)), (100, 100));
    }

    #[test]
    fn age_reports_none_before_the_first_feed() {
        assert_eq!(CommandWatchdog::new(TIMEOUT).age_ms(1_000), None);
        assert_eq!(fed_at(1_000).age_ms(1_150), Some(150));
    }

    #[test]
    fn gating_works_for_floats_as_well_as_duties() {
        // Same failsafe upstream of the kinematics, on a body twist.
        let w = fed_at(1_000);
        assert_eq!(w.gate(1_050, (0.45f64, 1.2)), (0.45, 1.2));
        assert_eq!(w.gate(9_999, (0.45f64, 1.2)), (0.0, 0.0));
    }

    #[test]
    fn a_zero_timeout_is_always_stale() {
        // Degenerate config: document it rather than pretend it cannot
        // happen. Always-stopped is the safe reading of "timeout 0".
        let w = fed_at(1_000);
        let z = CommandWatchdog { timeout_ms: 0, ..w };
        assert!(z.is_stale(1_000));
        assert_eq!(z.gate(1_000, (500i32, 500)), (0, 0));
    }
}

/// Notices that the robot is commanded to move and is not moving, and
/// backs it out.
///
/// # Why this exists
///
/// On 2026-08-10, measuring the real motor made the robot 3.86x slower,
/// which changed the path it took through the U-trap. It routed south
/// instead of north, nosed into a corner, and **stayed there** — pose
/// identical to two decimals for the remaining 40 seconds, 28,000 ticks
/// of wall contact.
///
/// The speed was not the bug. Isolating one variable at a time showed
/// that *any* perturbation which changes the route can wedge it: dropping
/// the derivative gain alone did it, and so did cruise speed 0.45 -> 0.30
/// with nothing else touched. **The U-trap had been passing because one
/// particular set of numbers happened to produce a path that never
/// wedged.** A green tick was measuring luck.
///
/// Nothing in the stack could recover, because a differential drive that
/// only ever drives forward has no move that gets it out of a corner.
/// Turning in place against a wall does not help: the wheels turn, the
/// odometry integrates, and the robot stays exactly where it is — which
/// is also why drift ran to 47 m. Reversing is the only escape, and
/// reversing is the one thing nothing ever asked for.
///
/// # How it decides
///
/// Commanded motion against measured motion — the same comparison that
/// found the inverted wheel the same day, and the same shape as
/// [`CommandWatchdog`]: **it never reads a clock or a sensor.** The
/// caller supplies both facts each tick, so it is testable without a
/// robot and identical in the simulator, the host and the firmware.
#[derive(Debug, Clone, Copy)]
pub struct StuckMonitor {
    /// Ticks of "asked to move, didn't" before declaring the robot stuck.
    patience_ticks: u32,
    /// How long an escape lasts once triggered.
    escape_ticks: u32,
    /// Speed below which the robot counts as not moving, in the same
    /// units as the speeds passed to [`Self::update`].
    ///
    /// A **detection threshold only.** It used to also set how hard the
    /// robot reversed, via an `escape_speed = still_speed * 20.0` that
    /// nobody would predict: lowering the noise floor would have quietly
    /// made every escape gentler. The escape magnitude is now given
    /// explicitly by the caller, which is the only place that knows what
    /// the robot can do.
    still_speed: f64,
    /// How to back out: reverse speed (positive; the sign is applied for
    /// you) and turn rate, in the caller's own units.
    escape: (f64, f64),
    no_progress: u32,
    escaping: u32,
    /// Consecutive escapes without a decent run of progress in between.
    ///
    /// **The first version of this had no such counter and it was not
    /// enough.** Backing out once and turning by a fixed amount left the
    /// planner aimed at the same corner, so the robot escaped, drove
    /// straight back in, escaped again — a limit cycle that reduced the
    /// wall contact by 95% and still never finished. An escape that is
    /// identical every time is a retry.
    attempts: u32,
    /// Ticks of real progress since the last escape, used to decide when
    /// the ladder has done its job and can reset.
    progress: u32,
}

impl StuckMonitor {
    /// `patience_ticks` at 50 Hz: 25 is half a second of trying and
    /// failing, which is long enough that a momentary stall on carpet or
    /// a motor's own lag never triggers it, and short enough that the
    /// robot does not grind.
    /// `escape` is `(reverse_speed, turn_rate)`. **Pick both from the
    /// robot's own limits, not by feel.** A reverse faster than the robot
    /// ever drives forward is a surprise waiting on a real floor, and a
    /// turn rate above what the wheels can deliver just gets scaled away.
    /// The policy, derived from the robot and the loop it runs in —
    /// **the one place that knows how to build one of these.**
    ///
    /// It was built by hand in two places: `sim-run`'s `Mission` at 50 Hz
    /// and `chase`'s camera loop at ~15 Hz, each converting seconds into
    /// its own ticks and each deriving the escape magnitudes from the spec
    /// separately. Two copies of one policy, differing only in a rate — so
    /// a change to the policy meant remembering the other one existed.
    ///
    /// The durations are the swept values: **0.5 s of trying and failing**
    /// before backing out, then **2.4 s of backing out.** The second is
    /// the load-bearing one. Against four configurations that each wedged
    /// the robot permanently, a 1 s escape cleared 2 of 4 and a 2.4 s
    /// escape cleared 4 of 4, while the patience barely mattered — because
    /// what clears a corner is the DISTANCE reversed, and distance is
    /// speed times duration.
    pub fn for_robot(spec: &RobotSpec, loop_hz: f64) -> Self {
        let ticks = |seconds: f64| ((seconds * loop_hz) as u32).max(1);
        StuckMonitor::new(
            ticks(0.5),
            ticks(2.4),
            // A centimetre per second: below any real commanded motion,
            // above the numerical noise of a robot pressed against a wall.
            0.01,
            (
                // Near TOP speed, not cruise. At half cruise the real
                // robot backed off 0.14 m in 2.4 s — under two robot radii
                // — and never cleared the corner. 0.85 of top gives ~5.
                spec.max_body_speed() * 0.85,
                spec.max_turn_rate() / 3.0,
            ),
        )
    }

    pub const fn new(
        patience_ticks: u32,
        escape_ticks: u32,
        still_speed: f64,
        escape: (f64, f64),
    ) -> Self {
        StuckMonitor {
            patience_ticks,
            escape_ticks,
            still_speed,
            escape,
            no_progress: 0,
            escaping: 0,
            attempts: 0,
            progress: 0,
        }
    }

    /// Feed one tick. Returns `Some(escape)` while backing out, `None`
    /// when the caller's own command should be used.
    ///
    /// `commanded_speed` is what the controller asked for and
    /// `measured_speed` is what actually happened — both as magnitudes.
    /// On a real robot the second comes from encoders; in the simulator
    /// it comes from the physics. **It must not come from odometry
    /// integrated off the commanded value**, which would agree with
    /// itself and never notice anything.
    pub fn update(&mut self, commanded_speed: f64, measured_speed: f64) -> Option<BodyTwist> {
        if self.escaping == 0 {
            self.detect(commanded_speed, measured_speed);
        }
        if self.escaping > 0 {
            self.escaping -= 1;
            // Back out AND turn. The turn is what makes it an escape
            // rather than a retry: reversing along the line it came in on
            // leaves the planner aimed at the same corner.
            //
            // The direction alternates and the duration grows with each
            // consecutive attempt — a recovery ladder. Escaping the same
            // way every time just retraces the path that failed.
            return Some(BodyTwist {
                forward_speed: -self.escape_speed(),
                turn_rate: self.escape_turn(),
            });
        }

        None
    }

    /// The detection half, split out so an escape begins on the tick it is
    /// triggered rather than the one after. A robot that has been wedged
    /// for half a second should not spend one more tick pushing.
    fn detect(&mut self, commanded_speed: f64, measured_speed: f64) {
        let asked_to_move = commanded_speed.abs() > self.still_speed;
        let is_moving = measured_speed.abs() > self.still_speed;
        if asked_to_move && !is_moving {
            self.no_progress += 1;
            self.progress = 0;
            if self.no_progress >= self.patience_ticks {
                self.no_progress = 0;
                self.attempts = self.attempts.saturating_add(1);
                // Each rung backs out longer, up to 4x the base. Bounded
                // because an unbounded reverse is its own hazard: the
                // robot cannot see behind itself.
                self.escaping = self.escape_ticks * self.attempts.min(4);
            }
        } else {
            // Any progress at all resets the patience. A robot inching
            // out of a tight spot is working, not stuck.
            self.no_progress = 0;
            if is_moving {
                self.progress += 1;
                // Sustained progress means the ladder worked; start again
                // from the bottom rung next time. Without this the robot
                // would carry an escalating reverse for the rest of the
                // run because of one corner it left minutes ago.
                if self.progress >= self.patience_ticks * 4 {
                    self.attempts = 0;
                    self.progress = 0;
                }
            }
        }
    }

    /// True while an escape is in progress — for telemetry, so a viewer
    /// can show *why* the robot suddenly drove backwards.
    pub fn is_escaping(&self) -> bool {
        self.escaping > 0
    }

    fn escape_speed(&self) -> f64 {
        self.escape.0.abs()
    }

    /// Alternates with each consecutive attempt, so a failed escape is
    /// never repeated identically.
    fn escape_turn(&self) -> f64 {
        if self.attempts % 2 == 0 {
            self.escape.1
        } else {
            -self.escape.1
        }
    }
}

#[cfg(test)]
mod stuck_tests {
    use super::*;

    fn wedged() -> StuckMonitor {
        StuckMonitor::new(3, 5, 0.01, (0.2, 1.0))
    }

    #[test]
    fn moving_normally_never_triggers() {
        let mut m = wedged();
        for _ in 0..100 {
            assert!(m.update(0.3, 0.29).is_none());
        }
        assert!(!m.is_escaping());
    }

    /// The signal is commanded-without-measured. A robot legitimately
    /// stopped — commanded zero, measured zero — is not stuck, and an
    /// escape there would drive a parked robot backwards.
    #[test]
    fn a_deliberately_stopped_robot_is_not_stuck() {
        let mut m = wedged();
        for _ in 0..100 {
            assert!(m.update(0.0, 0.0).is_none());
        }
    }

    #[test]
    fn commanded_but_not_moving_backs_out() {
        let mut m = wedged();
        assert!(m.update(0.3, 0.0).is_none());
        assert!(m.update(0.3, 0.0).is_none());
        let escape = m.update(0.3, 0.0);
        assert!(
            escape.is_some(),
            "3 ticks of patience should have expired, and the escape starts \
             on the tick it triggers"
        );
        assert!(
            escape.map(|e| e.forward_speed < 0.0).unwrap_or(false),
            "an escape must REVERSE — forward is the direction that failed"
        );
    }

    /// The bug the first version shipped with: escaping identically every
    /// time is a retry, and the robot re-entered the same corner forever.
    #[test]
    fn consecutive_escapes_turn_opposite_ways() {
        let mut m = wedged();
        for _ in 0..2 {
            m.update(0.3, 0.0);
        }
        let first = m.update(0.3, 0.0).map(|e| e.turn_rate).unwrap_or(0.0);
        while m.is_escaping() {
            m.update(0.3, 0.0);
        }
        for _ in 0..2 {
            m.update(0.3, 0.0);
        }
        let second = m.update(0.3, 0.0).map(|e| e.turn_rate).unwrap_or(0.0);
        assert!(
            first * second < 0.0,
            "escapes turned the same way twice: {first} then {second}"
        );
    }

    #[test]
    fn sustained_progress_resets_the_ladder() {
        let mut m = wedged();
        for _ in 0..2 {
            m.update(0.3, 0.0);
        }
        let first = m.update(0.3, 0.0).map(|e| e.turn_rate).unwrap_or(0.0);
        while m.is_escaping() {
            m.update(0.3, 0.0);
        }
        // Long run of real motion: the corner is behind us.
        for _ in 0..200 {
            m.update(0.3, 0.3);
        }
        for _ in 0..2 {
            m.update(0.3, 0.0);
        }
        let after_reset = m.update(0.3, 0.0).map(|e| e.turn_rate).unwrap_or(0.0);
        assert!(
            after_reset * first > 0.0,
            "after a clean run the ladder should start at the bottom rung \
             again — first escape turned {first}, this one {after_reset}"
        );
    }
}
