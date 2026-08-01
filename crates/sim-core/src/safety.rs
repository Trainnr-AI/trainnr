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
