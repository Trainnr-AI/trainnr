//! Turning a phone's thumb into a body twist, and deciding when to stop.
//!
//! The server and the serial port live in `main.rs`. Everything that can
//! be wrong *without a robot attached* lives here, because a control law
//! reachable only by holding a phone over a moving machine is a control
//! law nobody tests.
//!
//! # Two watchdogs, not one
//!
//! ```text
//!   phone ──WiFi──▶ laptop ──USB──▶ chip ──▶ motors
//!            │                 │
//!            │                 └── CommandWatchdog, 200 ms, ON THE CHIP
//!            └──────────────────── CommandWatchdog, 500 ms, here
//! ```
//!
//! The chip's watchdog is the one that actually stops the motors, and it
//! is deliberately short. The laptop therefore **sends continuously at
//! 50 Hz whatever the phone last said**, rather than forwarding phone
//! messages as they arrive.
//!
//! That indirection is the whole design. Phone WiFi jitters; a 300 ms gap
//! in a forwarded stream would trip the chip's 200 ms watchdog and read as
//! a stutter or a fault. Feeding the chip at a steady rate and judging the
//! *phone's* liveness on a slacker budget separates "the network hiccuped"
//! from "nobody is holding this any more".
//!
//! # Why the watchdog is `sim_core::CommandWatchdog`
//!
//! Because it already exists, is already tested, and is already the thing
//! running on the chip. It takes `now_ms` rather than reading a clock, so
//! it works unchanged on a host — and a second implementation of "has the
//! source gone quiet" is exactly the duplication this repo keeps paying
//! for. One definition, both hops.

use std::time::Instant;

use sim_core::{BodyTwist, CommandWatchdog, Freshness, RobotSpec};

/// How often the laptop repeats the current twist to the chip.
///
/// Four times inside the chip's 200 ms window, so three consecutive
/// datagrams can be lost before the wheels stop. Matches the rate
/// `twist_send` uses to demonstrate the failsafe.
pub const SEND_HZ: u64 = 50;

/// How long the phone may go quiet before the laptop starts commanding
/// zero.
///
/// Generously longer than the chip's 200 ms, because the two failures are
/// different: the chip is guarding against *the laptop dying*, which must
/// be caught fast, while this guards against *a thumb lifting or a phone
/// locking*, where half a second of coasting is not dangerous and a
/// hair-trigger makes the robot stutter on ordinary WiFi jitter.
pub const PHONE_TIMEOUT_MS: u64 = 500;

/// Fraction of the robot's physical top speed a full joystick deflection
/// commands.
///
/// ⚠️ Deliberately not 1.0. `RobotSpec::REAL_BOT` reports what the motors
/// can deliver, not what is sensible to hand a beginner with a
/// touchscreen in a room with furniture in it. Raise it once you have
/// watched it drive.
pub const SPEED_FRACTION: f64 = 0.35;

/// What the browser sends, `SEND_HZ`-ish times a second while a thumb is
/// down. Both axes are −1.0 to 1.0.
///
/// Named for the gesture rather than the physics — `forward`/`turn`, not
/// `v`/`w` — because the page that produces it knows about thumbs and
/// nothing about differential drive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stick {
    /// Positive is away from you — forward.
    pub forward: f64,
    /// Positive is **right**, as the thumb sees it. Note that this is the
    /// opposite sign to [`BodyTwist::turn_rate`]; see [`Stick::to_twist`].
    pub turn: f64,
}

impl Stick {
    pub const CENTRED: Stick = Stick {
        forward: 0.0,
        turn: 0.0,
    };

    /// Map a thumb position onto a body twist the chip will accept.
    ///
    /// # Why it clamps rather than trusting the page
    ///
    /// The input arrives from a browser over a socket. Anything can send
    /// anything: a bug in the page, a stale tab, a curl command, a NaN
    /// from a division by a zero-size touch area. The robot is not the
    /// place to find out. NaN in particular would propagate silently
    /// through the twist, through `DiffDrive::inverse`, and arrive at the
    /// motor duty as a number that compares false with everything —
    /// including the safety limits meant to catch it.
    pub fn to_twist(self, spec: &RobotSpec) -> BodyTwist {
        let top_speed = spec.max_body_speed() * SPEED_FRACTION;
        // A full-deflection spin, expressed as the body rotation you get
        // when the wheels run opposite at `SPEED_FRACTION` of top speed.
        let top_turn = 2.0 * top_speed / spec.track_width;
        BodyTwist {
            forward_speed: clamp_unit(self.forward) * top_speed,
            // ⚠️ NEGATED, and this is the whole reason `Stick` exists as a
            // separate type from `BodyTwist`.
            //
            // `turn_rate` is positive counter-CLOCKWISE, matching
            // `Pose::heading` and the right-hand rule. A thumb pushed
            // right means "go right", which is CLOCKWISE. Passing the
            // stick straight through made the robot steer away from the
            // direction of the push — an inversion that is obvious with a
            // robot in front of you and invisible in a duty percentage,
            // because `Status::duty_percent` is a magnitude.
            turn_rate: -clamp_unit(self.turn) * top_turn,
        }
    }
}

/// −1.0 to 1.0, with NaN treated as centred.
///
/// `f64::clamp` PANICS if either bound is NaN and silently passes NaN
/// through when the *value* is NaN, so it is the wrong tool here.
fn clamp_unit(x: f64) -> f64 {
    if x.is_nan() {
        return 0.0;
    }
    x.clamp(-1.0, 1.0)
}

/// What the laptop is currently telling the chip to do.
///
/// Holds the last thing the phone asked for and decides, at each tick,
/// whether that request is still fresh enough to act on.
pub struct Pilot {
    requested: Stick,
    phone: CommandWatchdog,
    spec: RobotSpec,
    /// When this pilot started, so that [`Pilot::now_ms`] is the ONE
    /// answer to "what time is it".
    ///
    /// ⚠️ This exists because the socket handler and the serial writer
    /// each created their own `Instant` and derived `now_ms` from it. The
    /// handler fed the watchdog at ~5,000 ms (its socket was young) while
    /// the writer asked `is_stale` at ~60,000 ms (its thread was old), so
    /// every command looked 55 seconds stale against a 500 ms timeout and
    /// the robot never moved. Two clocks that had to agree, with nothing
    /// comparing them.
    epoch: Instant,
}

impl Pilot {
    pub fn new(spec: RobotSpec) -> Self {
        Pilot {
            requested: Stick::CENTRED,
            // Starts stale, and that matters: a laptop that starts before
            // any phone connects must not command whatever happens to be
            // in `requested`. `CommandWatchdog` is built this way already.
            phone: CommandWatchdog::new(PHONE_TIMEOUT_MS),
            spec,
            epoch: Instant::now(),
        }
    }

    /// Milliseconds since this pilot started.
    ///
    /// Every caller of [`Pilot::steer`] and [`Pilot::command`] in the
    /// server reads the clock through here, so they cannot disagree about
    /// it. `steer` and `command` still TAKE `now_ms` rather than reading
    /// it themselves, which is what keeps them testable without a clock.
    pub fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// A message arrived from the browser.
    pub fn steer(&mut self, stick: Stick, now_ms: u64) {
        self.requested = stick;
        self.phone.feed(now_ms);
    }

    /// The phone said stop, or hung up.
    ///
    /// Distinct from letting the timeout expire: a released thumb or a
    /// closed socket is *known* to mean stop, and should not wait out
    /// [`PHONE_TIMEOUT_MS`] first.
    pub fn halt(&mut self) {
        self.requested = Stick::CENTRED;
    }

    /// What to send this tick, and why.
    ///
    /// Always returns a twist — never `None`. The chip must be fed
    /// continuously, and "stop" is a command, not an absence of one.
    ///
    /// The [`Freshness`] comes back so the caller can SAY when the reason
    /// is a clock fault rather than a quiet phone. Those were
    /// indistinguishable here on 2026-08-11, and the silence cost an
    /// evening: the answer "stale, command zero" was correct given its
    /// inputs and completely wrong about the world.
    pub fn command(&mut self, now_ms: u64) -> (BodyTwist, Freshness) {
        let freshness = self.phone.observe(now_ms);
        let twist = match freshness {
            Freshness::Fresh { .. } => self.requested.to_twist(&self.spec),
            // Every other verdict stops the robot. The distinction is for
            // the log, never for the actuator.
            _ => BodyTwist {
                forward_speed: 0.0,
                turn_rate: 0.0,
            },
        };
        (twist, freshness)
    }

    /// Whether the phone is currently considered present, for the status
    /// line the page displays.
    pub fn phone_present(&self, now_ms: u64) -> bool {
        !self.phone.is_stale(now_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bot() -> RobotSpec {
        RobotSpec::REAL_BOT
    }

    /// The sign convention, pinned. `BodyTwist::turn_rate` is positive
    /// counter-clockwise; a thumb pushed right means clockwise. Getting
    /// this backwards steers the robot away from the push, and no
    /// amount of staring at `duty_percent` reveals it — that field is a
    /// magnitude.
    #[test]
    fn pushing_the_stick_right_turns_the_robot_clockwise() {
        let right = Stick {
            forward: 0.0,
            turn: 1.0,
        }
        .to_twist(&bot());
        assert!(
            right.turn_rate < 0.0,
            "stick right must be clockwise (negative turn_rate), got {right:?}"
        );

        let left = Stick {
            forward: 0.0,
            turn: -1.0,
        }
        .to_twist(&bot());
        assert!(left.turn_rate > 0.0, "stick left must be counter-clockwise");
        assert_eq!(right.turn_rate, -left.turn_rate, "and symmetric");
    }

    /// Pulling back must reverse, not brake to zero.
    #[test]
    fn pulling_the_stick_back_commands_reverse() {
        let back = Stick {
            forward: -1.0,
            turn: 0.0,
        }
        .to_twist(&bot());
        assert!(back.forward_speed < 0.0, "got {back:?}");
    }

    #[test]
    fn a_centred_stick_commands_nothing() {
        let t = Stick::CENTRED.to_twist(&bot());
        assert_eq!(t.forward_speed, 0.0);
        assert_eq!(t.turn_rate, 0.0);
    }

    #[test]
    fn full_deflection_stays_under_what_the_motors_can_deliver() {
        let t = Stick {
            forward: 1.0,
            turn: 0.0,
        }
        .to_twist(&bot());
        assert!(
            t.forward_speed < bot().max_body_speed(),
            "teleop must not command the motors' ceiling: {t:?}"
        );
    }

    /// The page is untrusted input. A stale tab, a bug, or a curl command
    /// must not be able to ask for more than full deflection.
    #[test]
    fn an_over_range_stick_is_clamped_not_believed() {
        let sane = Stick {
            forward: 1.0,
            turn: 1.0,
        }
        .to_twist(&bot());
        let absurd = Stick {
            forward: 50.0,
            turn: -50.0,
        }
        .to_twist(&bot());
        assert_eq!(absurd.forward_speed, sane.forward_speed);
        assert_eq!(absurd.turn_rate, -sane.turn_rate);
    }

    /// NaN is the dangerous one: it would pass every `>` and `<` check
    /// downstream, including the ones meant to catch it.
    #[test]
    fn nan_is_treated_as_centred_rather_than_propagated() {
        let t = Stick {
            forward: f64::NAN,
            turn: f64::NAN,
        }
        .to_twist(&bot());
        assert_eq!(t.forward_speed, 0.0);
        assert_eq!(t.turn_rate, 0.0);
    }

    /// The property the whole two-watchdog design exists for.
    #[test]
    fn a_pilot_nobody_has_touched_commands_zero() {
        let mut p = Pilot::new(bot());
        let (t, freshness) = p.command(0);
        assert_eq!(t.forward_speed, 0.0);
        assert_eq!(t.turn_rate, 0.0);
        assert_eq!(freshness, Freshness::NeverFed, "and it says WHY");
        assert!(!p.phone_present(0));
    }

    #[test]
    fn a_fed_pilot_drives_and_then_stops_when_the_phone_goes_quiet() {
        let mut p = Pilot::new(bot());
        p.steer(
            Stick {
                forward: 1.0,
                turn: 0.0,
            },
            1_000,
        );
        assert!(p.command(1_000).0.forward_speed > 0.0, "should be driving");
        assert!(
            p.command(1_000 + PHONE_TIMEOUT_MS - 1).0.forward_speed > 0.0,
            "still inside the window"
        );
        assert_eq!(
            p.command(1_000 + PHONE_TIMEOUT_MS).0.forward_speed,
            0.0,
            "phone went quiet — must command zero"
        );
    }

    /// A released thumb is *known* to mean stop and must not wait out the
    /// timeout first.
    #[test]
    fn an_explicit_halt_stops_immediately_rather_than_waiting_for_the_timeout() {
        let mut p = Pilot::new(bot());
        p.steer(
            Stick {
                forward: 1.0,
                turn: 0.0,
            },
            1_000,
        );
        p.halt();
        assert_eq!(p.command(1_000).0.forward_speed, 0.0);
    }

    /// The laptop must feed the chip several times inside its window, or
    /// a single dropped message reads as a dead host.
    #[test]
    fn the_send_rate_fits_several_times_inside_the_chips_watchdog() {
        const CHIP_TIMEOUT_MS: u64 = 200; // firmware/pico-odom
        let period_ms = 1_000 / SEND_HZ;
        assert!(
            CHIP_TIMEOUT_MS / period_ms >= 4,
            "only {} sends fit inside the chip's window",
            CHIP_TIMEOUT_MS / period_ms
        );
        assert!(
            PHONE_TIMEOUT_MS > CHIP_TIMEOUT_MS,
            "the phone's budget must be slacker than the chip's, or WiFi \
             jitter reads as a dead laptop"
        );
    }
}
