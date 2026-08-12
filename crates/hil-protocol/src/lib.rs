//! The hardware-in-the-loop wire protocol — one definition, both ends.
//!
//! The chip decides; the host owns physics. Every 20 ms:
//!
//! ```text
//!   host -> chip   G <x> <y> <budget>     steer here, at most this fast
//!   host -> chip   R <x> <y> <budget>     the same, but reset state first
//!   host -> chip   T <v> <w>              or: just do exactly this
//!   chip -> host   P <x> <y> <heading>     believed pose (display only)
//!   chip -> host   M <duty_l> <duty_r>   motor command, ±1000
//!   host -> chip   S <dl> <dr>           encoder tick deltas
//!   host -> chip   I <x> <y> <heading>     begin a session here (once, first)
//!   chip -> host   H <worst_us>         new worst-case compute time
//!   chip -> host   J <i> <ticks> <mrad> <duty> <phase>   one joint's tick
//! ```
//!
//! # Why the host sends the goal
//!
//! The chip cannot plan. `OccupancyGrid` is a `Vec` and `plan()` uses a
//! `BinaryHeap`; bare metal has no allocator. So mapping and A* stay on
//! the host — which is not a workaround, it is the two-tier architecture
//! docs/00 describes and every real robot uses:
//!
//! ```text
//!   Tier 2 (host)   vision, mapping, planning     "steer at (x, y)"
//!   Tier 1 (chip)   hard-real-time control loop   "wheels do this"
//! ```
//!
//! `G` is the seam between those tiers. Before it existed the chip carried
//! a hardcoded waypoint list it could not possibly plan around, and the
//! simulator and the HIL rig ran different missions in different worlds —
//! not the digital twins they claimed to be.
//!
//! # Why this is a crate and not two `split_whitespace` calls
//!
//! It *was* two `split_whitespace` calls — one in `hil-host`, one in
//! `pico-robot`. Two hand-rolled parsers for one format is the classic
//! setup for a protocol bug that presents as a physics bug: the host reads
//! a field the chip never meant to send, the robot drives somewhere
//! strange, and you go looking in the control law. Neither side could be
//! tested, because each was a private function inside a `main.rs`.
//!
//! Now both ends call the same encoder and the same parser, and the format
//! has tests. See docs/10-hil-protocol.md and docs/13-architecture-review.md
//! (item 6).
//!
//! # Why ASCII and not `postcard`
//!
//! Deliberate, and documented in docs/10: a human can read this protocol
//! in a terminal, which is worth a great deal while learning. The types
//! below are the seam that makes swapping in a binary encoding later a
//! change to two functions rather than to both programs — the encoding
//! lives in [`Message::write_into`] and [`Message::parse`], nowhere else.
//!
//! # no_std, no alloc
//!
//! Compiled for the Mac and for ARM Cortex-M. Encoding writes into a
//! caller-supplied [`core::fmt::Write`] (a `heapless::String` on the chip,
//! a `String` or socket on the host) so this crate never allocates.

// `host` pulls in `serialport`, which is std-only. Firmware never
// enables it, so the no_std discipline this crate exists under is
// unchanged for the target that actually needs it.
#![cfg_attr(not(any(test, feature = "host")), no_std)]
#![forbid(unsafe_code)]

use core::fmt::Write;
use sim_core::{Directive, Point};

/// Motor commands are clamped to this range on both ends. A duty of
/// `±DUTY_FULL` means "full commanded wheel speed".
///
/// Re-exported from `sim-core`, where `RobotSpec::duty` — the only thing
/// that produces a duty — can name it. Defining it here left the encoder
/// unable to reference its own limit, so the literal `1000` ended up
/// written three times across two crates.
pub use sim_core::DUTY_FULL;

/// What a joint was doing when it reported.
///
/// On the wire as a **word**, not a number. A `2` would need a table
/// somewhere to read, and that table is the kind of thing that lives in
/// one place and rots in another — the shape of most bugs in this repo.
/// Six extra bytes buys a line anybody can read without a decoder ring.
///
/// This is the same argument `Verdict` makes one layer up: a hold that
/// says *why* sends you to the right connector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointPhase {
    /// Creeping toward a hard stop, looking for a zero.
    Homing,
    /// Found the stop and adopted it. Reported once, on the tick it
    /// happened.
    Homed,
    /// Under closed-loop control against a known zero, with the guard
    /// authorising motion.
    Holding,
    /// The guard **refused** this tick, so no new target was written and
    /// the joint is keeping the last one it was given. Distinct from
    /// [`Self::NoZero`]: the calibration is fine, the *commander* is not.
    ///
    /// This is the arm's failsafe seen from outside — and it is the
    /// opposite of the base's, where the same silence cuts the outputs.
    Held,
    /// Homing gave up. **No zero was adopted**, so the angle field is
    /// meaningless and the outputs are off.
    NoZero,
}

impl core::fmt::Display for JointPhase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            JointPhase::Homing => "homing",
            JointPhase::Homed => "homed",
            JointPhase::Holding => "holding",
            JointPhase::Held => "held",
            JointPhase::NoZero => "nozero",
        })
    }
}

impl core::str::FromStr for JointPhase {
    type Err = ();
    fn from_str(word: &str) -> Result<Self, Self::Err> {
        match word {
            "homing" => Ok(JointPhase::Homing),
            "homed" => Ok(JointPhase::Homed),
            "holding" => Ok(JointPhase::Holding),
            "held" => Ok(JointPhase::Held),
            "nozero" => Ok(JointPhase::NoZero),
            _ => Err(()),
        }
    }
}

/// One message in either direction.
///
/// A single enum rather than one per direction: the format is small, and
/// a shared type means a test can round-trip *every* variant without
/// knowing which way it travels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Message {
    /// `G x y budget` — steer at this point, at most this fast.
    ///
    /// Host to chip, recomputed every tick, so it silently carries
    /// replanning: the host may move the point around an obstacle and the
    /// chip simply follows, knowing nothing about the map.
    ///
    /// **`budget` is separate from the point on purpose.** The point is a
    /// *lookahead* — deliberately close, so steering is smooth. The speed
    /// must come from distance to the REAL goal, or the robot crawls
    /// whenever the next path node happens to be nearby. That is why
    /// `GotoController::steer` takes a budget rather than deriving one,
    /// and this message mirrors it. (Discovered the hard way: the chip
    /// stalled 1.28 m short of the goal, commanding zero duty.)
    Goal {
        x: f64,
        y: f64,
        budget: f64,
        /// Drop accumulated controller state first — see
        /// [`sim_core::Directive::Steer::fresh`].
        fresh: bool,
    },
    /// `T v w` — a body twist to apply directly, bypassing the chip's
    /// controller. Forward speed in m/s, turn rate in rad/s.
    ///
    /// For decisions the chip *cannot* make because they depend on
    /// sensors it does not have. The obstacle reflex is the case: it needs
    /// a depth scan, the scan lives on the host (and on the real robot,
    /// on the Jetson), so the host computes the escape twist and the chip
    /// simply obeys.
    ///
    /// The split is honest rather than convenient — planning and reactive
    /// sensing sit in Tier 2, the real-time control loop in Tier 1. Adding
    /// this fixed a rig that ground along a wall for 1019 ticks because
    /// the reflex had no way to reach the motors.
    Twist { v: f64, w: f64 },
    /// `P x y heading` — the chip's believed pose, in metres and radians.
    /// Display only; the host never steers with it.
    Pose { x: f64, y: f64, heading: f64 },
    /// `M duty_l duty_r` — motor command, each in `[-DUTY_FULL, DUTY_FULL]`.
    Motor { duty_l: i32, duty_r: i32 },
    /// `S dl dr` — encoder tick deltas since the previous step. Signed:
    /// a reversing wheel counts down.
    Sensors { dl: i64, dr: i64 },
    /// host → chip: **begin a session here.** Sets the believed pose and
    /// clears every scrap of accumulated state.
    ///
    /// # Why the chip cannot just know
    ///
    /// The firmware used to initialise its pose once, at boot, to a
    /// hardcoded start. That is correct exactly once. A board left powered
    /// between runs — which is every board on a bench, and every robot
    /// that is restarted without a power cycle — begins the next session
    /// believing it is wherever the last one left it.
    ///
    /// Found by replaying a recorded hardware session: the second line of
    /// the log was the chip announcing it was at (6.38, 3.09), the finish
    /// of the *previous* run, and commanding a full-speed spin to correct
    /// an error that did not exist. 0/1 waypoints, 4.97 m drift, 2036 wall
    /// bumps.
    ///
    /// Sent once, before the control loop, so it never shares the UART
    /// FIFO with an `S` line.
    Start { x: f64, y: f64, heading: f64 },
    /// chip → host: a NEW worst-case control-loop compute time, in
    /// microseconds.
    ///
    /// Sent only when the record is beaten, so a healthy run costs a
    /// handful of lines rather than one per tick. That matters: this
    /// direction is not FIFO-constrained, but a per-tick line would still
    /// be 1139 lines of noise to say nothing changed.
    ///
    /// # Why the chip reports this at all
    ///
    /// The 50 Hz deadline was measured **once**, on a bench, at 198 us
    /// against a 20 000 us budget. That is a snapshot of one binary on one
    /// day. When the loop grows — reading encoders, driving an H-bridge,
    /// a heavier planner — nothing would notice it eating the budget until
    /// the robot started missing ticks in a way that looks like bad
    /// tuning. Reporting it makes the headroom a continuously checked
    /// property instead of a remembered number.
    Health { worst_us: u32 },

    /// `J <index> <raw_ticks> <milliradians> <duty> <phase>` — one arm
    /// joint's state for one tick.
    ///
    /// # Why this lives on the SAME wire as the base's messages
    ///
    /// One vocabulary means one parser, one set of round-trip tests, and —
    /// the part that matters — record and replay for free. A hardware
    /// session with an arm becomes a committed fixture the way
    /// `rp2350-utrap.wire` did, rather than scrollback somebody described
    /// afterwards from memory.
    ///
    /// # Why milliradians and not radians
    ///
    /// The chip formats this, and `f64` formatting drags a float
    /// formatter into a binary that otherwise has none. Integer
    /// milliradians costs nothing and is exact — 1 mrad is ~0.057°, far
    /// finer than a 4290-count encoder resolves.
    ///
    /// `raw_ticks` rides along beside the angle deliberately: they are the
    /// same fact through two conversions, and when they disagree the fault
    /// is the zero, the tick count, or the gear ratio — which is exactly
    /// what a homing bug looks like.
    Joint {
        index: u8,
        raw_ticks: i64,
        milliradians: i32,
        duty: i32,
        phase: JointPhase,
    },
}

/// Why a line could not be parsed.
///
/// Distinct variants rather than a bare `None`, because "I received a
/// message I don't know" and "I received a corrupt message" call for
/// different responses: the first is a version mismatch, the second is a
/// dropped byte on the UART.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// The line was empty or only whitespace.
    Empty,
    /// The leading tag was not one of `P`, `M`, `S`.
    UnknownTag,
    /// The tag was recognised but a field was missing.
    MissingField,
    /// A field was present but not a number.
    BadNumber,
}

impl Message {
    /// Serialise as one line, **including** the trailing newline.
    ///
    /// Pose is written to 4 decimals: at our scale that is 0.1 mm, well
    /// under any real robot's odometry error, and it keeps a line short
    /// enough for a small UART buffer.
    pub fn write_into<W: Write>(&self, w: &mut W) -> core::fmt::Result {
        match *self {
            Message::Goal {
                x,
                y,
                budget,
                fresh,
            } => {
                // The reset bit rides in the TAG, not as a field. See the
                // FIFO note on this module: the wire has a hard byte
                // budget, and `R` costs nothing where ` 1` cost two.
                let tag = if fresh { 'R' } else { 'G' };
                writeln!(w, "{tag} {x:.4} {y:.4} {budget:.4}")
            }
            Message::Twist { v, w: tw } => writeln!(w, "T {v:.4} {tw:.4}"),
            Message::Pose { x, y, heading } => writeln!(w, "P {x:.4} {y:.4} {heading:.4}"),
            Message::Motor { duty_l, duty_r } => writeln!(w, "M {duty_l} {duty_r}"),
            Message::Sensors { dl, dr } => writeln!(w, "S {dl} {dr}"),
            Message::Start { x, y, heading } => writeln!(w, "I {x:.4} {y:.4} {heading:.4}"),
            Message::Health { worst_us } => writeln!(w, "H {worst_us}"),
            Message::Joint {
                index,
                raw_ticks,
                milliradians,
                duty,
                phase,
            } => writeln!(w, "J {index} {raw_ticks} {milliradians} {duty} {phase}"),
        }
    }

    /// Parse one line. Leading/trailing whitespace and a trailing `\r` or
    /// `\n` are tolerated — serial links add them unpredictably.
    pub fn parse(line: &str) -> Result<Message, ParseError> {
        let mut parts = line.split_whitespace();
        let tag = parts.next().ok_or(ParseError::Empty)?;

        match tag {
            // `R` is `G` plus "drop your accumulated state first".
            "G" | "R" => {
                let x = next_f64(&mut parts)?;
                let y = next_f64(&mut parts)?;
                let budget = next_f64(&mut parts)?;
                Ok(Message::Goal {
                    x,
                    y,
                    budget,
                    fresh: tag == "R",
                })
            }
            "T" => {
                let v = next_f64(&mut parts)?;
                let w = next_f64(&mut parts)?;
                Ok(Message::Twist { v, w })
            }
            "P" => {
                let x = next_f64(&mut parts)?;
                let y = next_f64(&mut parts)?;
                let heading = next_f64(&mut parts)?;
                Ok(Message::Pose { x, y, heading })
            }
            "M" => {
                let duty_l = next_i32(&mut parts)?;
                let duty_r = next_i32(&mut parts)?;
                Ok(Message::Motor { duty_l, duty_r })
            }
            "I" => {
                let x = next_f64(&mut parts)?;
                let y = next_f64(&mut parts)?;
                let heading = next_f64(&mut parts)?;
                Ok(Message::Start { x, y, heading })
            }
            "J" => {
                let index = next_i64(&mut parts)?;
                let raw_ticks = next_i64(&mut parts)?;
                let milliradians = next_i32(&mut parts)?;
                let duty = next_i32(&mut parts)?;
                let phase = parts
                    .next()
                    .ok_or(ParseError::MissingField)?
                    .parse()
                    .map_err(|_| ParseError::BadNumber)?;
                Ok(Message::Joint {
                    index: index.clamp(0, 255) as u8,
                    raw_ticks,
                    milliradians,
                    duty,
                    phase,
                })
            }
            "H" => {
                let worst_us = next_i64(&mut parts)?;
                Ok(Message::Health {
                    worst_us: worst_us.max(0) as u32,
                })
            }
            "S" => {
                let dl = next_i64(&mut parts)?;
                let dr = next_i64(&mut parts)?;
                Ok(Message::Sensors { dl, dr })
            }
            _ => Err(ParseError::UnknownTag),
        }
    }

    /// The single-character tag this message serialises with.
    pub fn tag(&self) -> char {
        match self {
            Message::Joint { .. } => 'J',
            Message::Goal { fresh: false, .. } => 'G',
            Message::Goal { fresh: true, .. } => 'R',
            Message::Twist { .. } => 'T',
            Message::Pose { .. } => 'P',
            Message::Motor { .. } => 'M',
            Message::Sensors { .. } => 'S',
            Message::Health { .. } => 'H',
            Message::Start { .. } => 'I',
        }
    }
}

/// Parse the next field as a **finite** number.
///
/// # Why non-finite is a parse error, not a value
///
/// `"NaN".parse::<f64>()` succeeds. So does `"inf"`. Without this check a
/// single `G NaN NaN NaN` on the wire parses as a perfectly valid
/// directive, and NaN then propagates into the PID's integral — where it
/// **stays**. Measured: three good ticks, one NaN tick, and every
/// subsequent tick outputs NaN even with a perfectly good target. Duty
/// maps NaN to 0, so the robot silently stops steering for the rest of the
/// session while still reporting healthy.
///
/// Rejecting here means the chip treats the line the way it treats any
/// other garbage — ignore it and wait for a good one — and if garbage is
/// all that arrives, `CommandWatchdog` goes stale and zeroes the output.
/// Both are paths that already exist and are already tested.
///
/// Every quantity this protocol carries is a physical measurement. None of
/// them has a meaningful infinite or undefined value.
fn next_f64<'a, I: Iterator<Item = &'a str>>(it: &mut I) -> Result<f64, ParseError> {
    let value: f64 = it
        .next()
        .ok_or(ParseError::MissingField)?
        .parse()
        .map_err(|_| ParseError::BadNumber)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ParseError::BadNumber)
    }
}

fn next_i32<'a, I: Iterator<Item = &'a str>>(it: &mut I) -> Result<i32, ParseError> {
    it.next()
        .ok_or(ParseError::MissingField)?
        .parse()
        .map_err(|_| ParseError::BadNumber)
}

fn next_i64<'a, I: Iterator<Item = &'a str>>(it: &mut I) -> Result<i64, ParseError> {
    it.next()
        .ok_or(ParseError::MissingField)?
        .parse()
        .map_err(|_| ParseError::BadNumber)
}

/// Accumulates bytes from a serial link into complete lines.
///
/// Firmware reads a UART one byte at a time and needs somewhere to put
/// them; that loop was previously written inline in `pico-robot` where it
/// could not be tested. Fixed capacity `N`, no allocation.
///
/// Overlong lines are **truncated, not silently split**: a split would
/// produce two half-messages that might each parse, which is far worse
/// than one dropped message.
pub struct LineReader<const N: usize> {
    buf: [u8; N],
    len: usize,
    overflowed: bool,
}

impl<const N: usize> Default for LineReader<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> LineReader<N> {
    pub const fn new() -> Self {
        LineReader {
            buf: [0; N],
            len: 0,
            overflowed: false,
        }
    }

    /// Feed one byte. Returns `Some(line)` when a newline completes one.
    ///
    /// Returns `None` for the overflowed remainder of an overlong line, so
    /// a corrupt burst can never be mistaken for a message.
    pub fn push(&mut self, byte: u8) -> Option<&str> {
        match byte {
            b'\n' => {
                let len = self.len;
                let overflowed = self.overflowed;
                self.len = 0;
                self.overflowed = false;
                if overflowed {
                    // Truncated line: drop it rather than hand back a
                    // fragment that might still parse into a valid command.
                    None
                } else {
                    self.buf
                        .get(..len)
                        .and_then(|b| core::str::from_utf8(b).ok())
                }
            }
            b'\r' => None,
            b => {
                if self.len < N {
                    self.buf[self.len] = b;
                    self.len += 1;
                } else {
                    self.overflowed = true;
                }
                None
            }
        }
    }
}

/// A planner's [`Directive`] becomes exactly one wire message.
///
/// This impl and [`Message::directive`] are the *only* translation between
/// the control vocabulary and the wire. Both ends of the link go through
/// them, so the host cannot encode something the chip decodes differently
/// — which is what happened when each side built its own commands from
/// the observation.
impl From<Directive> for Message {
    fn from(d: Directive) -> Message {
        match d {
            Directive::Steer {
                target,
                budget,
                fresh,
            } => Message::Goal {
                x: target.x,
                y: target.y,
                budget,
                fresh,
            },
            Directive::Twist { v, w } => Message::Twist { v, w },
        }
    }
}

impl Message {
    /// The directive this message carries, or `None` if it is telemetry
    /// rather than a command.
    ///
    /// `Pose`, `Motor`, `Sensors` and `Joint` flow the other way (chip →
    /// host) and are deliberately not directives: a robot that could be
    /// commanded by its own status report is a robot with a feedback loop
    /// nobody drew.
    pub fn directive(&self) -> Option<Directive> {
        match *self {
            Message::Goal {
                x,
                y,
                budget,
                fresh,
            } => Some(Directive::Steer {
                target: Point::new(x, y),
                budget,
                fresh,
            }),
            Message::Twist { v, w } => Some(Directive::Twist { v, w }),
            Message::Pose { .. }
            | Message::Motor { .. }
            | Message::Sensors { .. }
            | Message::Health { .. }
            | Message::Joint { .. }
            // A session start is a command, but not a *steering* one: it
            // resets state rather than producing motion, so it is handled
            // before the controller ever sees it.
            | Message::Start { .. } => None,
        }
    }
}

/// The human-readable status line `firmware/pico-odom` emits, 50 times a
/// second.
///
/// # Why this is a shared type and not a `write!` in the firmware
///
/// It was a `write!` in the firmware, and two hosts each grew their own
/// parser for it. On 2026-08-09 the firmware split its error counter into
/// `errL`/`errR`; `hil-host`'s viewer kept looking for `err`, dropped
/// every line, and **drew an empty screen while `tools/verify.sh` stayed
/// green at 25/25** — a parser that agrees with itself compiles fine.
///
/// That was fixed by pinning the viewer's parser to a captured line. Then,
/// hours later, `crates/vision` needed the same data and got a *second*
/// independent parser with its own copy of the same fixture. Three
/// implementations of one format, two of them added in response to a bug
/// caused by having two.
///
/// So it lives here, in the crate both ends already share, where
/// [`tests::a_status_line_survives_the_round_trip`] makes drift a
/// compile-and-test failure rather than a blank screen.
///
/// # Human-readable on purpose
///
/// Unlike [`Message`], which is byte-budgeted for a 32-byte UART FIFO,
/// this is meant to be read by a person with `screen` open. That is why
/// it spells out `errL=` rather than packing fields positionally — and
/// why it is a separate type rather than a `Message` variant.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Status {
    /// Report number, counted on the chip and never reset. **The only
    /// field that makes loss measurable.**
    ///
    /// Without it, a status line that never arrived and a robot that did
    /// not move are the same bytes: identical ticks, identical pose. That
    /// was tolerable while the only transport was a cable, which does not
    /// silently drop. It is not tolerable over UDP, where dropping is the
    /// designed behaviour — so comparing the two transports means being
    /// able to say *which* lines went missing, not just how many arrived.
    ///
    /// A gap in this sequence is a lost report. A repeat is a duplicate.
    /// A decrease is the chip having rebooted.
    pub seq: u64,
    /// The chip's own dead reckoning.
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub ticks_left: i64,
    pub ticks_right: i64,
    /// Per wheel, never summed. A decode error is a MISSED transition and
    /// therefore an undercount, so a combined figure cannot say which
    /// wheel reads low — which is exactly the confound that made the
    /// left/right speed comparison untrustworthy until they were split.
    pub errors_left: u64,
    pub errors_right: u64,
    /// Magnitude of the duty being applied, 0–100.
    pub duty_percent: u64,
    /// The chip commanded motion and the encoders disagreed.
    pub stalled: bool,
}

impl Status {
    /// The banner appended when [`Self::stalled`]. Prose, not a field:
    /// it is aimed at whoever is watching the terminal.
    pub const STALL_BANNER: &'static str =
        "*** STALLED: commanded but not moving — check power ***";

    /// Number of `name=value` fields a complete line carries.
    const FIELDS: usize = 9;

    pub fn write_into<W: Write>(&self, w: &mut W) -> core::fmt::Result {
        write!(
            w,
            "n={} pose x={:+.3} y={:+.3} th={:+.3}  ticks L={} R={}  errL={} errR={}  duty={}%",
            self.seq,
            self.x,
            self.y,
            self.heading,
            self.ticks_left,
            self.ticks_right,
            self.errors_left,
            self.errors_right,
            self.duty_percent
        )?;
        if self.stalled {
            write!(w, "  {}", Self::STALL_BANNER)?;
        }
        write!(w, "\r\n")
    }

    /// Parse one line, or `None` if any field is missing or malformed.
    ///
    /// Keyed on the `name=value` tokens rather than on position, so adding
    /// a field cannot silently shift what this reads. A line missing a key
    /// is rejected whole — half a pose plotted as though it were complete
    /// is worse than a dropped frame, and a cancelled USB write can
    /// genuinely truncate one.
    pub fn parse(line: &str) -> Option<Status> {
        let mut status = Status::default();
        let mut seen = 0;
        for token in line.split_whitespace() {
            // Tokens without an `=` are prose — the `pose` prefix, the
            // `ticks` label, the stall banner. Skipped, not fatal.
            let Some((key, value)) = token.split_once('=') else {
                continue;
            };
            match key {
                "n" => status.seq = value.parse().ok()?,
                "x" => status.x = value.parse().ok()?,
                "y" => status.y = value.parse().ok()?,
                "th" => status.heading = value.parse().ok()?,
                "L" => status.ticks_left = value.parse().ok()?,
                "R" => status.ticks_right = value.parse().ok()?,
                "errL" => status.errors_left = value.parse().ok()?,
                "errR" => status.errors_right = value.parse().ok()?,
                // The only field carrying a unit.
                "duty" => status.duty_percent = value.trim_end_matches('%').parse().ok()?,
                _ => continue,
            }
            seen += 1;
        }
        status.stalled = line.contains("STALLED");
        (seen == Self::FIELDS).then_some(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `core::fmt::Write` sink, so the encoder is tested exactly as
    /// firmware uses it — writing into a fixed buffer, not a String.
    struct Buf {
        data: [u8; 128],
        len: usize,
    }

    impl Buf {
        fn new() -> Self {
            Buf {
                data: [0; 128],
                len: 0,
            }
        }
        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.data[..self.len]).unwrap()
        }
    }

    impl Write for Buf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let b = s.as_bytes();
            if self.len + b.len() > self.data.len() {
                return Err(core::fmt::Error);
            }
            self.data[self.len..self.len + b.len()].copy_from_slice(b);
            self.len += b.len();
            Ok(())
        }
    }

    fn encode(m: Message) -> Buf {
        let mut b = Buf::new();
        m.write_into(&mut b).unwrap();
        b
    }

    // ---- round trips: the property that actually matters ----

    #[test]
    fn motor_and_sensors_round_trip_exactly() {
        for m in [
            Message::Motor {
                duty_l: 0,
                duty_r: 0,
            },
            Message::Motor {
                duty_l: 1000,
                duty_r: -1000,
            },
            Message::Joint {
                index: 0,
                raw_ticks: -4290,
                milliradians: -1571,
                duty: 250,
                phase: JointPhase::Homing,
            },
            Message::Joint {
                index: 5,
                raw_ticks: 0,
                milliradians: 0,
                duty: 0,
                phase: JointPhase::NoZero,
            },
            Message::Motor {
                duty_l: -7,
                duty_r: 999,
            },
            Message::Sensors { dl: 0, dr: 0 },
            Message::Sensors { dl: -5, dr: 12345 },
            Message::Sensors {
                dl: i64::MIN,
                dr: i64::MAX,
            },
        ] {
            let text = encode(m);
            let back = Message::parse(text.as_str()).unwrap();
            assert_eq!(
                back,
                m,
                "round trip failed for {text:?}",
                text = text.as_str()
            );
        }
    }

    #[test]
    fn goal_round_trips_within_the_wire_precision() {
        let m = Message::Goal {
            x: 6.5,
            y: 3.25,
            budget: 0.45,
            fresh: true,
        };
        let text = encode(m);
        assert_eq!(text.as_str(), "R 6.5000 3.2500 0.4500\n");
        assert_eq!(Message::parse(text.as_str()).unwrap(), m);
    }

    #[test]
    fn a_goal_needs_all_three_fields() {
        assert_eq!(Message::parse("G"), Err(ParseError::MissingField));
        assert_eq!(Message::parse("G 1"), Err(ParseError::MissingField));
        assert_eq!(Message::parse("G 1 2"), Err(ParseError::MissingField));
        assert_eq!(Message::parse("G 1 x 3"), Err(ParseError::BadNumber));
    }

    #[test]
    fn a_twist_round_trips() {
        let m = Message::Twist { v: 0.12, w: -1.8 };
        assert_eq!(encode(m).as_str(), "T 0.1200 -1.8000\n");
        assert_eq!(Message::parse("T 0.1200 -1.8000").unwrap(), m);
        assert_eq!(Message::parse("T 1"), Err(ParseError::MissingField));
    }

    #[test]
    fn pose_round_trips_within_the_wire_precision() {
        // Encoded at 4 decimals, so equality is to 0.1 mm — not exact.
        // Values chosen to exercise 4-decimal rounding in both
        // directions. Deliberately NOT near pi — clippy's approx_constant
        // reads a bare 3.14159 as a botched constant, and it is right to.
        let m = Message::Pose {
            x: 1.23456,
            y: -0.5,
            heading: 0.98765,
        };
        let text = encode(m);
        match Message::parse(text.as_str()).unwrap() {
            Message::Pose { x, y, heading } => {
                assert!((x - 1.2346).abs() < 1e-9, "x = {x}");
                assert!((y + 0.5).abs() < 1e-9, "y = {y}");
                assert!((heading - 0.9877).abs() < 1e-9, "heading = {heading}");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn encoded_lines_have_the_documented_shape() {
        assert_eq!(
            encode(Message::Motor {
                duty_l: 250,
                duty_r: -250
            })
            .as_str(),
            "M 250 -250\n"
        );
        assert_eq!(
            encode(Message::Sensors { dl: 3, dr: -4 }).as_str(),
            "S 3 -4\n"
        );
        assert_eq!(
            encode(Message::Pose {
                x: 1.0,
                y: 2.0,
                heading: 0.0
            })
            .as_str(),
            "P 1.0000 2.0000 0.0000\n"
        );
    }

    // ---- tolerance: real serial links are messy ----

    #[test]
    fn trailing_newline_and_carriage_return_are_tolerated() {
        for line in ["M 1 2", "M 1 2\n", "M 1 2\r\n", "  M   1   2  \r\n"] {
            assert_eq!(
                Message::parse(line).unwrap(),
                Message::Motor {
                    duty_l: 1,
                    duty_r: 2
                },
                "failed on {line:?}"
            );
        }
    }

    #[test]
    fn extra_trailing_fields_are_ignored() {
        // Forward compatibility: an older parser must not choke on a
        // newer sender that appended a field.
        assert_eq!(
            Message::parse("S 1 2 99 extra").unwrap(),
            Message::Sensors { dl: 1, dr: 2 }
        );
    }

    // ---- errors are distinguishable ----

    #[test]
    fn empty_input_is_reported_as_empty() {
        assert_eq!(Message::parse(""), Err(ParseError::Empty));
        assert_eq!(Message::parse("   \r\n"), Err(ParseError::Empty));
    }

    #[test]
    fn unknown_tag_is_distinct_from_corruption() {
        assert_eq!(Message::parse("Z 1 2"), Err(ParseError::UnknownTag));
        assert_eq!(Message::parse("hello"), Err(ParseError::UnknownTag));
    }

    #[test]
    fn missing_fields_are_reported() {
        assert_eq!(Message::parse("M"), Err(ParseError::MissingField));
        assert_eq!(Message::parse("M 1"), Err(ParseError::MissingField));
        assert_eq!(Message::parse("P 1 2"), Err(ParseError::MissingField));
        assert_eq!(Message::parse("S 1"), Err(ParseError::MissingField));
    }

    #[test]
    fn non_numeric_fields_are_reported() {
        assert_eq!(Message::parse("M x 2"), Err(ParseError::BadNumber));
        assert_eq!(Message::parse("M 1 y"), Err(ParseError::BadNumber));
        assert_eq!(Message::parse("P 1 2 z"), Err(ParseError::BadNumber));
        assert_eq!(Message::parse("S a b"), Err(ParseError::BadNumber));
    }

    #[test]
    fn a_float_where_an_int_belongs_is_rejected() {
        // Motor duty is an integer. "1.5" must not silently truncate.
        assert_eq!(Message::parse("M 1.5 2"), Err(ParseError::BadNumber));
    }

    /// A NaN on the wire is corruption, not a command.
    ///
    /// `"NaN".parse::<f64>()` succeeds, so without the check in `next_f64`
    /// this line parses as a valid `Directive::Steer` and poisons the
    /// chip's PID permanently.
    #[test]
    fn non_finite_numbers_are_rejected_as_corrupt() {
        for line in [
            "G NaN 3.0 0.5",
            "G 1.0 nan 0.5",
            "G 1.0 3.0 inf",
            "R 1.0 3.0 -inf",
            "T NaN 1.0",
            "T 0.1 infinity",
            "I NaN 0.0 0.0",
            "P 1.0 NaN 0.0",
        ] {
            assert!(
                Message::parse(line).is_err(),
                "{line:?} must not parse as a command"
            );
        }
        // ...and the ordinary values still do.
        assert!(Message::parse("G 1.0 3.0 0.5").is_ok());
        assert!(Message::parse("T -0.12 1.8").is_ok());
    }

    /// **The wire has a hard byte budget, and it is not documentation —
    /// it is the RP2040's 32-byte UART RX FIFO.**
    ///
    /// The emulator transport reads straight out of that FIFO
    /// (`Uart<Blocking>`; `BufferedUart` would fix it but storms the
    /// emulator's interrupt controller). The host writes `S` and then the
    /// next `G` back to back, so those two messages share the FIFO. Go
    /// over and bytes are silently dropped, the command line corrupts, the
    /// chip never sees a valid directive, and the host waits forever for
    /// an `M` that cannot come.
    ///
    /// That is not hypothetical. Adding a ` <fresh>` field to `G` took the
    /// burst from 31 to 33 bytes and cost hours: it presented as a hang,
    /// with a chip that was provably fast in isolation (52 ticks/s) and a
    /// host that was provably fast in isolation. The fix was to move the
    /// bit into the tag (`R`), where it costs nothing.
    ///
    /// So: any new field is a byte budget decision. This test is the thing
    /// that says so.
    #[test]
    fn a_command_and_a_sensor_report_fit_in_one_uart_fifo() {
        /// RP2040 / RP2350 UART RX FIFO depth, in bytes. Hardware.
        const FIFO: usize = 32;

        // Worst case actually reachable in the U-trap: an 8 m world, and
        // encoder deltas bounded by one tick at full motor speed
        // (30 rad/s * 0.02 s / 2pi * 1024 ~= 98 counts).
        let goal = Message::Goal {
            x: 7.9999,
            y: 5.9999,
            budget: 0.4500,
            fresh: true,
        };
        let sensors = Message::Sensors { dl: 98, dr: 98 };

        let burst = encode(goal).as_str().len() + encode(sensors).as_str().len();
        assert!(
            burst <= FIFO,
            "a back-to-back S+G burst is {burst} bytes against a {FIFO}-byte              FIFO. The emulator will drop bytes and the run will hang.              Shorten the encoding or move the transport to BufferedUart."
        );
    }

    /// ⚠️ The budget above is **not** met for negative coordinates, and
    /// that is a real latent limit rather than an oversight.
    ///
    /// A world with negative coordinates adds a sign per field. This test
    /// records the headroom that actually exists so the next person meets
    /// the constraint as a number rather than as a mystery hang.
    #[test]
    fn negative_coordinates_would_not_fit_and_we_know_it() {
        let worst = Message::Goal {
            x: -7.9999,
            y: -5.9999,
            budget: -0.4500,
            fresh: true,
        };
        let burst = encode(worst).as_str().len()
            + encode(Message::Sensors { dl: -98, dr: -98 }).as_str().len();
        assert!(
            burst > 32,
            "negative coordinates now fit in the FIFO ({burst} bytes) —              the encoding must have shrunk. Good, but update this test and              the note on the module so the limit stays honest."
        );
    }

    /// Every directive must survive encode → parse unchanged. If it does
    /// not, the chip acts on something the host did not decide.
    #[test]
    fn every_directive_survives_the_wire() {
        let directives = [
            Directive::Steer {
                target: Point::new(6.5, 3.25),
                budget: 0.45,
                fresh: false,
            },
            Directive::Steer {
                target: Point::new(-1.25, 0.0),
                budget: 0.0,
                fresh: true,
            },
            Directive::Twist { v: 0.12, w: -1.8 },
            Directive::Twist { v: 0.0, w: 0.0 },
        ];
        for d in directives {
            let text = encode(Message::from(d));
            let line = text.as_str().trim_end();
            let parsed = Message::parse(line).expect("must parse");
            assert_eq!(
                parsed.directive(),
                Some(d),
                "directive changed crossing the wire: {line:?}"
            );
        }
    }

    /// `fresh` is a single bit and the most forgettable field on the wire.
    /// Losing it means the chip keeps a stale integral across a new
    /// segment while the host drops it — a divergence with no symptom
    /// except the two slowly disagreeing.
    #[test]
    fn the_fresh_flag_is_not_dropped_in_transit() {
        for fresh in [true, false] {
            let d = Directive::Steer {
                target: Point::new(1.0, 2.0),
                budget: 0.3,
                fresh,
            };
            let text = encode(Message::from(d));
            match Message::parse(text.as_str().trim_end())
                .unwrap()
                .directive()
            {
                Some(Directive::Steer { fresh: got, .. }) => {
                    assert_eq!(got, fresh, "fresh={fresh} arrived as {got}")
                }
                other => panic!("expected a Steer, got {other:?}"),
            }
        }
    }

    /// Telemetry must never be mistaken for a command.
    #[test]
    fn chip_to_host_messages_are_not_directives() {
        for m in [
            Message::Pose {
                x: 1.0,
                y: 2.0,
                heading: 0.3,
            },
            Message::Motor {
                duty_l: 500,
                duty_r: -500,
            },
            Message::Sensors { dl: 7, dr: 9 },
            Message::Health { worst_us: 198 },
            Message::Start {
                x: 1.0,
                y: 3.0,
                heading: 0.0,
            },
        ] {
            assert_eq!(m.directive(), None, "{m:?} is telemetry, not a command");
        }
    }

    #[test]
    fn tags_match_what_is_encoded() {
        for (m, t) in [
            (
                Message::Goal {
                    x: 0.0,
                    y: 0.0,
                    budget: 0.0,
                    fresh: false,
                },
                'G',
            ),
            (
                Message::Pose {
                    x: 0.0,
                    y: 0.0,
                    heading: 0.0,
                },
                'P',
            ),
            (
                Message::Motor {
                    duty_l: 0,
                    duty_r: 0,
                },
                'M',
            ),
            (Message::Twist { v: 0.0, w: 0.0 }, 'T'),
            (Message::Sensors { dl: 0, dr: 0 }, 'S'),
        ] {
            assert_eq!(m.tag(), t);
            assert!(encode(m).as_str().starts_with(t));
        }
    }

    // ---- duty clamping ----

    /// The property survives; the duplicate function that used to hold it
    /// does not. `hil_protocol::clamp_duty` claimed "Both ends call this"
    /// and had no callers, because `RobotSpec::duty` — the only thing that
    /// makes a duty — clamped inline with its own copy of the literal.
    /// This now tests the real encoder against the shared constant.
    #[test]
    fn duty_saturates_at_the_protocol_limit() {
        let spec = sim_core::RobotSpec::SIM_BOT;
        let max = spec.max_wheel_speed;

        assert_eq!(spec.duty(0.0), 0);
        assert_eq!(spec.duty(max), DUTY_FULL);
        assert_eq!(spec.duty(-max), -DUTY_FULL);
        // Well past what the motor can do: saturate, never wrap.
        assert_eq!(spec.duty(max * 50.0), DUTY_FULL);
        assert_eq!(spec.duty(-max * 50.0), -DUTY_FULL);
        assert_eq!(spec.duty(f64::MAX), DUTY_FULL);
        assert_eq!(spec.duty(f64::MIN), -DUTY_FULL);
    }

    // ---- LineReader ----

    fn feed<'a, const N: usize>(r: &'a mut LineReader<N>, bytes: &str) -> Option<&'a str> {
        let b = bytes.as_bytes();
        let (last, head) = b.split_last().unwrap();
        for byte in head {
            assert!(r.push(*byte).is_none(), "line completed early");
        }
        r.push(*last)
    }

    #[test]
    fn line_reader_assembles_a_line() {
        let mut r: LineReader<64> = LineReader::new();
        assert_eq!(feed(&mut r, "M 1 2\n"), Some("M 1 2"));
    }

    #[test]
    fn line_reader_strips_carriage_returns() {
        let mut r: LineReader<64> = LineReader::new();
        assert_eq!(feed(&mut r, "S 3 4\r\n"), Some("S 3 4"));
    }

    #[test]
    fn line_reader_handles_back_to_back_lines() {
        let mut r: LineReader<64> = LineReader::new();
        assert_eq!(feed(&mut r, "M 1 2\n"), Some("M 1 2"));
        assert_eq!(feed(&mut r, "S 9 8\n"), Some("S 9 8"));
    }

    #[test]
    fn overlong_lines_are_dropped_not_split() {
        // A 4-byte buffer given 10 bytes must yield NOTHING, not a
        // truncated line that might still parse into a valid command.
        let mut r: LineReader<4> = LineReader::new();
        assert_eq!(feed(&mut r, "M 100 200\n"), None);
        // And it must recover cleanly on the next line.
        assert_eq!(feed(&mut r, "M 12\n"), Some("M 12"));
    }

    #[test]
    fn invalid_utf8_yields_no_line_rather_than_panicking() {
        let mut r: LineReader<16> = LineReader::new();
        for b in [0xffu8, 0xfe, b'\n'] {
            let out = r.push(b);
            if b == b'\n' {
                assert_eq!(out, None, "invalid UTF-8 must not produce a line");
            }
        }
    }

    /// Compared by BEHAVIOUR rather than by peeking at the buffer count.
    /// The accessor that used to be peeked at existed only for this test.
    #[test]
    fn line_reader_default_matches_new() {
        let mut a: LineReader<8> = LineReader::default();
        let mut b: LineReader<8> = LineReader::new();
        assert_eq!(feed(&mut a, "M 1 2\n"), feed(&mut b, "M 1 2\n"));
    }

    /// The end-to-end property: what the chip writes, the host reads.
    #[test]
    fn a_full_exchange_survives_a_byte_stream() {
        let outgoing = [
            Message::Pose {
                x: 1.0,
                y: 1.0,
                heading: 0.0,
            },
            Message::Motor {
                duty_l: 400,
                duty_r: 380,
            },
            Message::Sensors { dl: 12, dr: 11 },
        ];

        let mut wire = Buf::new();
        for m in outgoing {
            m.write_into(&mut wire).unwrap();
        }

        let mut reader: LineReader<64> = LineReader::new();
        let mut received = [None; 3];
        let mut i = 0;
        for byte in wire.as_str().bytes() {
            if let Some(line) = reader.push(byte) {
                received[i] = Some(Message::parse(line).unwrap());
                i += 1;
            }
        }

        assert_eq!(i, 3, "expected three complete lines");
        assert_eq!(received[1], Some(outgoing[1]));
        assert_eq!(received[2], Some(outgoing[2]));
    }

    // ---- Status ----

    /// Captured verbatim from `/dev/cu.usbmodem11` on 2026-08-10, from a
    /// board flashed **before** status lines carried a sequence number.
    /// Evidence, not an example.
    const CAPTURED_BEFORE_SEQ: &str =
        "pose x=-0.001 y=+0.003 th=-0.863  ticks L=-37793 R=38304  errL=235 errR=207  duty=0%";

    /// A board flashed before sequence numbers existed must be REJECTED,
    /// not read as `seq: 0`.
    ///
    /// This is the whole argument for `seq` being required rather than
    /// optional. If a missing `n=` parsed as zero, a stale binary would
    /// report every line as sequence 0 — which a loss detector reads as
    /// "50 duplicates a second", i.e. a transport fault, on a board whose
    /// only fault is needing a reflash. Tested against bytes a real board
    /// really sent, so the rejection is not merely this file agreeing with
    /// itself.
    ///
    #[test]
    fn a_line_from_firmware_without_sequence_numbers_is_rejected() {
        assert_eq!(Status::parse(CAPTURED_BEFORE_SEQ), None);
    }

    /// Captured verbatim from `/dev/cu.usbmodem11` on 2026-08-11, off
    /// board #1 (chipid `0x12ea158439ef5cea`) running `pico-odom teleop`
    /// **while a phone was driving the motors**. Evidence, not an example.
    ///
    /// The previous capture came off board #2, which has no encoders, so
    /// every pose and tick field was zero — it proved the keys parse and
    /// could not have caught `L` and `R` being swapped, because zero is
    /// zero either way. This one has a distinct value in every field:
    /// `L=6997` against `R=-1940`, `errL=47` against `errR=36`, opposite
    /// signs on the wheels, and a heading well away from zero. A
    /// transposition anywhere now fails.
    const CAPTURED: &str = "n=8632 pose x=+0.470 y=-0.041 th=-2.618  \
         ticks L=6997 R=-1940  errL=47 errR=36  duty=0%\r\n";

    #[test]
    fn a_line_the_board_actually_sent() {
        let Some(s) = Status::parse(CAPTURED) else {
            panic!("the firmware's real output no longer parses");
        };
        assert_eq!(s.seq, 8632);
        assert_eq!(s.x, 0.470);
        assert_eq!(s.y, -0.041);
        assert_eq!(s.heading, -2.618);
        assert_eq!(s.ticks_left, 6997);
        assert_eq!(s.ticks_right, -1940);
        assert_eq!(s.errors_left, 47);
        assert_eq!(s.errors_right, 36);
        assert_eq!(s.duty_percent, 0);
        assert!(!s.stalled);
    }

    /// **The test that makes the 2026-08-09 bug impossible.** Writer and
    /// parser are now one type, so a field renamed on one side fails here
    /// rather than silently emptying a viewer.
    #[test]
    fn a_status_line_survives_the_round_trip() {
        let sent = Status {
            seq: 9_001,
            x: -1.25,
            y: 0.5,
            heading: 3.0,
            ticks_left: -37793,
            ticks_right: 38304,
            errors_left: 235,
            errors_right: 207,
            duty_percent: 42,
            stalled: false,
        };
        let mut line = String::new();
        assert!(sent.write_into(&mut line).is_ok());
        assert_eq!(Status::parse(&line), Some(sent));
    }

    #[test]
    fn the_stall_banner_survives_it_too() {
        let sent = Status {
            stalled: true,
            ..Status::default()
        };
        let mut line = String::new();
        assert!(sent.write_into(&mut line).is_ok());
        assert_eq!(Status::parse(&line), Some(sent));
    }

    #[test]
    fn a_truncated_line_is_dropped_not_guessed() {
        assert!(Status::parse("pose x=-0.001 y=+0.003 th=-0.8").is_none());
        assert!(Status::parse("").is_none());
        // The superseded single-`err` format, which is what drifted.
        assert!(
            Status::parse("pose x=+0.0 y=+0.0 th=+0.0  ticks L=1 R=0  err=0  duty=0%").is_none()
        );
    }
}

/// Opening the serial link to a chip, with the facts that are the same
/// every time.
///
/// Five call sites across two crates each spelled out the baud rate and
/// called `.open()`. The baud is genuinely shared; the timeout genuinely
/// is not — a viewer waiting on 50 Hz status lines, a prober expecting a
/// reply, and a mission runner tolerating an emulator's startup all want
/// different patience, and they ranged from 200 ms to 5 s.
///
/// So this fixes the one thing that must not vary and takes the one that
/// must as an argument.
#[cfg(feature = "host")]
pub mod link {
    use std::time::Duration;

    /// Every firmware here runs its CDC/UART link at this rate.
    ///
    /// On USB CDC the number is ignored by the hardware — a CDC device
    /// does not have a baud rate — but it must still be passed, and
    /// passing the same one everywhere means a UART board and a USB board
    /// are opened by identical code.
    pub const BAUD: u32 = 115_200;

    /// Open the port to a chip.
    ///
    /// # ⚠️ This can start the motors
    ///
    /// Opening a CDC port **raises DTR**, and DTR is what the firmware
    /// waits on before it will drive anything — see `HOST_WATCHING` in
    /// `firmware/pico-odom`. That gate exists so a board on a charger sits
    /// still. From the moment this returns, a robot with power to its
    /// H-bridge can move.
    ///
    /// That warning previously appeared at exactly one of the five call
    /// sites, and it is true of all of them.
    pub fn open(
        port: &str,
        timeout: Duration,
    ) -> serialport::Result<Box<dyn serialport::SerialPort>> {
        serialport::new(port, BAUD).timeout(timeout).open()
    }
}
