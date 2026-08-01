//! The hardware-in-the-loop wire protocol — one definition, both ends.
//!
//! The chip decides; the host owns physics. Every 20 ms:
//!
//! ```text
//!   chip -> host   P <x> <y> <theta>     believed pose (display only)
//!   chip -> host   M <duty_l> <duty_r>   motor command, ±1000
//!   host -> chip   S <dl> <dr>           encoder tick deltas
//! ```
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

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use core::fmt::Write;

/// Motor commands are clamped to this range on both ends. A duty of
/// `±DUTY_FULL` means "full commanded wheel speed".
pub const DUTY_FULL: i32 = 1000;

/// One message in either direction.
///
/// A single enum rather than one per direction: the format is small, and
/// a shared type means a test can round-trip *every* variant without
/// knowing which way it travels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Message {
    /// `P x y theta` — the chip's believed pose, in metres and radians.
    /// Display only; the host never steers with it.
    Pose { x: f64, y: f64, theta: f64 },
    /// `M duty_l duty_r` — motor command, each in `[-DUTY_FULL, DUTY_FULL]`.
    Motor { duty_l: i32, duty_r: i32 },
    /// `S dl dr` — encoder tick deltas since the previous step. Signed:
    /// a reversing wheel counts down.
    Sensors { dl: i64, dr: i64 },
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
            Message::Pose { x, y, theta } => writeln!(w, "P {x:.4} {y:.4} {theta:.4}"),
            Message::Motor { duty_l, duty_r } => writeln!(w, "M {duty_l} {duty_r}"),
            Message::Sensors { dl, dr } => writeln!(w, "S {dl} {dr}"),
        }
    }

    /// Parse one line. Leading/trailing whitespace and a trailing `\r` or
    /// `\n` are tolerated — serial links add them unpredictably.
    pub fn parse(line: &str) -> Result<Message, ParseError> {
        let mut parts = line.split_whitespace();
        let tag = parts.next().ok_or(ParseError::Empty)?;

        match tag {
            "P" => {
                let x = next_f64(&mut parts)?;
                let y = next_f64(&mut parts)?;
                let theta = next_f64(&mut parts)?;
                Ok(Message::Pose { x, y, theta })
            }
            "M" => {
                let duty_l = next_i32(&mut parts)?;
                let duty_r = next_i32(&mut parts)?;
                Ok(Message::Motor { duty_l, duty_r })
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
            Message::Pose { .. } => 'P',
            Message::Motor { .. } => 'M',
            Message::Sensors { .. } => 'S',
        }
    }
}

fn next_f64<'a, I: Iterator<Item = &'a str>>(it: &mut I) -> Result<f64, ParseError> {
    it.next()
        .ok_or(ParseError::MissingField)?
        .parse()
        .map_err(|_| ParseError::BadNumber)
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

/// Clamp a motor command into the protocol's legal range.
///
/// Both ends call this. A duty outside `±DUTY_FULL` is not a protocol
/// error — it is a control law that asked for more than the motor has, and
/// saturating is the correct physical answer.
pub fn clamp_duty(duty: i32) -> i32 {
    duty.clamp(-DUTY_FULL, DUTY_FULL)
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
                    self.buf.get(..len).and_then(|b| core::str::from_utf8(b).ok())
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

    /// Bytes buffered so far in the current, incomplete line.
    pub fn pending(&self) -> usize {
        self.len
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
            assert_eq!(back, m, "round trip failed for {text:?}", text = text.as_str());
        }
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
            theta: 0.98765,
        };
        let text = encode(m);
        match Message::parse(text.as_str()).unwrap() {
            Message::Pose { x, y, theta } => {
                assert!((x - 1.2346).abs() < 1e-9, "x = {x}");
                assert!((y + 0.5).abs() < 1e-9, "y = {y}");
                assert!((theta - 0.9877).abs() < 1e-9, "theta = {theta}");
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
                theta: 0.0
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

    #[test]
    fn tags_match_what_is_encoded() {
        for (m, t) in [
            (
                Message::Pose {
                    x: 0.0,
                    y: 0.0,
                    theta: 0.0,
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
            (Message::Sensors { dl: 0, dr: 0 }, 'S'),
        ] {
            assert_eq!(m.tag(), t);
            assert!(encode(m).as_str().starts_with(t));
        }
    }

    // ---- duty clamping ----

    #[test]
    fn duty_saturates_at_the_protocol_limit() {
        assert_eq!(clamp_duty(0), 0);
        assert_eq!(clamp_duty(DUTY_FULL), DUTY_FULL);
        assert_eq!(clamp_duty(-DUTY_FULL), -DUTY_FULL);
        assert_eq!(clamp_duty(50_000), DUTY_FULL);
        assert_eq!(clamp_duty(-50_000), -DUTY_FULL);
        assert_eq!(clamp_duty(i32::MAX), DUTY_FULL);
        assert_eq!(clamp_duty(i32::MIN), -DUTY_FULL);
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
        assert_eq!(r.pending(), 0);
    }

    #[test]
    fn line_reader_reports_pending_bytes() {
        let mut r: LineReader<64> = LineReader::new();
        for b in b"M 1" {
            r.push(*b);
        }
        assert_eq!(r.pending(), 3);
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

    #[test]
    fn line_reader_default_matches_new() {
        let a: LineReader<8> = LineReader::default();
        let b: LineReader<8> = LineReader::new();
        assert_eq!(a.pending(), b.pending());
    }

    /// The end-to-end property: what the chip writes, the host reads.
    #[test]
    fn a_full_exchange_survives_a_byte_stream() {
        let outgoing = [
            Message::Pose {
                x: 1.0,
                y: 1.0,
                theta: 0.0,
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
}
