//! The link to the brain — live, recorded, or replayed.
//!
//! # Why this exists
//!
//! On 2026-08-07 a two-byte change to the protocol hung the rig. The chip
//! was provably fast in isolation (52 ticks/s), the host was provably fast
//! in isolation, and together they did three ticks in seventy seconds.
//! Finding it took hours of re-running with ad-hoc `eprintln`s, because
//! **the one thing nobody had was the conversation itself.**
//!
//! Recording it is nearly free: every decision and every sensor report
//! already crosses this seam as a line of ASCII.
//!
//! ```sh
//! cargo run -p hil-host -- --serial /dev/cu.usbmodem11 --record run.wire
//! cargo run -p hil-host -- --replay run.wire       # no hardware needed
//! ```
//!
//! # Replay is a regression test, not just a viewer
//!
//! A replay does not merely re-read the chip's answers — it also checks
//! **what the host says**. Every outgoing line is compared against the
//! recording, so if today's code would command something different from
//! the run that was captured, the replay says so and exits non-zero.
//!
//! That turns a session on real hardware into a permanent test: record the
//! robot doing the right thing once, and any later change that would have
//! made it behave differently is caught on a laptop, in milliseconds, with
//! no robot present.
//!
//! # What it cannot catch
//!
//! This is the host's view of the wire. Bytes the chip *received wrongly*
//! are invisible here — the corruption that caused the hang above happened
//! in the chip's UART FIFO, and a recording would have shown the host
//! sending a perfectly good line. What it would have shown, immediately,
//! is the chip going silent mid-conversation, which is the half of the
//! question that took longest to establish.

use hil_protocol::Message;
use std::fs::File;
use std::io::{BufRead, Write};
use std::path::Path;

/// Who said it. The recording is line-oriented and greppable on purpose —
/// `grep '^<' run.wire` is a complete transcript of the chip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    /// host → chip: a directive.
    ToChip,
    /// chip → host: pose, motor duty, health.
    FromChip,
}

impl Dir {
    fn marker(self) -> char {
        match self {
            Dir::ToChip => '>',
            Dir::FromChip => '<',
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) dir: Dir,
    pub(crate) text: String,
}

/// Parse a recording. Blank lines and `#` comments are skipped so a log
/// can be annotated by hand while debugging.
pub(crate) fn parse_log(body: &str) -> Result<Vec<Entry>, String> {
    body.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .enumerate()
        .map(|(i, line)| {
            let (marker, text) = line.split_at(1);
            let dir = match marker {
                ">" => Dir::ToChip,
                "<" => Dir::FromChip,
                _ => return Err(format!("line {}: expected '>' or '<', got {line:?}", i + 1)),
            };
            Ok(Entry {
                dir,
                text: text.trim().to_string(),
            })
        })
        .collect()
}

/// The link, in whichever of its three modes.
pub enum Wire {
    /// A real chip — over serial, or the emulator on a pipe.
    Live {
        to: Box<dyn Write>,
        from: Box<dyn BufRead>,
        /// Where to tee the conversation, if `--record` was given.
        log: Option<File>,
    },
    /// A recording standing in for the chip.
    Replay {
        entries: std::vec::IntoIter<Entry>,
        /// Every point where today's host disagreed with the recording.
        /// Collected rather than fatal, so one run reports all of them.
        divergences: Vec<String>,
    },
}

impl Wire {
    pub fn live(
        to: Box<dyn Write>,
        from: Box<dyn BufRead>,
        record: Option<&Path>,
    ) -> std::io::Result<Wire> {
        Ok(Wire::Live {
            to,
            from,
            log: record.map(File::create).transpose()?,
        })
    }

    pub fn replay(path: &Path) -> Result<Wire, Box<dyn std::error::Error>> {
        let body = std::fs::read_to_string(path)?;
        let wire = Wire::from_log(&body).map_err(|e| format!("{}: {e}", path.display()))?;
        eprintln!(
            "[host] replaying {} lines from {}",
            wire.unconsumed(),
            path.display()
        );
        Ok(wire)
    }

    /// A replay wire from a transcript already in memory.
    ///
    /// This is the seam that made the host loop testable at all: taking a
    /// `&str` means the whole exchange runs in `cargo test` with no chip,
    /// no emulator and no serial port. See [`crate::rig`].
    pub fn from_log(body: &str) -> Result<Wire, String> {
        let entries = parse_log(body)?;
        if entries.is_empty() {
            return Err("no protocol lines".into());
        }
        Ok(Wire::Replay {
            entries: entries.into_iter(),
            divergences: Vec::new(),
        })
    }

    /// Send a message — or, on replay, check that we *would have* sent the
    /// one that was recorded.
    pub fn send(&mut self, m: Message) -> std::io::Result<()> {
        let mut text = String::new();
        // Encoding into a String cannot fail; the protocol crate returns
        // `fmt::Result` only because it is generic over sinks.
        let _ = m.write_into(&mut text);
        let text = text.trim_end().to_string();

        match self {
            Wire::Live { to, log, .. } => {
                writeln!(to, "{text}")?;
                to.flush()?;
                record(log, Dir::ToChip, &text)
            }
            Wire::Replay {
                entries,
                divergences,
            } => {
                match entries.next() {
                    Some(e) if e.dir == Dir::ToChip && e.text == text => {}
                    Some(e) if e.dir == Dir::ToChip => divergences.push(format!(
                        "would send {text:?}, recording has {:?}",
                        e.text
                    )),
                    Some(e) => divergences.push(format!(
                        "would send {text:?}, but the recording expects the chip \
                         to speak next ({:?}) — the host's message ORDER changed",
                        e.text
                    )),
                    None => divergences
                        .push(format!("would send {text:?}, recording ended — the host runs longer than the recorded session")),
                }
                Ok(())
            }
        }
    }

    /// One line from the chip, or `None` when it has nothing more to say.
    pub fn recv_line(&mut self) -> std::io::Result<Option<String>> {
        match self {
            Wire::Live { from, log, .. } => {
                let mut line = String::new();
                if from.read_line(&mut line)? == 0 {
                    return Ok(None); // closed
                }
                let line = line.trim_end().to_string();
                record(log, Dir::FromChip, &line)?;
                Ok(Some(line))
            }
            Wire::Replay {
                entries,
                divergences,
            } => match entries.next() {
                Some(e) if e.dir == Dir::FromChip => Ok(Some(e.text)),
                Some(e) => {
                    divergences.push(format!(
                        "waiting on the chip, but the recording has the host \
                         sending {:?} next — the host's message ORDER changed",
                        e.text
                    ));
                    Ok(None)
                }
                None => Ok(None), // recording exhausted: a clean end
            },
        }
    }

    /// Divergences found, empty on a live run.
    pub fn divergences(&self) -> &[String] {
        match self {
            Wire::Live { .. } => &[],
            Wire::Replay { divergences, .. } => divergences,
        }
    }

    /// Recorded lines never consumed — the recording outlived the host,
    /// which means today's code stopped earlier than the captured run.
    pub fn unconsumed(&self) -> usize {
        match self {
            Wire::Live { .. } => 0,
            Wire::Replay { entries, .. } => entries.len(),
        }
    }
}

fn record(log: &mut Option<File>, dir: Dir, text: &str) -> std::io::Result<()> {
    match log {
        Some(f) => writeln!(f, "{}{text}", dir.marker()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire_from(body: &str) -> Wire {
        Wire::Replay {
            entries: parse_log(body).expect("valid log").into_iter(),
            divergences: Vec::new(),
        }
    }

    const SESSION: &str = "\
> G 1.0000 2.0000 0.4000
< P 1.0000 3.0000 0.0000
< M 383 525
> S 5 5
";

    #[test]
    fn a_replay_that_matches_reports_nothing() {
        let mut w = wire_from(SESSION);
        w.send(Message::Goal {
            x: 1.0,
            y: 2.0,
            budget: 0.4,
            fresh: false,
        })
        .unwrap();
        assert_eq!(
            w.recv_line().unwrap().as_deref(),
            Some("P 1.0000 3.0000 0.0000")
        );
        assert_eq!(w.recv_line().unwrap().as_deref(), Some("M 383 525"));
        w.send(Message::Sensors { dl: 5, dr: 5 }).unwrap();

        assert!(w.divergences().is_empty(), "{:?}", w.divergences());
        assert_eq!(w.unconsumed(), 0, "the whole session should be consumed");
    }

    /// The point of the whole module: a code change that alters what the
    /// robot commands is caught without a robot.
    #[test]
    fn a_different_decision_is_caught() {
        let mut w = wire_from(SESSION);
        w.send(Message::Goal {
            x: 1.0,
            y: 2.0,
            budget: 0.9, // the tuning changed
            fresh: false,
        })
        .unwrap();

        assert_eq!(w.divergences().len(), 1);
        assert!(
            w.divergences()[0].contains("0.9000") && w.divergences()[0].contains("0.4000"),
            "the message should show both sides: {:?}",
            w.divergences()[0]
        );
    }

    /// `fresh` rides in the tag (`R`), so a lost reset changes the tag and
    /// must be caught — it is exactly the kind of one-bit change that
    /// otherwise shows up weeks later as drift.
    #[test]
    fn a_lost_reset_flag_is_caught() {
        let mut w = wire_from(SESSION);
        w.send(Message::Goal {
            x: 1.0,
            y: 2.0,
            budget: 0.4,
            fresh: true, // recording says G, this is R
        })
        .unwrap();
        assert_eq!(w.divergences().len(), 1, "{:?}", w.divergences());
    }

    #[test]
    fn reordering_the_conversation_is_caught() {
        let mut w = wire_from(SESSION);
        // Read before sending — the host's turn-taking changed.
        assert_eq!(w.recv_line().unwrap(), None);
        assert_eq!(w.divergences().len(), 1);
        assert!(w.divergences()[0].contains("ORDER"));
    }

    #[test]
    fn running_longer_than_the_recording_is_caught() {
        let mut w = wire_from("> S 1 1\n");
        w.send(Message::Sensors { dl: 1, dr: 1 }).unwrap();
        w.send(Message::Sensors { dl: 2, dr: 2 }).unwrap();
        assert_eq!(w.divergences().len(), 1);
        assert!(w.divergences()[0].contains("recording ended"));
    }

    #[test]
    fn stopping_early_leaves_the_recording_unconsumed() {
        let mut w = wire_from(SESSION);
        w.send(Message::Goal {
            x: 1.0,
            y: 2.0,
            budget: 0.4,
            fresh: false,
        })
        .unwrap();
        assert_eq!(w.unconsumed(), 3, "three lines never reached");
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let log = "# a note\n\n> S 1 1\n\n# another\n< M 1 2\n";
        let entries = parse_log(log).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].dir, Dir::ToChip);
        assert_eq!(entries[1].text, "M 1 2");
    }

    #[test]
    fn a_log_without_direction_markers_is_rejected() {
        let err = parse_log("G 1 2 3\n").unwrap_err();
        assert!(err.contains("expected '>' or '<'"), "{err}");
    }
}
