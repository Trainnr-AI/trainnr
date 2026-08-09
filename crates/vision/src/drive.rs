//! Send the control loop's twists to a real chip, and read back what the
//! wheels actually did.
//!
//! This is the last link in the chain. Everything upstream — camera,
//! detector, bearing, PID — already existed and already produced a
//! [`BodyTwist`]; until now that twist drove a *simulated* robot in the
//! same process.
//!
//! ```text
//!   camera ─▶ Detector ─▶ bearing ─▶ GotoController ─▶ BodyTwist
//!                                                          │
//!                                          simulated Robot ◀┴▶ THIS
//!                                                              │
//!                                        T v w over USB ─▶ pico-odom
//!                                        ◀─ pose, ticks, duty
//! ```
//!
//! # One process owns the port
//!
//! Not a design preference — a constraint discovered by violating it. On
//! 2026-08-10 `odom_view` and `twist_send` were run together and the
//! second died with `Device or resource busy`: `serialport` opens
//! exclusively on macOS. So the thing that commands and the thing that
//! visualises have to be the same program, which is also the shape
//! `hil-host` already has.
//!
//! # What happens when this program dies
//!
//! **Nothing is sent, and that is the safe behaviour.** There is no
//! shutdown handler here on purpose. `pico-odom`'s `CommandWatchdog`
//! stops the motors after 200 ms of silence, and a failsafe that depends
//! on the dying process politely saying goodbye is not a failsafe — the
//! interesting deaths (panic, SIGKILL, unplugged cable) never get to say
//! anything. Measured on the bench: 160 ms to zero duty, then 8 ticks of
//! coast.

use anyhow::{Context, Result};
use hil_protocol::Message;
use sim_core::BodyTwist;
use std::io::{BufRead, BufReader, Write};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

/// What the chip reports back, parsed from `pico-odom`'s status line.
///
/// ⚠️ That line is ad-hoc text, not `hil-protocol`. The **commands** this
/// module sends are shared-vocabulary and therefore safe; the telemetry it
/// reads is the one format in this system that no compiler checks, and it
/// has already drifted from its parser once — see the header of
/// `crates/hil-host/examples/odom_view.rs`. Treated as best-effort here:
/// an unparseable line is skipped, never guessed at.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChipReport {
    pub ticks_left: i64,
    pub ticks_right: i64,
    pub errors_left: u64,
    pub errors_right: u64,
    /// Magnitude of the duty the chip is applying, 0–100.
    pub duty_percent: u64,
}

impl ChipReport {
    fn parse(line: &str) -> Option<ChipReport> {
        let mut report = ChipReport::default();
        let mut seen = 0;
        for token in line.split_whitespace() {
            let Some((key, value)) = token.split_once('=') else {
                continue;
            };
            match key {
                "L" => report.ticks_left = value.parse().ok()?,
                "R" => report.ticks_right = value.parse().ok()?,
                "errL" => report.errors_left = value.parse().ok()?,
                "errR" => report.errors_right = value.parse().ok()?,
                "duty" => report.duty_percent = value.trim_end_matches('%').parse().ok()?,
                _ => continue,
            }
            seen += 1;
        }
        (seen == 5).then_some(report)
    }
}

/// The serial link to `pico-odom --features teleop`.
pub struct Wheels {
    port: Box<dyn serialport::SerialPort>,
    /// Latest-wins, like the camera's frame channel. The chip reports at
    /// 50 Hz and the control loop runs at ~20 — draining to the newest is
    /// right, because a stale tick count is worse than none.
    reports: Receiver<ChipReport>,
    latest: ChipReport,
    line: String,
}

impl Wheels {
    /// Open the port and start reading telemetry in the background.
    ///
    /// ⚠️ Opening raises DTR, which is what un-gates the chip. **From this
    /// call on, the motors can move.**
    pub fn open(port_name: &str) -> Result<Wheels> {
        let port = serialport::new(port_name, 115_200)
            .timeout(Duration::from_millis(200))
            .open()
            .with_context(|| {
                format!(
                    "could not open {port_name}. Another program holding it? \
                     serialport opens exclusively — one process at a time."
                )
            })?;

        // A reader thread, so a quiet chip never stalls the control loop.
        // The same reason the camera has one: this loop's deadline belongs
        // to the robot, not to whichever I/O happens to be slowest.
        let reader_port = port
            .try_clone()
            .context("could not clone the serial handle")?;
        let (tx, reports) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(reader_port);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    // EOF: the board went away. Stop; the control loop
                    // will see the channel close.
                    Ok(0) => return,
                    // A timeout looks like an error and is normal.
                    Err(_) => continue,
                    Ok(_) => {}
                }
                if let Some(report) = ChipReport::parse(&line) {
                    if tx.send(report).is_err() {
                        return;
                    }
                }
            }
        });

        Ok(Wheels {
            port,
            reports,
            latest: ChipReport::default(),
            line: String::new(),
        })
    }

    /// Send one twist. Called every frame — the chip's watchdog is fed by
    /// arrival, so a loop that stops calling this stops the robot.
    pub fn command(&mut self, twist: BodyTwist) -> Result<()> {
        self.line.clear();
        // Built through `hil-protocol`, never formatted by hand. Both ends
        // parse with the same crate, so the command channel cannot drift
        // the way the telemetry line did.
        Message::Twist {
            v: twist.forward_speed,
            w: twist.turn_rate,
        }
        .write_into(&mut self.line)?;
        self.port.write_all(self.line.as_bytes())?;
        self.port.flush()?;
        Ok(())
    }

    /// The newest telemetry, or the last seen if none arrived this frame.
    ///
    /// Returns `None` once the board disconnects, which the caller should
    /// treat as fatal — a control loop still computing twists for a chip
    /// that is gone is a loop lying to its own viewer.
    pub fn latest(&mut self) -> Option<ChipReport> {
        loop {
            match self.reports.try_recv() {
                Ok(report) => self.latest = report,
                Err(TryRecvError::Empty) => return Some(self.latest),
                Err(TryRecvError::Disconnected) => return None,
            }
        }
    }
}

/// Draw what the wheels did, beside what they were asked to do.
///
/// The pairing is the point. Commanded duty rising while the tick lines
/// stay flat is a stall; ticks moving while the command is zero is a push
/// or a runaway. Neither is visible in either series alone.
pub fn log(rec: &rerun::RecordingStream, report: ChipReport, commanded: BodyTwist) -> Result<()> {
    rec.log(
        "chip/ticks_left",
        &rerun::Scalars::single(report.ticks_left as f64),
    )?;
    rec.log(
        "chip/ticks_right",
        &rerun::Scalars::single(report.ticks_right as f64),
    )?;
    rec.log(
        "chip/duty_percent",
        &rerun::Scalars::single(report.duty_percent as f64),
    )?;
    rec.log(
        "chip/errors",
        &rerun::Scalars::single((report.errors_left + report.errors_right) as f64),
    )?;
    rec.log(
        "chip/commanded_forward",
        &rerun::Scalars::single(commanded.forward_speed),
    )?;
    rec.log(
        "chip/commanded_turn",
        &rerun::Scalars::single(commanded.turn_rate),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ChipReport;

    /// Captured verbatim from `/dev/cu.usbmodem11`, 2026-08-10. Evidence,
    /// not an example — see the note on [`ChipReport`] about why this
    /// format gets a real line rather than an invented one.
    const CAPTURED: &str =
        "pose x=-0.001 y=+0.003 th=-0.863  ticks L=-37793 R=38304  errL=235 errR=207  duty=0%";

    #[test]
    fn a_line_the_board_actually_sent() {
        let Some(report) = ChipReport::parse(CAPTURED) else {
            panic!("the firmware's real output no longer parses");
        };
        assert_eq!(report.ticks_left, -37793);
        assert_eq!(report.ticks_right, 38304);
        assert_eq!(report.errors_left, 235);
        assert_eq!(report.errors_right, 207);
        assert_eq!(report.duty_percent, 0);
    }

    #[test]
    fn a_partial_line_is_skipped_not_guessed() {
        assert!(ChipReport::parse("pose x=-0.001 ticks L=5").is_none());
        assert!(ChipReport::parse("").is_none());
    }
}
