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
//! # The one plot that matters: commanded against achieved
//!
//! Logging what we asked for is easy and nearly worthless on its own. A
//! command of `v=0.25` proves the controller ran; it says nothing about
//! whether a wheel turned. This module therefore closes the comparison:
//! encoder deltas become wheel speeds, [`DiffDrive::forward`] turns those
//! into the twist the robot **actually performed**, and both go to the
//! viewer on the same axes.
//!
//! That is the repo's oldest lesson in its newest clothes. The blank duty
//! panels, the battery pack switched off, the `errL`/`errR` viewer that
//! drew nothing — every one was two facts held in the same program that
//! nobody compared. Commanded and achieved are the two facts this program
//! holds.
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
use hil_protocol::{Message, Status};
use sim_core::{BodyTwist, RobotSpec, WheelSpeeds};
use std::io::{BufRead, BufReader, Write};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

/// The robot the chip believes it is. **The same constant the firmware
/// reads**, so the achieved twist computed here and the pose computed on
/// the chip cannot disagree about geometry.
///
/// ⚠️ Still `SIM_BOT` placeholders, and **they hide each other.**
///
/// The command path divides by `max_wheel_speed` (30.0, measured 7.77);
/// the feedback path divides by `ticks_per_revolution` (1024, measured
/// 4290). Both are wrong in opposite directions by nearly the same
/// factor, so at v = 0.27 through the motor's measured response:
///
/// ```text
///   today             (1024, 30.0)   achieved 0.26   0.97x  "perfect"
///   fix max_wheel_speed only         achieved 1.13   4.19x  looks broken
///   fix ticks_per_rev only           achieved 0.06   0.23x  looks broken
///   fix BOTH          (4290, 7.77)   achieved 0.27   1.00x  correct
/// ```
///
/// **So [`log`]'s commanded-vs-achieved plot cannot detect this class of
/// error.** Both sides read these same constants, which makes a
/// correlated mistake structurally invisible to the comparison — the
/// robot is self-consistent and unanchored, doing what it is told in
/// units that are not metres. Only an external measurement, a ruler
/// against real travel, can anchor it.
///
/// That is a limit of the best instrument this system has, and it is
/// written here rather than in a commit message because the plot looks
/// most trustworthy exactly when it is least informative. Fix the two
/// numbers together or the robot will appear 4x broken.
const SPEC: RobotSpec = RobotSpec::REAL_BOT;

/// One status line from `pico-odom`, as the shared
/// [`hil_protocol::Status`] type.
///
/// **This was a second, independent parser** with its own copy of the
/// captured fixture — added hours after the first one's drift emptied a
/// viewer, which made it the third implementation of one format. It is
/// now an alias, so there is exactly one.
pub type ChipReport = Status;

/// What the robot actually did, derived from the encoders.
#[derive(Debug, Clone, Copy)]
pub struct Feedback {
    pub report: ChipReport,
    /// Measured wheel angular velocities, rad/s, from tick deltas over the
    /// wall time between reads.
    pub wheels: WheelSpeeds,
    /// The twist those wheel speeds correspond to — the robot's ACTUAL
    /// motion, to be read beside the commanded one.
    pub achieved: BodyTwist,
    /// How many fresh status lines arrived since the last call.
    ///
    /// **Zero is a warning, not a normal quiet frame.** The chip reports
    /// at 50 Hz and this loop runs at ~15, so two or three should arrive
    /// every time. Zero means the link is degrading while the channel is
    /// still nominally open — which no other signal here reveals.
    pub fresh_reports: usize,
    /// Age of the newest report when it was used, in seconds. This is the
    /// staleness the controller is actually acting on.
    pub age_seconds: f64,
}

/// The serial link to `pico-odom --features teleop`.
pub struct Wheels {
    port: Box<dyn serialport::SerialPort>,
    /// Latest-wins, like the camera's frame channel. A quiet chip must
    /// never stall the control loop — this loop's deadline belongs to the
    /// robot, not to whichever I/O happens to be slowest.
    reports: Receiver<(ChipReport, Instant)>,
    latest: Option<(ChipReport, Instant)>,
    /// The previous sample, kept so tick deltas can become speeds. Without
    /// it every rate here would be a difference against zero.
    previous: Option<(ChipReport, Instant)>,
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
                    // Stamped HERE, at arrival, not when the control loop
                    // gets round to reading it. Otherwise the age this
                    // reports is the age of the read, which is always zero
                    // and always useless.
                    if tx.send((report, Instant::now())).is_err() {
                        return;
                    }
                }
            }
        });

        // Block for the first report before returning.
        //
        // Two jobs. It makes `latest` always `Some` afterwards, so a
        // `None` from `feedback` means **disconnected** and nothing else —
        // without this, a control loop that asked for feedback before the
        // first line arrived would die at startup reporting that the board
        // had stopped, which it never started.
        //
        // And it turns "wrong firmware flashed" into an error here, with a
        // sentence saying so, instead of a puzzle three seconds into a run
        // with the motors already live.
        let first = reports.recv_timeout(Duration::from_secs(2)).map_err(|_| {
            anyhow::anyhow!(
                "opened {port_name} but no status line arrived in 2 s. Is \
                 pico-odom flashed with `teleop`? The sweep and USB builds \
                 report the same format, so check the duty column changes \
                 when a twist is sent."
            )
        })?;

        Ok(Wheels {
            port,
            reports,
            latest: Some(first),
            previous: None,
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

    /// Drain the telemetry channel and work out what the robot did.
    ///
    /// Returns `None` once the board disconnects, which the caller should
    /// treat as fatal — a control loop still computing twists for a chip
    /// that is gone is a loop lying to its own viewer.
    pub fn feedback(&mut self) -> Option<Feedback> {
        let mut fresh = 0;
        loop {
            match self.reports.try_recv() {
                Ok(sample) => {
                    // The one we are about to replace becomes the baseline
                    // for the delta, so speeds are measured over the real
                    // interval between two adjacent samples.
                    self.previous = self.latest.replace(sample);
                    fresh += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return None,
            }
        }

        let (report, stamped) = self.latest?;
        let wheels = match self.previous {
            Some((before, before_at)) => {
                let seconds = stamped.duration_since(before_at).as_secs_f64();
                // A zero interval would divide by zero; two samples with
                // the same timestamp carry no rate information anyway.
                if seconds <= 0.0 {
                    WheelSpeeds::STOPPED
                } else {
                    let turns = |now: i64, then: i64| {
                        (now - then) as f64 / SPEC.ticks_per_revolution * core::f64::consts::TAU
                            / seconds
                    };
                    WheelSpeeds::new(
                        turns(report.ticks_left, before.ticks_left),
                        turns(report.ticks_right, before.ticks_right),
                    )
                }
            }
            // First sample of the session: one point defines no rate.
            None => WheelSpeeds::STOPPED,
        };

        Some(Feedback {
            report,
            wheels,
            // Forward kinematics — the SAME function the simulator uses to
            // turn wheel speeds into robot motion. Nothing bespoke.
            achieved: SPEC.drive().forward(wheels),
            fresh_reports: fresh,
            age_seconds: stamped.elapsed().as_secs_f64(),
        })
    }
}

/// Draw what the wheels did, beside what they were asked to do.
///
/// The pairing is the whole point. Commanded duty rising while the tick
/// lines stay flat is a stall; ticks moving while the command is zero is a
/// push or a runaway; commanded and achieved diverging by a constant
/// factor is a wrong number in [`SPEC`]. None of the three is visible in
/// any one series alone.
pub fn log(rec: &rerun::RecordingStream, fb: &Feedback, commanded: BodyTwist) -> Result<()> {
    let r = &fb.report;

    // ---- the comparison, on shared axes ----
    rec.log(
        "robot/forward_speed/commanded",
        &rerun::Scalars::single(commanded.forward_speed),
    )?;
    rec.log(
        "robot/forward_speed/achieved",
        &rerun::Scalars::single(fb.achieved.forward_speed),
    )?;
    rec.log(
        "robot/turn_rate/commanded",
        &rerun::Scalars::single(commanded.turn_rate),
    )?;
    rec.log(
        "robot/turn_rate/achieved",
        &rerun::Scalars::single(fb.achieved.turn_rate),
    )?;

    // ---- per wheel, measured ----
    rec.log("wheels/left_rad_s", &rerun::Scalars::single(fb.wheels.left))?;
    rec.log(
        "wheels/right_rad_s",
        &rerun::Scalars::single(fb.wheels.right),
    )?;
    rec.log(
        "wheels/ticks_left",
        &rerun::Scalars::single(r.ticks_left as f64),
    )?;
    rec.log(
        "wheels/ticks_right",
        &rerun::Scalars::single(r.ticks_right as f64),
    )?;
    // Separate series, never summed — see `ChipReport::errors_left`.
    rec.log(
        "wheels/errors_left",
        &rerun::Scalars::single(r.errors_left as f64),
    )?;
    rec.log(
        "wheels/errors_right",
        &rerun::Scalars::single(r.errors_right as f64),
    )?;
    rec.log(
        "chip/duty_percent",
        &rerun::Scalars::single(r.duty_percent as f64),
    )?;

    // ---- the chip's OWN belief, which is not the simulated robot ----
    //
    // `robot/body` and `robot/trail` are the in-process simulation. THIS
    // is dead reckoning computed on the microcontroller from real
    // encoders. Drawing only the first would put a confident, entirely
    // synthetic trail on screen and call it the robot.
    rec.log(
        "chip/belief/body",
        &rerun::Points2D::new([[r.x as f32, r.y as f32]])
            .with_radii([0.02])
            .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
    )?;
    rec.log(
        "chip/belief/heading",
        &rerun::Arrows2D::from_vectors([[
            0.15 * r.heading.cos() as f32,
            0.15 * r.heading.sin() as f32,
        ]])
        .with_origins([[r.x as f32, r.y as f32]])
        .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
    )?;
    rec.log(
        "chip/belief/heading_rad",
        &rerun::Scalars::single(r.heading),
    )?;

    // ---- link health ----
    rec.log(
        "link/fresh_reports",
        &rerun::Scalars::single(fb.fresh_reports as f64),
    )?;
    rec.log(
        "link/telemetry_age_ms",
        &rerun::Scalars::single(fb.age_seconds * 1000.0),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parsing tests moved to `hil-protocol`, where the writer lives,
    /// so they cover the round trip rather than one direction. What stays
    /// here is the thing only this crate can check: that the alias really
    /// is the shared type, so re-introducing a private parser is a
    /// compile error rather than a slow drift.
    #[test]
    fn the_chip_report_is_the_shared_protocol_type() {
        let shared: Status = Status::default();
        let _: ChipReport = shared;
    }
}
