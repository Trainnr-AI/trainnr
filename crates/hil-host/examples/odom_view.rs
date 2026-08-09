//! Watch the chip's dead reckoning in Rerun, live.
//!
//! ```sh
//! cargo run -p hil-host --example odom_view -- /dev/cu.usbmodem11
//! ```
//!
//! `firmware/pico-odom` computes a pose on the microcontroller using
//! `sim-core`'s `Odometry` — the same integrator the simulator runs — and
//! prints one line every `REPORT_MS` (20 ms). This turns that text into
//! the same picture the other runners draw, so a real robot's belief can
//! be watched the way a simulated one is.
//!
//! # Why a viewer rather than a terminal
//!
//! Reading `th=-0.720` tells you a number changed. Watching the trail
//! curve tells you the *kinematics* are right — that one wheel produces an
//! arc and two produce a line — which is the property under test before
//! any dimension has been measured. This project has already learned once
//! that a bug can be invisible in a passing test and obvious on a screen
//! (docs/07, the blank duty panels).
//!
//! # Colours match the rest of the repo
//!
//! Blue is *belief* everywhere here — `sim-run` and `hil-host` both draw
//! the odometry estimate blue and ground truth yellow. There is no ground
//! truth on a bench, so everything drawn here is blue on purpose: it is a
//! reminder that dead reckoning is a claim, not a measurement.
//!
//! # ⚠️ This file is coupled to a format string in another crate
//!
//! The firmware's `write!` and [`Report::parse`] are two halves of one
//! wire format that no compiler checks. On 2026-08-09 the firmware split
//! its error counter into `errL`/`errR` and this parser kept looking for
//! `err`, so **every line was rejected and the viewer drew nothing** —
//! with `tools/verify.sh` green at 25/25 throughout, because a parser that
//! agrees with itself compiles fine.
//!
//! [`tests::a_line_the_board_actually_sent`] is the guard: it holds a line
//! captured verbatim off `/dev/cu.usbmodem11` and asserts the fields come
//! back. **When the firmware's format changes, paste a fresh line into
//! that test from the real board** — an invented line only proves the
//! parser matches this file's idea of the format, which is exactly the
//! belief that failed.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

/// Status lines per second, i.e. `1000 / REPORT_MS` in
/// `firmware/pico-odom`. Measured rather than assumed: 300 lines arrived
/// in 6.0 s off the bench board on 2026-08-10.
const REPORTS_PER_SECOND: u64 = 50;

/// One status line, as the shared [`hil_protocol::Status`] type.
///
/// **This file used to own a parser.** On 2026-08-09 the firmware split
/// its error counter per wheel, this parser kept looking for the old key,
/// and every line was silently dropped — an empty viewer with
/// `tools/verify.sh` green at 25/25, because a parser that agrees with
/// itself compiles fine.
///
/// Pinning it to a captured line fixed that instance. Sharing the type
/// with the firmware that writes it fixes the class: `hil-protocol`
/// round-trips writer against parser, so a renamed field is a failing
/// test rather than a blank screen.
use hil_protocol::Status as Report;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::args()
        .nth(1)
        .ok_or("usage: odom_view <serial-port>")?;

    let serial = serialport::new(&port, 115_200)
        .timeout(Duration::from_millis(2000))
        .open()?;
    let mut reader = BufReader::new(serial);

    let rec = rerun::RecordingStreamBuilder::new("robotiq_odom").spawn()?;
    println!("reading {port} — turn a wheel");

    let started = Instant::now();
    let mut trail: Vec<[f32; 2]> = Vec::new();
    let mut frame = 0u64;
    let mut last_errors_left = 0u64;
    let mut last_errors_right = 0u64;
    let mut announced_stall = false;
    let mut parsed_any = false;
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            // A timeout is normal when nothing is moving — the firmware
            // still reports, but a disconnect looks the same to `read_line`.
            // Keep going; the user will notice a frozen viewer faster than
            // they would notice a message here.
            Err(_) => continue,
        }
        let Some(report) = Report::parse(&line) else {
            // Say so ONCE. A viewer that silently draws nothing is the
            // exact failure this file's header describes: with no output
            // at all, "the board is idle" and "the format moved under us"
            // look identical.
            if !parsed_any {
                eprintln!("⚠️  cannot parse this line — has the firmware format changed?");
                eprintln!("    {}", line.trim_end());
                parsed_any = true;
            }
            continue;
        };
        parsed_any = true;

        rec.set_duration_secs("wall_time", started.elapsed().as_secs_f64());

        let (x, y) = (report.x as f32, report.y as f32);
        trail.push([x, y]);

        // The trail is CUMULATIVE: drawing it means re-sending every point
        // it has ever had. At 50 reports a second that is quadratic, and
        // `sim-run/src/viz.rs` records what it costs — a 22.5 s run pushed
        // 1.27 million points where 2,252 would do. Redraw at 1 Hz.
        if frame.is_multiple_of(REPORTS_PER_SECOND) {
            rec.log(
                "belief/trail",
                &rerun::LineStrips2D::new([trail.clone()])
                    .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
            )?;
        }

        rec.log(
            "belief/body",
            &rerun::Points2D::new([[x, y]])
                .with_radii([0.02])
                .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
        )?;
        rec.log(
            "belief/heading",
            &rerun::Arrows2D::from_vectors([[
                0.05 * report.heading.cos() as f32,
                0.05 * report.heading.sin() as f32,
            ]])
            .with_origins([[x, y]])
            .with_colors([rerun::Color::from_rgb(255, 90, 90)]),
        )?;

        rec.log(
            "ticks/left",
            &rerun::Scalars::single(report.ticks_left as f64),
        )?;
        rec.log(
            "ticks/right",
            &rerun::Scalars::single(report.ticks_right as f64),
        )?;
        rec.log(
            "belief/heading_rad",
            &rerun::Scalars::single(report.heading),
        )?;

        // Plotted beside the ticks, because this is the pair that reveals
        // a stall: commanded duty rising while the tick lines stay flat.
        // docs/07 records the run where restoring exactly this panel is
        // what made a bug visible that the tests could not see.
        rec.log(
            "motor/duty_percent",
            &rerun::Scalars::single(report.duty_percent as f64),
        )?;

        // Plotted, not just printed. A rising error count means the tick
        // stream is being undercounted, so every pose after it is wrong by
        // an amount nothing else on screen reveals.
        rec.log(
            "ticks/errors_left",
            &rerun::Scalars::single(report.errors_left as f64),
        )?;
        rec.log(
            "ticks/errors_right",
            &rerun::Scalars::single(report.errors_right as f64),
        )?;
        // Named per wheel in the message too. "Errors rose" sends you to
        // both channels; "the left rose" sends you to one connector.
        for (wheel, was, now) in [
            ("left", &mut last_errors_left, report.errors_left),
            ("right", &mut last_errors_right, report.errors_right),
        ] {
            if now > *was {
                rec.log(
                    "events",
                    &rerun::TextLog::new(format!(
                        "{wheel} decode errors {was} -> {now} — pose is now an UNDERCOUNT"
                    )),
                )?;
                println!("⚠️  {wheel} decode errors: {was} -> {now}");
                *was = now;
            }
        }

        // The firmware stops driving when this latches, so it is reported
        // once rather than 50 times a second for as long as it is parked.
        if report.stalled && !announced_stall {
            rec.log(
                "events",
                &rerun::TextLog::new(
                    "STALLED — commanded but not moving. Check the battery switch.",
                ),
            )?;
            println!("⚠️  STALLED — commanded but not moving. Check the battery switch.");
            announced_stall = true;
        }

        frame += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Report;

    /// The parsing tests live in `hil-protocol` now, beside the writer,
    /// where they cover the round trip instead of one direction. Keeping a
    /// second copy of the captured fixture here would recreate exactly the
    /// duplication that emptied this viewer in the first place.
    ///
    /// What is worth checking here is that this file has not quietly grown
    /// its own parser again.
    #[test]
    fn the_viewer_uses_the_shared_protocol_type() {
        let shared = hil_protocol::Status::default();
        let _: Report = shared;
    }
}
