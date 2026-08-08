//! Watch the chip's dead reckoning in Rerun, live.
//!
//! ```sh
//! cargo run -p hil-host --example odom_view -- /dev/cu.usbmodem11
//! ```
//!
//! `firmware/pico-odom` computes a pose on the microcontroller using
//! `sim-core`'s `Odometry` — the same integrator the simulator runs — and
//! prints one line every 100 ms. This turns that text into the same
//! picture the other runners draw, so a real robot's belief can be watched
//! the way a simulated one is.
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

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

/// A parsed `pose … ticks … err=` line.
struct Report {
    x: f64,
    y: f64,
    heading: f64,
    left: i64,
    right: i64,
    errors: u64,
}

impl Report {
    /// Parse one line of `pico-odom` output.
    ///
    /// ```text
    /// pose x=+0.013 y=-0.001 th=-0.180  ticks L=147 R=0  err=0
    /// ```
    ///
    /// Deliberately keyed on the `name=value` tokens rather than on
    /// position, so adding a field to the firmware's format cannot
    /// silently shift what this reads. Any line missing a key is skipped
    /// whole — a half-parsed pose plotted as if it were complete is worse
    /// than a dropped frame.
    fn parse(line: &str) -> Option<Report> {
        let mut x = None;
        let mut y = None;
        let mut heading = None;
        let mut left = None;
        let mut right = None;
        let mut errors = None;

        for token in line.split_whitespace() {
            let Some((key, value)) = token.split_once('=') else {
                continue;
            };
            match key {
                "x" => x = value.parse().ok(),
                "y" => y = value.parse().ok(),
                "th" => heading = value.parse().ok(),
                "L" => left = value.parse().ok(),
                "R" => right = value.parse().ok(),
                "err" => errors = value.parse().ok(),
                _ => {}
            }
        }

        Some(Report {
            x: x?,
            y: y?,
            heading: heading?,
            left: left?,
            right: right?,
            errors: errors?,
        })
    }
}

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
    let mut last_errors = 0u64;
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
            continue;
        };

        rec.set_duration_secs("wall_time", started.elapsed().as_secs_f64());

        let (x, y) = (report.x as f32, report.y as f32);
        trail.push([x, y]);

        // The trail is CUMULATIVE: drawing it means re-sending every point
        // it has ever had. At 10 reports a second that is quadratic, and
        // `sim-run/src/viz.rs` records what it costs — a 22.5 s run pushed
        // 1.27 million points where 2,252 would do. Redraw at 1 Hz.
        if frame.is_multiple_of(10) {
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

        rec.log("ticks/left", &rerun::Scalars::single(report.left as f64))?;
        rec.log("ticks/right", &rerun::Scalars::single(report.right as f64))?;
        rec.log(
            "belief/heading_rad",
            &rerun::Scalars::single(report.heading),
        )?;

        // Plotted, not just printed. A rising error count means the tick
        // stream is being undercounted, so every pose after it is wrong by
        // an amount nothing else on screen reveals.
        rec.log(
            "ticks/errors",
            &rerun::Scalars::single(report.errors as f64),
        )?;
        if report.errors > last_errors {
            rec.log(
                "events",
                &rerun::TextLog::new(format!(
                    "decode errors {last_errors} -> {} — pose is now an UNDERCOUNT",
                    report.errors
                )),
            )?;
            println!("⚠️  decode errors: {last_errors} -> {}", report.errors);
            last_errors = report.errors;
        }

        frame += 1;
    }
    Ok(())
}
