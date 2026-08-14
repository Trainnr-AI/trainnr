//! The whole rig in one window: what the camera sees, what the robot
//! believes, and what the motors are doing about it.
//!
//! ```sh
//! cargo run -p hil-host --example rig_view -- /dev/cu.usbmodem11
//! ```
//!
//! Needs `firmware/pico-odom` built with `camera` (typically `chase`).
//!
//! # Why one viewer and not two
//!
//! The serial port has one reader. `odom_view` drew the pose and never
//! the camera; `camera_view` drew the camera and never the pose — so the
//! question the chase loop actually raises, *"did the wheels do that
//! because of what the camera saw?"*, could not be answered by either.
//! Cause and effect belong on one screen, on one clock.
//!
//! (`odom_view` remains the transport-comparison instrument — USB against
//! radio — a job this viewer does not attempt, and a build this viewer's
//! firmware cannot even carry: `camera,wifi` is refused at compile time.)
//!
//! # Everything here is the shared vocabulary
//!
//! `hil_protocol::Status` for pose lines, `hil_protocol::thumbnail` for
//! the picture, `belief_viz::Trail` for the drawing, `blob` for the
//! pixel conversion. This file adds no parsing of its own, so it cannot
//! drift from what the chip writes.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use hil_protocol::{thumbnail, Status};

/// Status lines per second — `1000 / REPORT_MS` in `firmware/pico-odom`.
const REPORTS_PER_SECOND: u64 = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: rig_view <serial-port>");
        std::process::exit(2);
    });

    let rec = rerun::RecordingStreamBuilder::new("robotiq_rig")
        .with_blueprint(layout())
        .spawn()?;

    // ⚠️ Opening raises DTR, which un-gates the chip — on a `chase`
    // build, the robot is ARMED from this line on.
    let reader = BufReader::new(hil_protocol::link::open(&port, Duration::from_millis(500))?);
    println!("watching {port} — pick `robotiq_rig` in the viewer");
    println!("⚠️  a chase build is armed the moment this opened the port");

    let started = Instant::now();
    let mut trail = belief_viz::Trail::new(
        belief_viz::belief(),
        belief_viz::Size::BENCH,
        REPORTS_PER_SECOND,
    );
    let mut expecting: Option<(u32, u32)> = None;
    let mut pixels: Vec<u8> = Vec::new();
    let mut frames = 0u64;

    for line in reader.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        rec.set_duration_secs("wall_time", started.elapsed().as_secs_f64());

        // ---- the picture ----
        if let Some(dims) = thumbnail::parse_header(line) {
            expecting = Some(dims);
            pixels.clear();
            continue;
        }
        if let Some((width, height)) = expecting {
            if let Some(row) = thumbnail::parse_row(line, width) {
                pixels.extend(row.flat_map(blob::rgb565_to_rgb888));
                if pixels.len() >= (width * height * 3) as usize {
                    frames += 1;
                    rec.log(
                        "camera/image",
                        &rerun::Image::from_rgb24(pixels.clone(), [width, height]),
                    )?;
                    expecting = None;
                }
                continue;
            }
        }

        // ---- the words: camera identity, blob, anything else the chip
        // wants read. Pixel rows never reach here; they were consumed
        // above, which is what keeps the log legible. ----
        if let Some(note) = line.strip_prefix("# ") {
            rec.log("events", &rerun::TextLog::new(note.to_string()))?;
            continue;
        }

        // ---- the belief and the motors ----
        let Some(report) = Status::parse(line) else {
            continue;
        };
        trail.draw(
            &rec,
            "belief",
            sim_core::Pose::new(report.x, report.y, report.heading),
        )?;
        // The stall-revealing pair: commanded duty rising while the tick
        // lines stay flat. docs/07 records the run where exactly this
        // panel made a bug visible that the tests could not see.
        rec.log(
            "motor/duty_percent",
            &rerun::Scalars::single(report.duty_percent as f64),
        )?;
        rec.log(
            "ticks/left",
            &rerun::Scalars::single(report.ticks_left as f64),
        )?;
        rec.log(
            "ticks/right",
            &rerun::Scalars::single(report.ticks_right as f64),
        )?;
    }

    println!("{frames} camera frames drawn");
    Ok(())
}

/// Cause on the left, effect on the right: the camera above the pose it
/// steers, the motor evidence beside them.
fn layout() -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{
        Blueprint, Horizontal, Spatial2DView, TextLogView, TimeSeriesView, Vertical,
    };
    Blueprint::new(
        Horizontal::new([
            Vertical::new([
                Spatial2DView::new("what the camera sees")
                    .with_origin("/camera")
                    .into(),
                Spatial2DView::new("where the robot believes it is")
                    .with_origin("/belief")
                    .into(),
            ])
            .into(),
            Vertical::new([
                TimeSeriesView::new("duty (%)").with_origin("/motor").into(),
                TimeSeriesView::new("encoder ticks")
                    .with_origin("/ticks")
                    .into(),
                TextLogView::new("what the chip says")
                    .with_origin("/events")
                    .into(),
            ])
            .into(),
        ])
        .with_column_shares([0.5, 0.5]),
    )
    .with_auto_views(false)
}
