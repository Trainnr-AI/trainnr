//! Watch what the camera actually sees, in Rerun.
//!
//! ```sh
//! cargo run -p hil-host --example camera_view -- /dev/cu.usbmodem11
//! ```
//!
//! Needs `firmware/pico-odom` built with `camera`:
//!
//! ```sh
//! tools/build-pico2.sh pico-odom teleop,camera
//! ```
//!
//! # Why this exists, and why it exists *late*
//!
//! Everything about the camera up to this point was inferred from
//! statistics — an identity register, a frame count, a min/max byte
//! range, a blob centroid. Every one of those can be confidently wrong,
//! and on 2026-08-14 several were: a hue tracker reported a stable
//! 685-pixel blob while the camera was pointed at a **black table**.
//!
//! The number was steady, which read as convincing. It was steady noise.
//!
//! A picture settles in one glance what no further statistic can:
//! whether the byte order is right, whether the sensor is really in
//! RGB565, whether the exposure is usable, and what the thing is even
//! looking at. This project's own rule, learned from the arm being drawn
//! upside down: **check the trajectory, not the printout.**
//!
//! # The wire format
//!
//! `hil_protocol::thumbnail`, shared with the firmware — so the header
//! this parses and the header the chip writes cannot drift apart. The
//! thumbnail is deliberately small (thirty-odd rows, under a second on
//! the wire): the full frame as hex would be 600 lines and twelve
//! seconds at 50 Hz, and a picture only needs to be *looked at*.

use std::io::{BufRead, BufReader};
use std::time::Duration;

use hil_protocol::thumbnail;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: camera_view <serial-port>");
        std::process::exit(2);
    });

    let rec = rerun::RecordingStreamBuilder::new("robotiq_camera")
        .with_blueprint(layout())
        .spawn()?;

    // ⚠️ Opening raises DTR, which un-gates the chip's reports.
    let reader = BufReader::new(hil_protocol::link::open(&port, Duration::from_millis(500))?);
    println!("watching {port} — pick `robotiq_camera` in the viewer");

    let mut expecting: Option<(u32, u32)> = None;
    let mut rows: Vec<u8> = Vec::new();
    let mut frames = 0u64;

    for line in reader.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();

        if let Some(dims) = thumbnail::parse_header(line) {
            expecting = Some(dims);
            rows.clear();
            continue;
        }

        // A blob line is worth showing beside the picture, since the two
        // disagreeing is exactly the failure this viewer exists to catch.
        if line.contains("blob") || line.contains("no blob") {
            rec.log("events", &rerun::TextLog::new(line.to_string()))?;
        }

        let Some((width, height)) = expecting else {
            continue;
        };
        let Some(pixels) = thumbnail::parse_row(line, width) else {
            continue;
        };
        rows.extend(pixels.flat_map(blob::rgb565_to_rgb888));

        if rows.len() >= (width * height * 3) as usize {
            frames += 1;
            rec.set_duration_secs("frame", frames as f64);
            rec.log(
                "camera/image",
                &rerun::Image::from_rgb24(rows.clone(), [width, height]),
            )?;
            println!("frame {frames}");
            expecting = None;
        }
    }

    println!("{frames} frames drawn");
    Ok(())
}

/// The picture, and the words that came with it.
fn layout() -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{Blueprint, Spatial2DView, TextLogView, Vertical};
    Blueprint::new(
        Vertical::new([
            Spatial2DView::new("what the camera sees")
                .with_origin("/camera")
                .into(),
            TextLogView::new("what the chip says")
                .with_origin("/events")
                .into(),
        ])
        .with_row_shares([0.7, 0.3]),
    )
    .with_auto_views(false)
}
