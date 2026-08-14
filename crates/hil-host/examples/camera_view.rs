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
//! The chip cannot send binary through a line-based protocol, so a
//! thumbnail arrives as hex inside ordinary `#` notes:
//!
//! ```text
//!   # IMG 40 30 rgb565
//!   # 0841F80007E0...        <- 40 pixels, 4 hex digits each
//!   ...                         30 such rows
//! ```
//!
//! ⚠️ 40x30, not the full 160x120. The full frame is 38,400 bytes, which
//! as hex is 600 lines — twelve seconds at 50 Hz. A thumbnail is thirty
//! lines and under a second, and it is enough to *look at*, which is the
//! whole job.

use std::io::{BufRead, BufReader};
use std::time::Duration;

/// What the chip announces before a picture.
const HEADER: &str = "# IMG";

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

        if let Some(rest) = line.strip_prefix(HEADER) {
            // `# IMG 40 30 rgb565`
            let mut parts = rest.split_whitespace();
            let width: u32 = parts.next().and_then(|w| w.parse().ok()).unwrap_or(0);
            let height: u32 = parts.next().and_then(|h| h.parse().ok()).unwrap_or(0);
            if width == 0 || height == 0 {
                continue;
            }
            expecting = Some((width, height));
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
        let Some(hex) = line.strip_prefix("# ") else {
            continue;
        };
        // Only hex of the right length is a pixel row; anything else is
        // an ordinary note that happened to arrive mid-picture.
        if hex.len() != width as usize * 4 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            continue;
        }

        for chunk in hex.as_bytes().chunks(4) {
            let Ok(text) = std::str::from_utf8(chunk) else {
                continue;
            };
            let Ok(pixel) = u16::from_str_radix(text, 16) else {
                continue;
            };
            let [red, green, blue] = rgb565_to_rgb888(pixel);
            rows.extend_from_slice(&[red, green, blue]);
        }

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

/// RGB565 to three bytes.
///
/// ⚠️ The channel widths are **5, 6, 5** — green gets the extra bit
/// because the eye is most sensitive to it. Shifting all three by the
/// same amount produces a picture with a green cast that looks like a
/// white-balance problem rather than a decoding one.
///
/// The low bits are replicated rather than zero-filled so that full-scale
/// input reaches full-scale output: `0b11111` becomes 255, not 248.
fn rgb565_to_rgb888(pixel: u16) -> [u8; 3] {
    let red = ((pixel >> 11) & 0x1F) as u8;
    let green = ((pixel >> 5) & 0x3F) as u8;
    let blue = (pixel & 0x1F) as u8;
    [
        (red << 3) | (red >> 2),
        (green << 2) | (green >> 4),
        (blue << 3) | (blue >> 2),
    ]
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
