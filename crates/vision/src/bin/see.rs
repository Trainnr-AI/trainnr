//! P0 — "can the robot see?"
//!
//! Streams webcam frames to the Rerun viewer. Nothing else. If this works,
//! the three hard parts of Rust webcam capture on macOS are solved
//! (permissions, format negotiation, threading) and everything after it is
//! just models.
//!
//! ## Run it from Terminal.app or iTerm2 — NOT a VS Code/Cursor terminal
//!
//! macOS attaches the camera permission prompt to the *parent* process. In
//! an embedded editor terminal, or over SSH, the prompt goes to the wrong
//! place and you get a silent failure. Run:
//!
//! ```sh
//! cargo run -p vision --bin see
//! ```
//!
//! and click **Allow** the first time. No app bundle, no Info.plist, no
//! entitlements, no codesigning needed — verified.

use anyhow::Result;
use std::time::Instant;
use vision::{Source, Stream};

/// Hint only — the camera negotiates. We read back what we actually get.
/// Which camera. `cargo run -p vision --bin BIN -- 1` for the second one;
/// run the `probe` binary to list what is attached.
fn camera_index() -> u32 {
    std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0)
}

const DESIRED: (u32, u32) = (640, 480);
const FPS: u32 = 30;

fn main() -> Result<()> {
    vision::logging::init();
    let rec = rerun::RecordingStreamBuilder::new("robotiq_vision").spawn()?;

    // Permission (macOS TCC prompt), device open, and the capture thread.
    // `Camera` is !Send and a viewer loop must never block on one, so the
    // device lives on its own thread — see `vision::camera::Stream`.
    let stream = Stream::start(Source::Index(camera_index()), DESIRED, FPS)?;
    let (w, h) = stream.resolution;
    println!(
        "camera open at {w}x{h} (requested {}x{})",
        DESIRED.0, DESIRED.1
    );
    println!("streaming to Rerun — close the viewer or press Ctrl-C to stop");

    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_report = Instant::now();

    for frame in stream.frames {
        rec.set_duration_secs("capture_time", start.elapsed().as_secs_f64());
        rec.log(
            "camera/image",
            &rerun::Image::from_rgb24(frame.rgb, [frame.width, frame.height]),
        )?;

        frames += 1;
        if last_report.elapsed().as_secs_f64() >= 1.0 {
            let fps = frames as f64 / start.elapsed().as_secs_f64();
            rec.log("telemetry/fps", &rerun::Scalars::single(fps))?;
            println!("{frames} frames, {fps:.1} fps average");
            last_report = Instant::now();
        }
    }

    println!("camera stream ended after {frames} frames");
    Ok(())
}
