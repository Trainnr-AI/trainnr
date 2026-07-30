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
use std::sync::mpsc;
use std::time::Instant;
use vision::{CameraSource, Frame, NokhwaCamera};

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

    // macOS: this triggers the TCC permission prompt. It is asynchronous —
    // the callback fires once the user answers, so we wait on a channel
    // rather than racing ahead and failing to open the device.
    let (tx, rx) = mpsc::channel();
    eprintln!("requesting camera access (click Allow if macOS asks)...");
    nokhwa::nokhwa_initialize(move |granted| {
        let _ = tx.send(granted);
    });
    let granted = rx.recv().unwrap_or(false);
    if !granted {
        anyhow::bail!(
            "camera permission denied.\n\
             If no prompt appeared, you are probably running inside an editor's \n\
             integrated terminal or over SSH — try Terminal.app or iTerm2.\n\
             To reset a previous denial: tccutil reset Camera"
        );
    }

    // Capture runs on its own thread: nokhwa's Camera is !Send, and more
    // importantly a control loop must never block waiting on a camera.
    // Frames cross to the main thread over a channel.
    let (frame_tx, frame_rx) = mpsc::channel::<Frame>();
    let (res_tx, res_rx) = mpsc::channel::<(u32, u32)>();

    let index = camera_index();
    std::thread::spawn(move || {
        let mut cam = match NokhwaCamera::open(index, DESIRED, FPS) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("could not open camera: {e:#}");
                return;
            }
        };
        let _ = res_tx.send(cam.resolution());
        loop {
            match cam.next_frame() {
                Ok(frame) => {
                    if frame_tx.send(frame).is_err() {
                        return; // main thread hung up
                    }
                }
                Err(e) => {
                    eprintln!("capture error: {e:#}");
                    return;
                }
            }
        }
    });

    let (w, h) = res_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("camera thread died before opening the device"))?;
    println!(
        "camera open at {w}x{h} (requested {}x{})",
        DESIRED.0, DESIRED.1
    );
    println!("streaming to Rerun — close the viewer or press Ctrl-C to stop");

    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_report = Instant::now();

    for frame in frame_rx {
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
