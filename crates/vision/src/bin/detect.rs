//! P1 — "what is the robot looking at?"
//!
//! Live webcam → D-FINE object detection → boxes drawn in Rerun, with the
//! bearing of each object printed. That bearing is the whole point: it is
//! the number that will drive the robot in P2.
//!
//! Run from Terminal.app or iTerm2 (not an editor terminal — see `see.rs`):
//!
//! ```sh
//! cargo run --release -p vision --bin detect
//! ```
//!
//! **Use `--release`.** Debug-build inference is roughly an order of
//! magnitude slower; this is one of the few places where that matters.
//! First run downloads the model weights (a few MB, cached afterwards).

use anyhow::Result;
use std::time::Instant;
use vision::{Detection, Detector, ObjectDetector, Source, Stream};

const DESIRED: (u32, u32) = (640, 480);
const FPS: u32 = 30;
/// Below this, a detection is noise. A false positive costs a control loop
/// more than a missed frame does.
const MIN_CONFIDENCE: f32 = 0.40;
/// Rough horizontal field of view of a typical laptop webcam, in radians
/// (~60°). Only used to turn pixel positions into bearings; measure yours
/// properly before trusting it for control.
const HORIZONTAL_FOV: f32 = 1.05;

fn main() -> Result<()> {
    vision::logging::init();
    let rec = rerun::RecordingStreamBuilder::new("robotiq_vision").spawn()?;

    // Colour + label per class, logged ONCE. After this, boxes carry only
    // a class id and Rerun colours them consistently — the ergonomic win
    // that keeps the per-frame logging to two calls.
    rec.log_static(
        "/",
        &rerun::AnnotationContext::new([
            (0, "person", rerun::Rgba32::from_rgb(255, 90, 90)),
            (39, "bottle", rerun::Rgba32::from_rgb(90, 200, 255)),
            (41, "cup", rerun::Rgba32::from_rgb(120, 255, 120)),
            (67, "cell phone", rerun::Rgba32::from_rgb(255, 200, 60)),
            (73, "book", rerun::Rgba32::from_rgb(200, 140, 255)),
        ]),
    )?;

    // Permission, device open, and a latest-wins capture thread — all in
    // one call. A bounded channel of 1 means the camera drops stale frames
    // rather than queueing them: we always detect on the *newest* frame,
    // which is what a control loop wants. A growing queue means acting on
    // the past. See `vision::camera::Stream`.
    let stream = Stream::start(Source::Index(0), DESIRED, FPS)?;
    let (w, h) = stream.resolution;
    println!("camera open at {w}x{h}");

    println!("loading D-FINE-N (first run downloads weights)...");
    let load_start = Instant::now();
    let mut detector = ObjectDetector::new(MIN_CONFIDENCE)?;
    println!("model ready in {:.1}s", load_start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_report = Instant::now();

    for frame in stream.frames {
        let t_infer = Instant::now();
        let detections = detector.detect(&frame)?;
        let infer_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        rec.set_duration_secs("capture_time", start.elapsed().as_secs_f64());
        rec.log(
            "camera/image",
            &rerun::Image::from_rgb24(frame.rgb, [frame.width, frame.height]),
        )?;
        log_detections(&rec, &detections)?;
        rec.log("telemetry/inference_ms", &rerun::Scalars::single(infer_ms))?;
        rec.log(
            "telemetry/detections",
            &rerun::Scalars::single(detections.len() as f64),
        )?;

        // The bearing of the most confident object — the number that will
        // steer the robot in P2.
        if let Some(best) = detections
            .iter()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
        {
            let bearing = best.bearing(frame.width, HORIZONTAL_FOV);
            rec.log(
                "telemetry/target_bearing_rad",
                &rerun::Scalars::single(bearing as f64),
            )?;
            if last_report.elapsed().as_secs_f64() >= 1.0 {
                println!(
                    "{:>12} conf={:.2}  bearing={:+.3} rad ({:+.0}°)  {:.0} ms",
                    best.label,
                    best.confidence,
                    bearing,
                    bearing.to_degrees(),
                    infer_ms
                );
            }
        }

        frames += 1;
        if last_report.elapsed().as_secs_f64() >= 1.0 {
            println!(
                "  {frames} frames, {:.1} fps overall, {:.0} ms/inference",
                frames as f64 / start.elapsed().as_secs_f64(),
                infer_ms
            );
            last_report = Instant::now();
        }
    }
    Ok(())
}

/// Boxes are logged UNDER the image entity path so Rerun renders them
/// inside the 2D view rather than as a separate scene.
fn log_detections(rec: &rerun::RecordingStream, detections: &[Detection]) -> Result<()> {
    let mins: Vec<(f32, f32)> = detections.iter().map(|d| (d.x, d.y)).collect();
    let sizes: Vec<(f32, f32)> = detections.iter().map(|d| (d.width, d.height)).collect();
    let labels: Vec<String> = detections
        .iter()
        .map(|d| format!("{} {:.0}%", d.label, d.confidence * 100.0))
        .collect();
    let class_ids: Vec<u16> = detections.iter().map(|d| d.class_id).collect();

    rec.log(
        "camera/image/detections",
        &rerun::Boxes2D::from_mins_and_sizes(mins, sizes)
            .with_labels(labels)
            .with_class_ids(class_ids),
    )?;
    Ok(())
}
