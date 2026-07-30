//! P3 — say what to look for.
//!
//! ```sh
//! cargo run --release -p vision --bin find -- "cube" "coffee mug" "pen"
//! ```
//!
//! Grounding DINO takes free-form phrases, so the target set is no longer
//! whatever COCO decided in 2014. It is also **slow** — this measures how
//! slow, because the research could find no published latency figures.
//!
//! Each detection also gets its dominant colour computed with plain HSV
//! arithmetic rather than asked of the model. Attributes ("red cube") are
//! exactly where text-embedding detectors are weakest, and ten lines of
//! deterministic maths beat a confident guess.

use anyhow::Result;
use std::sync::mpsc;
use std::time::Instant;
use vision::{
    dominant_hue, hue_name, CameraSource, Detection, Detector, Frame, NokhwaCamera,
    OpenVocabDetector, Promptable,
};

const RESOLUTION: (u32, u32) = (640, 480);
const FPS: u32 = 30;
const MIN_CONFIDENCE: f32 = 0.30;

fn main() -> Result<()> {
    vision::logging::init();

    let phrases: Vec<String> = std::env::args().skip(1).collect();
    let phrases: Vec<&str> = if phrases.is_empty() {
        vec!["cube", "bottle", "coffee mug", "hand"]
    } else {
        phrases.iter().map(|s| s.as_str()).collect()
    };

    let rec = rerun::RecordingStreamBuilder::new("robotiq_find").spawn()?;

    let (tx, rx) = mpsc::channel();
    eprintln!("requesting camera access...");
    nokhwa::nokhwa_initialize(move |g| {
        let _ = tx.send(g);
    });
    if !rx.recv().unwrap_or(false) {
        anyhow::bail!("camera permission denied — run from Terminal.app");
    }

    let (frame_tx, frame_rx) = mpsc::sync_channel::<Frame>(1);
    std::thread::spawn(move || {
        let mut cam = match NokhwaCamera::open_named("Brio", RESOLUTION, FPS)
            .or_else(|_| NokhwaCamera::open(0, RESOLUTION, FPS))
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("camera: {e:#}");
                return;
            }
        };
        loop {
            match cam.next_frame() {
                Ok(f) => match frame_tx.try_send(f) {
                    Ok(()) | Err(mpsc::TrySendError::Full(_)) => {}
                    Err(_) => return,
                },
                Err(e) => {
                    eprintln!("capture: {e:#}");
                    return;
                }
            }
        }
    });

    println!("looking for: {phrases:?}");
    println!("loading Grounding DINO (first run downloads a large model)...");
    let t0 = Instant::now();
    let mut detector = OpenVocabDetector::new(&phrases, MIN_CONFIDENCE)?;
    println!("ready in {:.1}s\n", t0.elapsed().as_secs_f64());
    println!("vocabulary: {:?}\n", detector.vocabulary());

    let start = Instant::now();
    let mut n = 0u64;
    let mut total_ms = 0.0;

    for frame in frame_rx {
        let t = Instant::now();
        let dets = detector.detect(&frame)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;
        n += 1;

        rec.set_duration_secs("time", start.elapsed().as_secs_f64());
        rec.log(
            "camera/image",
            &rerun::Image::from_rgb24(frame.rgb.clone(), [frame.width, frame.height]),
        )?;
        log_boxes(&rec, &dets, &frame)?;
        rec.log("telemetry/inference_ms", &rerun::Scalars::single(ms))?;

        for d in &dets {
            // Colour decided by arithmetic, not by the model.
            let colour = dominant_hue(&frame, d)
                .map(|(h, s)| format!("{} (hue {h:.0}°, sat {s:.2})", hue_name(h)))
                .unwrap_or_else(|| "unsaturated".into());
            println!("  {:>14}  {:.0}%  {colour}", d.label, d.confidence * 100.0);
        }
        println!(
            "[{n}] {ms:.0} ms  ({:.1} fps, mean {:.0} ms)  {} found",
            1000.0 / ms,
            total_ms / n as f64,
            dets.len()
        );
    }
    Ok(())
}

fn log_boxes(rec: &rerun::RecordingStream, dets: &[Detection], frame: &Frame) -> Result<()> {
    let mins: Vec<(f32, f32)> = dets.iter().map(|d| (d.x, d.y)).collect();
    let sizes: Vec<(f32, f32)> = dets.iter().map(|d| (d.width, d.height)).collect();
    let labels: Vec<String> = dets
        .iter()
        .map(|d| {
            let colour = dominant_hue(frame, d)
                .map(|(h, _)| format!("{} ", hue_name(h)))
                .unwrap_or_default();
            format!("{colour}{} {:.0}%", d.label, d.confidence * 100.0)
        })
        .collect();
    rec.log(
        "camera/image/found",
        &rerun::Boxes2D::from_mins_and_sizes(mins, sizes)
            .with_labels(labels)
            .with_colors([rerun::Color::from_rgb(120, 255, 120)]),
    )?;
    Ok(())
}
