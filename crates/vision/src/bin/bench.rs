//! Benchmark the inference backends on THIS machine.
//!
//! The research reported CoreML being 3–7× faster than CPU on Apple
//! Silicon — and also a documented case where `CPUOnly` *beat*
//! `CPUAndGPU`, because CoreML's win comes from graph partitioning and a
//! model whose operators don't map cleanly gets split and shuttled
//! between processors. Both claims are about *some* model on *some*
//! machine. This measures ours.
//!
//! Both backends see the same captured frames, so the comparison is fair.
//!
//! ```sh
//! cargo run --release -p vision --bin bench
//! ```

use anyhow::Result;
use std::sync::mpsc;
use std::time::Instant;
use vision::{Backend, CameraSource, DFineDetector, Detector, Frame, NokhwaCamera};

const WARMUP: usize = 5;
const RUNS: usize = 40;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn measure(name: &str, backend: Backend, frames: &[Frame]) -> Result<Vec<f64>> {
    println!("\n=== {name} ===");
    let load = Instant::now();
    let mut detector = match DFineDetector::with_backend(0.4, backend) {
        Ok(d) => d,
        Err(e) => {
            println!("  unavailable: {e:#}");
            return Ok(Vec::new());
        }
    };
    println!("  model load: {:.1}s", load.elapsed().as_secs_f64());

    // CoreML compiles the model on first inference (1-5s per the research)
    // and caches it. Warmup absorbs that so it doesn't pollute the timings.
    print!("  warmup");
    for i in 0..WARMUP {
        let _ = detector.detect(&frames[i % frames.len()])?;
    }
    println!(" done");

    let mut times = Vec::with_capacity(RUNS);
    let mut total_detections = 0usize;
    for i in 0..RUNS {
        let f = &frames[i % frames.len()];
        let t = Instant::now();
        let dets = detector.detect(f)?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        total_detections += dets.len();
    }

    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "  mean {:.1} ms   p50 {:.1}   p90 {:.1}   min {:.1}   max {:.1}",
        mean,
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.9),
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!(
        "  -> {:.1} fps ceiling, {} detections over {RUNS} runs",
        1000.0 / mean,
        total_detections
    );
    Ok(times)
}

fn main() -> Result<()> {
    // ort logs through `tracing`. WITHOUT a subscriber installed, execution
    // provider failures are completely silent — the research flagged this as
    // costing someone days. Run with RUST_LOG=ort=debug to see which ops
    // CoreML actually took and which fell back to CPU.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let index: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);

    let (tx, rx) = mpsc::channel();
    nokhwa::nokhwa_initialize(move |g| {
        let _ = tx.send(g);
    });
    if !rx.recv().unwrap_or(false) {
        anyhow::bail!("camera permission denied — run from Terminal.app");
    }

    // Capture a fixed set of real frames once, then replay them through
    // every backend. Same input, fair comparison.
    println!("capturing {} frames from camera {index}...", WARMUP + 4);
    let (frame_tx, frame_rx) = mpsc::channel::<Frame>();
    std::thread::spawn(move || {
        let mut cam = match NokhwaCamera::open(index, (640, 480), 30) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("camera: {e:#}");
                return;
            }
        };
        for _ in 0..(WARMUP + 4) {
            match cam.next_frame() {
                Ok(f) => {
                    if frame_tx.send(f).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("capture: {e:#}");
                    return;
                }
            }
        }
    });
    let frames: Vec<Frame> = frame_rx.into_iter().collect();
    if frames.is_empty() {
        anyhow::bail!("captured no frames");
    }
    println!(
        "captured {} frames at {}x{}",
        frames.len(),
        frames[0].width,
        frames[0].height
    );

    let cpu = measure("CPU execution provider", Backend::Cpu, &frames)?;
    let coreml = measure("CoreML execution provider", Backend::CoreMl, &frames)?;

    if !cpu.is_empty() && !coreml.is_empty() {
        let mean_cpu = cpu.iter().sum::<f64>() / cpu.len() as f64;
        let mean_ml = coreml.iter().sum::<f64>() / coreml.len() as f64;
        println!("\n=== verdict ===");
        let ratio = mean_cpu / mean_ml;
        if ratio > 1.05 {
            println!("  CoreML is {ratio:.2}x faster — use it.");
        } else if ratio < 0.95 {
            println!(
                "  CoreML is {:.2}x SLOWER — stay on CPU.\n  \
                 (Partitioning overhead can exceed the kernel win; this is a\n  \
                 documented outcome, not a misconfiguration.)",
                1.0 / ratio
            );
        } else {
            println!("  No meaningful difference ({ratio:.2}x). Prefer CPU for predictability.");
        }
    }
    Ok(())
}
