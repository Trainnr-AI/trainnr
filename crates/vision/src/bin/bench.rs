//! Benchmark detectors on THIS machine.
//!
//! Published mAP is portable; published latency is not. Every number in
//! docs/14-model-landscape.md came from a project README and was measured
//! on a GPU. This is the only instrument that tells us what any of them
//! cost on an M-series laptop, which is the number that decides whether
//! DEIMv2-S's **+8.1 mAP over D-FINE-N** is affordable in a control loop.
//!
//! ```sh
//! cargo run --release -p vision --bin bench                 # sweep models
//! cargo run --release -p vision --bin bench -- --all        # all 7
//! cargo run --release -p vision --bin bench -- --backends   # CPU vs CoreML
//! cargo run --release -p vision --bin bench -- --synthetic  # no camera
//! ```
//!
//! Every model sees the **same frames**, captured once up front, so the
//! comparison is fair. First run of a given model downloads its weights.
//!
//! # A warning that used to be a crash
//!
//! This binary used to run `Backend::CoreMlDynamic` unconditionally, and
//! that backend **aborts the process** (SIGABRT inside Apple's MIL
//! compiler — a C++ exception Rust cannot catch, the same failure mode as
//! nokhwa#247). So `bench` reliably killed itself before printing its
//! verdict. It is now behind `--dangerous-coreml-dynamic`.

use anyhow::Result;
use std::time::Instant;
use vision::{Backend, Detector, DetectorModel, Frame, ObjectDetector, Source, Stats, Stream};

const WARMUP: usize = 5;
const RUNS: usize = 40;
const CAPTURE: usize = 9;
const MIN_CONFIDENCE: f32 = 0.4;

/// The models swept by default.
///
/// Chosen to answer one question — *is DEIMv2-S's accuracy affordable?* —
/// without downloading the whole registry. `--all` adds the rest.
const DEFAULT_SWEEP: [DetectorModel; 4] = [
    DetectorModel::DFineN,
    DetectorModel::Deimv2Pico,
    DetectorModel::Deimv2N,
    DetectorModel::Deimv2S,
];

struct Args {
    all: bool,
    backends: bool,
    synthetic: bool,
    dangerous_dynamic: bool,
    camera: Option<u32>,
}

fn parse_args() -> Args {
    let mut a = Args {
        all: false,
        backends: false,
        synthetic: false,
        dangerous_dynamic: false,
        camera: None,
    };
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--all" => a.all = true,
            "--backends" => a.backends = true,
            "--synthetic" => a.synthetic = true,
            "--dangerous-coreml-dynamic" => a.dangerous_dynamic = true,
            other => a.camera = other.parse().ok(),
        }
    }
    a
}

/// A deterministic frame set for when no camera is attached.
///
/// Detection *latency* for a fixed-shape NMS-free DETR is essentially
/// input-independent — there is no variable-length NMS loop — so synthetic
/// input gives honest timings. It gives meaningless *detection counts*,
/// and the report says so rather than quietly printing zero.
fn synthetic_frames() -> Vec<Frame> {
    (0..CAPTURE)
        .map(|k| {
            let (w, h) = (640u32, 480u32);
            let mut rgb = vec![0u8; (w * h * 3) as usize];
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    rgb[i] = ((x + k as u32 * 7) % 256) as u8;
                    rgb[i + 1] = ((y * 2) % 256) as u8;
                    rgb[i + 2] = ((x ^ y) % 256) as u8;
                }
            }
            Frame {
                width: w,
                height: h,
                rgb,
            }
        })
        .collect()
}

/// Returns the frames and whether they came from a real camera.
fn capture(args: &Args) -> Result<(Vec<Frame>, bool)> {
    if args.synthetic {
        println!("using synthetic frames (latency is valid; detection counts are not)");
        return Ok((synthetic_frames(), false));
    }

    let source = match args.camera {
        Some(i) => Source::Index(i),
        None => Source::Name("Brio"),
    };
    let opened = match Stream::start(source, (640, 480), 30) {
        Ok(s) => Ok(s),
        Err(_) if args.camera.is_none() => Stream::open(Source::Index(0), (640, 480), 30),
        Err(e) => Err(e),
    };
    let stream = match opened {
        Ok(s) => s,
        Err(e) => {
            eprintln!("camera unavailable ({e:#}); falling back to synthetic frames");
            return Ok((synthetic_frames(), false));
        }
    };
    println!(
        "capturing {CAPTURE} frames at {}x{}...",
        stream.resolution.0, stream.resolution.1
    );
    let frames: Vec<Frame> = stream.frames.into_iter().take(CAPTURE).collect();
    anyhow::ensure!(!frames.is_empty(), "captured no frames");
    Ok((frames, true))
}

/// One measured configuration.
struct Row {
    label: String,
    stats: Option<Stats>,
    detections: usize,
}

fn measure(label: &str, model: DetectorModel, backend: Backend, frames: &[Frame]) -> Row {
    use std::io::Write as _;
    print!("  {label:<16} loading...");
    let _ = std::io::stdout().flush();

    let load = Instant::now();
    let mut detector = match ObjectDetector::load(model, MIN_CONFIDENCE, backend) {
        Ok(d) => d,
        Err(e) => {
            println!("\r  {label:<16} UNAVAILABLE: {e:#}");
            return Row {
                label: label.to_string(),
                stats: None,
                detections: 0,
            };
        }
    };
    let load_s = load.elapsed().as_secs_f64();

    // CoreML compiles the graph on first inference and caches it; warmup
    // absorbs that so it doesn't pollute the timings.
    print!("\r  {label:<16} warmup...   ");
    let _ = std::io::stdout().flush();
    for i in 0..WARMUP {
        if detector.detect(&frames[i % frames.len()]).is_err() {
            println!("\r  {label:<16} FAILED during warmup");
            return Row {
                label: label.to_string(),
                stats: None,
                detections: 0,
            };
        }
    }

    let mut times = Vec::with_capacity(RUNS);
    let mut detections = 0usize;
    for i in 0..RUNS {
        let f = &frames[i % frames.len()];
        let t = Instant::now();
        match detector.detect(f) {
            Ok(d) => {
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                detections += d.len();
            }
            Err(e) => {
                println!("\r  {label:<16} FAILED at run {i}: {e:#}");
                break;
            }
        }
    }

    let stats = Stats::from_samples(&times);
    match stats {
        Some(s) => println!(
            "\r  {label:<16} {:>7.1} ms  {:>5.1} fps   p90 {:>6.1} ms   load {:>4.1}s",
            s.mean,
            s.fps(),
            s.p90,
            load_s
        ),
        None => println!("\r  {label:<16} no samples"),
    }
    Row {
        label: label.to_string(),
        stats,
        detections,
    }
}

fn main() -> Result<()> {
    // ort logs through `tracing`. WITHOUT a subscriber installed, execution
    // provider failures are completely silent. Run with
    // ORT_LOG=verbose RUST_LOG=ort=trace to see which ops CoreML took.
    vision::logging::init();
    let args = parse_args();
    let (frames, real) = capture(&args)?;
    println!(
        "{} frames at {}x{}\n",
        frames.len(),
        frames[0].width,
        frames[0].height
    );

    let rows = if args.backends {
        backend_sweep(&args, &frames)
    } else {
        model_sweep(&args, &frames)
    };

    report(&rows, real, args.backends);
    Ok(())
}

fn model_sweep(args: &Args, frames: &[Frame]) -> Vec<Row> {
    let models: Vec<DetectorModel> = if args.all {
        DetectorModel::ALL.to_vec()
    } else {
        DEFAULT_SWEEP.to_vec()
    };
    println!("=== model sweep (CPU provider) ===");
    println!("first run of each model downloads its weights\n");
    models
        .into_iter()
        .map(|m| measure(m.name(), m, Backend::Cpu, frames))
        .collect()
}

fn backend_sweep(args: &Args, frames: &[Frame]) -> Vec<Row> {
    println!("=== backend sweep (D-FINE-N) ===\n");
    let model = DetectorModel::DFineN;
    let mut rows = vec![
        measure("cpu", model, Backend::Cpu, frames),
        measure("coreml", model, Backend::CoreMl, frames),
    ];
    if args.dangerous_dynamic {
        eprintln!(
            "\n  !! CoreMlDynamic aborts the process on this machine (SIGABRT\n  \
             !! inside Apple's MIL compiler). Anything after this may not print.\n"
        );
        rows.push(measure(
            "coreml-dynamic",
            model,
            Backend::CoreMlDynamic,
            frames,
        ));
    }
    rows
}

fn report(rows: &[Row], real_frames: bool, backends: bool) {
    let measured: Vec<&Row> = rows.iter().filter(|r| r.stats.is_some()).collect();
    if measured.is_empty() {
        println!("\nnothing was measured.");
        return;
    }

    println!("\n=== results ===");
    println!(
        "  {:<16} {:>9} {:>8} {:>9} {:>9} {:>7}",
        "config", "mean ms", "fps", "p50 ms", "p90 ms", "dets"
    );
    for r in &measured {
        let s = r.stats.unwrap();
        println!(
            "  {:<16} {:>9.1} {:>8.1} {:>9.1} {:>9.1} {:>7}",
            r.label,
            s.mean,
            s.fps(),
            s.p50,
            s.p90,
            if real_frames {
                r.detections.to_string()
            } else {
                "-".into()
            }
        );
    }

    println!("\n=== verdict ===");
    if backends {
        let cpu = measured.iter().find(|r| r.label == "cpu");
        let ml = measured.iter().find(|r| r.label == "coreml");
        if let (Some(c), Some(m)) = (cpu, ml) {
            let ratio = c.stats.unwrap().mean / m.stats.unwrap().mean;
            if ratio > 1.05 {
                println!("  CoreML is {ratio:.2}x faster — use it.");
            } else if ratio < 0.95 {
                println!(
                    "  CoreML is {:.2}x SLOWER — stay on CPU.\n  \
                     (Partitioning overhead can exceed the kernel win; a\n  \
                     documented outcome, not a misconfiguration.)",
                    1.0 / ratio
                );
            } else {
                println!(
                    "  No meaningful difference ({ratio:.2}x). Prefer CPU for\n  \
                     predictability. Check with ORT_LOG=verbose RUST_LOG=ort=trace\n  \
                     for 'All nodes placed on [CPUExecutionProvider]'."
                );
            }
        }
    } else {
        let fastest = measured
            .iter()
            .min_by(|a, b| a.stats.unwrap().mean.total_cmp(&b.stats.unwrap().mean))
            .unwrap();
        let slowest = measured
            .iter()
            .max_by(|a, b| a.stats.unwrap().mean.total_cmp(&b.stats.unwrap().mean))
            .unwrap();
        let f = fastest.stats.unwrap();
        let s = slowest.stats.unwrap();
        println!(
            "  fastest: {} at {:.1} ms ({:.1} fps)",
            fastest.label,
            f.mean,
            f.fps()
        );
        if measured.len() > 1 {
            println!(
                "  slowest: {} at {:.1} ms ({:.1} fps) — {:.2}x the cost",
                slowest.label,
                s.mean,
                s.fps(),
                s.mean / f.mean
            );
        }
        println!(
            "\n  A 50 Hz control loop has a 20 ms budget; anything above that\n  \
             sets the perception rate, not the control rate. Weigh against\n  \
             published mAP in docs/14-model-landscape.md before choosing."
        );
    }

    if !real_frames {
        println!(
            "\n  NOTE: synthetic frames. Latency is representative;\n  \
             detection counts are meaningless and shown as '-'."
        );
    }
}
