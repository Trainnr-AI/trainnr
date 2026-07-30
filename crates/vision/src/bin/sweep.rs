//! Sweep permissive detectors for one that CoreML can actually accelerate.
//!
//! D-FINE-N's ONNX export has a dynamic batch dimension (`{-1,3,640,640}`),
//! which makes CoreML reject all 731 nodes and fall back to CPU — see
//! docs/11-perception-stack.md. Other exports in the usls zoo may declare a
//! fixed batch. This tries each one on both providers and reports which,
//! if any, gets a speedup.
//!
//! All candidates are **Apache-2.0** — no Ultralytics/AGPL models here.
//!
//! ```sh
//! cargo run --release -p vision --bin sweep
//! ```

use anyhow::Result;
use std::sync::mpsc;
use std::time::Instant;
use usls::models::{Model, RTDETR};
use usls::{Config, Image};
use vision::{CameraSource, Frame, NokhwaCamera};

const WARMUP: usize = 3;
const RUNS: usize = 20;

/// (label, config factory). All Apache-2.0, all NMS-free DETR-family.
fn candidates() -> Vec<(&'static str, fn() -> Config)> {
    vec![
        ("d-fine-n (current)", || Config::d_fine_n_coco()),
        ("d-fine-s", || Config::d_fine_s_coco()),
        ("rtdetr-v1-r18", || Config::rtdetr_v1_r18()),
        ("rtdetr-v2-s", || Config::rtdetr_v2_s()),
        ("deim-dfine-s", || Config::deim_dfine_s_coco()),
        ("rfdetr-nano", || Config::rfdetr_nano()),
        ("rfdetr-small", || Config::rfdetr_small()),
    ]
}

fn time_model(cfg: Config, coreml: bool, frames: &[Frame]) -> Result<f64> {
    let cfg = if coreml {
        cfg.with_model_device(usls::Device::CoreMl)
    } else {
        cfg
    };
    let mut model = RTDETR::new(cfg.commit()?)?;

    for i in 0..WARMUP {
        let f = &frames[i % frames.len()];
        let img = Image::from_u8s(&f.rgb, f.width, f.height)?;
        let _ = model.run(&[img])?;
    }
    let mut total = 0.0;
    for i in 0..RUNS {
        let f = &frames[i % frames.len()];
        let img = Image::from_u8s(&f.rgb, f.width, f.height)?;
        let t = Instant::now();
        let _ = model.run(&[img])?;
        total += t.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(total / RUNS as f64)
}

fn main() -> Result<()> {
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

    let (frame_tx, frame_rx) = mpsc::channel::<Frame>();
    std::thread::spawn(move || {
        if let Ok(mut cam) = NokhwaCamera::open(index, (640, 480), 30) {
            for _ in 0..4 {
                match cam.next_frame() {
                    Ok(f) => {
                        if frame_tx.send(f).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
        }
    });
    let frames: Vec<Frame> = frame_rx.into_iter().collect();
    anyhow::ensure!(!frames.is_empty(), "no frames captured");
    println!(
        "{} frames at {}x{}\n",
        frames.len(),
        frames[0].width,
        frames[0].height
    );

    println!(
        "{:<22} {:>10} {:>10} {:>9}",
        "model", "cpu (ms)", "coreml", "speedup"
    );
    println!("{}", "-".repeat(54));

    for (name, make) in candidates() {
        // First download/CPU. A failure here is usually a missing weight file.
        let cpu = match time_model(make(), false, &frames) {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                println!(
                    "{name:<22} {:>10}  {}",
                    "n/a",
                    msg.lines().next().unwrap_or("")
                );
                continue;
            }
        };
        // CoreML may abort the process on a dynamic-shape model, so this is
        // the risky call — but with static shapes required it degrades to
        // CPU instead of dying.
        let ml = time_model(make(), true, &frames).ok();
        match ml {
            Some(ml) => println!("{name:<22} {cpu:>10.1} {ml:>10.1} {:>8.2}x", cpu / ml),
            None => println!("{name:<22} {cpu:>10.1} {:>10}", "failed"),
        }
    }

    println!("\nA speedup near 1.00x means CoreML took no nodes (dynamic shapes).");
    println!("Anything above ~1.2x is a model worth switching to.");
    Ok(())
}
