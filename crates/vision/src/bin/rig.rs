//! Two cameras, one timeline — and the honest skew between them.
//!
//! This is the Stage 5 interface built early: π0 and SmolVLA both take an
//! external view plus a wrist view in a single forward pass. Today the
//! second camera is your built-in; later it becomes a wrist camera on the
//! arm, and nothing above this layer changes.
//!
//! ```sh
//! cargo run --release -p vision --bin rig            # cameras 0 and 1
//! cargo run --release -p vision --bin rig -- 1 0     # swap which is which
//! ```
//!
//! Watch `rig/skew_ms` in Rerun. That number is how far apart in *time*
//! the two views are — the thing most multi-camera code silently ignores.

use anyhow::Result;
use std::time::Instant;
use vision::CameraRig;

const RESOLUTION: (u32, u32) = (640, 480);
const FPS: u32 = 30;

fn main() -> Result<()> {
    vision::logging::init();

    // Which physical device plays which role. Run `probe` to list them.
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let external_idx = args.first().copied().unwrap_or(0);
    let wrist_idx = args.get(1).copied().unwrap_or(1);

    let rec = rerun::RecordingStreamBuilder::new("robotiq_rig").spawn()?;

    let (tx, rx) = std::sync::mpsc::channel();
    eprintln!("requesting camera access...");
    nokhwa::nokhwa_initialize(move |g| {
        let _ = tx.send(g);
    });
    if !rx.recv().unwrap_or(false) {
        anyhow::bail!("camera permission denied — run from Terminal.app");
    }

    let mut rig = CameraRig::new();
    // Names are roles, not devices. Downstream code — and eventually a
    // policy's observation dict — refers to "external"/"wrist", never to
    // an index. Swapping hardware is then a launch argument.
    rig.add("external", external_idx, RESOLUTION, FPS)?;
    println!("opened 'external' on camera {external_idx}");

    match rig.add("wrist", wrist_idx, RESOLUTION, FPS) {
        Ok(()) => println!("opened 'wrist' on camera {wrist_idx}"),
        Err(e) => {
            // Degraded, not dead. A rig with one camera is still useful,
            // and saying so beats crashing.
            println!("could not open 'wrist' on camera {wrist_idx}: {e:#}");
            println!("continuing with one camera");
        }
    }

    println!("\nstreaming — watch rig/skew_ms in Rerun\n");

    let start = Instant::now();
    let mut sets = 0u64;
    let mut skew_total = 0.0f64;
    let mut skew_max = 0.0f64;
    let mut last_print = Instant::now();

    while let Some(set) = rig.next_set() {
        let t = start.elapsed().as_secs_f64();
        rec.set_duration_secs("time", t);

        for name in set.names().map(String::from).collect::<Vec<_>>() {
            if let Some(frame) = set.get(&name) {
                rec.log(
                    format!("camera/{name}/image"),
                    &rerun::Image::from_rgb24(frame.rgb.clone(), [frame.width, frame.height]),
                )?;
                if let Some(age) = set.age_of(&name) {
                    rec.log(
                        format!("rig/age_ms/{name}"),
                        &rerun::Scalars::single(age.as_secs_f64() * 1000.0),
                    )?;
                }
            }
        }

        let skew_ms = set.skew().as_secs_f64() * 1000.0;
        rec.log("rig/skew_ms", &rerun::Scalars::single(skew_ms))?;
        skew_total += skew_ms;
        skew_max = skew_max.max(skew_ms);
        sets += 1;

        if last_print.elapsed().as_secs_f64() >= 2.0 {
            println!(
                "{sets} sets, {:.1} sets/s | skew now {:.1} ms, mean {:.1}, max {:.1} | triggered by '{}'",
                sets as f64 / t,
                skew_ms,
                skew_total / sets as f64,
                skew_max,
                set.triggered_by
            );
            for (cam, err) in rig.failures() {
                println!("  ⚠ camera '{cam}' died: {err}");
            }
            last_print = Instant::now();
        }
    }

    println!("\nall cameras stopped after {sets} sets");
    Ok(())
}
