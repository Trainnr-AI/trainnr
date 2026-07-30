//! Camera diagnostic: what does THIS machine's camera actually accept?
//!
//! macOS returns an empty format list from enumeration, so the only
//! reliable way to learn what works is to try opening with each request
//! strategy and see which succeeds. Run this once when a camera misbehaves.
//!
//!   cargo run -p vision --bin probe

use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::{query, Camera};
use std::sync::mpsc;

fn try_open(index: u32, label: &str, req: RequestedFormatType) {
    let format = RequestedFormat::new::<RgbFormat>(req);
    match Camera::new(CameraIndex::Index(index), format) {
        Ok(mut cam) => match cam.open_stream() {
            Ok(()) => {
                let res = cam.resolution();
                let fmt = cam.camera_format();
                match cam.frame() {
                    Ok(buf) => println!(
                        "  OK   {label:<34} -> {}x{} {:?} @{}fps, {} bytes/frame",
                        res.width(),
                        res.height(),
                        fmt.format(),
                        fmt.frame_rate(),
                        buf.buffer().len()
                    ),
                    Err(e) => println!("  open-but-no-frame  {label:<20} -> {e}"),
                }
            }
            Err(e) => println!("  stream failed      {label:<20} -> {e}"),
        },
        Err(e) => {
            let msg = e.to_string();
            let brief = msg.split(':').next().unwrap_or(&msg);
            println!("  FAIL {label:<34} -> {brief}");
        }
    }
}

fn main() {
    vision::logging::init();
    let (tx, rx) = mpsc::channel();
    nokhwa::nokhwa_initialize(move |granted| {
        let _ = tx.send(granted);
    });
    println!("camera permission granted: {}", rx.recv().unwrap_or(false));

    println!("\ndevices:");
    match query(ApiBackend::Auto) {
        Ok(devices) if devices.is_empty() => println!("  (none found)"),
        Ok(devices) => {
            for d in &devices {
                println!("  index={} name={:?}", d.index(), d.human_name());
            }
        }
        Err(e) => println!("  query failed: {e}"),
    }

    // Which device to probe: `cargo run -p vision --bin probe -- 1`
    let index: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);

    // NOTE: AbsoluteHighestResolution / AbsoluteHighestFrameRate are
    // deliberately NOT tried — they hang on M-series Macs.
    println!("\nrequest strategies (index {index}):");
    try_open(
        index,
        "None (let the driver decide)",
        RequestedFormatType::None,
    );
    try_open(
        index,
        "HighestResolution(640x480)",
        RequestedFormatType::HighestResolution(Resolution::new(640, 480)),
    );
    for fmt in [
        FrameFormat::YUYV,
        FrameFormat::MJPEG,
        FrameFormat::NV12,
        FrameFormat::RAWRGB,
    ] {
        try_open(
            index,
            &format!("Closest(640x480 {fmt:?} 30)"),
            RequestedFormatType::Closest(CameraFormat::new(Resolution::new(640, 480), fmt, 30)),
        );
    }

    // DANGEROUS — opt in with a second argument:
    //   cargo run -p vision --bin probe -- 1 danger
    //
    // HighestFrameRate ABORTS THE PROCESS on some external webcams
    // (reproduced on a Logitech Brio 100: SIGABRT, exit 134, "panic in a
    // function that cannot unwind"). The panic happens inside an
    // AVFoundation C callback, which Rust cannot unwind through — so it
    // is not catchable, and any strategy listed after it never runs.
    // This is nokhwa#247. Never use it in real code.
    if std::env::args().nth(2).as_deref() == Some("danger") {
        println!("\n  (trying HighestFrameRate — this may abort the process)");
        try_open(
            index,
            "HighestFrameRate(30)",
            RequestedFormatType::HighestFrameRate(30),
        );
    }
}
