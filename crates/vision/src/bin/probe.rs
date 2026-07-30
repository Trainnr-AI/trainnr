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

fn try_open(label: &str, req: RequestedFormatType) {
    let format = RequestedFormat::new::<RgbFormat>(req);
    match Camera::new(CameraIndex::Index(0), format) {
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

    // NOTE: AbsoluteHighestResolution / AbsoluteHighestFrameRate are
    // deliberately NOT tried — they hang on M-series Macs.
    println!("\nrequest strategies (index 0):");
    try_open("None (let the driver decide)", RequestedFormatType::None);
    try_open(
        "HighestResolution(640x480)",
        RequestedFormatType::HighestResolution(Resolution::new(640, 480)),
    );
    try_open(
        "HighestFrameRate(30)",
        RequestedFormatType::HighestFrameRate(30),
    );
    for fmt in [
        FrameFormat::NV12,
        FrameFormat::YUYV,
        FrameFormat::MJPEG,
        FrameFormat::RAWRGB,
    ] {
        try_open(
            &format!("Closest(640x480 {fmt:?} 30)"),
            RequestedFormatType::Closest(CameraFormat::new(Resolution::new(640, 480), fmt, 30)),
        );
    }
}
