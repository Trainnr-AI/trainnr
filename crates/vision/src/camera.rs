//! Camera capture, behind a trait.
//!
//! # Why a trait for one implementation
//!
//! `nokhwa`'s macOS backend is **disowned by its own maintainer** — 0.11
//! will not fix it, other projects (Cap, videocall-rs) have migrated away,
//! and the process-abort bug on external webcams (nokhwa#247) is unmerged.
//! We use it anyway because nothing else on macOS is close: zero system
//! dependencies, no ffmpeg, no OpenCV, no brew.
//!
//! So the escape hatch gets built first, not later. Everything downstream
//! depends on `CameraSource`, never on nokhwa, which keeps a future swap to
//! a direct `objc2-av-foundation` backend contained to this one file. Same
//! discipline as `RobotWorld` in sim-core: name the seam before you need it.

use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
};
use nokhwa::Camera;

/// One captured frame: tightly packed RGB8, row-major, no padding.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Anything that can hand us frames. The whole perception pipeline talks
/// to this, so the capture implementation stays swappable.
pub trait CameraSource {
    /// Block until the next frame is available.
    fn next_frame(&mut self) -> Result<Frame>;
    /// Actual negotiated resolution — NOT what was requested. On macOS the
    /// camera decides; see `open()`.
    fn resolution(&self) -> (u32, u32);
}

/// nokhwa-backed capture (AVFoundation on macOS).
pub struct NokhwaCamera {
    camera: Camera,
    width: u32,
    height: u32,
    name: String,
}

impl NokhwaCamera {
    /// Open a camera by device index.
    ///
    /// `desired` is a *hint*. We try a ladder of request strategies and
    /// keep the first that opens, then read back what we actually got.
    /// This is not defensive programming for its own sake — on macOS:
    ///
    /// 1. **Format enumeration returns an empty list**, so nothing can
    ///    reason ahead of time about what the device supports. The only
    ///    way to find out is to try. (`cargo run -p vision --bin probe`
    ///    reports what works on a given machine.)
    /// 2. **`RequestedFormatType::None` fails** on the built-in FaceTime
    ///    camera — it picks 1920x1080@15 and then cannot set it.
    /// 3. **`AbsoluteHighest*` hangs** on M-series. Never used here.
    ///
    /// Measured on this machine: FaceTime HD Camera accepts **YUYV only**;
    /// MJPEG, NV12 and RAWRGB are all rejected. USB webcams commonly
    /// prefer MJPEG, hence the ladder rather than a hardcoded format.
    pub fn open(index: u32, desired: (u32, u32), fps: u32) -> Result<Self> {
        let resolution = Resolution::new(desired.0, desired.1);
        // Order matters: YUYV first (built-in Mac cameras), then MJPEG
        // (most USB webcams), then a resolution-only request as a
        // last resort that lets the driver choose the pixel format.
        let ladder: [(&str, RequestedFormatType); 4] = [
            (
                "Closest YUYV",
                RequestedFormatType::Closest(CameraFormat::new(resolution, FrameFormat::YUYV, fps)),
            ),
            (
                "Closest MJPEG",
                RequestedFormatType::Closest(CameraFormat::new(
                    resolution,
                    FrameFormat::MJPEG,
                    fps,
                )),
            ),
            (
                "Closest NV12",
                RequestedFormatType::Closest(CameraFormat::new(resolution, FrameFormat::NV12, fps)),
            ),
            (
                "HighestResolution",
                RequestedFormatType::HighestResolution(resolution),
            ),
        ];

        let mut last_error = None;
        for (label, request) in ladder {
            let requested = RequestedFormat::new::<RgbFormat>(request);
            match Camera::new(CameraIndex::Index(index), requested) {
                Ok(mut camera) => match camera.open_stream() {
                    Ok(()) => {
                        let res = camera.resolution();
                        // Report the device we ACTUALLY got, not the index we
                        // asked for — on macOS these can differ, and two roles
                        // silently landing on one device looks like success.
                        eprintln!(
                            "camera: index {index} -> \"{}\" via {label} at {}x{}",
                            camera.info().human_name(),
                            res.width(),
                            res.height()
                        );
                        let name = camera.info().human_name();
                        return Ok(NokhwaCamera {
                            camera,
                            width: res.width(),
                            height: res.height(),
                            name,
                        });
                    }
                    Err(e) => last_error = Some(anyhow::anyhow!("{label}: stream failed: {e}")),
                },
                Err(e) => last_error = Some(anyhow::anyhow!("{label}: {e}")),
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no request strategy attempted"))).context(
            "could not open the camera with any format. \
                 Run `cargo run -p vision --bin probe` to see what this device accepts",
        )
    }
}

impl NokhwaCamera {
    /// The device this camera actually opened. **Not necessarily the one
    /// you asked for** — see [`NokhwaCamera::open_named`].
    pub fn device_name(&self) -> String {
        self.name.clone()
    }

    /// Open the camera whose name contains `fragment`, verifying what we
    /// actually got.
    ///
    /// # Why this exists — nokhwa's macOS indices are not trustworthy
    ///
    /// Measured on this machine:
    ///
    /// ```text
    /// query() reports:            open by index actually gives:
    ///   index 0 = FaceTime HD       index 0 -> "Brio 100"
    ///   index 1 = Brio 100          index 1 -> "FaceTime HD Camera"
    /// ```
    ///
    /// Inverted. And worse, *unstable*: opening two cameras in one process
    /// produced "Brio 100" for **both** roles, silently — two views of the
    /// same device, which looks like a working multi-camera rig until you
    /// notice both pictures are identical.
    ///
    /// (This is the same class of bug the research flagged in OpenCV,
    /// which sorts devices by opaque `uniqueID` so index 0 is not "the
    /// built-in camera".)
    ///
    /// So: never trust an index. Ask for a name, try candidate indices,
    /// and **verify with `info().human_name()`** before accepting.
    pub fn open_named(fragment: &str, desired: (u32, u32), fps: u32) -> Result<Self> {
        let devices = nokhwa::query(nokhwa::utils::ApiBackend::Auto)
            .map_err(|e| anyhow::anyhow!("could not enumerate cameras: {e}"))?;
        anyhow::ensure!(!devices.is_empty(), "no cameras found");

        let mut tried = Vec::new();

        // Indices are unreliable, so try them all and check what came back.
        for i in 0..devices.len() as u32 {
            match Self::open(i, desired, fps) {
                Ok(cam) => {
                    let got = cam.device_name();
                    if name_matches(&got, fragment) {
                        return Ok(cam);
                    }
                    tried.push(format!("index {i} -> \"{got}\""));
                    drop(cam); // release before trying the next
                }
                Err(e) => tried.push(format!("index {i} -> failed: {e}")),
            }
        }

        let available: Vec<String> = devices.iter().map(|d| d.human_name()).collect();
        anyhow::bail!(
            "no camera matching \"{fragment}\".\n  enumerated: {available:?}\n  tried: {tried:?}"
        )
    }
}

impl CameraSource for NokhwaCamera {
    fn next_frame(&mut self) -> Result<Frame> {
        let buffer = self.camera.frame().context("capturing a frame")?;
        let image = buffer
            .decode_image::<RgbFormat>()
            .context("decoding the frame to RGB")?;
        Ok(Frame {
            width: image.width(),
            height: image.height(),
            rgb: image.into_raw(),
        })
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

// ---------------------------------------------------------------------
// Running a camera in the background — the boilerplate, written once.
// ---------------------------------------------------------------------
//
// Every binary in this crate needs the same three things: ask macOS for
// permission, open a device, and pump frames from a dedicated thread
// (`Camera` is `!Send`, and a control loop must never block on one).
//
// That was copy-pasted into six binaries, which is how five of them ended
// up still calling `open(index, ..)` after we proved macOS indices are
// inverted AND unstable — the bug that had `rig` streaming one physical
// camera into two windows. A fix that lives in one place gets applied
// once; a fix that lives in six places gets applied to whichever file you
// happened to be editing.

use std::sync::mpsc::{self, Receiver};

/// Ask macOS for camera access and block until the user answers.
///
/// Must be called before opening any device. Returns an error rather than
/// panicking so callers can print something more useful than a backtrace.
pub fn request_access() -> Result<()> {
    let (tx, rx) = mpsc::channel();
    eprintln!("requesting camera access...");
    nokhwa::nokhwa_initialize(move |granted| {
        let _ = tx.send(granted);
    });
    if rx.recv().unwrap_or(false) {
        Ok(())
    } else {
        anyhow::bail!("camera permission denied — run from Terminal.app, not an editor terminal")
    }
}

/// Which physical camera to open.
pub enum Source<'a> {
    /// By substring of the device's human-readable name. **Prefer this.**
    Name(&'a str),
    /// By nokhwa index — unreliable on macOS, see [`NokhwaCamera::open_named`].
    Index(u32),
}

/// A running camera: latest-wins frames, plus the resolution actually
/// negotiated (which is often not the one you asked for).
pub struct Stream {
    pub frames: Receiver<Frame>,
    pub resolution: (u32, u32),
}

impl Stream {
    /// Open `source` and start pumping frames on a background thread.
    ///
    /// Blocks until the device opens, so a bad camera fails here with a
    /// useful message instead of hanging the caller's loop forever.
    ///
    /// The channel has capacity 1 and drops on backpressure: a control
    /// loop wants the *newest* frame, never a queue of stale ones.
    pub fn open(source: Source<'_>, resolution: (u32, u32), fps: u32) -> Result<Stream> {
        let (frame_tx, frames) = mpsc::sync_channel::<Frame>(1);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(u32, u32), String>>();

        // The device is opened ON the thread that will own it: `Camera` is
        // !Send, so it can never cross this boundary.
        let owned = match source {
            Source::Name(n) => OwnedSource::Name(n.to_string()),
            Source::Index(i) => OwnedSource::Index(i),
        };

        std::thread::Builder::new()
            .name("camera".into())
            .spawn(move || {
                let opened = match &owned {
                    OwnedSource::Name(frag) => NokhwaCamera::open_named(frag, resolution, fps),
                    OwnedSource::Index(i) => NokhwaCamera::open(*i, resolution, fps),
                };
                let mut cam = match opened {
                    Ok(c) => {
                        let _ = ready_tx.send(Ok(c.resolution()));
                        c
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("{e:#}")));
                        return;
                    }
                };
                loop {
                    match cam.next_frame() {
                        Ok(f) => match frame_tx.try_send(f) {
                            // Full means the consumer is still working on
                            // the previous frame. Dropping is correct.
                            Ok(()) | Err(mpsc::TrySendError::Full(_)) => {}
                            Err(mpsc::TrySendError::Disconnected(_)) => return,
                        },
                        Err(e) => {
                            eprintln!("capture stopped: {e:#}");
                            return;
                        }
                    }
                }
            })
            .context("spawning the camera thread")?;

        match ready_rx.recv() {
            Ok(Ok(resolution)) => Ok(Stream { frames, resolution }),
            Ok(Err(e)) => anyhow::bail!("opening camera: {e}"),
            Err(_) => anyhow::bail!("camera thread died before reporting readiness"),
        }
    }

    /// Convenience: permission + open, the two calls every binary makes.
    pub fn start(source: Source<'_>, resolution: (u32, u32), fps: u32) -> Result<Stream> {
        request_access()?;
        Stream::open(source, resolution, fps)
    }
}

enum OwnedSource {
    Name(String),
    Index(u32),
}

/// Does `device_name` satisfy a request for `fragment`?
///
/// Case-insensitive substring match. Extracted from
/// [`NokhwaCamera::open_named`] so the rule that fixed the
/// two-roles-one-camera bug is pinned by tests rather than living inside
/// a function that cannot run without a webcam attached.
///
/// Deliberately permissive: users type "brio", the device calls itself
/// "Brio 100", and macOS has been known to append vendor noise. Requiring
/// an exact match would break on a firmware update.
pub fn name_matches(device_name: &str, fragment: &str) -> bool {
    device_name.to_lowercase().contains(&fragment.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_is_case_insensitive_both_ways() {
        assert!(name_matches("Brio 100", "brio"));
        assert!(name_matches("brio 100", "BRIO"));
        assert!(name_matches("FaceTime HD Camera", "facetime"));
    }

    #[test]
    fn partial_fragments_match() {
        // Users type a short name; devices have long ones.
        assert!(name_matches("Logitech BRIO 100 (046d:0942)", "brio"));
        assert!(name_matches("FaceTime HD Camera (Built-in)", "HD"));
    }

    #[test]
    fn the_two_cameras_on_this_machine_do_not_collide() {
        // The actual bug: both roles opened one device. These two names
        // must never satisfy each other's fragment.
        let brio = "Brio 100";
        let facetime = "FaceTime HD Camera";
        assert!(name_matches(brio, "Brio"));
        assert!(!name_matches(facetime, "Brio"));
        assert!(name_matches(facetime, "FaceTime"));
        assert!(!name_matches(brio, "FaceTime"));
    }

    #[test]
    fn unrelated_names_do_not_match() {
        assert!(!name_matches("FaceTime HD Camera", "kinect"));
    }

    #[test]
    fn an_empty_fragment_matches_anything() {
        // Documents the edge rather than pretending it cannot happen:
        // `contains("")` is true, so callers must not pass an empty
        // fragment expecting "no camera".
        assert!(name_matches("Brio 100", ""));
    }

    #[test]
    fn frame_reports_its_own_dimensions() {
        let f = Frame {
            width: 4,
            height: 2,
            rgb: vec![0u8; 4 * 2 * 3],
        };
        assert_eq!(f.rgb.len(), (f.width * f.height * 3) as usize);
    }
}
