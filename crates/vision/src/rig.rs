//! Multiple cameras, one timeline.
//!
//! Stage 5's policies require it: π0 takes 2–3 RGB images per timestep
//! (`observation/exterior_image_1_left`, `observation/wrist_image_left`),
//! SmolVLA takes top + wrist. Both feed every view into a single forward
//! pass. See docs/12-model-choice.md.
//!
//! # The design decision: skew is reported, not hidden
//!
//! Two cameras never deliver frames at the same instant — they free-run at
//! their own rates (measured here: Brio ~30 fps, built-in ~14). Three
//! policies were possible:
//!
//! - **wait for all fresh** — every set is simultaneous, but you run at the
//!   slowest camera's rate and add latency
//! - **latest-of-each, silently** — never blocks, but hands downstream code
//!   two frames of unknown age as if they were one moment
//! - **latest-of-each with timestamps** ← what this does
//!
//! The third is what real robots do, and it is barely harder. The point is
//! that a consumer which *knows* two views are 40 ms apart can compensate;
//! one that assumes simultaneity silently computes wrong geometry. Hiding
//! the skew would be the same mistake as a simulator that reports success
//! while driving through walls.
//!
//! Reassuringly, LeRobot records at ~30 fps and pairs whatever frames are
//! current — policies are *trained* on the same skew they meet at
//! inference, so consistency matters more than perfect synchronisation.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::camera::{CameraSource, Frame, NokhwaCamera};

/// A frame plus the moment it arrived.
struct Stamped {
    frame: Arc<Frame>,
    at: Instant,
}

/// The latest frame from every camera, with the age of each.
pub struct FrameSet {
    frames: HashMap<String, Arc<Frame>>,
    timestamps: HashMap<String, Instant>,
    /// Which camera's arrival triggered this set.
    pub triggered_by: String,
}

impl FrameSet {
    pub fn get(&self, name: &str) -> Option<&Frame> {
        self.frames.get(name).map(|f| f.as_ref())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.frames.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Age of a given view relative to the newest one in the set.
    pub fn age_of(&self, name: &str) -> Option<Duration> {
        let newest = self.timestamps.values().max()?;
        let t = self.timestamps.get(name)?;
        Some(newest.duration_since(*t))
    }

    /// Spread between the oldest and newest frame here. **Check this before
    /// treating the views as one moment.** A large skew means the two
    /// cameras disagree about *when*, not just *what*.
    pub fn skew(&self) -> Duration {
        let (Some(min), Some(max)) = (
            self.timestamps.values().min(),
            self.timestamps.values().max(),
        ) else {
            return Duration::ZERO;
        };
        max.duration_since(*min)
    }
}

/// Several cameras, each on its own thread, funnelled into one stream.
pub struct CameraRig {
    tx: Sender<(String, Stamped)>,
    rx: Receiver<(String, Stamped)>,
    latest: HashMap<String, Stamped>,
    /// Cameras whose thread died, and why. One failing camera must not
    /// take down the rig — you run degraded and report it.
    failures: Arc<Mutex<Vec<(String, String)>>>,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraRig {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        CameraRig {
            tx,
            rx,
            latest: HashMap::new(),
            failures: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Open a camera and start streaming it under `name`.
    ///
    /// Blocks until the device opens (so failures surface here, not later),
    /// then hands it to a dedicated thread — `Camera` is `!Send`, and a
    /// control loop must never block waiting on one.
    pub fn add(&mut self, name: &str, index: u32, resolution: (u32, u32), fps: u32) -> Result<()> {
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let tx = self.tx.clone();
        let failures = Arc::clone(&self.failures);
        let name_owned = name.to_string();

        std::thread::Builder::new()
            .name(format!("camera-{name}"))
            .spawn(move || {
                let mut cam = match NokhwaCamera::open(index, resolution, fps) {
                    Ok(c) => {
                        let _ = ready_tx.send(Ok(()));
                        c
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("{e:#}")));
                        return;
                    }
                };
                loop {
                    match cam.next_frame() {
                        Ok(frame) => {
                            let stamped = Stamped {
                                frame: Arc::new(frame),
                                at: Instant::now(),
                            };
                            if tx.send((name_owned.clone(), stamped)).is_err() {
                                return; // rig dropped
                            }
                        }
                        Err(e) => {
                            // Record and stop THIS camera only.
                            if let Ok(mut f) = failures.lock() {
                                f.push((name_owned.clone(), format!("{e:#}")));
                            }
                            return;
                        }
                    }
                }
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => anyhow::bail!("camera '{name}' (index {index}): {e}"),
            Err(_) => anyhow::bail!("camera '{name}' thread died during startup"),
        }
    }

    /// Block until *some* camera delivers a frame, then return the newest
    /// frame from every camera seen so far.
    ///
    /// Returns `None` when every camera has stopped.
    pub fn next_set(&mut self) -> Option<FrameSet> {
        let (name, stamped) = self.rx.recv().ok()?;
        self.latest.insert(name.clone(), stamped);

        // Drain anything else already queued, so we always act on the
        // freshest data rather than working through a backlog.
        while let Ok((n, s)) = self.rx.try_recv() {
            self.latest.insert(n, s);
        }

        Some(FrameSet {
            frames: self
                .latest
                .iter()
                .map(|(k, v)| (k.clone(), Arc::clone(&v.frame)))
                .collect(),
            timestamps: self.latest.iter().map(|(k, v)| (k.clone(), v.at)).collect(),
            triggered_by: name,
        })
    }

    /// Cameras that have died, with the reason. Poll this — a rig running
    /// on one of two cameras looks fine until you ask.
    pub fn failures(&self) -> Vec<(String, String)> {
        self.failures.lock().map(|f| f.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Arc<Frame> {
        Arc::new(Frame {
            width: 2,
            height: 2,
            rgb: vec![0; 12],
        })
    }

    fn set_with(times: &[(&str, Instant)]) -> FrameSet {
        FrameSet {
            frames: times
                .iter()
                .map(|(n, _)| (n.to_string(), frame()))
                .collect(),
            timestamps: times.iter().map(|(n, t)| (n.to_string(), *t)).collect(),
            triggered_by: times
                .first()
                .map(|(n, _)| n.to_string())
                .unwrap_or_default(),
        }
    }

    #[test]
    fn skew_of_one_camera_is_zero() {
        let s = set_with(&[("only", Instant::now())]);
        assert_eq!(s.skew(), Duration::ZERO);
    }

    #[test]
    fn skew_is_the_spread_between_oldest_and_newest() {
        let t0 = Instant::now();
        let s = set_with(&[
            ("a", t0),
            ("b", t0 + Duration::from_millis(40)),
            ("c", t0 + Duration::from_millis(10)),
        ]);
        assert_eq!(s.skew(), Duration::from_millis(40));
    }

    #[test]
    fn age_is_measured_against_the_newest_frame() {
        let t0 = Instant::now();
        let s = set_with(&[("old", t0), ("new", t0 + Duration::from_millis(25))]);
        assert_eq!(s.age_of("new"), Some(Duration::ZERO));
        assert_eq!(s.age_of("old"), Some(Duration::from_millis(25)));
        assert_eq!(s.age_of("absent"), None);
    }

    #[test]
    fn empty_set_has_zero_skew_and_no_ages() {
        let s = set_with(&[]);
        assert!(s.is_empty());
        assert_eq!(s.skew(), Duration::ZERO);
        assert_eq!(s.age_of("anything"), None);
    }

    #[test]
    fn a_rig_with_no_cameras_reports_no_failures() {
        let rig = CameraRig::new();
        assert!(rig.failures().is_empty());
    }
}
