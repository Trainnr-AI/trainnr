//! Stage 1 — perception.
//!
//! P0: camera frames onto the Rerun timeline.
//! P1: object detection drawn over them.
//! P2: detections steering the Stage 0 robot.
//!
//! Both capture and detection sit behind traits, because both are expected
//! to be replaced: nokhwa's macOS backend is maintainer-disowned, and the
//! detector itself is a stepping stone toward a VLA that subsumes
//! perception entirely (docs/12-model-choice.md).
//!
//! Research and rationale: docs/11-perception-stack.md

pub mod camera;
pub mod cli;
pub mod detect;
pub mod filter;
pub mod lock;
pub mod logging;
pub mod openvocab;
pub mod rig;
pub mod stats;
pub mod target;

pub use camera::{name_matches, request_access, CameraSource, Frame, NokhwaCamera, Source, Stream};
pub use cli::Args;
pub use detect::{Backend, Detection, Detector, DetectorModel, ObjectDetector};
pub use filter::{deadband, LowPass};
pub use lock::{hue_distance, iou, TargetLock};
pub use openvocab::{dominant_hue, hue_name, OpenVocabDetector, Promptable};
pub use rig::{CameraRig, FrameSet};
pub use stats::{percentile, Stats};
pub use target::{approach_factor, pick_target};
