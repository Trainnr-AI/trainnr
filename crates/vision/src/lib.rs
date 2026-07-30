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
pub mod detect;
pub mod filter;
pub mod logging;
pub mod rig;

pub use camera::{CameraSource, Frame, NokhwaCamera};
pub use detect::{Backend, DFineDetector, Detection, Detector};
pub use filter::{deadband, LowPass};
pub use rig::{CameraRig, FrameSet};
