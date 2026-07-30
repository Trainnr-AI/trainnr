//! Stage 1 — perception.
//!
//! P0: get camera frames onto the Rerun timeline. No ML yet — this exists
//! to settle the three things that sink Rust webcam projects on macOS
//! (permissions, format negotiation, threading) before a single inference
//! dependency is added.
//!
//! Research and rationale: docs/11-perception-stack.md

pub mod camera;

pub use camera::{CameraSource, Frame, NokhwaCamera};
