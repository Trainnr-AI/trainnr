//! Stage 0 simulator core — and, from the same source, the shared robot
//! math that runs on the microcontroller.
//!
//! Pure, deterministic, no I/O. Knows nothing about rendering or
//! wall-clock time — those live in the frontends (`sim-run`, firmware).
//!
//! # Two shapes, one source
//!
//! ```text
//! cargo build                        -> std: everything below
//! cargo build --no-default-features  -> no_std: the MCU-safe subset
//! ```
//!
//! The `std`-only modules are the ones that ALLOCATE (`Vec`, `BinaryHeap`):
//! a world of walls, a camera returning a scan, an occupancy grid, A*.
//! Bare-metal firmware has no allocator by default, so those stay on the
//! laptop — which is also where they belong architecturally: the big
//! computer maps and plans, the chip does real-time control.
//!
//! Everything else — poses, kinematics, odometry, PID, motor model, RNG,
//! encoders — is fixed-size math and compiles for both. That is how the
//! PID tuned by eye in the Rerun viewer ends up running on the Pico
//! unmodified.

#![cfg_attr(not(feature = "std"), no_std)]

// ---- Always available: fixed-size math, no allocation. ----
pub mod control;
pub mod exercises;
pub mod motor;
pub mod nav;
pub mod odometry;
pub mod pose;
pub mod rng;
pub mod robot;
pub mod safety;
pub mod sensors;
pub mod spec;
pub mod world;

pub use control::{Directive, GotoController, Pid};
pub use motor::Motor;
pub use nav::{lookahead_point, summarize_scan, AvoidHysteresis, Mode, ScanSummary};
pub use odometry::Odometry;
pub use pose::{wrap_angle, Pose};
pub use rng::Rng;
pub use robot::{BodyTwist, DiffDrive, Robot, WheelSpeeds};
pub use safety::{CommandWatchdog, Millis};
pub use sensors::Encoders;
pub use spec::{ControlGains, RobotSpec, DUTY_FULL};
pub use world::Segment;

// ---- std only: these allocate. ----
#[cfg(feature = "std")]
pub mod camera;
#[cfg(feature = "std")]
pub mod grid;
#[cfg(feature = "std")]
pub mod planner;

#[cfg(feature = "std")]
pub use camera::DepthCamera;
#[cfg(feature = "std")]
pub use grid::{Cell, OccupancyGrid};
#[cfg(feature = "std")]
pub use planner::plan;
#[cfg(feature = "std")]
pub use world::World;

/// Float math methods (`cos`, `sqrt`, `hypot`, `atan2`, …) come from `std`
/// on the laptop and from `libm` via this trait on bare metal.
///
/// Re-exported so downstream `no_std` users — our firmware — get the same
/// methods with `use sim_core::Float as _;` instead of each having to
/// depend on `num-traits` and pick matching features. A library that needs
/// a trait to be usable should hand that trait to its callers.
pub use num_traits::Float;
