//! Stage 0's mission, separated from the viewer that draws it.
//!
//! `main.rs` used to be one 400-line function: world setup, the control
//! loop, physics, and ~20 Rerun calls interleaved. That is unrunnable in a
//! test — it spawns a viewer and sleeps in real time — so the project's
//! headline result ("mapping + A* beats the U-trap in 22.5 s") was
//! verified only by a human watching a window.
//!
//! Here the mission is a state machine with a `step()` method and no I/O.
//! `main` drives it and draws each [`Tick`]; the regression test drives it
//! as fast as the CPU allows and asserts the outcome.
//!
//! # Determinism
//!
//! Everything is seeded ([`MissionConfig::seed`]) and stepped at a fixed
//! `dt`. There is no wall-clock time and no randomness beyond the seeded
//! [`Rng`], so the same config always produces the same trajectory —
//! which is what makes a regression test on the outcome meaningful rather
//! than flaky.

pub mod mission;

pub use mission::{Mission, MissionConfig, Outcome, Tick};
