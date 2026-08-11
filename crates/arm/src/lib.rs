//! Joint-space control for a serial-chain arm.
//!
//! # What this is not
//!
//! It is not a servo driver, and it holds no geometry. There are no link
//! lengths here and no forward kinematics, because those need a physical
//! arm to measure and guessing them would mean writing tests that encode
//! a fiction. Everything below is arithmetic over joint angles, so it all
//! runs in `cargo test` with nothing plugged in.
//!
//! # The one thing an arm changes about everything you know
//!
//! ```text
//!   a wheeled base    command stops arriving ─▶ cut the outputs ─▶ SAFE
//!   an arm            command stops arriving ─▶ cut the outputs ─▶ IT FALLS
//! ```
//!
//! Every failsafe in this project so far has been built on a machine
//! where "stop driving" is the safe state. `CommandWatchdog` zeroes the
//! duty and drops STBY; the wheels coast; the robot sits there. Do that
//! to an arm holding anything and it collapses under gravity — into the
//! table, into itself, or onto a hand.
//!
//! So the failsafe here **inverts**: the safe response to silence is to
//! *hold the last commanded position and stop accepting new ones*. Stop
//! listening, not stop driving. See [`safety::Guard`].
//!
//! # And the tension that has no library answer
//!
//! Holding costs current, and current makes heat. `docs/e2e-research/24`
//! records that neither Feetech nor the SO-ARM100 repository publishes a
//! duty cycle, thermal rating or MTBF — *"that absence is the finding"*.
//!
//! So an arm must hold to stay safe and must not hold forever to stay
//! alive, and nothing published says where the line is. What makes this
//! tractable is that the servo does report its own temperature —
//! `Present_Temperature`, register 63, one byte — which LeRobot does not
//! surface (issue #1319, closed "not planned"). [`safety::Thermal`] is
//! built around reading it.
//!
//! # Why the shape here is ready for a VLA
//!
//! A policy does not emit one action. π0-class models emit an **action
//! chunk** — a sequence of future joint targets — and the controller
//! executes it until a fresher chunk arrives. Teleoperation emits a chunk
//! of length one. "Move to this pose" emits an interpolated chunk.
//!
//! All three are the same object, so this crate has exactly one input
//! type: [`plan::Plan`]. Nothing needs to grow a second path when a
//! policy is plugged in — a VLA is simply another producer of plans.
//!
//! ```text
//!   teleop  ─┐
//!   goto    ─┼─▶ Plan ─▶ Guard ─▶ Joint::command
//!   a VLA   ─┘           (limits, steps, staleness, heat)
//! ```

#![forbid(unsafe_code)]

pub mod plan;
pub mod safety;
pub mod sim;
pub mod spec;

pub use plan::Plan;
pub use safety::{Guard, Verdict};
pub use spec::{ArmSpec, JointSpec};

/// Something that can be told to hold an angle.
///
/// Deliberately the smaller of the two traits: a cheap PWM hobby servo
/// can do this and nothing more. See [`SensingJoint`] for why that
/// distinction is a type rather than a runtime flag.
pub trait Joint {
    /// Command this joint to `radians`, measured from its calibrated
    /// zero. Positive is the direction its [`JointSpec`] documents.
    ///
    /// Returning `Err` means the command did not reach the hardware. It
    /// does **not** mean the joint stopped — a servo that loses its link
    /// holds its last target, which is the whole reason the failsafe here
    /// is about refusing to send rather than about sending zero.
    fn command(&mut self, radians: f64) -> Result<(), JointError>;
}

/// A joint that can also say where it **actually** is.
///
/// # Why this is a separate trait
///
/// Because feedback is a capability, not an implementation detail. An
/// SG90 cannot tell you its angle — ever — and a `measured()` that
/// returned a guess, a default, or the last commanded value would be a
/// lie that reads exactly like a measurement.
///
/// Splitting it means the compiler enforces the difference. Recording
/// demonstrations, closing a loop, and the step limiter in
/// [`safety::Guard`] all require `SensingJoint`, so they cannot be built
/// against hardware that has no idea where it is.
pub trait SensingJoint: Joint {
    /// Where the joint reports itself to be, in radians from its zero.
    fn measured(&mut self) -> Result<f64, JointError>;

    /// The joint's own temperature in degrees Celsius, if it reports one.
    ///
    /// `None` means "this hardware has no thermometer", which is a
    /// different statement from "it is cold". [`safety::Thermal`] treats
    /// the two differently on purpose: an arm whose heat cannot be
    /// observed is not an arm that is safe to hold indefinitely.
    fn temperature_celsius(&mut self) -> Result<Option<f64>, JointError> {
        Ok(None)
    }
}

/// Why a joint could not be commanded or read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JointError {
    /// The transport failed — cable out, bus silent, checksum bad.
    Link(String),
    /// The joint answered, but with something the caller cannot use.
    Protocol(String),
    /// No joint with this index exists on this arm.
    NoSuchJoint { index: usize, joints: usize },
}

impl core::fmt::Display for JointError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            JointError::Link(why) => write!(f, "link: {why}"),
            JointError::Protocol(why) => write!(f, "protocol: {why}"),
            JointError::NoSuchJoint { index, joints } => {
                write!(f, "joint {index} does not exist — this arm has {joints}")
            }
        }
    }
}

impl std::error::Error for JointError {}
