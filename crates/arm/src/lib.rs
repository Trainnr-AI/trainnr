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
#![cfg_attr(not(feature = "std"), no_std)]

// ---- Always available: fixed-size math, no allocation, no OS. ----
pub mod homing;
pub mod safety;
pub mod sim;
pub mod spec;

pub use homing::{Homing, HomingRun, Seek};
pub use safety::{Guard, Verdict};
pub use spec::{ArmSpec, JointSpec};

// ---- std only: `Plan` holds a chunk of arbitrary length. ----
#[cfg(feature = "std")]
pub mod plan;
#[cfg(feature = "std")]
pub use plan::Plan;

/// The most joints this crate carries.
///
/// Six, because that is a full SO-101 — pan, lift, elbow, wrist flex,
/// wrist roll, gripper. `ArmSpec::so101_four_dof()` uses four of them and
/// the two spare slots are what "adding the wrist is two more servos and
/// no code change here" actually means.
pub const MAX_JOINTS: usize = 6;

/// One value per joint, sized at compile time.
///
/// A `Vec` would need an allocator, and the chip has no heap and will not
/// be given one: a control loop that can fail to allocate mid-tick has a
/// failure mode nobody tests and everybody discovers at 3 a.m.
pub type Joints<T> = heapless::Vec<T, MAX_JOINTS>;

/// Collect into a [`Joints`], refusing to truncate.
///
/// ⚠️ `heapless`'s own `FromIterator` **silently drops** anything past
/// capacity. For a list of joint angles that is the worst available
/// failure: the arm would move the joints that fit and leave the rest —
/// exactly the "act on the overlap" mistake that [`ArmSpec::clamp_all`]
/// and `Plan::chunk` already refuse by returning `None`.
pub fn collect_joints<T>(items: impl IntoIterator<Item = T>) -> Option<Joints<T>> {
    let mut collected = Joints::new();
    for item in items {
        collected.push(item).ok()?;
    }
    Some(collected)
}

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

/// Proof that the arm is somewhere it may be switched off.
///
/// # Why cutting torque needs a witness and holding does not
///
/// Torque-off is simultaneously the **best** thing we can do for these
/// servos and the **worst** thing we can do to a raised arm.
/// `docs/e2e-research/24` rates it *"probably the single biggest win"*
/// for duty cycle — nothing published says how long an STS3215 may hold a
/// load, and a joint that is off makes no heat at all. The same call, one
/// pose earlier, drops the arm on the table.
///
/// So the two are separated by making one of them **unspeakable**:
///
/// ```text
///   something failed   ─▶  hold          (Verdict, every tick, free)
///   idle and parked    ─▶  release       (needs a Parked, issued once)
/// ```
///
/// This type has a private field, so the only way to obtain one is
/// [`safety::Guard::parked`], which checks the arm is at rest at a pose
/// the caller nominated. "Torque off because it is idle" stays easy;
/// "torque off because something went wrong" will not compile.
///
/// # And it cannot be kept
///
/// The lifetime borrows the guard that issued it, so a proof is only
/// usable before that guard is touched again — which in a control loop
/// means *this tick*. Without it a caller could stash a `Parked` and cut
/// torque minutes later, with the arm somewhere else entirely: a proof
/// that was true when written and a lie when used. That is the same
/// staleness bug this repo has hit with clocks and with recordings, and
/// here the borrow checker refuses it outright.
///
/// Stashing the proof and using it after the guard has moved on does not
/// compile, and this is a test rather than a claim:
///
/// ```compile_fail
/// use arm::{sim::SimJoint, ArmSpec, Guard, Torque};
/// let mut guard = Guard::new(ArmSpec::so101_four_dof(), 0.02, 500);
/// let park = vec![0.0; 4];
/// guard.parked(&park, &park, 1e-3);
/// let proof = guard.parked(&park, &park, 1e-3).unwrap();
/// guard.fed(0); // the guard moves on — the proof is now historical
/// SimJoint::at(0.0).release(proof).unwrap();
/// ```
///
/// Releasing within the tick that issued the proof compiles and runs:
///
/// ```
/// use arm::{sim::SimJoint, ArmSpec, Guard, Torque};
/// let mut guard = Guard::new(ArmSpec::so101_four_dof(), 0.02, 500);
/// let park = vec![0.0; 4];
/// guard.parked(&park, &park, 1e-3);
/// let mut joints = [SimJoint::at(0.0), SimJoint::at(0.0)];
/// if let Some(proof) = guard.parked(&park, &park, 1e-3) {
///     for joint in &mut joints {
///         joint.release(proof).unwrap();
///     }
/// }
/// assert!(joints.iter().all(|joint| !joint.is_powered()));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parked<'guard>(core::marker::PhantomData<&'guard ()>);

impl Parked<'_> {
    /// Crate-private: `safety::Guard::parked` is the only issuer.
    pub(crate) fn new() -> Self {
        Parked(core::marker::PhantomData)
    }
}

/// A joint whose holding torque can be switched.
///
/// Separate from [`Joint`] because it is a capability, not an
/// implementation detail — the same reason [`SensingJoint`] is separate.
/// It requires `SensingJoint` because [`Torque::engage`] cannot be made
/// safe without reading where the joint actually is.
///
/// A driver implements one method. The two safe operations are derived
/// from it here, once, rather than in every driver.
pub trait Torque: SensingJoint {
    /// Write the torque-enable register. **The driver's job, not the
    /// controller's** — callers want [`Self::release`] or
    /// [`Self::engage`], which are these semantics done safely.
    fn write_torque(&mut self, on: bool) -> Result<(), JointError>;

    /// Switch the joint off. Only callable with proof of a parked arm.
    fn release(&mut self, _parked: Parked<'_>) -> Result<(), JointError> {
        self.write_torque(false)
    }

    /// Switch the joint back on, without a lurch.
    ///
    /// ⚠️ Commands the present position **first**. A servo remembers its
    /// goal register across a torque-off, so enabling torque on a joint
    /// that has since been moved by hand — or by gravity — snaps it back
    /// to a goal nobody asked for, at whatever speed it can manage. The
    /// step limiter cannot help: that motion never passes through it.
    fn engage(&mut self) -> Result<(), JointError> {
        let here = self.measured()?;
        self.command(here)?;
        self.write_torque(true)
    }
}

/// Why a joint could not be commanded or read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JointError {
    /// The transport failed — cable out, bus silent, checksum bad.
    Link(&'static str),
    /// The joint answered, but with something the caller cannot use.
    Protocol(&'static str),
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

#[cfg(feature = "std")]
impl std::error::Error for JointError {}
