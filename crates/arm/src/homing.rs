//! Finding zero, for joints that do not know where they are.
//!
//! # The difference an absolute encoder hides
//!
//! An STS3215 has a magnetic absolute encoder: the instant it powers up it
//! knows its angle. A quadrature encoder — an N20, or any DC motor with a
//! wheel on the back — only counts *changes*. On power-up its position is
//! whatever it happens to be, and the count starts at nought there.
//!
//! So for an incremental joint the contract on
//! [`crate::SensingJoint::measured`] — *"radians from its zero"* — is a
//! claim the hardware cannot make until something establishes that zero.
//! Servos hide this so completely that it is easy to design as though it
//! never existed; the STS3215 spells it `Homing_Offset`, register 31.
//!
//! # Why homing needs a motion primitive that [`crate::Joint`] lacks
//!
//! `Joint::command` takes an **angle**. Before a zero exists, an angle
//! means nothing — "go to 0.5 rad" is unanswerable when nothing knows
//! where 0 is. So homing cannot use it, and [`Homing::creep`] exists as
//! the one open-loop motion available beforehand.
//!
//! ```text
//!   after homing    command(radians)   closed loop, against a known zero
//!   before homing   creep(effort)      open loop, direction and no more
//! ```
//!
//! # The procedure, and the failure that matters
//!
//! Creep gently toward a hard stop; when the joint stops making progress,
//! it has arrived, and that place is zero.
//!
//! ⚠️ The failure to design for is a joint that **never** stalls — a
//! disconnected motor, a slipping coupling, a joint with no stop in the
//! direction chosen. It would creep forever and, without a bound, be
//! declared homed at whatever the timeout happened to catch. So
//! [`Seek::TimedOut`] is a distinct outcome and adopts nothing: an arm
//! with no zero is useless, and an arm with a *wrong* zero drives itself
//! into its own hard stops at full commanded speed.

use crate::{Joint, JointError};

/// A joint that must find its zero before its readings mean anything.
pub trait Homing: Joint {
    /// Encoder position since power-up, radians. **Not** from a
    /// calibrated zero — that is the whole point of this trait.
    fn raw_radians(&mut self) -> Result<f64, JointError>;

    /// Adopt `raw_radians` as the new zero. Everything the joint reports
    /// through [`crate::SensingJoint::measured`] afterwards is relative to
    /// it.
    fn adopt_zero(&mut self, raw_radians: f64) -> Result<(), JointError>;

    /// Move open-loop at a signed fraction of full effort, −1.0 to 1.0.
    /// Zero means stop.
    ///
    /// The only motion available before a zero exists. Deliberately not
    /// expressed in radians per second: an un-homed joint has no
    /// calibration to convert effort into speed with.
    fn creep(&mut self, effort: f64) -> Result<(), JointError>;
}

/// Where a homing run has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seek {
    /// Still moving toward the stop.
    Creeping,
    /// Found it, and adopted it as zero.
    Homed,
    /// Gave up. **No zero was adopted** — see the module note.
    TimedOut,
}

/// Drives a joint into its hard stop and calls that zero.
#[derive(Debug, Clone, PartialEq)]
pub struct HomingRun {
    effort: f64,
    still_radians: f64,
    stall_ticks: u32,
    limit_ticks: u32,
    no_progress: u32,
    elapsed: u32,
    last_raw: Option<f64>,
}

impl HomingRun {
    /// `effort` is signed: its sign picks which hard stop to seek.
    ///
    /// `still_radians` is the movement below which a tick counts as no
    /// progress — a detection floor for encoder noise and nothing more,
    /// the same role `still_speed` plays in `sim_core::StuckMonitor`.
    /// ⚠️ Non-finite inputs fail **closed**, toward timing out rather than
    /// toward homing.
    ///
    /// A NaN `still_radians` makes `(raw - previous).abs() > NaN` false,
    /// which counts as *no progress* — so a joint that never moved would
    /// stall-detect immediately and adopt whatever position it powered up
    /// in as zero. A negative threshold instead makes every tick count as
    /// movement, so the run times out and adopts nothing. Given the module
    /// note — *an arm with a wrong zero drives itself into its own hard
    /// stops at full commanded speed* — that is the only acceptable
    /// direction to fail in.
    pub fn new(effort: f64, still_radians: f64, stall_ticks: u32, limit_ticks: u32) -> Self {
        HomingRun {
            effort: if effort.is_finite() { effort } else { 0.0 },
            still_radians: if still_radians.is_finite() {
                still_radians
            } else {
                -1.0
            },
            stall_ticks,
            limit_ticks,
            no_progress: 0,
            elapsed: 0,
            last_raw: None,
        }
    }

    /// One control period of the search.
    pub fn step<J: Homing>(&mut self, joint: &mut J) -> Result<Seek, JointError> {
        self.elapsed = self.elapsed.saturating_add(1);
        if self.elapsed > self.limit_ticks {
            joint.creep(0.0)?;
            return Ok(Seek::TimedOut);
        }

        let raw = joint.raw_radians()?;
        // The first tick has nothing to compare against, so it counts as
        // movement. Starting a stall count from a sample that does not
        // exist would home a joint that never moved at all.
        let moved = self
            .last_raw
            .replace(raw)
            .is_none_or(|previous| (raw - previous).abs() > self.still_radians);
        self.no_progress = if moved {
            0
        } else {
            self.no_progress.saturating_add(1)
        };

        if self.no_progress >= self.stall_ticks {
            joint.creep(0.0)?;
            joint.adopt_zero(raw)?;
            // ⚠️ Stopping the creep is not enough. The joint still holds
            // whatever target it had before homing began — a target set
            // against a zero that no longer exists — and would drive
            // straight to it the moment open-loop motion ends. Now that a
            // zero DOES exist, `command` finally means something: hold
            // here. Same hazard as `Torque::engage`, one layer earlier.
            joint.command(0.0)?;
            return Ok(Seek::Homed);
        }
        joint.creep(self.effort)?;
        Ok(Seek::Creeping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimJoint;
    use crate::SensingJoint;

    /// 50 Hz, a gentle effort, half a second of stillness to call it a
    /// stop, and ten seconds before giving up.
    fn run(effort: f64) -> HomingRun {
        HomingRun::new(effort, 1e-4, 25, 500)
    }

    fn seek(joint: &mut SimJoint, effort: f64) -> Seek {
        let mut homing = run(effort);
        loop {
            let state = homing.step(joint).unwrap();
            joint.step();
            if state != Seek::Creeping {
                return state;
            }
        }
    }

    #[test]
    fn homing_drives_into_the_stop_and_calls_that_place_zero() {
        let mut joint = SimJoint::at(0.5).with_hard_stop(-0.4);
        assert_eq!(seek(&mut joint, -0.3), Seek::Homed);
        assert!(
            joint.measured().unwrap().abs() < 1e-9,
            "the stop is now zero, whatever the raw count says there"
        );
        assert!(
            (joint.raw_radians().unwrap() + 0.4).abs() < 1e-9,
            "and the raw reading is unchanged — homing moves the ORIGIN, not the joint"
        );
    }

    /// The failure that matters. A joint with nothing to stop it must not
    /// be handed a zero, because a wrong zero drives the arm into its own
    /// hard stops at full commanded speed.
    #[test]
    fn a_joint_that_never_stalls_times_out_rather_than_inventing_a_zero() {
        let mut joint = SimJoint::at(0.0); // free — no hard stop
        assert_eq!(seek(&mut joint, -0.3), Seek::TimedOut);
        assert!(
            joint.measured().unwrap() < -1.0,
            "it kept moving, and nothing was adopted as zero"
        );
    }

    /// The failure direction that matters. A NaN threshold must not make
    /// a motionless joint look stalled and hand it a zero.
    #[test]
    fn a_nonsense_stillness_threshold_times_out_rather_than_inventing_a_zero() {
        let mut joint = SimJoint::at(0.3).with_hard_stop(-0.4);
        let mut homing = HomingRun::new(-0.3, f64::NAN, 25, 200);
        let mut state = Seek::Creeping;
        for _ in 0..300 {
            state = homing.step(&mut joint).unwrap();
            joint.step();
            if state != Seek::Creeping {
                break;
            }
        }
        assert_eq!(
            state,
            Seek::TimedOut,
            "NaN must fail toward adopting nothing"
        );
        assert!(
            joint.measured().unwrap() != 0.0,
            "and no zero was adopted, so the reading is still the raw one"
        );
    }

    #[test]
    fn the_sign_of_the_effort_picks_which_stop_is_sought() {
        let mut joint = SimJoint::at(0.0).with_hard_stop(0.6);
        assert_eq!(seek(&mut joint, 0.3), Seek::Homed);
        assert!((joint.raw_radians().unwrap() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn a_homed_joint_is_left_stopped_rather_than_still_creeping() {
        let mut joint = SimJoint::at(0.2).with_hard_stop(-0.1);
        assert_eq!(seek(&mut joint, -0.3), Seek::Homed);
        let settled = joint.raw_radians().unwrap();
        for _ in 0..100 {
            joint.step();
        }
        assert!(
            (joint.raw_radians().unwrap() - settled).abs() < 1e-12,
            "creep must be cancelled on the tick the stop is found"
        );
    }
}
