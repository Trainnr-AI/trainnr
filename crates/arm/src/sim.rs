//! An arm that exists only in memory, so everything above it can be
//! tested before a servo is bought.
//!
//! The same move `sim-run` made for the base: the U-trap was solved in a
//! simulator before a motor existed, which is why the first hardware run
//! worked. A joint here is a first-order lag toward its target — the same
//! shape `sim_core::Motor` uses for a wheel, because a servo closing its
//! own position loop behaves the same way from outside.

use crate::{Joint, JointError, Parked, SensingJoint, Torque};

/// A joint that obeys instantly-ish and reports honestly.
#[derive(Debug, Clone)]
pub struct SimJoint {
    radians: f64,
    target: f64,
    /// Fraction of the remaining error closed per step. 1.0 is a perfect
    /// joint; real servos are slower and that is worth simulating,
    /// because a controller tuned against a perfect joint discovers lag
    /// on the bench instead of in a test.
    responsiveness: f64,
    celsius: Option<f64>,
    /// Whether holding torque is on. A released joint neither tracks a
    /// target nor accepts one — the same refusal a real servo gives,
    /// which is what makes `engage`'s command-then-enable order testable.
    powered: bool,
}

impl Default for SimJoint {
    fn default() -> Self {
        SimJoint {
            radians: 0.0,
            target: 0.0,
            responsiveness: 0.4,
            celsius: Some(25.0),
            powered: true,
        }
    }
}

impl SimJoint {
    pub fn at(radians: f64) -> Self {
        SimJoint {
            radians,
            target: radians,
            ..SimJoint::default()
        }
    }

    /// A joint with no thermometer — a cheap PWM servo, or one whose
    /// temperature register was never read.
    pub fn without_thermometer(mut self) -> Self {
        self.celsius = None;
        self
    }

    pub fn heat_to(&mut self, celsius: f64) {
        self.celsius = Some(celsius);
    }

    /// Advance one control period.
    ///
    /// ⚠️ Note what this does NOT do: it never moves toward zero when
    /// nothing commands it. A real servo holds its last target under
    /// power, and a simulator that quietly relaxed to zero would hide
    /// exactly the failure this crate exists to get right.
    pub fn step(&mut self) {
        if self.powered {
            self.radians += (self.target - self.radians) * self.responsiveness;
        }
    }

    /// Is holding torque on? For tests and for the viewer.
    pub fn is_powered(&self) -> bool {
        self.powered
    }

    /// Move the joint from outside the control loop — a hand, or gravity
    /// on a released arm. Leaves `target` alone on purpose: that gap
    /// between where a limp joint IS and where it was last TOLD to go is
    /// exactly what makes re-enabling torque dangerous.
    pub fn nudge_to(&mut self, radians: f64) {
        self.radians = radians;
    }
}

impl Joint for SimJoint {
    fn command(&mut self, radians: f64) -> Result<(), JointError> {
        if radians.is_nan() {
            return Err(JointError::Protocol("commanded angle was NaN".into()));
        }
        self.target = radians;
        Ok(())
    }
}

impl SensingJoint for SimJoint {
    fn measured(&mut self) -> Result<f64, JointError> {
        Ok(self.radians)
    }

    fn temperature_celsius(&mut self) -> Result<Option<f64>, JointError> {
        Ok(self.celsius)
    }
}

impl Torque for SimJoint {
    fn write_torque(&mut self, on: bool) -> Result<(), JointError> {
        self.powered = on;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_commanded_joint_converges_on_its_target() {
        let mut joint = SimJoint::at(0.0);
        joint.command(1.0).unwrap();
        for _ in 0..100 {
            joint.step();
        }
        assert!((joint.measured().unwrap() - 1.0).abs() < 1e-6);
    }

    /// The property that makes this a useful stand-in for a servo: an
    /// uncommanded joint HOLDS. A simulator that relaxed to zero would
    /// hide the exact failure this crate exists to prevent.
    #[test]
    fn an_uncommanded_joint_holds_its_position_rather_than_falling() {
        let mut joint = SimJoint::at(0.8);
        for _ in 0..1_000 {
            joint.step();
        }
        assert!((joint.measured().unwrap() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn a_nan_command_is_rejected_at_the_joint() {
        let mut joint = SimJoint::at(0.0);
        assert!(joint.command(f64::NAN).is_err());
        assert_eq!(joint.measured().unwrap(), 0.0, "and it did not move");
    }

    /// The order inside `engage` is the whole point: a servo remembers
    /// its goal across a torque-off, so enabling torque before writing a
    /// fresh goal snaps the joint to wherever it was last told to go.
    #[test]
    fn engaging_torque_adopts_the_present_position_before_switching_on() {
        let mut joint = SimJoint::at(0.0);
        joint.command(1.0).unwrap();
        for _ in 0..100 {
            joint.step();
        }

        // Released, then moved by hand (or by gravity) while limp.
        joint.write_torque(false).unwrap();
        joint.nudge_to(0.2);

        joint.engage().unwrap();
        for _ in 0..100 {
            joint.step();
        }
        assert!(
            (joint.measured().unwrap() - 0.2).abs() < 1e-9,
            "engage must hold where the joint IS, not snap back to the stale 1.0 goal"
        );
    }

    #[test]
    fn a_released_joint_stops_tracking_its_target() {
        let mut joint = SimJoint::at(0.5);
        joint.write_torque(false).unwrap();
        joint.command(1.5).unwrap();
        for _ in 0..1_000 {
            joint.step();
        }
        assert_eq!(joint.measured().unwrap(), 0.5, "no torque, no motion");
    }

    /// "Cannot measure" and "is cold" are different answers, and the
    /// safety layer treats them differently.
    #[test]
    fn a_joint_without_a_thermometer_reports_none_not_a_comfortable_number() {
        let mut blind = SimJoint::at(0.0).without_thermometer();
        assert_eq!(blind.temperature_celsius().unwrap(), None);

        let mut sensing = SimJoint::at(0.0);
        assert_eq!(sensing.temperature_celsius().unwrap(), Some(25.0));
    }
}
