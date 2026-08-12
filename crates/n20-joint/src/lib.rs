//! An arm joint built from a DC gearmotor, for testing the arm before
//! servos exist.
//!
//! # What a servo does that this has to do itself
//!
//! Commanding an STS3215 an angle is one bus write: the servo closes its
//! own position loop internally, holds, and reports back. An N20 gearmotor
//! takes a **duty**, not an angle, and holds nothing. So this crate closes
//! that loop — and the point of doing it *here*, below
//! [`arm::Joint`], is that everything above the trait is then identical
//! for both. `Guard`, `Plan`, `Parked`, the whole stack, cannot tell them
//! apart, which is what makes swapping in a servo a drop-in later.
//!
//! ```text
//!   STS3215   command(angle) ─▶ [servo's own loop] ─▶ holds
//!   N20       command(angle) ─▶ [N20Joint::tick]   ─▶ duty ─▶ TB6612
//! ```
//!
//! # Three things it genuinely cannot fake
//!
//! - **No thermometer.** N20s report no temperature, so
//!   `temperature_celsius()` returns `None` and `Thermal`'s
//!   `trust_joints_without_a_thermometer` decides what happens. That
//!   default was written on paper; this is the first hardware that
//!   actually exercises it.
//! - **No absolute encoder.** Quadrature counts *changes*, so position at
//!   power-up is unknown and [`arm::Homing`] is mandatory rather than
//!   optional. A servo hides this behind `Homing_Offset`, register 31.
//! - **Holding is not free.** A servo holds electrically; a 153:1 N20
//!   holds by whatever the gearbox's static friction gives, and past that
//!   it costs continuous duty. ⚠️ Whether it holds a link at all is a
//!   **measurement nobody has taken yet** — do not assume it.
//!
//! # The seam below
//!
//! [`MotorPort`] is what firmware supplies: a duty, a standby line, and a
//! tick count. Everything above it is arithmetic, so the whole driver is
//! host-testable against a mock — the same shape `Mpu6050<I2C>` uses for
//! its bus.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

use arm::{Homing, Joint, JointError, SensingJoint, Torque};
use core::f64::consts::TAU;
use sim_core::{Pid, DUTY_FULL};

/// The hardware under a joint: one motor channel and its encoder.
///
/// Deliberately three methods and no lifecycle. Opening the port,
/// configuring PWM and wiring interrupts belong to firmware; by the time
/// a `MotorPort` exists it works, and a failure afterwards is a
/// [`JointError`].
pub trait MotorPort {
    /// Signed drive, `-DUTY_FULL..=DUTY_FULL`. Positive must move the
    /// joint in the direction its `JointSpec` documents as positive —
    /// getting this backwards is a wiring fault, not a code one, and the
    /// base has already been bitten by it once.
    fn set_duty(&mut self, duty: i32) -> Result<(), JointError>;

    /// The TB6612's STBY line. `false` cuts the H-bridge outputs, which
    /// is what makes the motor free to be back-driven.
    fn set_standby(&mut self, powered: bool) -> Result<(), JointError>;

    /// Encoder count since power-up. **Cumulative and signed** — a delta
    /// would force every caller to keep its own running total, which is
    /// the same fact stored in as many places as there are callers.
    fn ticks(&mut self) -> Result<i64, JointError>;
}

/// A gearmotor pretending to be a servo, well enough for the layers above
/// to be unable to tell.
pub struct N20Joint<P: MotorPort> {
    port: P,
    /// Measured, never taken from a listing. 4290 for the motors on the
    /// bench — `4290 / 28 = 153:1`, a ratio no supplier published.
    ticks_per_revolution: f64,
    /// The count that means zero. `None` until [`Homing`] establishes it,
    /// which is why `measured()` refuses before then.
    zero_ticks: Option<i64>,
    position: Pid,
    /// Where the joint has been told to go, in calibrated radians.
    target: Option<f64>,
    /// Open-loop effort during homing. `Some` overrides the position loop
    /// entirely, because before a zero exists there is no position to loop
    /// around.
    creeping: Option<f64>,
    powered: bool,
}

impl<P: MotorPort> N20Joint<P> {
    pub fn new(port: P, ticks_per_revolution: f64, position: Pid) -> Self {
        N20Joint {
            port,
            ticks_per_revolution,
            zero_ticks: None,
            position,
            target: None,
            creeping: None,
            powered: true,
        }
    }

    /// Run one control period: read where the joint is, decide a duty,
    /// write it.
    ///
    /// The order matters. Creep outranks the position loop, and an
    /// unpowered or un-homed joint is driven with **zero** rather than
    /// whatever the loop last wanted — a stale duty surviving a change of
    /// state is how the arm crate has already been bitten twice.
    pub fn tick(&mut self, dt: f64) -> Result<(), JointError> {
        if !self.powered {
            return self.port.set_duty(0);
        }
        if let Some(effort) = self.creeping {
            let duty = (effort.clamp(-1.0, 1.0) * f64::from(DUTY_FULL)) as i32;
            return self.port.set_duty(duty);
        }
        let (Some(target), Ok(here)) = (self.target, self.measured()) else {
            return self.port.set_duty(0);
        };
        let duty = self.position.update(target - here, dt) * f64::from(DUTY_FULL);
        self.port
            .set_duty(duty.clamp(-f64::from(DUTY_FULL), f64::from(DUTY_FULL)) as i32)
    }

    fn radians_per_tick(&self) -> f64 {
        TAU / self.ticks_per_revolution.max(1.0)
    }

    /// Give the port back — for firmware that needs to reconfigure it.
    pub fn release_port(self) -> P {
        self.port
    }

    /// The port, borrowed. Firmware needs this to service the encoder
    /// between ticks: on a polled quadrature input somebody has to read
    /// the pins, and that somebody cannot be this crate, which has no
    /// notion of a pin.
    pub fn port_mut(&mut self) -> &mut P {
        &mut self.port
    }
}

impl<P: MotorPort> Joint for N20Joint<P> {
    fn command(&mut self, radians: f64) -> Result<(), JointError> {
        if radians.is_nan() {
            return Err(JointError::Protocol("commanded angle was NaN"));
        }
        if self.zero_ticks.is_none() {
            return Err(JointError::NotHomed);
        }
        self.target = Some(radians);
        Ok(())
    }
}

impl<P: MotorPort> SensingJoint for N20Joint<P> {
    /// ⚠️ Refuses before homing rather than reporting a number.
    ///
    /// An incremental encoder reads zero wherever it powered up. Returning
    /// that as "radians from the calibrated zero" would be a measurement
    /// that is confidently, silently wrong — and the step limiter would
    /// then compute its distances against a fiction.
    fn measured(&mut self) -> Result<f64, JointError> {
        let zero = self.zero_ticks.ok_or(JointError::NotHomed)?;
        Ok((self.port.ticks()? - zero) as f64 * self.radians_per_tick())
    }

    /// Always `None`. An N20 has no thermometer, and saying so is the
    /// point — see the module note.
    fn temperature_celsius(&mut self) -> Result<Option<f64>, JointError> {
        Ok(None)
    }
}

impl<P: MotorPort> Torque for N20Joint<P> {
    /// ⚠️ Zeroes the duty **before** dropping standby, and clears the
    /// target on the way out.
    ///
    /// Dropping STBY with a duty still set leaves the H-bridge inputs
    /// asserted against a disabled output stage — harmless on a TB6612,
    /// untidy everywhere, and the sort of thing that differs per driver
    /// chip. Clearing the target matters more: re-enabling into a stale
    /// goal is exactly what [`Torque::engage`] exists to prevent.
    fn write_torque(&mut self, on: bool) -> Result<(), JointError> {
        if !on {
            self.port.set_duty(0)?;
            self.target = None;
            self.position.reset();
        }
        self.powered = on;
        self.port.set_standby(on)
    }
}

impl<P: MotorPort> Homing for N20Joint<P> {
    fn raw_radians(&mut self) -> Result<f64, JointError> {
        Ok(self.port.ticks()? as f64 * self.radians_per_tick())
    }

    fn adopt_zero(&mut self, raw_radians: f64) -> Result<(), JointError> {
        self.zero_ticks = Some((raw_radians / self.radians_per_tick()) as i64);
        self.position.reset();
        Ok(())
    }

    fn creep(&mut self, effort: f64) -> Result<(), JointError> {
        self.creeping = (effort != 0.0).then_some(effort);
        if self.creeping.is_none() {
            self.port.set_duty(0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arm::{HomingRun, Seek};

    const TICKS_PER_REV: f64 = 4290.0; // measured on the bench motors
    const DT: f64 = 0.02;

    /// A motor that integrates duty into encoder counts, with an optional
    /// wall to home against.
    struct MockPort {
        ticks: i64,
        duty: i32,
        standby: bool,
        stop_ticks: Option<i64>,
    }

    impl MockPort {
        fn new() -> Self {
            MockPort {
                ticks: 0,
                duty: 0,
                standby: true,
                stop_ticks: None,
            }
        }

        fn with_stop(mut self, ticks: i64) -> Self {
            self.stop_ticks = Some(ticks);
            self
        }

        /// Advance the motor by whatever duty was last written.
        fn spin(&mut self) {
            if !self.standby {
                return;
            }
            self.ticks += i64::from(self.duty) / 20;
            if let Some(stop) = self.stop_ticks {
                self.ticks = self.ticks.max(stop);
            }
        }
    }

    impl MotorPort for MockPort {
        fn set_duty(&mut self, duty: i32) -> Result<(), JointError> {
            self.duty = duty;
            Ok(())
        }
        fn set_standby(&mut self, powered: bool) -> Result<(), JointError> {
            self.standby = powered;
            Ok(())
        }
        fn ticks(&mut self) -> Result<i64, JointError> {
            Ok(self.ticks)
        }
    }

    fn joint(port: MockPort) -> N20Joint<MockPort> {
        N20Joint::new(port, TICKS_PER_REV, Pid::new(4.0, 0.0, 0.05, 1.0))
    }

    /// The whole reason `measured()` returns a Result.
    #[test]
    fn an_unhomed_joint_refuses_to_report_a_position_or_take_a_command() {
        let mut j = joint(MockPort::new());
        assert_eq!(j.measured(), Err(JointError::NotHomed));
        assert_eq!(j.command(0.5), Err(JointError::NotHomed));
    }

    #[test]
    fn homing_finds_the_stop_and_the_joint_then_reads_zero_there() {
        let mut j = joint(MockPort::new().with_stop(-600));
        let mut run = HomingRun::new(-0.3, 1e-6, 25, 500);
        loop {
            let state = run.step(&mut j).unwrap();
            j.tick(DT).unwrap();
            j.port.spin();
            if state != Seek::Creeping {
                assert_eq!(state, Seek::Homed);
                break;
            }
        }
        assert!(
            j.measured().unwrap().abs() < 1e-9,
            "the hard stop is zero now"
        );
        assert_eq!(j.port.duty, 0, "and it stopped creeping");
    }

    /// Closing the loop is this crate's whole job.
    #[test]
    fn a_homed_joint_drives_toward_its_target_and_settles_there() {
        let mut j = joint(MockPort::new());
        j.adopt_zero(0.0).unwrap();
        j.command(0.5).unwrap();
        for _ in 0..400 {
            j.tick(DT).unwrap();
            j.port.spin();
        }
        assert!(
            (j.measured().unwrap() - 0.5).abs() < 0.02,
            "settled at {:?}",
            j.measured()
        );
    }

    #[test]
    fn the_duty_points_the_way_the_error_does() {
        let mut j = joint(MockPort::new());
        j.adopt_zero(0.0).unwrap();

        j.command(1.0).unwrap();
        j.tick(DT).unwrap();
        assert!(j.port.duty > 0, "target above: drive positive");

        j.command(-1.0).unwrap();
        j.tick(DT).unwrap();
        assert!(j.port.duty < 0, "target below: drive negative");
    }

    /// Releasing must not leave a duty asserted against a disabled bridge,
    /// and must not keep a goal that re-engaging would snap back to.
    #[test]
    fn releasing_zeroes_the_duty_and_forgets_the_goal() {
        let mut j = joint(MockPort::new());
        j.adopt_zero(0.0).unwrap();
        j.command(1.0).unwrap();
        j.tick(DT).unwrap();
        assert!(j.port.duty != 0);

        j.write_torque(false).unwrap();
        assert_eq!(j.port.duty, 0);
        assert!(!j.port.standby);

        for _ in 0..50 {
            j.tick(DT).unwrap();
            j.port.spin();
        }
        assert_eq!(j.port.ticks, 0, "an unpowered joint does not move");
    }

    /// The first hardware that actually exercises `Thermal`'s deliberately
    /// arguable default.
    #[test]
    fn an_n20_reports_no_temperature_because_it_has_no_thermometer() {
        let mut j = joint(MockPort::new());
        assert_eq!(j.temperature_celsius().unwrap(), None);
    }

    #[test]
    fn a_nan_command_never_reaches_the_motor() {
        let mut j = joint(MockPort::new());
        j.adopt_zero(0.0).unwrap();
        assert!(j.command(f64::NAN).is_err());
        j.tick(DT).unwrap();
        assert_eq!(j.port.duty, 0);
    }
}
