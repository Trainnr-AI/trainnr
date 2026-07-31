//! One definition of *what the robot is*, and *how it is tuned*.
//!
//! Before this module existed, `WHEEL_RADIUS` was written in four files and
//! the heading gains in three — and the gains had already drifted apart
//! (6.0/0.6 in the simulator and the firmware, 3.0/0.3 in the camera
//! chase) with nothing to say whether that was deliberate. It was. But you
//! could not tell that from the code, and a difference you cannot
//! distinguish from a bug is a bug waiting to be "fixed".
//!
//! So the rule here is:
//!
//! - **Geometry** ([`RobotSpec`]) is a property of the *machine*. It is the
//!   same number everywhere or the robot is lying to itself. One `const`.
//! - **Gains** ([`ControlGains`]) are a property of the *control problem*.
//!   Steering to a known map coordinate and steering onto a jittery camera
//!   bearing are genuinely different problems, so they get genuinely
//!   different named profiles — and the name carries the reason.
//!
//! What does *not* belong here: per-experiment dials like `BEARING_ALPHA`
//! or `AVOID_ENTER`. Those exist to be edited while watching the viewer.
//! DRY applies to definitions, not to knobs.

use crate::robot::DiffDrive;

/// The physical robot: the numbers firmware and simulator must agree on.
///
/// If these ever disagree between the chip and the host, odometry silently
/// integrates a robot that does not exist — the drift looks like sensor
/// noise and is nearly impossible to diagnose from a plot. Hence: one
/// definition, imported by both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobotSpec {
    /// Wheel radius, metres. Converts wheel *angle* to ground *distance*.
    pub wheel_radius: f64,
    /// Distance between the wheel contact patches, metres. Sets how much
    /// turning you get per unit of left/right wheel-speed difference.
    pub track_width: f64,
    /// Encoder counts per full wheel revolution (post-quadrature).
    pub ticks_per_rev: f64,
    /// Wheel speed at full motor command, rad/s. The duty-cycle scale.
    pub max_wheel_rad_s: f64,
}

impl RobotSpec {
    /// The robot this project has been simulating since Stage 0, and which
    /// the Pico drives in the HIL rig. The hardware order (docs/09) is
    /// specced to match these numbers so the transition costs nothing.
    pub const SIM_BOT: RobotSpec = RobotSpec {
        wheel_radius: 0.03,
        track_width: 0.15,
        ticks_per_rev: 1024.0,
        max_wheel_rad_s: 30.0,
    };

    /// The kinematic model implied by this geometry.
    pub fn drive(&self) -> DiffDrive {
        DiffDrive {
            wheel_radius: self.wheel_radius,
            track_width: self.track_width,
        }
    }

    /// Wheel speed (rad/s) → motor command in ±1000 duty units, saturated.
    ///
    /// Lives here because the conversion is only meaningful in terms of
    /// [`Self::max_wheel_rad_s`], and firmware was doing it inline.
    pub fn duty(&self, wheel_rad_s: f64) -> i32 {
        ((wheel_rad_s / self.max_wheel_rad_s) * 1000.0).clamp(-1000.0, 1000.0) as i32
    }
}

/// A tuned control profile: gains plus the speed policy they were tuned
/// against.
///
/// Gains are meaningless without the plant they were tuned for, so the
/// speed limits travel with them rather than sitting in a separate const.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlGains {
    pub heading_kp: f64,
    pub heading_ki: f64,
    pub heading_kd: f64,
    /// Anti-windup clamp on the heading integral.
    pub heading_i_limit: f64,
    /// Forward speed per metre of remaining distance, m/s per m.
    pub kp_dist: f64,
    /// Speed cap, m/s.
    pub v_max: f64,
    /// Close enough to call a waypoint reached, metres.
    pub arrive_radius: f64,
}

impl ControlGains {
    /// Driving to a **known coordinate** — Stage 0's waypoint follower and
    /// the Pico's tour.
    ///
    /// The target is exact and stationary, so the derivative term can be
    /// aggressive: there is no measurement noise for it to amplify.
    /// Tuned by eye in the Rerun viewer during M3.
    pub const WAYPOINT: ControlGains = ControlGains {
        heading_kp: 6.0,
        heading_ki: 0.0,
        heading_kd: 0.6,
        heading_i_limit: 1.0,
        kp_dist: 0.8,
        v_max: 0.45,
        arrive_radius: 0.15,
    };

    /// Steering onto a **camera bearing** — visual servoing.
    ///
    /// Half the gain of [`Self::WAYPOINT`], **on purpose**. The setpoint
    /// here is a detector output that jitters box-to-box every frame, and
    /// D differentiates whatever jitter survives filtering. The plant is
    /// also slower: detection runs at ~20 fps against the sim's 50 Hz, so
    /// the loop has less authority per unit time and a stiff controller
    /// oscillates. See docs/11-perception-stack.md.
    pub const VISUAL_SERVO: ControlGains = ControlGains {
        heading_kp: 3.0,
        heading_ki: 0.0,
        heading_kd: 0.3,
        heading_i_limit: 1.0,
        kp_dist: 0.8,
        v_max: 0.35,
        arrive_radius: 0.15,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duty_saturates_symmetrically() {
        let s = RobotSpec::SIM_BOT;
        assert_eq!(s.duty(s.max_wheel_rad_s), 1000);
        assert_eq!(s.duty(-s.max_wheel_rad_s), -1000);
        assert_eq!(s.duty(s.max_wheel_rad_s * 10.0), 1000, "must clamp");
        assert_eq!(s.duty(0.0), 0);
    }

    #[test]
    fn drive_matches_the_spec() {
        let d = RobotSpec::SIM_BOT.drive();
        assert_eq!(d.wheel_radius, RobotSpec::SIM_BOT.wheel_radius);
        assert_eq!(d.track_width, RobotSpec::SIM_BOT.track_width);
    }

    #[test]
    fn visual_servo_is_gentler_than_waypoint() {
        // Not a style assertion — this encodes the reason the two profiles
        // exist. If someone "unifies" them, this test explains the cost.
        assert!(ControlGains::VISUAL_SERVO.heading_kp < ControlGains::WAYPOINT.heading_kp);
        assert!(ControlGains::VISUAL_SERVO.heading_kd < ControlGains::WAYPOINT.heading_kd);
    }
}
