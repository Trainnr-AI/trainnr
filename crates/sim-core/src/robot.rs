//! The differential-drive robot: two independently driven wheels.
//!
//! Kinematics in one picture (top view, robot pointing right):
//!
//! ```text
//!        left wheel  ────○────
//!                         │      track_width (distance between wheels)
//!        right wheel ────○────
//! ```
//!
//! Each wheel spins at some angular velocity (rad/s). Wheel rim speed is
//! `wheel_radius * omega` (m/s). From the two rim speeds:
//!
//! - forward speed  v = (v_right + v_left) / 2        (average)
//! - turn rate      w = (v_right - v_left) / track    (difference)
//!
//! Same speeds → straight. Right faster → turns left (CCW, positive w).

use crate::pose::Pose;

/// How the whole robot is moving: forward speed and turn rate.
///
/// # Why this is a type and not `(f64, f64)`
///
/// It used to be a bare pair, and so did [`WheelSpeeds`] — which made
/// [`DiffDrive::forward`] and [`DiffDrive::inverse`], exact inverses of
/// each other, **interchangeable to the compiler**. Swapping them at the
/// one line that turns a decision into wheel commands compiled cleanly
/// and took the Stage 0 mission from 1/1 waypoints to 0/1. Measured, not
/// imagined.
///
/// Now that swap is a type error. The fields are spelled out for the same
/// reason: `v` and `w` are what the textbooks call these, and neither says
/// what it is to someone reading the code for the first time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyTwist {
    /// Forward speed along the robot's own heading, m/s. Negative is
    /// reverse.
    pub forward_speed: f64,
    /// Rotation rate about the robot's centre, rad/s. Positive is
    /// counter-clockwise, matching [`Pose::heading`].
    pub turn_rate: f64,
}

impl BodyTwist {
    /// Stopped: no forward motion, no rotation.
    pub const STOPPED: BodyTwist = BodyTwist {
        forward_speed: 0.0,
        turn_rate: 0.0,
    };

    pub const fn new(forward_speed: f64, turn_rate: f64) -> BodyTwist {
        BodyTwist {
            forward_speed,
            turn_rate,
        }
    }
}

/// How fast each wheel is turning, rad/s.
///
/// Not to be confused with a [`BodyTwist`] — that is the *whole robot's*
/// motion, this is the *two wheels'*. They are related by
/// [`DiffDrive::forward`] and [`DiffDrive::inverse`], and keeping them
/// distinct types is what stops those two being used interchangeably.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelSpeeds {
    /// Left wheel angular velocity, rad/s. Positive drives forward.
    pub left: f64,
    /// Right wheel angular velocity, rad/s. Positive drives forward.
    pub right: f64,
}

impl WheelSpeeds {
    pub const STOPPED: WheelSpeeds = WheelSpeeds {
        left: 0.0,
        right: 0.0,
    };

    pub const fn new(left: f64, right: f64) -> WheelSpeeds {
        WheelSpeeds { left, right }
    }

    /// The faster wheel, ignoring direction — what saturation is measured
    /// against.
    pub fn peak(&self) -> f64 {
        self.left.abs().max(self.right.abs())
    }
}

/// The physical parameters of a differential-drive robot.
#[derive(Debug, Clone, Copy)]
pub struct DiffDrive {
    /// Wheel radius in meters.
    pub wheel_radius: f64,
    /// Distance between the two wheel contact points, in meters.
    pub track_width: f64,
}

impl DiffDrive {
    /// Forward kinematics: what the wheels are doing → what the robot is
    /// doing.
    pub fn forward(&self, wheels: WheelSpeeds) -> BodyTwist {
        let left_rim = self.wheel_radius * wheels.left;
        let right_rim = self.wheel_radius * wheels.right;
        BodyTwist {
            forward_speed: (right_rim + left_rim) / 2.0,
            turn_rate: (right_rim - left_rim) / self.track_width,
        }
    }

    /// Inverse kinematics: what we want the robot to do → what to tell the
    /// wheels. This is what a controller uses: "I want to go 0.5 m/s while
    /// turning 0.2 rad/s — what do I tell each wheel?"
    pub fn inverse(&self, twist: BodyTwist) -> WheelSpeeds {
        let half_track = self.track_width / 2.0;
        WheelSpeeds {
            left: (twist.forward_speed - twist.turn_rate * half_track) / self.wheel_radius,
            right: (twist.forward_speed + twist.turn_rate * half_track) / self.wheel_radius,
        }
    }
}

/// A robot instance: the model plus its true pose in the world.
///
/// "True" matters: later, the robot's *believed* pose (odometry) will live
/// separately and drift away from this one. Only the simulator ever sees
/// the true pose — exactly like reality, where no one does.
#[derive(Debug, Clone)]
pub struct Robot {
    pub model: DiffDrive,
    pub pose: Pose,
}

impl Robot {
    /// Advance the true pose by one time step with the given wheel speeds.
    pub fn step(&mut self, wheels: WheelSpeeds, dt: f64) {
        self.pose = self.pose.integrate(self.model.forward(wheels), dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn model() -> DiffDrive {
        DiffDrive {
            wheel_radius: 0.03, // 3 cm — typical N20-motor hobby wheel
            track_width: 0.15,  // 15 cm between wheels
        }
    }

    #[test]
    fn equal_wheels_go_straight() {
        let twist = model().forward(WheelSpeeds::new(10.0, 10.0));
        assert!((twist.forward_speed - 0.3).abs() < 1e-12); // 0.03 m * 10 rad/s
        assert!(twist.turn_rate.abs() < 1e-12);
    }

    #[test]
    fn opposite_wheels_spin_in_place() {
        let twist = model().forward(WheelSpeeds::new(-5.0, 5.0));
        assert!(twist.forward_speed.abs() < 1e-12);
        assert!(twist.turn_rate > 0.0); // right wheel faster → CCW
    }

    #[test]
    fn inverse_then_forward_round_trips() {
        let m = model();
        let wheels = m.inverse(BodyTwist::new(0.4, -1.3));
        let twist = m.forward(wheels);
        assert!((twist.forward_speed - 0.4).abs() < 1e-12);
        assert!((twist.turn_rate + 1.3).abs() < 1e-12);
    }

    #[test]
    fn robot_drives_a_circle_back_to_start() {
        let m = model();
        let mut robot = Robot {
            model: m,
            pose: Pose::ORIGIN,
        };
        // Command v = 0.3 m/s, w = 0.6 rad/s → circle, period 2π/0.6 s.
        let wheels = m.inverse(BodyTwist::new(0.3, 0.6));
        let dt = 0.001;
        let steps = ((2.0 * PI / 0.6) / dt).round() as usize;
        for _ in 0..steps {
            robot.step(wheels, dt);
        }
        assert!(robot.pose.distance_to(&Pose::ORIGIN) < 1e-3);
    }
}
