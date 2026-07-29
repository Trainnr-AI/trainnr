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

/// The physical parameters of a differential-drive robot.
#[derive(Debug, Clone, Copy)]
pub struct DiffDrive {
    /// Wheel radius in meters.
    pub wheel_radius: f64,
    /// Distance between the two wheel contact points, in meters.
    pub track_width: f64,
}

impl DiffDrive {
    /// Forward kinematics: wheel angular velocities (rad/s) → body twist.
    /// Returns `(v, w)`: forward speed (m/s) and turn rate (rad/s).
    pub fn forward(&self, omega_left: f64, omega_right: f64) -> (f64, f64) {
        let v_l = self.wheel_radius * omega_left;
        let v_r = self.wheel_radius * omega_right;
        let v = (v_r + v_l) / 2.0;
        let w = (v_r - v_l) / self.track_width;
        (v, w)
    }

    /// Inverse kinematics: desired body twist → wheel angular velocities.
    /// This is what a controller uses: "I want to go 0.5 m/s while turning
    /// 0.2 rad/s — what do I tell each wheel?"
    pub fn inverse(&self, v: f64, w: f64) -> (f64, f64) {
        let v_l = v - w * self.track_width / 2.0;
        let v_r = v + w * self.track_width / 2.0;
        (v_l / self.wheel_radius, v_r / self.wheel_radius)
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
    pub fn step(&mut self, omega_left: f64, omega_right: f64, dt: f64) {
        let (v, w) = self.model.forward(omega_left, omega_right);
        self.pose = self.pose.integrate(v, w, dt);
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
        let (v, w) = model().forward(10.0, 10.0);
        assert!((v - 0.3).abs() < 1e-12); // 0.03 m * 10 rad/s
        assert!(w.abs() < 1e-12);
    }

    #[test]
    fn opposite_wheels_spin_in_place() {
        let (v, w) = model().forward(-5.0, 5.0);
        assert!(v.abs() < 1e-12);
        assert!(w > 0.0); // right wheel faster → CCW
    }

    #[test]
    fn inverse_then_forward_round_trips() {
        let m = model();
        let (wl, wr) = m.inverse(0.4, -1.3);
        let (v, w) = m.forward(wl, wr);
        assert!((v - 0.4).abs() < 1e-12);
        assert!((w + 1.3).abs() < 1e-12);
    }

    #[test]
    fn robot_drives_a_circle_back_to_start() {
        let m = model();
        let mut robot = Robot {
            model: m,
            pose: Pose::ORIGIN,
        };
        // Command v = 0.3 m/s, w = 0.6 rad/s → circle, period 2π/0.6 s.
        let (wl, wr) = m.inverse(0.3, 0.6);
        let dt = 0.001;
        let steps = ((2.0 * PI / 0.6) / dt).round() as usize;
        for _ in 0..steps {
            robot.step(wl, wr, dt);
        }
        assert!(robot.pose.distance_to(&Pose::ORIGIN) < 1e-3);
    }
}
