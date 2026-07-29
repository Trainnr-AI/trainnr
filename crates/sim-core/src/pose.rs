//! Pose = where the robot is, in the world frame.
//!
//! Conventions (used everywhere in this repo, worth memorizing):
//! - Units: meters, seconds, radians.
//! - World frame: x right, y up, angles counter-clockwise from the +x axis.
//! - `theta` is always kept wrapped to (-π, π].

// Float math (`cos`, `sqrt`, `exp`, ...) lives in `std`. On bare metal it
// comes from libm through this trait, with identical method syntax.
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use core::f64::consts::PI;

/// Position + heading in the world frame (an element of SE(2)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose {
    pub const ORIGIN: Pose = Pose {
        x: 0.0,
        y: 0.0,
        theta: 0.0,
    };

    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Pose {
            x,
            y,
            theta: wrap_angle(theta),
        }
    }

    /// Advance this pose by body velocities held constant for `dt`:
    /// `v` = forward speed (m/s), `w` = turn rate (rad/s, CCW positive).
    ///
    /// Uses the exact arc solution: with constant (v, w) the robot traces a
    /// circular arc of radius v/w. Euler integration (x += v·cosθ·dt) would
    /// drift outward on curves — with a 50 Hz loop the error is small but
    /// there is no reason to accept it when the closed form is this short.
    pub fn integrate(&self, v: f64, w: f64, dt: f64) -> Pose {
        // Below this turn rate the arc radius v/w blows up numerically;
        // the motion is indistinguishable from a straight line.
        const STRAIGHT_EPS: f64 = 1e-9;

        if w.abs() < STRAIGHT_EPS {
            Pose {
                x: self.x + v * dt * self.theta.cos(),
                y: self.y + v * dt * self.theta.sin(),
                theta: self.theta,
            }
        } else {
            let r = v / w; // signed arc radius
            let theta_next = self.theta + w * dt;
            Pose {
                x: self.x + r * (theta_next.sin() - self.theta.sin()),
                y: self.y - r * (theta_next.cos() - self.theta.cos()),
                theta: wrap_angle(theta_next),
            }
        }
    }

    /// Straight-line distance to another pose (ignores heading).
    pub fn distance_to(&self, other: &Pose) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Wrap an angle to (-π, π].
///
/// THE classic robotics gotcha: heading error computed as `target - current`
/// without wrapping makes a robot at +179° turn 358° the long way to reach
/// -179°. Every angle difference in this codebase must go through here.
pub fn wrap_angle(a: f64) -> f64 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn straight_line_travels_v_times_dt() {
        let p = Pose::new(0.0, 0.0, 0.0).integrate(1.0, 0.0, 2.0);
        assert!((p.x - 2.0).abs() < TOL);
        assert!(p.y.abs() < TOL);
        assert!(p.theta.abs() < TOL);
    }

    #[test]
    fn straight_line_follows_heading() {
        let p = Pose::new(0.0, 0.0, PI / 2.0).integrate(1.0, 0.0, 3.0);
        assert!(p.x.abs() < TOL);
        assert!((p.y - 3.0).abs() < TOL);
    }

    #[test]
    fn full_circle_returns_to_start() {
        // v = 1 m/s, w = 1 rad/s → circle of radius 1 m, period 2π s.
        let start = Pose::new(0.5, -0.25, 0.7);
        let p = start.integrate(1.0, 1.0, 2.0 * PI);
        assert!(p.distance_to(&start) < 1e-6);
        assert!((wrap_angle(p.theta - start.theta)).abs() < 1e-6);
    }

    #[test]
    fn arc_radius_is_v_over_w() {
        // Quarter turn: after θ sweeps 90°, the center of the circle is at
        // distance R to the robot's left; check the chord length R·√2.
        let p = Pose::new(0.0, 0.0, 0.0).integrate(2.0, 1.0, PI / 2.0);
        let r = 2.0;
        let chord = (p.x * p.x + p.y * p.y).sqrt();
        assert!((chord - r * 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn spin_in_place_moves_nothing() {
        let p = Pose::new(1.0, 2.0, 0.0).integrate(0.0, 3.0, 0.5);
        assert!((p.x - 1.0).abs() < TOL);
        assert!((p.y - 2.0).abs() < TOL);
        assert!((p.theta - 1.5).abs() < TOL);
    }

    #[test]
    fn many_small_steps_match_one_big_step() {
        // The exact integrator must be consistent under subdivision.
        let big = Pose::ORIGIN.integrate(1.0, 0.8, 1.0);
        let mut small = Pose::ORIGIN;
        for _ in 0..1000 {
            small = small.integrate(1.0, 0.8, 0.001);
        }
        assert!(big.distance_to(&small) < 1e-9);
    }

    #[test]
    fn wrap_angle_cases() {
        assert!((wrap_angle(0.0)).abs() < TOL);
        assert!((wrap_angle(3.0 * PI) - PI).abs() < TOL);
        assert!((wrap_angle(-3.0 * PI) - PI).abs() < TOL);
        assert!((wrap_angle(2.0 * PI)).abs() < TOL);
        // Shortest-way check: from +170° to -170° should be +20°, not -340°.
        let err = wrap_angle((-170.0_f64).to_radians() - (170.0_f64).to_radians());
        assert!((err - (20.0_f64).to_radians()).abs() < 1e-9);
    }
}
