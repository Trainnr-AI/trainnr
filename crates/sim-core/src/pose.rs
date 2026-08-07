//! Pose = where the robot is, in the world frame.
//!
//! Conventions (used everywhere in this repo, worth memorizing):
//! - Units: meters, seconds, radians.
//! - World frame: x right, y up, angles counter-clockwise from the +x axis.
//! - `heading` is always kept wrapped to (-π, π].

// Float math (`cos`, `sqrt`, `exp`, ...) lives in `std`. On bare metal it
// comes from libm through this trait, with identical method syntax.
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use core::f64::consts::PI;

use crate::robot::BodyTwist;

/// A location in the world, metres.
///
/// A [`Pose`] is a place *and a facing*; a `Point` is just the place. The
/// distinction matters because most of the navigation code deals in
/// places — waypoints, path nodes, the ends of a wall — and only the robot
/// itself has a heading.
///
/// `x` and `y` keep their names: unlike `v`/`w`/`kp`, they are not jargon
/// standing in for a longer word — they *are* the words for Cartesian
/// coordinates, and `horizontal`/`vertical` would read worse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Metres along the world's +x axis (to the right on screen).
    pub x: f64,
    /// Metres along the world's +y axis (up on screen).
    pub y: f64,
}

impl Point {
    pub const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

    pub const fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    /// Straight-line distance to another point, metres.
    pub fn distance_to(&self, other: Point) -> f64 {
        (other.x - self.x).hypot(other.y - self.y)
    }
}

/// Position + heading in the world frame (an element of SE(2)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
}

impl Pose {
    pub const ORIGIN: Pose = Pose {
        x: 0.0,
        y: 0.0,
        heading: 0.0,
    };

    /// Where this pose is, discarding which way it faces.
    pub fn position(&self) -> Point {
        Point {
            x: self.x,
            y: self.y,
        }
    }

    pub fn new(x: f64, y: f64, heading: f64) -> Self {
        Pose {
            x,
            y,
            heading: wrap_angle(heading),
        }
    }

    /// Advance this pose by a body twist held constant for `dt`.
    ///
    /// Uses the exact arc solution: with a constant twist the robot traces
    /// a circular arc of radius `forward_speed / turn_rate`. Euler
    /// integration (x += v·cosθ·dt) would drift outward on curves — with a
    /// 50 Hz loop the error is small but there is no reason to accept it
    /// when the closed form is this short.
    pub fn integrate(&self, twist: BodyTwist, dt: f64) -> Pose {
        // Locals keep their short names on purpose: below, the arithmetic
        // is the textbook arc formula and reads best in the textbook's
        // letters. The *stored* values say what they are; the working
        // inside one formula does not need to.
        let (v, w) = (twist.forward_speed, twist.turn_rate);

        // Below this turn rate the arc radius v/w blows up numerically;
        // the motion is indistinguishable from a straight line.
        const STRAIGHT_EPS: f64 = 1e-9;

        if w.abs() < STRAIGHT_EPS {
            Pose {
                x: self.x + v * dt * self.heading.cos(),
                y: self.y + v * dt * self.heading.sin(),
                heading: self.heading,
            }
        } else {
            let r = v / w; // signed arc radius
            let theta_next = self.heading + w * dt;
            Pose {
                x: self.x + r * (theta_next.sin() - self.heading.sin()),
                y: self.y - r * (theta_next.cos() - self.heading.cos()),
                heading: wrap_angle(theta_next),
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
    use crate::robot::BodyTwist;

    const TOL: f64 = 1e-9;

    #[test]
    fn straight_line_travels_v_times_dt() {
        let p = Pose::new(0.0, 0.0, 0.0).integrate(BodyTwist::new(1.0, 0.0), 2.0);
        assert!((p.x - 2.0).abs() < TOL);
        assert!(p.y.abs() < TOL);
        assert!(p.heading.abs() < TOL);
    }

    #[test]
    fn straight_line_follows_heading() {
        let p = Pose::new(0.0, 0.0, PI / 2.0).integrate(BodyTwist::new(1.0, 0.0), 3.0);
        assert!(p.x.abs() < TOL);
        assert!((p.y - 3.0).abs() < TOL);
    }

    #[test]
    fn full_circle_returns_to_start() {
        // v = 1 m/s, w = 1 rad/s → circle of radius 1 m, period 2π s.
        let start = Pose::new(0.5, -0.25, 0.7);
        let p = start.integrate(BodyTwist::new(1.0, 1.0), 2.0 * PI);
        assert!(p.distance_to(&start) < 1e-6);
        assert!((wrap_angle(p.heading - start.heading)).abs() < 1e-6);
    }

    #[test]
    fn arc_radius_is_v_over_w() {
        // Quarter turn: after θ sweeps 90°, the center of the circle is at
        // distance R to the robot's left; check the chord length R·√2.
        let p = Pose::new(0.0, 0.0, 0.0).integrate(BodyTwist::new(2.0, 1.0), PI / 2.0);
        let r = 2.0;
        let chord = (p.x * p.x + p.y * p.y).sqrt();
        assert!((chord - r * 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn spin_in_place_moves_nothing() {
        let p = Pose::new(1.0, 2.0, 0.0).integrate(BodyTwist::new(0.0, 3.0), 0.5);
        assert!((p.x - 1.0).abs() < TOL);
        assert!((p.y - 2.0).abs() < TOL);
        assert!((p.heading - 1.5).abs() < TOL);
    }

    #[test]
    fn many_small_steps_match_one_big_step() {
        // The exact integrator must be consistent under subdivision.
        let big = Pose::ORIGIN.integrate(BodyTwist::new(1.0, 0.8), 1.0);
        let mut small = Pose::ORIGIN;
        for _ in 0..1000 {
            small = small.integrate(BodyTwist::new(1.0, 0.8), 0.001);
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
