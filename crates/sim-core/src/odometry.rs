//! Odometry: the robot's *belief* about where it is, built by integrating
//! encoder ticks. This is dead reckoning — no outside reference, ever.
//!
//! The believed pose and the true pose start identical and then diverge,
//! because odometry only knows: (a) quantized ticks, (b) its *assumed*
//! wheel radius and track width (never exactly the real ones), and (c)
//! nothing about wheel slip. Watching that divergence in the viewer is the
//! whole point of milestone M2.
//!
//! Math: see docs/learning/math-03-odometry.md.

use crate::pose::Pose;
use crate::robot::{BodyTwist, DiffDrive};
use core::f64::consts::PI;

pub struct Odometry {
    /// The parameters odometry BELIEVES the robot has. Deliberately a
    /// separate copy from the true robot's — in Stage 0 experiments we'll
    /// give it a slightly wrong wheel radius and watch the consequences.
    pub model: DiffDrive,
    pub ticks_per_revolution: f64,
    /// The believed pose (starts wherever we're told the robot starts).
    pub pose: Pose,
}

impl Odometry {
    /// EXERCISE 3 — turn encoder tick deltas into an updated believed pose.
    ///
    /// Given the new ticks observed on each wheel since the last update,
    /// advance `self.pose`. The recipe (derive each step in math-03):
    ///
    /// 1. ticks → wheel angle traveled:  Δφ = 2π · dticks / ticks_per_revolution
    /// 2. angle → distance rolled:       d  = wheel_radius · Δφ
    ///    (do 1–2 for each wheel: d_l, d_r)
    /// 3. the two distances → body motion:
    ///    d_center = (d_r + d_l) / 2          (how far forward)
    ///    d_theta  = (d_r - d_l) / track_width (how much turned)
    /// 4. advance the pose along that arc. Trick: `Pose::integrate(v, w, dt)`
    ///    with dt = 1.0 treats v and w as *distances* — so
    ///    `self.pose.integrate(d_center, d_theta, 1.0)` is exactly step 4.
    ///
    /// Hints: `core::f64::consts::PI` is in scope via `PI`;
    /// `dticks` are `i64` — convert with `as f64` before math.
    pub fn update(&mut self, dticks_l: i64, dticks_r: i64) {
        let dphi_l = 2.0 * PI * dticks_l as f64 / self.ticks_per_revolution;
        let dphi_r = 2.0 * PI * dticks_r as f64 / self.ticks_per_revolution;

        let d_l = self.model.wheel_radius * dphi_l;
        let d_r = self.model.wheel_radius * dphi_r;

        let d_center = (d_r + d_l) / 2.0;
        let d_theta = (d_r - d_l) / self.model.track_width;

        // dt = 1.0 because these are already DISTANCES, not speeds: one
        // "second" of travelling at d_center m/s covers d_center metres.
        // The arc formula is the same either way.
        self.pose = self.pose.integrate(BodyTwist::new(d_center, d_theta), 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::Robot;
    use crate::sensors::Encoders;
    use core::f64::consts::PI;

    fn model() -> DiffDrive {
        DiffDrive {
            wheel_radius: 0.03,
            track_width: 0.15,
        }
    }

    fn odo() -> Odometry {
        Odometry {
            model: model(),
            ticks_per_revolution: 1000.0,
            pose: Pose::ORIGIN,
        }
    }

    #[test]
    fn equal_ticks_drive_straight() {
        let mut o = odo();
        // 1000 ticks = one revolution = 2π · 0.03 m of travel.
        o.update(1000, 1000);
        assert!((o.pose.x - 2.0 * PI * 0.03).abs() < 1e-9);
        assert!(o.pose.y.abs() < 1e-9);
        assert!(o.pose.heading.abs() < 1e-9);
    }

    #[test]
    fn opposite_ticks_spin_in_place() {
        let mut o = odo();
        o.update(-100, 100);
        assert!(o.pose.x.abs() < 1e-9);
        assert!(o.pose.y.abs() < 1e-9);
        // Each wheel rolled 0.1 rev = 2π·0.03·0.1 m; dθ = (d_r - d_l)/track.
        let d = 2.0 * PI * 0.03 * 0.1;
        assert!((o.pose.heading - 2.0 * d / 0.15).abs() < 1e-9);
    }

    #[test]
    fn zero_ticks_zero_motion() {
        let mut o = odo();
        o.update(0, 0);
        assert_eq!(o.pose, Pose::ORIGIN);
    }

    /// The design-doc property (flaw #7): with NO noise and MATCHED
    /// parameters, belief must track truth to within quantization error.
    #[test]
    fn matches_truth_exactly_when_noise_free() {
        let m = model();
        let mut robot = Robot {
            model: m,
            pose: Pose::ORIGIN,
        };
        let mut enc = Encoders::new(4096.0);
        let mut o = Odometry {
            model: m,
            ticks_per_revolution: 4096.0,
            pose: Pose::ORIGIN,
        };

        // Drive a wandering path for 10 simulated seconds at 50 Hz.
        let wheels = m.inverse(BodyTwist::new(0.3, 0.5));
        for _ in 0..500 {
            robot.step(wheels, 0.02);
            let (dl, dr) = enc.advance(wheels, 0.02);
            o.update(dl, dr);
        }
        // Only quantization separates belief from truth here (ticks are
        // read in whole steps but nothing is ever lost) — sub-millimeter.
        assert!(o.pose.distance_to(&robot.pose) < 1e-3);
    }
}
