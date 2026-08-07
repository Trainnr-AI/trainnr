//! Simulated wheel encoders.
//!
//! A real encoder is a disc on the motor shaft with (say) 1024 slots and a
//! light gate counting slot edges — "ticks." It cannot report "the wheel
//! turned 13.7°"; it reports *whole ticks*: 38, then 39, then 39 again...
//! That rounding is **quantization**, and it's one of the reasons odometry
//! drifts. This module reproduces it faithfully.

// Float math (`cos`, `sqrt`, `exp`, ...) lives in `std`. On bare metal it
// comes from libm through this trait, with identical method syntax.
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use core::f64::consts::PI;

use crate::robot::WheelSpeeds;

/// A pair of quantizing encoders (left + right wheel).
pub struct Encoders {
    /// Ticks per full wheel revolution (real hobby encoders: 300–4096).
    pub ticks_per_rev: f64,
    // True accumulated wheel angles (radians) — the encoders' internal
    // ground truth that the outside world never sees directly.
    angle_l: f64,
    angle_r: f64,
    // Whole ticks already reported, so we can emit only the *new* ones.
    reported_l: i64,
    reported_r: i64,
}

impl Encoders {
    pub fn new(ticks_per_rev: f64) -> Self {
        Encoders {
            ticks_per_rev,
            angle_l: 0.0,
            angle_r: 0.0,
            reported_l: 0,
            reported_r: 0,
        }
    }

    /// Advance the wheels at the given speeds over `dt`, and return the
    /// NEW whole ticks observed on (left, right).
    ///
    /// Fractional progress is never lost — it stays in the accumulated
    /// angle and surfaces as a tick later. (Real encoders work the same
    /// way: the disc position remembers everything, you just read it in
    /// steps.)
    pub fn advance(&mut self, wheels: WheelSpeeds, dt: f64) -> (i64, i64) {
        self.angle_l += wheels.left * dt;
        self.angle_r += wheels.right * dt;

        let total_l = (self.angle_l / (2.0 * PI) * self.ticks_per_rev).floor() as i64;
        let total_r = (self.angle_r / (2.0 * PI) * self.ticks_per_rev).floor() as i64;

        let delta = (total_l - self.reported_l, total_r - self.reported_r);
        self.reported_l = total_l;
        self.reported_r = total_r;
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_revolution_yields_ticks_per_rev() {
        let mut enc = Encoders::new(1000.0);
        // Spin both wheels one full turn (2π rad) in 100 steps.
        let mut total = (0, 0);
        for _ in 0..100 {
            let (l, r) = enc.advance(WheelSpeeds::new(2.0 * PI, 2.0 * PI), 0.01);
            total.0 += l;
            total.1 += r;
        }
        // NOT assert_eq!(total, (1000, 1000)): summing 2π·0.01 a hundred
        // times lands a few billionths short of 2π, and floor() then reports
        // 999. Quantization's honest contract is "within one tick" — asserting
        // the exact boundary made the test flaky. (First float lesson: never
        // assert exact equality on accumulated floats, even via floor.)
        assert!((999..=1000).contains(&total.0));
        assert_eq!(total.0, total.1);
    }

    #[test]
    fn slow_rotation_quantizes() {
        // So slow that most steps produce 0 ticks — but nothing is lost.
        let mut enc = Encoders::new(360.0);
        let mut zero_steps = 0;
        let mut total = 0;
        for _ in 0..1000 {
            let (l, _) = enc.advance(WheelSpeeds::new(0.01, 0.0), 0.01); // 0.0001 rad/step
            if l == 0 {
                zero_steps += 1;
            }
            total += l;
        }
        assert!(zero_steps > 900); // quantization: mostly silent
                                   // 0.1 rad total = 0.1/2π rev ≈ 5.7 ticks worth → 5 whole ticks
        assert_eq!(total, 5);
    }

    #[test]
    fn reverse_counts_negative() {
        let mut enc = Encoders::new(1000.0);
        let (l, r) = enc.advance(WheelSpeeds::new(-2.0 * PI, 0.0), 1.0);
        assert_eq!(l, -1000);
        assert_eq!(r, 0);
    }
}
