//! Taming a noisy sensor before it reaches the controller.
//!
//! Stage 0's "sensor" was the exact simulated pose — perfect and
//! noiseless — so a derivative gain was pure benefit. A real detector's
//! box jitters a few pixels every frame, and `Pid`'s D term differentiates
//! that jitter into loud turn commands. The controller is not broken; it
//! is faithfully tracking noise.
//!
//! Two standard, complementary tools. Both are pure functions of their
//! input, so both are unit-testable on the host — no camera required.

/// Exponential moving average: a one-line low-pass filter.
///
/// ```text
/// out = alpha * new + (1 - alpha) * previous_out
/// ```
///
/// `alpha` is the trust you place in each new sample:
/// - `1.0` — no filtering at all (output = input)
/// - `0.3` — responsive, takes the edge off
/// - `0.05` — very smooth, visibly laggy
///
/// The tradeoff is unavoidable and worth feeling: **every bit of smoothing
/// costs responsiveness.** Filter too hard and the robot chases where the
/// object *was*. There is no setting that gives both; the art is choosing
/// where on that line your system should sit.
#[derive(Debug, Clone)]
pub struct LowPass {
    alpha: f64,
    state: Option<f64>,
}

impl LowPass {
    /// `alpha` is clamped to (0, 1].
    pub fn new(alpha: f64) -> Self {
        LowPass {
            alpha: alpha.clamp(1e-6, 1.0),
            state: None,
        }
    }

    /// Feed a sample, get the smoothed value.
    ///
    /// The first sample passes through untouched — starting from zero
    /// would make the filter ramp up from nothing and report a large fake
    /// change on frame one.
    pub fn update(&mut self, x: f64) -> f64 {
        let out = match self.state {
            Some(prev) => self.alpha * x + (1.0 - self.alpha) * prev,
            None => x,
        };
        self.state = Some(out);
        out
    }

    /// Forget history (use when the target is lost, so a reappearing
    /// object doesn't get blended with a stale position).
    pub fn reset(&mut self) {
        self.state = None;
    }

    pub fn value(&self) -> Option<f64> {
        self.state
    }
}

/// Suppress errors too small to be real, without introducing a jump.
///
/// The naive deadband — `if |x| < t { 0 } else { x }` — creates a
/// discontinuity: the command leaps from 0 to `t` the instant the error
/// crosses the threshold, which makes the robot twitch at exactly the
/// boundary you were trying to quiet.
///
/// This version *shifts* instead: output is zero at the threshold and
/// grows continuously beyond it. Same noise rejection, no cliff.
///
/// ```text
///        naive                        shifted
///   out │   ╱                    out │    ╱
///       │  ╱                         │   ╱
///       │ ┌╯  ← jump                 │  ╱
///   ────┼─┴──── in               ────┼─╯──── in
/// ```
pub fn deadband(x: f64, threshold: f64) -> f64 {
    if x.abs() <= threshold {
        0.0
    } else {
        x - threshold * x.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_passes_through() {
        let mut f = LowPass::new(0.2);
        assert_eq!(f.update(5.0), 5.0);
    }

    #[test]
    fn alpha_one_is_a_passthrough() {
        let mut f = LowPass::new(1.0);
        f.update(0.0);
        assert_eq!(f.update(7.0), 7.0);
        assert_eq!(f.update(-3.0), -3.0);
    }

    #[test]
    fn converges_toward_a_constant_input() {
        let mut f = LowPass::new(0.3);
        f.update(0.0);
        for _ in 0..200 {
            f.update(10.0);
        }
        assert!((f.value().unwrap() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn attenuates_alternating_noise() {
        // A signal flipping +1/-1 every sample is pure noise around zero.
        // The filter should crush it toward zero.
        let mut f = LowPass::new(0.2);
        let mut last = 0.0;
        for i in 0..500 {
            last = f.update(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        assert!(last.abs() < 0.2, "noise not attenuated: {last}");
    }

    #[test]
    fn reset_forgets_history() {
        let mut f = LowPass::new(0.1);
        f.update(100.0);
        f.reset();
        assert_eq!(f.update(1.0), 1.0); // passes through again
    }

    #[test]
    fn deadband_silences_small_errors() {
        assert_eq!(deadband(0.01, 0.02), 0.0);
        assert_eq!(deadband(-0.01, 0.02), 0.0);
        assert_eq!(deadband(0.02, 0.02), 0.0);
    }

    #[test]
    fn deadband_is_continuous_at_the_threshold() {
        // Just past the threshold the output must be ~0, not a jump to 0.021.
        let just_past = deadband(0.0201, 0.02);
        assert!(just_past.abs() < 1e-4, "discontinuity: {just_past}");
    }

    #[test]
    fn deadband_preserves_large_errors_minus_the_offset() {
        assert!((deadband(0.5, 0.02) - 0.48).abs() < 1e-12);
        assert!((deadband(-0.5, 0.02) + 0.48).abs() < 1e-12);
    }
}
