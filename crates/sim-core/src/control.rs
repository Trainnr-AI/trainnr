//! PID — the workhorse controller of all engineering.
//!
//! Given an error (how far off target we are), produce a corrective effort:
//!
//!   output = Kp·error + Ki·∫error dt + Kd·d(error)/dt
//!
//! - **P** pushes proportionally to how wrong we are now.
//! - **I** accumulates persistent error over time (kills steady-state bias).
//! - **D** reacts to how fast the error is changing (damps overshoot).
//!
//! Math and intuition: docs/learning/math-04-pid-and-lag.md.

pub struct Pid {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    /// Anti-windup clamp: the integral term is kept in [-i_limit, +i_limit].
    /// Without it, a long-blocked robot accumulates a huge integral and then
    /// lurches wildly once freed ("integral windup").
    pub i_limit: f64,
    // Internal state — private, reset via `reset()`.
    integral: f64,
    prev_error: Option<f64>,
}

impl Pid {
    pub fn new(kp: f64, ki: f64, kd: f64, i_limit: f64) -> Self {
        Pid {
            kp,
            ki,
            kd,
            i_limit,
            integral: 0.0,
            prev_error: None,
        }
    }

    /// EXERCISE 4 — implement the PID update.
    ///
    /// Called once per control tick with the current `error` and the tick
    /// duration `dt`. Returns the control output. The recipe:
    ///
    /// 1. Accumulate the integral: add `error * dt` to `self.integral`,
    ///    then clamp it into [-i_limit, +i_limit]. Rust:
    ///    `x.clamp(lo, hi)` returns x limited to that range.
    /// 2. Compute the derivative: `(error - previous_error) / dt` — but on
    ///    the very first call there IS no previous error, so use 0.0.
    ///    `self.prev_error` is an `Option<f64>`: it starts as `None` and
    ///    holds `Some(value)` after that. One clean way:
    ///    `match self.prev_error { Some(prev) => ..., None => 0.0 }`.
    /// 3. Remember this error for next time:
    ///    `self.prev_error = Some(error);`
    /// 4. Return `kp*error + ki*integral + kd*derivative` (all via `self.`).
    pub fn update(&mut self, error: f64, dt: f64) -> f64 {
        self.integral += error * dt;
        self.integral = self.integral.clamp(-self.i_limit, self.i_limit);
        let derivative = match self.prev_error {
            Some(prev) => (error - prev) / dt,
            None => 0.0,
        };
        self.prev_error = Some(error);
        self.kp * error + self.ki * self.integral + self.kd * derivative
    }

    /// Forget accumulated state (use when switching targets/modes).
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_p_is_proportional() {
        let mut pid = Pid::new(2.0, 0.0, 0.0, 1.0);
        assert!((pid.update(1.5, 0.02) - 3.0).abs() < 1e-12);
        assert!((pid.update(-0.5, 0.02) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn integral_accumulates_persistent_error() {
        let mut pid = Pid::new(0.0, 1.0, 0.0, 10.0);
        // Constant error of 1.0 for two ticks of 0.1 s.
        let out1 = pid.update(1.0, 0.1);
        let out2 = pid.update(1.0, 0.1);
        assert!((out1 - 0.1).abs() < 1e-12); // ∫ = 0.1
        assert!((out2 - 0.2).abs() < 1e-12); // ∫ = 0.2 — growing
    }

    #[test]
    fn integral_clamps_against_windup() {
        let mut pid = Pid::new(0.0, 1.0, 0.0, 0.5);
        for _ in 0..1000 {
            pid.update(1.0, 0.1); // would integrate to 100 unclamped
        }
        let out = pid.update(1.0, 0.1);
        assert!(out <= 0.5 + 1e-12); // held at i_limit
    }

    #[test]
    fn derivative_sees_change() {
        let mut pid = Pid::new(0.0, 0.0, 1.0, 1.0);
        pid.update(1.0, 0.1); // first call: no previous error
        let out = pid.update(0.5, 0.1); // error fell by 0.5 in 0.1 s
        assert!((out + 5.0).abs() < 1e-12); // derivative = -5
    }

    #[test]
    fn first_call_has_no_derivative_kick() {
        let mut pid = Pid::new(0.0, 0.0, 1.0, 1.0);
        // Big first error must NOT produce a derivative spike.
        assert!(pid.update(100.0, 0.02).abs() < 1e-12);
    }

    #[test]
    fn reset_clears_state() {
        let mut pid = Pid::new(0.0, 1.0, 1.0, 10.0);
        pid.update(1.0, 0.1);
        pid.update(1.0, 0.1);
        pid.reset();
        // After reset: no integral, no derivative memory.
        assert!(pid.update(0.0, 0.1).abs() < 1e-12);
    }
}
