//! A motor that doesn't do what it's told — immediately.
//!
//! Real motors have inertia and electrical lag: command a speed and the
//! motor *approaches* it, fast at first, then slower — a "first-order lag."
//! The time constant τ (tau) sets the sluggishness: after τ seconds the
//! motor has covered ~63% of the gap to the target; after 3τ, ~95%.
//!
//! This is THE missing ingredient that makes PID meaningful (docs/06 flaw
//! #2): with instant motors a plain P-controller is perfect and there is
//! nothing to learn. With lag, overshoot and oscillation become real.

// Float math (`cos`, `sqrt`, `exp`, ...) lives in `std`. On bare metal it
// comes from libm through this trait, with identical method syntax.
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

pub struct Motor {
    /// Time constant τ in seconds.
    ///
    /// # ⚠️ Measured 2026-08-09: **≈0.03–0.05 s**, not 0.15
    ///
    /// `crates/sim-run` still passes `motor_tau: 0.15`, which was never
    /// sourced — it sat in a defaults literal. The bench says otherwise.
    ///
    /// Both GA12-N20 motors were stepped 0→25→50→75→100% duty while their
    /// encoders were sampled every 20 ms, and the time to cover 63% of each
    /// rise was read off directly. Seven of the eight transitions landed at
    /// 30 ms and one at 50 ms, with the two motors agreeing.
    ///
    /// The sampling interval bounds the precision: 20 ms samples can only
    /// place τ to ±20 ms, so "30 or 50" really means "one or two samples to
    /// reach 63%". What makes it trustworthy is eight independent
    /// transitions clustering, not the precision of any one fit.
    ///
    /// **The simulator has therefore modelled a motor ~4× more sluggish
    /// than the real one.** That matters here more than elsewhere, because
    /// this module's own header argues lag is what makes PID meaningful at
    /// all — so `ControlGains::WAYPOINT`, "tuned by eye in the Rerun viewer
    /// during M3", was tuned against a robot that responds far more slowly
    /// than this hardware does. Expect those gains to want revisiting once
    /// `RobotSpec::REAL_BOT` is complete.
    pub tau: f64,
    /// Physical speed limit (rad/s) — commands beyond it saturate.
    ///
    /// Measured 2026-08-09: **7.71 rad/s** (motor ①) and **7.79** (motor
    /// ②) on four AA cells. `SIM_BOT` assumes 30.0 — 3.9× too fast.
    pub max_speed: f64,
    /// Actual current speed (rad/s). Private: the world reads it via
    /// the return value of `step` (or `speed()`), never sets it.
    omega: f64,
}

impl Motor {
    pub fn new(tau: f64, max_speed: f64) -> Self {
        Motor {
            tau,
            max_speed,
            omega: 0.0,
        }
    }

    /// Command a speed, advance one time step, return what the motor
    /// ACTUALLY does this step.
    ///
    /// # ⚠️ Not modelled: the deadband
    ///
    /// A real motor does nothing at all below the duty needed to overcome
    /// friction and cogging. Measured on this hardware, 2026-08-09:
    ///
    /// ```text
    ///     motor ①   speed = 55.06 × (duty − 4.29%)
    ///     motor ②   speed = 55.37 × (duty − 3.91%)
    /// ```
    ///
    /// Linear above it, to within 7 ticks/s at every point. Note the
    /// *gains* differ by 0.6% while the *deadbands* differ by 10% — the two
    /// wheels are near-identical in speed-per-volt and differ mainly in
    /// friction, which is why the mismatch shrinks as duty rises.
    ///
    /// Adding this would make the simulator refuse to move at low command,
    /// as the real robot does. It is deliberately left out until
    /// `REAL_BOT` is complete, because changing the model moves the Stage 0
    /// baseline and that should happen once, not twice.
    pub fn step(&mut self, command: f64, dt: f64) -> f64 {
        // Saturation: physics doesn't care what you ask for.
        let target = command.clamp(-self.max_speed, self.max_speed);
        // Exact discretization of the first-order lag dω/dt = (target-ω)/τ:
        // each step closes a fixed *fraction* of the remaining gap.
        let alpha = 1.0 - (-dt / self.tau).exp();
        self.omega += (target - self.omega) * alpha;
        self.omega
    }

    pub fn speed(&self) -> f64 {
        self.omega
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaches_63_percent_after_one_tau() {
        let mut m = Motor::new(0.1, 100.0);
        // Step for exactly τ seconds toward a target of 10 rad/s.
        for _ in 0..100 {
            m.step(10.0, 0.001);
        }
        assert!((m.speed() - 6.32).abs() < 0.02); // 1 - 1/e ≈ 0.632
    }

    #[test]
    fn converges_to_target() {
        let mut m = Motor::new(0.1, 100.0);
        for _ in 0..5000 {
            m.step(10.0, 0.001); // 5 s = 50τ: fully settled
        }
        assert!((m.speed() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn saturates_at_max_speed() {
        let mut m = Motor::new(0.05, 20.0);
        for _ in 0..10_000 {
            m.step(1000.0, 0.001); // ask for the impossible
        }
        assert!(m.speed() <= 20.0 + 1e-9);
        assert!((m.speed() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn responds_in_both_directions() {
        let mut m = Motor::new(0.1, 50.0);
        for _ in 0..3000 {
            m.step(-8.0, 0.001);
        }
        assert!((m.speed() + 8.0).abs() < 1e-6);
    }
}
