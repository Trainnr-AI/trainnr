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
//!
//! [`GotoController`] sits on top of [`Pid`] and is the robot's actual
//! steering law — the one that was, until the 2026-07-31 architecture
//! review, copy-pasted into the simulator, the firmware, and the camera
//! chase (docs/13-architecture-review.md).

#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use core::f64::consts::FRAC_PI_2;

use crate::exercises::shortest_turn;
use crate::pose::Pose;
use crate::spec::ControlGains;

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

/// Turn a heading error into a body twist — **turn first, then drive**.
///
/// This is the whole steering policy of the robot, and it is three lines:
///
/// ```text
/// w         = pid(heading_error)                       // how hard to turn
/// alignment = 1 - |heading_error| / (pi/2), floored at 0   // are we facing it?
/// v         = speed_budget * alignment                  // how fast to go
/// ```
///
/// # Why the alignment throttle matters
///
/// A differential-drive robot cannot move sideways. If it drives at full
/// speed while badly misaligned, it carves a wide arc *away* from the
/// target and the heading error stays large — the classic slow spiral.
/// Scaling forward speed by alignment makes the robot pivot nearly in
/// place when it is facing the wrong way, and commit to speed only once
/// it is pointed correctly.
///
/// The `.max(0.0)` floor is what stops it driving **backwards** when the
/// target is behind it (error > 90°, so the raw factor goes negative).
/// Without that floor the robot reverses away from its own goal.
///
/// # What this deliberately does not know
///
/// Where the heading error came from. Stage 0 computes it from a map
/// coordinate; `chase` computes it from a camera bearing; a future VLA may
/// emit it directly. All three are the same control problem downstream of
/// the error, which is exactly why this type takes the error as an
/// argument instead of a goal.
pub struct GotoController {
    pub heading_pid: Pid,
    pub gains: ControlGains,
}

impl GotoController {
    pub fn new(gains: ControlGains) -> Self {
        GotoController {
            heading_pid: Pid::new(
                gains.heading_kp,
                gains.heading_ki,
                gains.heading_kd,
                gains.heading_i_limit,
            ),
            gains,
        }
    }

    /// The primitive: heading error + a forward-speed budget → `(v, w)`.
    ///
    /// `speed_budget` is what the caller *would* drive at if perfectly
    /// aligned — distance-proportional for a waypoint, size-proportional
    /// for a camera target. It is capped at `v_max` here so no caller can
    /// forget to.
    pub fn steer(&mut self, heading_error: f64, speed_budget: f64, dt: f64) -> (f64, f64) {
        let w = self.heading_pid.update(heading_error, dt);
        let alignment = (1.0 - heading_error.abs() / FRAC_PI_2).max(0.0);
        let v = speed_budget.min(self.gains.v_max) * alignment;
        (v, w)
    }

    /// Drive toward a known point: bearing → error → [`Self::steer`], with
    /// the speed budget proportional to remaining distance.
    pub fn goto_point(&mut self, pose: &Pose, target: (f64, f64), dt: f64) -> (f64, f64) {
        let (tx, ty) = target;
        let bearing = (ty - pose.y).atan2(tx - pose.x);
        let error = shortest_turn(pose.theta, bearing);
        let budget = self.gains.kp_dist * Self::distance(pose, target);
        self.steer(error, budget, dt)
    }

    /// Straight-line distance from `pose` to `target`, metres.
    pub fn distance(pose: &Pose, target: (f64, f64)) -> f64 {
        (target.0 - pose.x).hypot(target.1 - pose.y)
    }

    /// Has the robot reached `target`?
    pub fn arrived(&self, pose: &Pose, target: (f64, f64)) -> bool {
        Self::distance(pose, target) < self.gains.arrive_radius
    }

    /// Drop accumulated PID state — on a new waypoint, or when the target
    /// is lost, so a reappearing object does not inherit a stale integral.
    pub fn reset(&mut self) {
        self.heading_pid.reset();
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

    fn ctrl() -> GotoController {
        GotoController::new(ControlGains::WAYPOINT)
    }

    #[test]
    fn facing_the_target_gives_full_speed() {
        let mut c = ctrl();
        let (v, w) = c.steer(0.0, 10.0, 0.02);
        assert!((v - ControlGains::WAYPOINT.v_max).abs() < 1e-12, "v = {v}");
        assert!(w.abs() < 1e-12, "no turn needed, got {w}");
    }

    #[test]
    fn misalignment_throttles_forward_speed() {
        let mut c = ctrl();
        // 45 degrees off: alignment = 1 - (pi/4)/(pi/2) = 0.5
        let (v, _) = c.steer(core::f64::consts::FRAC_PI_4, 10.0, 0.02);
        assert!(
            (v - ControlGains::WAYPOINT.v_max * 0.5).abs() < 1e-12,
            "v = {v}"
        );
    }

    #[test]
    fn target_behind_never_drives_backwards() {
        let mut c = ctrl();
        // Anything past 90 degrees would give a negative alignment factor.
        for err in [FRAC_PI_2 + 0.1, 2.0, core::f64::consts::PI] {
            let (v, _) = c.steer(err, 10.0, 0.02);
            assert!(v >= 0.0, "error {err} drove backwards at {v}");
            assert!(v.abs() < 1e-12, "should pivot in place, got v = {v}");
        }
    }

    #[test]
    fn speed_budget_is_capped_at_v_max() {
        let mut c = ctrl();
        let (v, _) = c.steer(0.0, 1000.0, 0.02);
        assert!(v <= ControlGains::WAYPOINT.v_max + 1e-12, "v = {v}");
    }

    #[test]
    fn goto_point_turns_toward_the_target() {
        let mut c = ctrl();
        // Robot at origin facing +x; target is directly to its left (+y).
        let pose = Pose {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
        };
        let (_, w) = c.goto_point(&pose, (0.0, 1.0), 0.02);
        assert!(w > 0.0, "should turn left (positive w), got {w}");
    }

    #[test]
    fn arrival_uses_the_configured_radius() {
        let c = ctrl();
        let pose = Pose {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
        };
        let r = ControlGains::WAYPOINT.arrive_radius;
        assert!(c.arrived(&pose, (r * 0.5, 0.0)));
        assert!(!c.arrived(&pose, (r * 2.0, 0.0)));
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
