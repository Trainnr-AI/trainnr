//! What the arm has been asked to do — from a thumb, a waypoint, or a
//! policy.
//!
//! # Why there is only one input type
//!
//! Because a VLA does not emit an action. π0-class policies emit an
//! **action chunk**: a sequence of future joint targets, produced every
//! few hundred milliseconds, which the controller executes until a
//! fresher chunk arrives.
//!
//! Teleoperation emits a chunk of length one. "Move to this pose" emits
//! an interpolated chunk. A policy emits a chunk of twenty or fifty.
//! They are the same object at different lengths, so this crate accepts
//! exactly one thing and nothing has to grow a second path when a policy
//! is plugged in.
//!
//! ```text
//!   teleop  ─┐
//!   goto    ─┼─▶ Plan ─▶ Guard ─▶ Joint::command
//!   a VLA   ─┘
//! ```
//!
//! # Superseding, not queueing
//!
//! A newer plan **replaces** the current one outright; it is never
//! appended. Executing a backlog means driving the arm through decisions
//! that were made about a world that has since moved — which is the same
//! reason the radio's outbox is a two-deep drop queue rather than a
//! buffer.

use crate::spec::ArmSpec;

/// A sequence of joint targets to be executed one per control period.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// One entry per step; each entry is one angle per joint.
    steps: Vec<Vec<f64>>,
    /// How far through `steps` execution has got.
    cursor: usize,
}

impl Plan {
    /// A plan that holds a single pose — what teleoperation emits.
    pub fn hold(radians: Vec<f64>) -> Self {
        Plan {
            steps: vec![radians],
            cursor: 0,
        }
    }

    /// A chunk straight from a policy: `steps[i][j]` is joint `j` at step
    /// `i`.
    ///
    /// Returns `None` if any step is not the arm's width, rather than
    /// executing the rows that happen to fit — a chunk of the wrong shape
    /// means the policy and the robot disagree about what machine this
    /// is, and running the overlap moves the wrong joints.
    pub fn chunk(spec: &ArmSpec, steps: Vec<Vec<f64>>) -> Option<Self> {
        if steps.is_empty() || steps.iter().any(|step| step.len() != spec.joints()) {
            return None;
        }
        Some(Plan { steps, cursor: 0 })
    }

    /// A straight-line interpolation from `from` to `to`, in joint space.
    ///
    /// # Why every joint gets the same number of steps
    ///
    /// So that they **arrive together**. Commanding each joint its final
    /// angle and letting the servos run at their own speeds means the
    /// gripper traces whatever path falls out of the slowest joint
    /// finishing last. Splitting the same count across all joints keeps
    /// each one proportionally through its own travel at every instant.
    ///
    /// ⚠️ This matters for a one-shot "go here". It matters much less for
    /// a streaming source, where each plan is already one small step and
    /// the stream *is* the trajectory.
    pub fn interpolate(from: &[f64], to: &[f64], steps: usize) -> Option<Self> {
        if from.len() != to.len() || from.is_empty() {
            return None;
        }
        let steps = steps.max(1);
        let path = (1..=steps)
            .map(|step| {
                let fraction = step as f64 / steps as f64;
                from.iter()
                    .zip(to)
                    .map(|(&start, &end)| start + (end - start) * fraction)
                    .collect()
            })
            .collect();
        Some(Plan {
            steps: path,
            cursor: 0,
        })
    }

    /// The targets for this control period, without advancing.
    ///
    /// Once the plan runs out this keeps returning its final step, which
    /// is a **hold**, not a stop — the arm stays where the plan left it
    /// until something supersedes it or the guard decides otherwise.
    pub fn current(&self) -> &[f64] {
        let last = self.steps.len() - 1;
        &self.steps[self.cursor.min(last)]
    }

    /// Move to the next step. Saturates at the end rather than wrapping.
    pub fn advance(&mut self) {
        if self.cursor + 1 < self.steps.len() {
            self.cursor += 1;
        }
    }

    /// Has every step been reached?
    pub fn finished(&self) -> bool {
        self.cursor + 1 >= self.steps.len()
    }

    pub fn steps(&self) -> usize {
        self.steps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm() -> ArmSpec {
        ArmSpec::so101_four_dof()
    }

    #[test]
    fn a_held_pose_is_a_plan_of_one_step() {
        let plan = Plan::hold(vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(plan.steps(), 1);
        assert_eq!(plan.current(), &[0.1, 0.2, 0.3, 0.4]);
        assert!(plan.finished());
    }

    /// The shape a policy emits, and the shape teleop emits, are the
    /// same type at different lengths.
    #[test]
    fn a_policy_chunk_and_a_teleop_command_are_the_same_type() {
        let chunk = Plan::chunk(&arm(), vec![vec![0.0; 4]; 20]).unwrap();
        let teleop = Plan::hold(vec![0.0; 4]);
        assert_eq!(chunk.steps(), 20);
        assert_eq!(teleop.steps(), 1);
    }

    #[test]
    fn a_chunk_of_the_wrong_width_is_refused_not_truncated() {
        assert!(Plan::chunk(&arm(), vec![vec![0.0; 3]]).is_none());
        assert!(Plan::chunk(&arm(), vec![]).is_none());
        // One bad row spoils the chunk — the policy and the robot
        // disagree, and running the good rows moves the wrong joints.
        assert!(Plan::chunk(&arm(), vec![vec![0.0; 4], vec![0.0; 5]]).is_none());
    }

    /// The whole point of interpolating: proportional progress, so the
    /// joints arrive together whatever their travel.
    #[test]
    fn every_joint_arrives_together_regardless_of_how_far_it_travels() {
        let from = [0.0, 0.0];
        let to = [1.0, 10.0]; // one joint travels ten times as far
        let mut plan = Plan::interpolate(&from, &to, 10).unwrap();
        for _ in 0..5 {
            plan.advance();
        }
        let half = plan.current();
        assert!(
            (half[0] / to[0] - half[1] / to[1]).abs() < 1e-12,
            "both joints must be the same FRACTION through their travel: {half:?}"
        );
    }

    #[test]
    fn an_interpolation_ends_exactly_on_the_target() {
        let mut plan = Plan::interpolate(&[0.0, 0.0], &[1.0, -2.0], 7).unwrap();
        while !plan.finished() {
            plan.advance();
        }
        let end = plan.current();
        assert!((end[0] - 1.0).abs() < 1e-12);
        assert!((end[1] + 2.0).abs() < 1e-12);
    }

    /// Asking to go where you already are must not produce motion.
    #[test]
    fn a_plan_to_the_current_pose_never_moves() {
        let here = [0.3, -0.4];
        let mut plan = Plan::interpolate(&here, &here, 5).unwrap();
        for _ in 0..10 {
            assert_eq!(plan.current(), here);
            plan.advance();
        }
    }

    /// Running out of plan is a hold, not a stop — the arm stays where
    /// the plan left it rather than releasing.
    #[test]
    fn running_off_the_end_of_a_plan_holds_the_last_step() {
        let mut plan = Plan::interpolate(&[0.0], &[1.0], 3).unwrap();
        for _ in 0..50 {
            plan.advance();
        }
        assert!((plan.current()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_zero_step_interpolation_is_clamped_to_one_rather_than_dividing_by_zero() {
        let plan = Plan::interpolate(&[0.0], &[1.0], 0).unwrap();
        assert_eq!(plan.steps(), 1);
        assert!((plan.current()[0] - 1.0).abs() < 1e-12);
    }
}
