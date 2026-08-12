//! What stands between a plan and the hardware.
//!
//! Three independent limits, applied in order, each answering a failure
//! the others cannot see:
//!
//! ```text
//!   travel     "that angle is outside the joint"        ← spec::JointSpec
//!   step       "that is too far from where you ARE"     ← Guard::step
//!   staleness  "nobody has said anything recently"      ← CommandWatchdog
//!   heat       "holding this is cooking the servo"      ← Thermal
//! ```
//!
//! The ordering matters. Clamping travel first and stepping second means
//! a command aimed past the end of a joint becomes a *small move toward*
//! the end, not a small move toward a place the joint cannot reach.

use sim_core::{CommandWatchdog, Freshness, Millis};

use crate::spec::ArmSpec;
use crate::{collect_joints, Joints, Parked, MAX_JOINTS};

/// How far a joint may be commanded from where it currently is, per tick.
///
/// # The bug this exists to not have
///
/// LeRobot has this concept — `max_relative_target` — and the Rust port
/// of it gets it wrong in a way worth writing down. Its limiter clamps
/// the **absolute goal**:
///
/// ```text
///     if val > max_delta { *value = max_delta }
/// ```
///
/// It never reads present position. On a servo with 0-4095 counts of
/// travel and a limit of 100, that does not restrain a step — it drives
/// every joint to near zero on the first command. The one safety feature
/// in the file slams the arm to one end of its range.
///
/// A step limit is only meaningful against **where the joint actually
/// is**, which is why [`Guard::step`] takes `measured` and why the whole
/// thing needs [`crate::SensingJoint`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepLimit {
    pub radians: f64,
}

impl StepLimit {
    /// The step a joint may take in one control period at its top speed.
    pub fn for_speed(max_speed_radians_per_second: f64, period_seconds: f64) -> Self {
        StepLimit {
            radians: (max_speed_radians_per_second * period_seconds).abs(),
        }
    }

    /// `wanted`, pulled back toward `measured` until it is within reach.
    pub fn apply(&self, measured: f64, wanted: f64) -> f64 {
        if wanted.is_nan() || measured.is_nan() {
            return measured;
        }
        let step = wanted - measured;
        if step.abs() <= self.radians {
            wanted
        } else {
            measured + self.radians.copysign(step)
        }
    }
}

/// What [`Guard::authorise`] decided, and why.
///
/// Every variant except [`Self::Move`] means **hold the present
/// position**. That is the arm's safe state and it is the opposite of
/// the base's: cutting the outputs on a wheeled robot makes it coast to
/// a stop, and cutting them on an arm makes it fall.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Send these angles.
    Move { radians: Joints<f64> },
    /// Hold. Nobody has commanded anything yet.
    HoldUncommanded,
    /// Hold. The source went quiet.
    HoldStale { age_ms: Millis },
    /// Hold. The times being compared never shared a clock — see
    /// `sim_core::Freshness::ClockMismatch`. Distinct from `HoldStale`
    /// because the source may be perfectly healthy and the fault ours.
    HoldClockFault { age_ms: Millis },
    /// Hold, and shed load if you can. A joint is too hot to keep
    /// holding, and holding is what made it hot.
    HoldOverheated { joint: &'static str, celsius: f64 },
    /// Hold. This joint has no thermometer and [`Thermal`] was told not to
    /// trust one that cannot be measured.
    ///
    /// A separate variant rather than an overheat with a missing reading:
    /// "too hot at 61 °C" and "cannot tell how hot" are different facts,
    /// they call for different fixes, and squeezing the second into the
    /// first needs a sentinel temperature. This repo has already lost an
    /// evening to a NaN sentinel that silenced an entire output stream.
    HoldUnmeasurable { joint: &'static str },
    /// Refused: the command was not this arm's width.
    Refused { wanted: usize, joints: usize },
}

/// Everything that may stop a plan reaching the joints.
#[derive(Debug, Clone)]
pub struct Guard {
    spec: ArmSpec,
    steps: Joints<StepLimit>,
    source: CommandWatchdog,
    thermal: Thermal,
    /// The pose seen at the previous `parked` check — the only way to
    /// tell "at rest here" from "passing through here".
    last_measured: Option<Joints<f64>>,
}

impl Guard {
    /// `period_seconds` is the control period the caller actually runs
    /// at; step limits are derived from it and each joint's top speed.
    pub fn new(spec: ArmSpec, period_seconds: f64, source_timeout_ms: Millis) -> Self {
        let steps =
            collect_joints(spec.joints.iter().map(|joint| {
                StepLimit::for_speed(joint.max_speed_radians_per_second, period_seconds)
            }))
            .expect("an ArmSpec never exceeds MAX_JOINTS");
        Guard {
            spec,
            steps,
            source: CommandWatchdog::new(source_timeout_ms),
            thermal: Thermal::default(),
            last_measured: None,
        }
    }

    pub fn with_thermal(mut self, thermal: Thermal) -> Self {
        self.thermal = thermal;
        self
    }

    /// A plan arrived and is worth acting on.
    pub fn fed(&mut self, now_ms: Millis) {
        self.source.feed(now_ms);
    }

    /// Decide what may actually be sent.
    ///
    /// `measured` is where the joints report themselves to be *now*;
    /// `wanted` is where the plan says they should go. Both must be this
    /// arm's width.
    pub fn authorise(&mut self, now_ms: Millis, measured: &[f64], wanted: &[f64]) -> Verdict {
        let joints = self.spec.joints();
        if measured.len() != joints || wanted.len() != joints {
            return Verdict::Refused {
                wanted: wanted.len(),
                joints,
            };
        }

        // Heat first: a joint that is cooking must stop being asked to
        // hold, and no amount of freshness in the command changes that.
        if let Some(objection) = self.thermal.objection(&self.spec) {
            return objection;
        }

        match self.source.observe(now_ms) {
            Freshness::NeverFed => return Verdict::HoldUncommanded,
            Freshness::Stale { age_ms } => return Verdict::HoldStale { age_ms },
            Freshness::ClockMismatch { age_ms } => return Verdict::HoldClockFault { age_ms },
            Freshness::Fresh { .. } => {}
        }

        // Travel BEFORE step, so a command aimed past the end of a joint
        // becomes a small move toward the end rather than a small move
        // toward somewhere unreachable.
        let Some(reachable) = self.spec.clamp_all(wanted) else {
            return Verdict::Refused {
                wanted: wanted.len(),
                joints,
            };
        };
        let Some(radians) = collect_joints(
            reachable
                .iter()
                .zip(measured)
                .zip(&self.steps)
                .map(|((&goal, &now), step)| step.apply(now, goal)),
        ) else {
            return Verdict::Refused {
                wanted: wanted.len(),
                joints,
            };
        };
        Verdict::Move { radians }
    }

    /// Proof that torque may be cut, or `None`.
    ///
    /// Both conditions are required, and each catches what the other
    /// cannot:
    ///
    /// ```text
    ///   at the park pose   the arm is somewhere going limp is harmless
    ///   and not moving     it is not merely PASSING THROUGH that pose
    /// ```
    ///
    /// The park pose is the caller's to nominate, deliberately. It is a
    /// property of the bench — a hard stop, a cradle, a folded-down
    /// rest — not of the servos, and this crate holds no geometry with
    /// which to guess one.
    ///
    /// "Not moving" is measured against the previous call, so this must
    /// be called on the control tick like everything else; called twice
    /// in a row within one tick it would see stillness that is really
    /// just a stale reading.
    pub fn parked(&mut self, measured: &[f64], park: &[f64], tolerance: f64) -> Option<Parked<'_>> {
        let still = self
            .last_measured
            .replace(collect_joints(measured.iter().copied())?)
            .is_some_and(|previous| Self::within(&previous, measured, tolerance));
        (measured.len() == self.spec.joints() && still && Self::within(measured, park, tolerance))
            .then_some(Parked::new())
    }

    /// Are two poses the same pose, to within `tolerance`?
    fn within(a: &[f64], b: &[f64], tolerance: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
    }

    /// Record a joint's reported temperature. See [`Thermal`].
    pub fn observe_temperature(&mut self, joint: usize, celsius: Option<f64>) {
        self.thermal.observe(joint, celsius);
    }

    pub fn spec(&self) -> &ArmSpec {
        &self.spec
    }
}

/// The heat limit, and what to do about a joint that cannot report one.
///
/// # Why this is not just a number
///
/// An arm must hold to stay safe and must not hold forever to stay
/// alive. `docs/e2e-research/24` records that no duty cycle, thermal
/// rating or MTBF is published for these servos by anyone — so there is
/// no authority to defer to, only measurement.
///
/// The measurement is available: `Present_Temperature` is register 63 on
/// an STS3215, one byte. LeRobot does not expose it (issue #1319, closed
/// "not planned"), which is why its absence reads as impossible rather
/// than merely unimplemented.
#[derive(Debug, Clone, PartialEq)]
pub struct Thermal {
    /// Hold is abandoned above this. Conservative by default because the
    /// real limit is unpublished — raise it once you have watched a joint
    /// hold a load and logged what it actually reaches.
    pub ceiling_celsius: f64,
    /// What to conclude about a joint with no thermometer.
    ///
    /// ⚠️ Defaults to *trusting* it, and that is a deliberate, arguable
    /// choice. Treating "cannot measure" as "too hot" would make every
    /// PWM-servo arm refuse to move at all, which is useless; treating it
    /// as "fine" means an unmeasurable arm has no thermal protection and
    /// the operator must know that. The honest thing is that neither is
    /// safe, so it is named rather than buried.
    pub trust_joints_without_a_thermometer: bool,
    reported: [Option<f64>; MAX_JOINTS],
}

impl Default for Thermal {
    fn default() -> Self {
        Thermal {
            ceiling_celsius: 55.0,
            trust_joints_without_a_thermometer: true,
            reported: [None; MAX_JOINTS],
        }
    }
}

impl Thermal {
    /// Record what a joint says about itself. Out-of-range indices are
    /// dropped rather than growing anything: the array is already the
    /// widest arm this crate builds.
    pub fn observe(&mut self, joint: usize, celsius: Option<f64>) {
        if let Some(slot) = self.reported.get_mut(joint) {
            *slot = celsius;
        }
    }

    /// The first joint that is a reason not to keep holding, if any.
    ///
    /// Takes its joint count from `spec` alone. It used to also receive a
    /// length from the caller, which is the same fact stored twice with
    /// nothing comparing the two — the shape of most bugs in this repo.
    fn objection(&self, spec: &ArmSpec) -> Option<Verdict> {
        spec.joints.iter().enumerate().find_map(|(index, joint)| {
            match self.reported.get(index).copied().flatten() {
                Some(celsius) if celsius >= self.ceiling_celsius => Some(Verdict::HoldOverheated {
                    joint: joint.name,
                    celsius,
                }),
                None if !self.trust_joints_without_a_thermometer => {
                    Some(Verdict::HoldUnmeasurable { joint: joint.name })
                }
                _ => None,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: f64 = 0.02; // 50 Hz
    const TIMEOUT: Millis = 500;

    fn guard() -> Guard {
        Guard::new(ArmSpec::so101_four_dof(), PERIOD, TIMEOUT)
    }

    fn at_rest() -> Vec<f64> {
        vec![0.0; 4]
    }

    /// The bug in the Rust port of LeRobot: its limiter clamps the
    /// absolute goal instead of the distance from present position, so a
    /// small limit drives every joint to near zero.
    #[test]
    fn the_step_limit_is_measured_from_present_position_not_from_zero() {
        let limit = StepLimit { radians: 0.1 };
        // A joint already far from zero, asked to move a little further.
        let commanded = limit.apply(1.5, 1.55);
        assert!(
            (commanded - 1.55).abs() < 1e-12,
            "a small step from 1.5 is allowed; got {commanded}"
        );
        // The broken version would have clamped this to 0.1.
        assert!(commanded > 1.0, "must not be dragged toward zero");
    }

    #[test]
    fn a_step_larger_than_the_limit_is_shortened_toward_the_target() {
        let limit = StepLimit { radians: 0.1 };
        assert!((limit.apply(0.0, 5.0) - 0.1).abs() < 1e-12);
        assert!((limit.apply(0.0, -5.0) + 0.1).abs() < 1e-12);
    }

    #[test]
    fn step_limits_come_from_each_joints_own_top_speed() {
        let arm = ArmSpec::so101_four_dof();
        let shoulder = StepLimit::for_speed(
            arm.joints[arm.index_of("shoulder_lift").unwrap()].max_speed_radians_per_second,
            PERIOD,
        );
        let pan = StepLimit::for_speed(
            arm.joints[arm.index_of("shoulder_pan").unwrap()].max_speed_radians_per_second,
            PERIOD,
        );
        assert!(
            shoulder.radians < pan.radians,
            "the joint holding the arm up must take smaller steps"
        );
    }

    /// The inversion. Silence must not release the arm.
    #[test]
    fn an_unfed_guard_holds_rather_than_commanding_zero() {
        let mut g = guard();
        assert_eq!(
            g.authorise(0, &at_rest(), &[1.0; 4]),
            Verdict::HoldUncommanded
        );
    }

    #[test]
    fn a_source_that_goes_quiet_holds_and_says_how_long() {
        let mut g = guard();
        g.fed(1_000);
        assert!(matches!(
            g.authorise(1_000, &at_rest(), &[0.05; 4]),
            Verdict::Move { .. }
        ));
        // Aged smoothly at 50 Hz until past the timeout.
        let mut last = Verdict::HoldUncommanded;
        for tick in 1..=40 {
            last = g.authorise(1_000 + tick * 20, &at_rest(), &[0.05; 4]);
        }
        assert!(
            matches!(last, Verdict::HoldStale { .. }),
            "got {last:?} — a quiet source is Stale, not a clock fault"
        );
    }

    /// Reusing `sim_core::Freshness` means the arm inherits the
    /// clock-mismatch detection that cost an evening on the base.
    #[test]
    fn two_clocks_that_never_agreed_are_named_as_such() {
        let mut g = guard();
        g.fed(5_000);
        assert!(matches!(
            g.authorise(60_000, &at_rest(), &[0.05; 4]),
            Verdict::HoldClockFault { .. }
        ));
    }

    #[test]
    fn a_command_of_the_wrong_width_is_refused_not_truncated() {
        let mut g = guard();
        g.fed(0);
        assert!(matches!(
            g.authorise(0, &at_rest(), &[0.0; 3]),
            Verdict::Refused { .. }
        ));
    }

    /// Travel is applied BEFORE the step, so an unreachable goal becomes
    /// a step toward the joint's limit rather than toward nowhere.
    #[test]
    fn a_goal_past_the_joints_travel_still_moves_toward_the_limit() {
        let mut g = guard();
        g.fed(0);
        let verdict = g.authorise(0, &at_rest(), &[99.0, 99.0, 99.0, 99.0]);
        let Verdict::Move { radians } = verdict else {
            panic!("expected a move, got {verdict:?}");
        };
        for angle in &radians {
            assert!(*angle > 0.0, "should have moved toward the limit");
            assert!(*angle < 0.1, "but only by one step: {angle}");
        }
    }

    #[test]
    fn a_joint_over_the_ceiling_stops_being_asked_to_hold() {
        let mut g = guard().with_thermal(Thermal {
            ceiling_celsius: 55.0,
            ..Thermal::default()
        });
        g.fed(0);
        g.observe_temperature(1, Some(61.0));
        assert_eq!(
            g.authorise(0, &at_rest(), &[0.01; 4]),
            Verdict::HoldOverheated {
                joint: "shoulder_lift",
                celsius: 61.0
            },
            "and it names the joint, because 'the arm is hot' sends you to four connectors"
        );
    }

    /// Heat outranks freshness: a perfectly fresh command must not be
    /// allowed to keep cooking a joint.
    #[test]
    fn heat_outranks_a_fresh_command() {
        let mut g = guard();
        g.fed(1_000);
        g.observe_temperature(0, Some(90.0));
        assert!(matches!(
            g.authorise(1_000, &at_rest(), &[0.01; 4]),
            Verdict::HoldOverheated { .. }
        ));
    }

    /// The condition that a pose check alone cannot see: an arm sweeping
    /// *through* the park pose is momentarily at it, and switching off
    /// there drops it mid-swing.
    #[test]
    fn passing_through_the_park_pose_is_not_being_parked_at_it() {
        let mut g = guard();
        let park = at_rest();
        assert!(
            g.parked(&[0.0, 0.0, 0.0, 0.0], &park, 1e-3).is_none(),
            "the first check has no previous pose to compare against"
        );
        assert!(
            g.parked(&[0.02, 0.0, 0.0, 0.0], &park, 1e-3).is_none(),
            "still moving — at the pose, but not stopped there"
        );
        assert!(
            g.parked(&park, &park, 1e-3).is_none(),
            "arrived, but the previous sample was elsewhere"
        );
        assert!(
            g.parked(&park, &park, 1e-3).is_some(),
            "two identical samples at the park pose: now it is parked"
        );
    }

    #[test]
    fn a_still_arm_away_from_its_park_pose_is_not_parked() {
        let mut g = guard();
        let held = vec![0.9, 0.4, -0.2, 0.1];
        g.parked(&held, &at_rest(), 1e-3);
        assert!(
            g.parked(&held, &at_rest(), 1e-3).is_none(),
            "perfectly still, and switching off here would drop it"
        );
    }

    #[test]
    fn a_pose_of_the_wrong_width_is_never_parked() {
        let mut g = guard();
        g.parked(&[0.0; 3], &[0.0; 3], 1e-3);
        assert!(g.parked(&[0.0; 3], &[0.0; 3], 1e-3).is_none());
    }

    /// A joint with no thermometer is trusted by default — and that
    /// choice is deliberate, so it is pinned rather than assumed.
    #[test]
    fn a_joint_that_cannot_report_heat_is_trusted_by_default_and_distrusted_on_request() {
        let mut trusting = guard();
        trusting.fed(0);
        assert!(matches!(
            trusting.authorise(0, &at_rest(), &[0.01; 4]),
            Verdict::Move { .. }
        ));

        let mut strict = guard().with_thermal(Thermal {
            trust_joints_without_a_thermometer: false,
            ..Thermal::default()
        });
        strict.fed(0);
        assert_eq!(
            strict.authorise(0, &at_rest(), &[0.01; 4]),
            Verdict::HoldUnmeasurable {
                joint: "shoulder_pan"
            },
            "'cannot tell how hot' is its own answer, not an overheat with a missing number"
        );
    }
}
