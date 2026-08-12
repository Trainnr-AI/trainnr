//! What an arm is made of: its joints, their travel, and how fast each is
//! allowed to move.
//!
//! The sibling of `sim_core::RobotSpec`, and it exists for the same
//! reason — so that one description of the machine is read by the
//! controller, the safety layer and the simulator, rather than each
//! holding a private copy of the numbers that drifts.

use core::f64::consts::PI;

use crate::{collect_joints, Joints};

/// One joint's identity and limits.
///
/// # Why `name` is here and not just an index
///
/// Because every layer above this speaks names — LeRobot's observation
/// keys are `shoulder_pan.pos`, a demonstration dataset is keyed by
/// joint name, and a person debugging an arm says "joint 2 is hot", not
/// "index 1". An arm whose joints are anonymous forces every one of
/// those to re-derive the mapping, which is how a transposed pair of
/// joints becomes a mystery instead of an error.
#[derive(Debug, Clone, PartialEq)]
pub struct JointSpec {
    /// The name this joint answers to everywhere above this crate.
    /// Matches LeRobot's `so_follower` naming so a dataset recorded here
    /// and a dataset recorded there describe the same joint.
    pub name: &'static str,

    /// The servo's address on the shared bus, 1-253.
    ///
    /// SO-101's follower assigns these in kinematic order from the base
    /// outward: `shoulder_pan`=1 … `gripper`=6. Keeping that means a
    /// calibration file written by LeRobot loads here unchanged.
    pub bus_id: u8,

    /// Travel limits in radians from the calibrated zero, `(min, max)`.
    ///
    /// These come from calibration, not from a datasheet: a joint's real
    /// range is where the printed parts stop it, which differs per build.
    pub travel_radians: (f64, f64),

    /// The fastest this joint may be commanded to move, radians/second.
    ///
    /// ⚠️ Per joint, not one number for the arm, because the joints do
    /// not carry equal loads. SO-101's own BOM proves the point: the
    /// leader arm mixes 1/191, 1/345 and 1/147 gearing across identical
    /// servo bodies, because the shoulder holds everything above it and
    /// the base rotation holds nothing.
    pub max_speed_radians_per_second: f64,

    /// Encoder counts across the servo's full mechanical travel.
    ///
    /// ⚠️ **Not a constant.** STS3215 and STS3250 are 4096; SCS0009 is
    /// **1024**. Code that assumes 4096 silently quadruples every angle
    /// on the cheap servos, which looks like a calibration fault and is
    /// not one.
    pub encoder_counts: u32,
}

impl JointSpec {
    /// Clamp `radians` into this joint's travel.
    ///
    /// NaN is treated as the joint's zero rather than propagated.
    /// `f64::clamp` passes NaN straight through, and a NaN angle would
    /// then compare false against every limit downstream — including the
    /// ones meant to catch it.
    pub fn clamp(&self, radians: f64) -> f64 {
        if radians.is_nan() {
            return 0.0;
        }
        radians.clamp(self.travel_radians.0, self.travel_radians.1)
    }

    /// Radians per encoder count, for turning a raw servo reading into an
    /// angle.
    pub fn radians_per_count(&self) -> f64 {
        (2.0 * PI) / f64::from(self.encoder_counts.max(1))
    }
}

/// A whole arm: its joints, in order from the base outward.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmSpec {
    pub joints: Joints<JointSpec>,
}

impl ArmSpec {
    /// The four joints that make a working pick-and-place arm.
    ///
    /// # Why four and not six
    ///
    /// Base rotation, shoulder, elbow and a gripper reach anywhere in a
    /// half-sphere and can pick things up. The two wrist joints buy
    /// *orientation* at the end effector, which matters for insertion and
    /// for pouring, and `docs/e2e-research/27` already flags sub-millimetre
    /// insertion as a disqualifier for a first wedge.
    ///
    /// The IDs and names are SO-101's, so adding `wrist_flex`(4) and
    /// `wrist_roll`(5) later is two more servos on the same bus and no
    /// code change here.
    pub fn so101_four_dof() -> Self {
        // Four into a capacity of six cannot fail; `expect` here is a
        // statement about MAX_JOINTS, not about runtime input.
        let joints = collect_joints([
            JointSpec {
                name: "shoulder_pan",
                bus_id: 1,
                // Rotation about the base carries no gravity load, so
                // it gets the widest travel and the highest speed.
                travel_radians: (-PI, PI),
                max_speed_radians_per_second: 2.0,
                encoder_counts: 4096,
            },
            JointSpec {
                name: "shoulder_lift",
                bus_id: 2,
                // ⚠️ The joint that holds up everything above it, and
                // therefore the one that gets hot first. Slowest on
                // purpose. SO-101 counterweights this joint on the
                // leader arm for the same reason.
                travel_radians: (-PI / 2.0, PI / 2.0),
                max_speed_radians_per_second: 1.0,
                encoder_counts: 4096,
            },
            JointSpec {
                name: "elbow_flex",
                bus_id: 3,
                travel_radians: (-PI / 2.0, PI / 2.0),
                max_speed_radians_per_second: 1.5,
                encoder_counts: 4096,
            },
            JointSpec {
                name: "gripper",
                bus_id: 6,
                // ⚠️ Keeps SO-101's id 6 even though joints 4 and 5
                // are absent. The gripper is the gripper; renumbering
                // it to 4 would make this arm's calibration file
                // silently incompatible with a six-joint one.
                //
                // Travel is one-sided because a gripper has no
                // meaningful negative — LeRobot normalises it 0..100
                // where the arm joints are -100..100.
                travel_radians: (0.0, PI / 2.0),
                max_speed_radians_per_second: 2.0,
                encoder_counts: 4096,
            },
        ])
        .expect("four joints fit in MAX_JOINTS");
        ArmSpec { joints }
    }

    /// The two N20 gearmotors on the bench, as a 2-DOF arm.
    ///
    /// Names borrowed from SO-101 rather than invented, so a recording
    /// made here describes the same joints a servo arm would. Bus ids are
    /// meaningless — these are PWM channels, not a serial bus — and are
    /// left at SO-101's so nothing downstream has to special-case them.
    ///
    /// ⚠️ `max_speed_radians_per_second` is a **policy cap, not a
    /// measurement**. These motors reach 7.77 rad/s, which at a 20 ms tick
    /// would let a single step cross 0.155 rad and make the step limiter
    /// invisible. 1.5 rad/s is the speed this arm is *allowed*, which is
    /// what the field has always meant.
    ///
    /// Travel is ±π because a gearmotor has no hard limit of its own —
    /// unlike a servo, where travel is where the printed parts stop it.
    pub fn bench_two_joint() -> Self {
        let joints = collect_joints([
            JointSpec {
                name: "shoulder_lift",
                bus_id: 2,
                travel_radians: (-PI, PI),
                max_speed_radians_per_second: 1.5,
                encoder_counts: 4290,
            },
            JointSpec {
                name: "elbow_flex",
                bus_id: 3,
                travel_radians: (-PI, PI),
                max_speed_radians_per_second: 1.5,
                encoder_counts: 4290,
            },
        ])
        .expect("two joints fit in MAX_JOINTS");
        ArmSpec { joints }
    }

    pub fn joints(&self) -> usize {
        self.joints.len()
    }

    /// Look a joint up by the name every layer above this one uses.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|joint| joint.name == name)
    }

    /// Clamp a whole set of angles into their joints' travel.
    ///
    /// Returns `None` when the slice is not this arm's width, rather than
    /// silently working on the joints that happen to line up — a
    /// mismatched length means the caller and the spec disagree about
    /// what machine this is, and acting on the overlap would move the
    /// wrong joints.
    pub fn clamp_all(&self, radians: &[f64]) -> Option<Joints<f64>> {
        if radians.len() != self.joints.len() {
            return None;
        }
        collect_joints(
            self.joints
                .iter()
                .zip(radians)
                .map(|(joint, &wanted)| joint.clamp(wanted)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_four_dof_arm_keeps_so101_names_and_bus_ids() {
        let arm = ArmSpec::so101_four_dof();
        let named: std::vec::Vec<_> = arm.joints.iter().map(|j| (j.name, j.bus_id)).collect();
        assert_eq!(
            named,
            vec![
                ("shoulder_pan", 1),
                ("shoulder_lift", 2),
                ("elbow_flex", 3),
                ("gripper", 6),
            ],
            "a calibration file from a six-joint arm must still line up"
        );
    }

    /// The shoulder holds up every joint above it, so it must never be
    /// the fastest thing on the arm.
    /// The bench arm is real hardware, so its encoder count is the
    /// measured 4290 — not the servo's 4096. Code that assumes either
    /// silently rescales every angle on the other.
    #[test]
    fn the_bench_arm_carries_the_measured_encoder_count_not_a_servos() {
        let bench = ArmSpec::bench_two_joint();
        assert_eq!(bench.joints(), 2);
        assert!(bench.joints.iter().all(|j| j.encoder_counts == 4290));
        assert!(ArmSpec::so101_four_dof()
            .joints
            .iter()
            .all(|j| j.encoder_counts == 4096));
    }

    #[test]
    fn the_shoulder_is_the_slowest_joint() {
        let arm = ArmSpec::so101_four_dof();
        let shoulder = &arm.joints[arm.index_of("shoulder_lift").unwrap()];
        for joint in &arm.joints {
            assert!(
                shoulder.max_speed_radians_per_second <= joint.max_speed_radians_per_second,
                "{} is allowed to move faster than the shoulder",
                joint.name
            );
        }
    }

    /// A gripper has no meaningful negative — it is open or closed.
    #[test]
    fn the_gripper_cannot_be_commanded_past_closed() {
        let arm = ArmSpec::so101_four_dof();
        let gripper = &arm.joints[arm.index_of("gripper").unwrap()];
        assert_eq!(gripper.clamp(-5.0), 0.0, "closed is the floor");
        assert!(gripper.travel_radians.0 >= 0.0);
    }

    #[test]
    fn an_angle_beyond_travel_stops_at_the_limit() {
        let arm = ArmSpec::so101_four_dof();
        let elbow = &arm.joints[arm.index_of("elbow_flex").unwrap()];
        assert_eq!(elbow.clamp(99.0), elbow.travel_radians.1);
        assert_eq!(elbow.clamp(-99.0), elbow.travel_radians.0);
    }

    /// NaN would pass every `>` and `<` downstream, including the limits
    /// meant to catch it.
    #[test]
    fn a_nan_angle_becomes_zero_rather_than_propagating() {
        let arm = ArmSpec::so101_four_dof();
        assert_eq!(arm.joints[0].clamp(f64::NAN), 0.0);
    }

    /// A slice of the wrong width means the caller and the spec disagree
    /// about what machine this is.
    #[test]
    fn commanding_the_wrong_number_of_joints_is_refused_not_truncated() {
        let arm = ArmSpec::so101_four_dof();
        assert!(arm.clamp_all(&[0.0, 0.0, 0.0]).is_none());
        assert!(arm.clamp_all(&[0.0; 5]).is_none());
        assert!(arm.clamp_all(&[0.0; 4]).is_some());
    }

    /// STS3215 is 4096 counts, SCS0009 is 1024. Code that assumes one
    /// silently scales every angle on the other.
    #[test]
    fn resolution_drives_the_angle_scale() {
        let fine = JointSpec {
            encoder_counts: 4096,
            ..ArmSpec::so101_four_dof().joints[0].clone()
        };
        let coarse = JointSpec {
            encoder_counts: 1024,
            ..fine.clone()
        };
        assert!(
            (coarse.radians_per_count() / fine.radians_per_count() - 4.0).abs() < 1e-12,
            "a 1024-count servo must read 4x coarser, not the same"
        );
    }
}
