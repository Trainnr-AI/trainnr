//! The whole crate driving a simulated arm, with no hardware and no
//! geometry — the arm equivalent of `sim-run` solving the U-trap before
//! a motor existed.

use arm::{sim::SimJoint, ArmSpec, Guard, Joint, Plan, SensingJoint, Verdict};

const PERIOD: f64 = 0.02;
const TIMEOUT_MS: u64 = 500;

fn rig() -> (Guard, Vec<SimJoint>) {
    let spec = ArmSpec::so101_four_dof();
    let joints = vec![SimJoint::at(0.0); spec.joints()];
    (Guard::new(spec, PERIOD, TIMEOUT_MS), joints)
}

fn measured(joints: &mut [SimJoint]) -> Vec<f64> {
    joints.iter_mut().map(|j| j.measured().unwrap()).collect()
}

/// A plan executed tick by tick reaches its target and stays there.
#[test]
fn an_arm_follows_a_plan_to_its_target_and_holds() {
    let (mut guard, mut joints) = rig();
    let goal = vec![0.5, 0.4, 0.3, 0.2];
    let mut plan = Plan::interpolate(&vec![0.0; 4], &goal, 40).unwrap();

    for tick in 0..400u64 {
        let now_ms = tick * 20;
        guard.fed(now_ms);
        let here = measured(&mut joints);
        if let Verdict::Move { radians } = guard.authorise(now_ms, &here, plan.current()) {
            for (joint, angle) in joints.iter_mut().zip(&radians) {
                joint.command(*angle).unwrap();
            }
        }
        joints.iter_mut().for_each(SimJoint::step);
        plan.advance();
    }

    for (angle, want) in measured(&mut joints).iter().zip(&goal) {
        assert!((angle - want).abs() < 1e-3, "got {angle}, wanted {want}");
    }
}

/// The inversion, end to end — in two halves, because both matter.
///
/// After the commander dies the arm keeps executing its last plan for
/// exactly the watchdog timeout, and only then freezes. Asserting only
/// the freeze would let a bug that stops instantly pass; asserting only
/// the grace period would let one that never stops pass.
#[test]
fn an_arm_whose_commander_dies_finishes_its_grace_period_then_holds_forever() {
    let (mut guard, mut joints) = rig();
    let mut plan = Plan::interpolate(&vec![0.0; 4], &vec![0.6; 4], 200).unwrap();

    // Driven normally.
    let mut tick = 0u64;
    while tick < 25 {
        let now_ms = tick * 20;
        guard.fed(now_ms);
        let here = measured(&mut joints);
        if let Verdict::Move { radians } = guard.authorise(now_ms, &here, plan.current()) {
            for (joint, angle) in joints.iter_mut().zip(&radians) {
                joint.command(*angle).unwrap();
            }
        }
        joints.iter_mut().for_each(SimJoint::step);
        plan.advance();
        tick += 1;
    }
    let when_it_died = measured(&mut joints);
    assert!(
        when_it_died[0] > 0.0,
        "should have moved before the source died"
    );

    // The commander is gone. Within the timeout the arm may still move —
    // silence for less than TIMEOUT_MS is not yet a fault.
    let mut moved_during_grace = false;
    while tick < 48 {
        let now_ms = tick * 20;
        let here = measured(&mut joints);
        if let Verdict::Move { radians } = guard.authorise(now_ms, &here, plan.current()) {
            moved_during_grace = true;
            for (joint, angle) in joints.iter_mut().zip(&radians) {
                joint.command(*angle).unwrap();
            }
        }
        joints.iter_mut().for_each(SimJoint::step);
        plan.advance();
        tick += 1;
    }
    assert!(
        moved_during_grace,
        "a gap shorter than the timeout must not stop the arm — that would \
         make every scheduling hiccup look like a fault"
    );

    // Past the timeout, nothing may ever authorise motion again.
    tick = 60;
    while tick < 500 {
        let now_ms = tick * 20;
        let here = measured(&mut joints);
        let verdict = guard.authorise(now_ms, &here, plan.current());
        assert!(
            !matches!(verdict, Verdict::Move { .. }),
            "a dead source must never authorise motion: {verdict:?}"
        );
        joints.iter_mut().for_each(SimJoint::step);
        tick += 1;
    }

    // And the arm is exactly where it stopped — it did not release.
    let settled = measured(&mut joints);
    for angle in &settled {
        assert!(*angle > 0.0, "the arm collapsed toward zero: {settled:?}");
    }
}

/// A fresher plan replaces the running one outright — it is never queued
/// behind decisions made about a world that has moved on.
#[test]
fn a_newer_plan_supersedes_the_running_one() {
    let (mut guard, mut joints) = rig();
    let mut plan = Plan::interpolate(&vec![0.0; 4], &vec![1.0; 4], 500).unwrap();

    for tick in 0..10u64 {
        let now_ms = tick * 20;
        guard.fed(now_ms);
        let here = measured(&mut joints);
        if let Verdict::Move { radians } = guard.authorise(now_ms, &here, plan.current()) {
            for (joint, angle) in joints.iter_mut().zip(&radians) {
                joint.command(*angle).unwrap();
            }
        }
        joints.iter_mut().for_each(SimJoint::step);
        plan.advance();
    }

    // A policy emits a fresh chunk pointing the other way.
    plan = Plan::chunk(guard.spec(), vec![vec![-0.5; 4]; 20]).unwrap();
    for tick in 10..300u64 {
        let now_ms = tick * 20;
        guard.fed(now_ms);
        let here = measured(&mut joints);
        if let Verdict::Move { radians } = guard.authorise(now_ms, &here, plan.current()) {
            for (joint, angle) in joints.iter_mut().zip(&radians) {
                joint.command(*angle).unwrap();
            }
        }
        joints.iter_mut().for_each(SimJoint::step);
        plan.advance();
    }

    assert!(
        measured(&mut joints)[0] < 0.0,
        "the newer plan should have won"
    );
}

/// A joint that overheats stops being asked to hold, even while the
/// commander is perfectly healthy.
#[test]
fn a_hot_joint_stops_the_arm_even_with_a_live_commander() {
    let (mut guard, mut joints) = rig();
    guard.fed(0);
    assert!(matches!(
        guard.authorise(0, &measured(&mut joints), &[0.1; 4]),
        Verdict::Move { .. }
    ));

    joints[1].heat_to(70.0);
    let hot = joints[1].temperature_celsius().unwrap();
    guard.observe_temperature(1, hot);
    guard.fed(20);

    assert!(
        matches!(
            guard.authorise(20, &measured(&mut joints), &[0.1; 4]),
            Verdict::HoldOverheated {
                joint: "shoulder_lift",
                ..
            }
        ),
        "heat must outrank a fresh command"
    );
}
