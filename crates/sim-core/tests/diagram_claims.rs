//! Every quantitative claim made by the diagrams in
//! `docs/learning/math-00-symbol-decoder.md`, checked against the real code.
//!
//! # Why this file exists
//!
//! A confident, well-drawn diagram that teaches the wrong thing is worse
//! than no diagram — and one already slipped through: a graph annotation
//! labelled the 90° mark "perfectly aimed" when 90° is where speed reaches
//! *zero*. That was caught by re-reading, which is not a process.
//!
//! So the diagrams are now executable. If a picture and the code ever
//! disagree, this fails, and whichever one is wrong gets fixed.

use sim_core::exercises::shortest_turn;
use sim_core::{ControlGains, DiffDrive, GotoController, Pose, RobotSpec};

const SPEC: RobotSpec = RobotSpec::SIM_BOT;

// ---- "The machine itself": the turning panel ----

#[test]
fn faster_wheel_is_on_the_outside_of_the_turn() {
    // Diagram: "the faster wheel is on the OUTSIDE — the robot turns away
    // from it. Left wheel faster → turns right." Right turn means w < 0,
    // because θ grows anticlockwise.
    let d = SPEC.drive();

    let (_, w) = d.forward(10.0, 5.0); // left faster
    assert!(w < 0.0, "left faster should turn RIGHT (w < 0), got w = {w}");

    let (_, w) = d.forward(5.0, 10.0); // right faster
    assert!(w > 0.0, "right faster should turn LEFT (w > 0), got w = {w}");
}

#[test]
fn equal_wheels_go_straight_and_opposite_wheels_spin_in_place() {
    // Diagram cases 1 and 4: "v > 0, w = 0" and "v = 0, w ≠ 0".
    let d = SPEC.drive();

    let (v, w) = d.forward(10.0, 10.0);
    assert!(v > 0.0, "equal wheels should advance, got v = {v}");
    assert!(w.abs() < 1e-12, "equal wheels should not turn, got w = {w}");

    let (v, w) = d.forward(-10.0, 10.0);
    assert!(v.abs() < 1e-12, "opposite wheels should not advance, got v = {v}");
    assert!(w != 0.0, "opposite wheels should turn, got w = {w}");
}

#[test]
fn kinematics_equations_in_the_doc_are_the_ones_in_the_code() {
    // Doc §2:  v = r(ω_R + ω_L)/2      w = r(ω_R − ω_L)/L
    let d = DiffDrive {
        wheel_radius: SPEC.wheel_radius,
        track_width: SPEC.track_width,
    };
    let (wl, wr) = (3.0, 7.0);
    let (v, w) = d.forward(wl, wr);

    let v_doc = SPEC.wheel_radius * (wr + wl) / 2.0;
    let w_doc = SPEC.wheel_radius * (wr - wl) / SPEC.track_width;

    assert!((v - v_doc).abs() < 1e-12, "v: code {v}, doc {v_doc}");
    assert!((w - w_doc).abs() < 1e-12, "w: code {w}, doc {w_doc}");
}

// ---- The loop: ③ bearing and ④ heading error ----

#[test]
fn a_target_on_the_left_produces_a_positive_error() {
    // Diagram ③: b = atan2(gy − y, gx − x)   ④: e = wrap(b − θ)
    // and "e > 0 → turn left". Robot at origin facing +x; goal at +y,
    // which the anatomy drawing calls the robot's LEFT.
    let pose = Pose::new(0.0, 0.0, 0.0);
    let (gx, gy) = (0.0, 1.0);

    let b = (gy - pose.y).atan2(gx - pose.x);
    assert!(
        (b - core::f64::consts::FRAC_PI_2).abs() < 1e-12,
        "bearing to +y should be π/2, got {b}"
    );

    let e = shortest_turn(pose.theta, b);
    assert!(e > 0.0, "a target on the left must give e > 0, got {e}");
}

#[test]
fn a_target_on_the_right_produces_a_negative_error() {
    let pose = Pose::new(0.0, 0.0, 0.0);
    let e = shortest_turn(pose.theta, (-1.0f64).atan2(0.0));
    assert!(e < 0.0, "a target on the right must give e < 0, got {e}");
}

// ---- The loop: ⑤ the alignment throttle ----

#[test]
fn the_plotted_throttle_curve_is_accurate() {
    // The graph claims: 0° → 1.0, 45° → 0.5, 90° → 0.0, 135° → 0.0.
    let mut c = GotoController::new(ControlGains::WAYPOINT);
    let v_max = ControlGains::WAYPOINT.v_max;
    let generous_budget = 1000.0; // capped to v_max inside, so v/v_max is the factor

    for (degrees, expected) in [(0.0, 1.0), (45.0, 0.5), (90.0, 0.0), (135.0, 0.0)] {
        c.reset();
        let (v, _) = c.steer(f64::to_radians(degrees), 0.0, generous_budget, 0.02);
        let factor = v / v_max;
        assert!(
            (factor - expected).abs() < 1e-9,
            "{degrees}° gives factor {factor}, the graph says {expected}"
        );
    }
}

#[test]
fn the_flat_section_of_the_curve_never_reverses() {
    // "Right of 90° the floor holds — this is .max(0.0)."
    let mut c = GotoController::new(ControlGains::WAYPOINT);
    for degrees in [91.0, 120.0, 180.0, -180.0] {
        c.reset();
        let (v, _) = c.steer(f64::to_radians(degrees), 0.0, 1000.0, 0.02);
        assert!(v >= 0.0, "{degrees}° gave v = {v}; the robot would reverse");
    }
}

#[test]
fn speed_budget_is_capped_at_v_max_inside_steer() {
    // Box ⑤: "Capped at v_max inside, so no caller can ask for more than
    // the robot has."
    let mut c = GotoController::new(ControlGains::WAYPOINT);
    let (v, _) = c.steer(0.0, 0.0, 99.0, 0.02);
    assert!(
        (v - ControlGains::WAYPOINT.v_max).abs() < 1e-12,
        "expected the cap at {}, got {v}",
        ControlGains::WAYPOINT.v_max
    );
}

// ---- The radian section ----

#[test]
// The doc prints rounded values (`2π ≈ 6.283`) and this test exists to
// check *those printed values* are right. Clippy sees a literal near TAU
// and assumes a botched constant — correct in general, wrong here. Allowed
// narrowly rather than by weakening the lint globally.
#[allow(clippy::approx_constant)]
fn the_radian_conversion_numbers_are_right() {
    // "radians × 57.3 ≈ degrees"; the table's entries.
    assert!((1.0f64.to_degrees() - 57.2957).abs() < 1e-3);
    assert!((0.02f64.to_degrees() - 1.1459).abs() < 1e-3, "the deadband row");
    assert!((core::f64::consts::FRAC_PI_2.to_degrees() - 90.0).abs() < 1e-9);
    assert!((core::f64::consts::PI.to_degrees() - 180.0).abs() < 1e-9);
    assert!((core::f64::consts::TAU - 6.2832).abs() < 1e-4);
}

#[test]
fn a_wheel_turning_once_rolls_its_circumference() {
    // "0.03 × 6.283 = 0.188 m — exactly the circumference, as it must be."
    let circumference = core::f64::consts::TAU * SPEC.wheel_radius;
    assert!((circumference - 0.1885).abs() < 1e-3, "got {circumference}");
    assert!((SPEC.wheel_circumference_m() - circumference).abs() < 1e-12);
}

// ---- The loop: ⑨ ⑩ odometry ----

#[test]
fn the_odometry_chain_in_the_doc_is_the_one_in_the_code() {
    // Doc: Δφ = 2π·Δticks/N ,  d = r·Δφ
    let one_turn_of_ticks = SPEC.ticks_per_rev;
    let dphi = core::f64::consts::TAU * one_turn_of_ticks / SPEC.ticks_per_rev;
    assert!((dphi - core::f64::consts::TAU).abs() < 1e-12, "one turn is 2π rad");

    let d = SPEC.wheel_radius * dphi;
    assert!((d - SPEC.wheel_circumference_m()).abs() < 1e-12);

    // and metres_per_tick is that distance spread over N counts
    assert!(
        (SPEC.metres_per_tick() * SPEC.ticks_per_rev - SPEC.wheel_circumference_m()).abs() < 1e-12
    );
}

#[test]
fn the_constants_printed_on_the_diagrams_are_the_real_ones() {
    // The anatomy drawing and box ⑥ print r = 0.03, L = 0.15, N = 1024.
    assert!((SPEC.wheel_radius - 0.03).abs() < 1e-12, "r on the drawing");
    assert!((SPEC.track_width - 0.15).abs() < 1e-12, "L on the drawing");
    assert!((SPEC.ticks_per_rev - 1024.0).abs() < 1e-12, "N on the drawing");
}
