//! **The digital-twin property, extended across the wire.**
//!
//! `sim-run/src/mission.rs` already tests that `observe → decide → advance`
//! matches `step()`. That covers the *split*, and it covered the decision
//! not at all: it called `Mission::decide` on both sides of the comparison,
//! so a rig that built its commands differently would sail straight past.
//!
//! And the rig did build them differently — it reconstructed `Goal` and
//! `Twist` from the observation by hand, mirroring `decide` in a second
//! place, with the firmware matching on a third. Three expressions of one
//! policy, and the twin claim resting on all three agreeing.
//!
//! This drives the *actual* path: policy → wire bytes → parse → the
//! controller the chip runs. What it cannot catch is a firmware build that
//! does something else entirely — for that, the HIL run against real
//! silicon is still the evidence.

use hil_protocol::Message;
use sim_core::{BodyTwist, Directive, GotoController};
use sim_run::{Mission, MissionConfig};

/// Encode and parse a directive exactly as the link does.
fn through_the_wire(d: Directive) -> Directive {
    let mut text = String::new();
    Message::from(d)
        .write_into(&mut text)
        .expect("encoding cannot fail into a String");
    Message::parse(text.trim_end())
        .expect("what we encode, we must be able to parse")
        .directive()
        .expect("a command must survive as a command")
}

/// The wire is 4-decimal ASCII, so it is *lossy by design*. This pins how
/// much a command can move crossing it — small enough to be irrelevant to
/// control, large enough not to fail on a formatting tweak.
///
/// Measured max over the whole U-trap mission: printed on failure.
const WIRE_TOLERANCE: f64 = 1e-2;

#[test]
fn the_wire_does_not_change_the_decision() {
    let config = MissionConfig::default();
    let (gains, dt) = (config.gains, config.tick_seconds);
    let mut mission = Mission::new(config);

    // Two controllers fed identical observations: one standing in for the
    // simulator, one for the chip. Separate instances because each carries
    // its own PID state, exactly as the real pair does.
    let mut local = GotoController::new(gains);
    let mut chip = GotoController::new(gains);

    let (mut worst_v, mut worst_w) = (0.0f64, 0.0f64);
    let mut ticks = 0usize;

    while let Some(obs) = mission.observe() {
        let directive = mission.plan(&obs);

        let on_laptop = local.execute(directive, &obs.pose, dt);
        let on_chip = chip.execute(through_the_wire(directive), &obs.pose, dt);

        worst_v = worst_v.max((on_laptop.forward_speed - on_chip.forward_speed).abs());
        worst_w = worst_w.max((on_laptop.turn_rate - on_chip.turn_rate).abs());
        ticks += 1;

        // Drive on the local decision so the trajectory is the simulator's
        // and both controllers keep seeing the same observations.
        let commanded = mission
            .config
            .spec
            .fit_wheels(mission.config.spec.drive().inverse(on_laptop));
        mission.advance(obs, commanded);
    }

    assert!(ticks > 500, "only {ticks} ticks — the mission did not run");
    assert!(
        worst_v < WIRE_TOLERANCE && worst_w < WIRE_TOLERANCE,
        "the wire changed the decision: worst Δv {worst_v:.2e} m/s, \
         worst Δw {worst_w:.2e} rad/s over {ticks} ticks (tolerance \
         {WIRE_TOLERANCE:.0e}). Either the encoding lost precision or the \
         two sides stopped running the same policy."
    );
}

/// Driving the whole mission through the wire must still solve the U-trap.
///
/// The trajectory is not bit-identical — 4-decimal ASCII guarantees that —
/// so this asserts the *outcome*, which is what actually matters and what
/// a real divergence destroys.
#[test]
fn a_mission_flown_entirely_through_the_wire_still_succeeds() {
    let reference = Mission::new(MissionConfig::default()).run();

    let config = MissionConfig::default();
    let (gains, dt) = (config.gains, config.tick_seconds);
    let mut mission = Mission::new(config);
    let mut chip = GotoController::new(gains);

    while let Some(obs) = mission.observe() {
        let directive = through_the_wire(mission.plan(&obs));
        let twist = chip.execute(directive, &obs.pose, dt);
        let commanded = mission
            .config
            .spec
            .fit_wheels(mission.config.spec.drive().inverse(twist));
        mission.advance(obs, commanded);
    }
    let wired = mission.outcome();

    assert!(
        wired.succeeded(),
        "flying through the wire failed the U-trap: {}/{} waypoints",
        wired.waypoints_reached,
        wired.waypoints_total
    );
    assert!(
        (wired.drift - reference.drift).abs() < 0.02,
        "drift through the wire {:.3} m vs {:.3} m in process",
        wired.drift,
        reference.drift
    );
    assert!(
        wired.bumps < 20,
        "{} wall bumps through the wire — it is feeling its way",
        wired.bumps
    );
}

/// `fresh` exists so a waypoint boundary reaches the chip. With one
/// waypoint that never happens, so this uses two — the configuration the
/// bug would actually appear in.
#[test]
fn a_waypoint_boundary_tells_the_controller_to_reset() {
    let config = MissionConfig {
        waypoints: vec![(3.0, 1.2), (6.5, 3.0)],
        duration: 90.0,
        ..MissionConfig::default()
    };
    let mut mission = Mission::new(config);

    let mut fresh_steers = 0usize;
    while let Some(obs) = mission.observe() {
        if let Directive::Steer { fresh: true, .. } = mission.plan(&obs) {
            fresh_steers += 1;
        }
        let twist = mission.decide(&obs);
        let commanded = mission
            .config
            .spec
            .fit_wheels(mission.config.spec.drive().inverse(twist));
        mission.advance(obs, commanded);
    }

    assert!(
        fresh_steers > 0,
        "no Steer ever carried fresh=true across two waypoints and mode \
         changes — the reset the host performs would never reach the chip"
    );
}
