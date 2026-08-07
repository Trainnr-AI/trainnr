//! The rig and the chip must model the **same robot**, checked against the
//! sources rather than trusted.
//!
//! # Why this is a test and not a comment
//!
//! `hil-host` used to declare `const SPEC: RobotSpec = REAL_BOT` for its
//! duty scale while running the mission on `MissionConfig::default()`,
//! whose spec is `SIM_BOT`. Two different robots in one program: the host
//! converted the chip's duty using one machine's motor limit and then
//! simulated the consequences on another's.
//!
//! It was invisible, because `REAL_BOT` is currently *defined as*
//! `SIM_BOT`. It would have become a silent, confusing divergence the
//! moment the measured values land in `spec.rs` — which is the documented
//! plan for the day the motor arrives, i.e. exactly when someone is
//! already suspicious of their own wiring and least able to afford a
//! second unrelated bug.
//!
//! A comment saying "keep these in sync" is the thing that failed. This
//! reads both files.

use std::path::PathBuf;

fn read(relative: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/hil-host has a workspace root two levels up")
        .join(relative);
    std::fs::read_to_string(&root).unwrap_or_else(|e| panic!("cannot read {}: {e}", root.display()))
}

/// Which `RobotSpec` constants a file mentions, ignoring the ones inside
/// `spec.rs` itself (where both are naturally defined).
fn specs_named(source: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    for name in ["REAL_BOT", "SIM_BOT"] {
        if source.contains(&format!("RobotSpec::{name}")) {
            found.push(name);
        }
    }
    found
}

#[test]
fn the_rig_and_the_firmware_drive_the_same_robot() {
    let host = read("crates/hil-host/src/main.rs");
    let chip = read("firmware/pico-robot/src/main.rs");

    let host_specs = specs_named(&host);
    let chip_specs = specs_named(&chip);

    assert_eq!(
        host_specs.len(),
        1,
        "hil-host names {host_specs:?}. It must commit to exactly ONE \
         robot — naming two is the bug this test exists for."
    );
    assert_eq!(
        chip_specs.len(),
        1,
        "pico-robot names {chip_specs:?}; the firmware must commit to one."
    );
    assert_eq!(
        host_specs, chip_specs,
        "the rig simulates {host_specs:?} while the chip believes it is \
         {chip_specs:?} — the HIL run would be testing the controller \
         against a machine that does not exist"
    );
}

/// The duty scale must be derived from the mission's own spec and from
/// the shared full-scale constant — never from a second copy of either.
///
/// Deliberately checks the two *ingredients* rather than one exact
/// expression. The first version pinned the literal string
/// `"...max_wheel_rad_s / 1000.0"` and failed the moment that `1000.0`
/// was replaced by the shared `DUTY_FULL` — i.e. it failed on an
/// improvement. A guard that fires on the fix it was asking for is worse
/// than no guard.
#[test]
fn the_duty_scale_comes_from_the_simulated_robot() {
    // Scans the whole crate rather than one named file. The first version
    // read `main.rs`, and broke the moment the loop moved into `rig.rs` —
    // the second time this guard has failed on a refactor rather than on a
    // regression. What matters is that the scale exists somewhere in the
    // crate and is built from the right two things.
    let host: String = std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"))
        .expect("hil-host has a src directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();

    let scale = host
        .lines()
        .find(|l| l.contains("duty_scale ="))
        .expect("hil-host must compute a duty_scale somewhere in src/");

    assert!(
        scale.contains("mission.config.spec"),
        "duty_scale must come from the mission's own spec, or the rig \
         scales for one robot and simulates another: {scale:?}"
    );
    assert!(
        scale.contains("DUTY_FULL"),
        "duty_scale must use the shared DUTY_FULL, not its own copy of \
         1000 — that literal was once written three times: {scale:?}"
    );
}

/// Both sides must convert duty the same way, or every command is scaled
/// wrong by a constant factor — which looks exactly like a mistuned gain.
#[test]
fn both_sides_agree_on_what_a_duty_count_means() {
    use sim_core::RobotSpec;
    let spec = RobotSpec::REAL_BOT;
    let scale = spec.max_wheel_rad_s / 1000.0;

    // Full scale in each direction must round-trip through the wire's
    // integer duty back to the motor limit.
    for wheel in [spec.max_wheel_rad_s, -spec.max_wheel_rad_s] {
        let duty = spec.duty(wheel);
        assert_eq!(duty.abs(), 1000, "full speed should be full duty");
        assert!(
            (f64::from(duty) * scale - wheel).abs() < 1e-9,
            "duty {duty} scales back to {} not {wheel}",
            f64::from(duty) * scale
        );
    }
}
