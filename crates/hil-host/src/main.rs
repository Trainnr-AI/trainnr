//! LEVEL 3 host — the robot's *body*, for a brain that lives on a chip.
//!
//! # Digital twin, not a second simulator
//!
//! This used to be a separate simulation: its own 3.4×4.6 room with a
//! pillar, its own four-waypoint tour, and its own copy of the physics
//! loop — while `sim-run` simulated an 8×6 U-trap with one waypoint. Two
//! programs that shared primitives and nothing else.
//!
//! Now both run the **same `sim_run::Mission`**: same world, same physics,
//! same maths, same Rerun drawing. The only difference is who fills the
//! gap between `observe()` and `advance()`:
//!
//! ```text
//!   sim-run    observe ──▶ mission.decide() ─────────────▶ advance
//!   hil-host   observe ──▶ [ serial cable · a real chip ] ▶ advance
//! ```
//!
//! # Who plans, and why
//!
//! The chip cannot plan — `OccupancyGrid` is a `Vec`, `plan()` uses a
//! `BinaryHeap`, and bare metal has no allocator. So the host senses, maps
//! and runs A*, then sends the chip a single point to steer at; the chip
//! runs the real-time control loop and answers with motor duty.
//!
//! That split is not a workaround. It is the two-tier architecture in
//! docs/00 — Tier 2 plans, Tier 1 controls — and it is what the physical
//! robot will do with a Jetson in place of this laptop.
//!
//! # Two transports
//!
//! ```sh
//! cargo run -p hil-host                                  # the emulator
//! cargo run -p hil-host -- --serial /dev/cu.usbmodem11   # a REAL Pico
//! ```

mod rig;
mod wire;

use rig::Rig;
use sim_core::{ControlGains, RobotSpec};
use sim_run::{viz, MissionConfig};
use std::io::BufReader;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use wire::Wire;

/// Every flag this binary takes. Listed once so that [`flag`] and
/// [`positional`] cannot disagree about what counts as a flag — they did,
/// briefly, and `--record run.wire` was read as a path to a UF2 image.
const FLAGS: [&str; 3] = ["--serial", "--record", "--replay"];

/// `--name <value>`, or `None`.
fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// The first argument that is neither a flag nor a flag's value — the UF2
/// to hand the emulator.
fn positional(args: &[String]) -> Option<String> {
    let mut skip_next = false;
    for a in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if FLAGS.contains(&a.as_str()) {
            skip_next = true;
            continue;
        }
        if !a.starts_with("--") {
            return Some(a.clone());
        }
    }
    None
}

/// The mission this rig runs — **the single definition**, so nothing has
/// to keep a second copy of it in step by hand.
///
/// It was inline in `main`, and a test in `sim-run` then reproduced it
/// field by field with a comment saying "keep in step with hil-host".
/// That is the same trap this function's own comments describe twice
/// over: two things that must agree, agreeing by hand. `sim-run` cannot
/// depend on `hil-host`, so the definition lives here and the test reaches
/// for it.
pub fn mission_config() -> MissionConfig {
    MissionConfig {
        // The rig must simulate the SAME robot the firmware believes in.
        //
        // This used to be a `const SPEC = REAL_BOT` used only for the duty
        // scale, while the mission ran on `MissionConfig::default()`'s
        // `SIM_BOT` — two different robots in one program. Invisible while
        // `REAL_BOT == SIM_BOT`, and a silent trap the moment the measured
        // values land in `spec.rs`, which is the documented plan.
        spec: RobotSpec::REAL_BOT,
        // ⚠️ The SAME profile `firmware/pico-robot` runs, for the same
        // reason the spec above is shared: this rig exists to model the
        // robot the firmware believes in.
        //
        // This was never set, so the host silently ran `WAYPOINT` while
        // the chip ran `HIL` — two different controllers in one loop.
        // Undetectable until 2026-08-10, because `HIL` was `..WAYPOINT`
        // differing only in `arrive_radius`, which the host does not use;
        // the chip decides arrival. The moment `HIL` was retuned for the
        // measured motor, `RobotSpec::check` refused the config outright.
        //
        // The exact shape of the `REAL_BOT`/`SIM_BOT` trap described
        // below, one layer up: two things that must agree, agreeing by
        // coincidence, in a rig whose whole purpose is that they agree.
        gains: ControlGains::HIL,
        // The measured motor, not the invented one. `MissionConfig`'s
        // defaults describe sim-run's teaching robot; this rig is a twin
        // of hardware, so it gets the bench numbers of 2026-08-09.
        //
        //   motor_tau   0.15 -> ~0.04 s   (eight step responses clustered)
        //   deadband    none -> 4.3%      (0.0429 and 0.0391 per motor)
        //
        // The deadband is the one that changes behaviour visibly: below
        // ~4.3% of full command the real robot does not move, so a twin
        // without it lets the chip's controller creep at speeds the
        // hardware cannot produce — and creeping is exactly what a
        // waypoint follower does as it arrives.
        // ⚠️ 60 s was the budget for a robot that could do 0.45 m/s. The
        // measured one does 0.117, so the same journey across the same
        // 8x6 m room takes about four times as long. Not a fudge factor —
        // the identical trip at a quarter of the speed.
        //
        // **This makes the emulator gate step roughly 4x slower**, which
        // is the honest cost of simulating the robot that exists rather
        // than the one that was assumed. If that becomes intolerable the
        // fix is a nearer waypoint for the emulator run, not a faster
        // robot in the config.
        duration: 300.0,
        motor_tau: 0.04,
        motor_deadband_fraction: 0.043,
        // THE one deliberate difference from sim-run, and it makes the rig
        // MORE realistic, not less.
        //
        // sim-run plans and steers on ground truth — a teaching
        // simplification that isolates control from estimation. Here the
        // chip can only steer on its own odometry belief, because that is
        // all a real robot ever has. If the host planned on truth we would
        // have two different poses in one loop: the host aims at a point
        // computed from where the robot really is, the chip turns toward
        // it from where it *thinks* it is, and the error feeds itself.
        // Measured: drift ran to 5.7 m and the belief left the room.
        //
        // Both sides integrate the same encoder ticks from the same start,
        // so host and chip odometry stay in lockstep by construction.
        control_on_belief: true,
        ..MissionConfig::default()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let serial_port = flag(&args, "--serial");
    let record = flag(&args, "--record").map(PathBuf::from);
    let replay = flag(&args, "--replay").map(PathBuf::from);

    let rec = rerun::RecordingStreamBuilder::new("robotiq_hil").spawn()?;
    // Held so the child can be killed when the run ends.
    let mut emulator: Option<Child> = None;

    // The SAME mission sim-run runs. Not a copy — the same type, the same
    // default config, the same world.
    let config = mission_config();
    let mut rig = Rig::new(
        config,
        wire_for(&args, &serial_port, &replay, &record, &mut emulator)?,
    );
    viz::draw_world(&rec, &rig.mission)?;
    if let Some(p) = &record {
        eprintln!("[host] recording to {}", p.display());
    }

    let mut trail_true: Vec<[f32; 2]> = Vec::new();
    let mut trail_belief: Vec<[f32; 2]> = Vec::new();

    rig.start()?;
    while let Some(tick) = rig.tick()? {
        // Drawing stays here: it is the one part of the loop that needs a
        // window, and keeping it out of `Rig` is what lets the whole
        // exchange run in `cargo test`.
        viz::draw(
            &rec,
            &rig.mission,
            &tick,
            &mut trail_true,
            &mut trail_belief,
        )?;
        if rig.ticks.is_multiple_of(50) {
            eprintln!(
                "[{:5.1}s] true ({:.2}, {:.2})  chip ({:.2}, {:.2})  drift {:.3} m",
                tick.obs.elapsed_seconds,
                tick.true_pose.x,
                tick.true_pose.y,
                rig.belief_from_chip.x,
                rig.belief_from_chip.y,
                tick.drift
            );
        }
    }

    if let Some(child) = &mut emulator {
        let _ = child.kill();
    }

    let verdict = rig.verdict();
    println!("{}", verdict.report(replay.is_some()));
    if verdict.failed() {
        std::process::exit(1);
    }
    Ok(())
}

/// Open the link the arguments ask for: a recording, a serial port, or a
/// freshly spawned emulator.
fn wire_for(
    args: &[String],
    serial_port: &Option<String>,
    replay: &Option<PathBuf>,
    record: &Option<PathBuf>,
    emulator: &mut Option<Child>,
) -> Result<Wire, Box<dyn std::error::Error>> {
    match (replay, serial_port) {
        // A recording stands in for the chip entirely: no serial port, no
        // emulator, no 20 s wait. And it checks what the host says.
        (Some(path), _) => Wire::replay(path),
        (None, Some(port)) => {
            eprintln!("[host] opening {port}");
            let sp = serialport::new(port, 115_200)
                .timeout(std::time::Duration::from_secs(5))
                .open()?;
            // Two handles: the read side blocks, and we must be able to
            // write while it does.
            let reader = sp.try_clone()?;
            Ok(Wire::live(
                Box::new(sp),
                Box::new(BufReader::new(reader)),
                record.as_deref(),
            )?)
        }
        (None, None) => {
            let uf2 =
                positional(args).unwrap_or_else(|| "firmware/pico-robot/pico-robot.uf2".into());
            eprintln!("[host] spawning emulator for {uf2}");
            let mut child: Child = Command::new("npx")
                .args(["tsx", "demo/hil-bridge.ts", &format!("../../{uf2}")])
                .current_dir("tools/rp2040js")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()?;
            let w = child.stdin.take().ok_or("emulator stdin was not piped")?;
            let r = child.stdout.take().ok_or("emulator stdout was not piped")?;
            *emulator = Some(child);
            Ok(Wire::live(
                Box::new(w),
                Box::new(BufReader::new(r)),
                record.as_deref(),
            )?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This rig's own config, solved natively in under a second.
    ///
    /// # Why it is worth a test of its own
    ///
    /// `control_on_belief: true` is strictly harder than the ground-truth
    /// case every `sim-run` test uses — the chip steers on its odometry,
    /// so drift feeds back into the steering. And the emulator step that
    /// exercises this same config takes minutes, so a failure there is
    /// slow to find and slower to iterate on. Here it costs 0.4 s.
    ///
    /// It also pins `duration`. That budget was 60 s for a robot that
    /// could do 0.45 m/s; the measured one does 0.117, and this test is
    /// what says the new number is enough rather than merely larger.
    #[test]
    fn the_rig_config_can_actually_be_solved() {
        let outcome = sim_run::Mission::new(mission_config()).run();
        assert_eq!(
            outcome.waypoints_reached, outcome.waypoints_total,
            "the rig's own mission is unsolvable: {outcome:?}"
        );
    }

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    /// The bug this guards: `--record run.wire` was read as a positional
    /// UF2 path, so the emulator was asked to boot a file called
    /// `--record` and the run died after one line.
    #[test]
    fn a_flags_value_is_not_mistaken_for_the_uf2() {
        assert_eq!(positional(&argv("--record run.wire")), None);
        assert_eq!(
            positional(&argv("--serial /dev/cu.x --record r.wire")),
            None
        );
        assert_eq!(
            positional(&argv("--record r.wire firmware/x.uf2")).as_deref(),
            Some("firmware/x.uf2")
        );
        assert_eq!(
            positional(&argv("firmware/x.uf2 --record r.wire")).as_deref(),
            Some("firmware/x.uf2")
        );
    }

    #[test]
    fn flags_read_their_own_values() {
        let a = argv("--serial /dev/cu.usbmodem11 --record run.wire");
        assert_eq!(flag(&a, "--serial").as_deref(), Some("/dev/cu.usbmodem11"));
        assert_eq!(flag(&a, "--record").as_deref(), Some("run.wire"));
        assert_eq!(flag(&a, "--replay"), None);
    }

    #[test]
    fn a_trailing_flag_with_no_value_is_not_a_panic() {
        assert_eq!(flag(&argv("--record"), "--record"), None);
        assert_eq!(positional(&argv("--record")), None);
    }
}
