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

mod wire;

use hil_protocol::Message;
use sim_core::RobotSpec;
use sim_run::{viz, Mission, MissionConfig};
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let serial_port = flag(&args, "--serial");
    let record = flag(&args, "--record").map(PathBuf::from);
    let replay = flag(&args, "--replay").map(PathBuf::from);

    let rec = rerun::RecordingStreamBuilder::new("robotiq_hil").spawn()?;

    // The SAME mission sim-run runs. Not a copy — the same type, the same
    // default config, the same world.
    let config = MissionConfig {
        // The rig must simulate the SAME robot the firmware believes in.
        //
        // This used to be a `const SPEC = REAL_BOT` used only for the duty
        // scale, while the mission ran on `MissionConfig::default()`'s
        // `SIM_BOT` — two different robots in one program. Invisible while
        // `REAL_BOT == SIM_BOT`, and a silent trap the moment the measured
        // values land in `spec.rs`, which is the documented plan.
        spec: RobotSpec::REAL_BOT,
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
    };
    let dt = config.dt;
    let mut mission = Mission::new(config);
    // Duty ±1000 maps to ±`max_wheel_rad_s`. Read off the mission's own
    // spec so there is exactly one robot in this program — the firmware
    // scales by the same quantity from the same constant.
    let duty_scale = mission.config.spec.max_wheel_rad_s / f64::from(sim_core::DUTY_FULL);
    viz::draw_world(&rec, &mission)?;

    // ---- connect to the brain, or to a recording of one ----
    let mut emulator: Option<Child> = None;
    let mut wire = match (&replay, &serial_port) {
        // A recording stands in for the chip entirely: no serial port, no
        // emulator, no 20 s wait. And it checks what the host says.
        (Some(path), _) => Wire::replay(path)?,
        (None, Some(port)) => {
            eprintln!("[host] opening {port}");
            let sp = serialport::new(port, 115_200)
                .timeout(std::time::Duration::from_secs(5))
                .open()?;
            // Two handles: the read side blocks, and we must be able to
            // write while it does.
            let reader = sp.try_clone()?;
            Wire::live(Box::new(sp), Box::new(BufReader::new(reader)), record.as_deref())?
        }
        (None, None) => {
            let uf2 = positional(&args)
                .unwrap_or_else(|| "firmware/pico-robot/pico-robot.uf2".into());
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
            emulator = Some(child);
            Wire::live(Box::new(w), Box::new(BufReader::new(r)), record.as_deref())?
        }
    };
    if let Some(p) = &record {
        eprintln!("[host] recording to {}", p.display());
    }

    let mut trail_true: Vec<[f32; 2]> = Vec::new();
    let mut trail_belief: Vec<[f32; 2]> = Vec::new();
    let mut belief_from_chip = mission.config.start;
    // The chip's worst control-loop compute time, straight from the chip.
    let mut worst_us = 0u32;
    let mut ticks = 0usize;

    // Tell the chip where this session begins, before anything else.
    // Without it the chip carries its previous run's belief into this one
    // — measured: 0/1 waypoints, 4.97 m drift, 2036 wall bumps.
    wire.send(Message::Start {
        x: mission.config.start.x,
        y: mission.config.start.y,
        theta: mission.config.start.theta,
    })?;
    // Wait for the chip to confirm before saying anything else. Sending
    // the first goal straight after would put both lines in its UART FIFO
    // at once — 46 bytes against 32 — and shred the command.
    while let Some(line) = wire.recv_line()? {
        if let Ok(Message::Pose { x, y, theta }) = Message::parse(line.trim_end()) {
            belief_from_chip = sim_core::Pose::new(x, y, theta);
            break;
        }
    }

    // ---- the loop: plan here, control there ----
    while let Some(obs) = mission.observe() {
        // 1. Send the planner's decision. Note what is NOT here any more:
        //    a second copy of the policy. This used to rebuild the Goal
        //    and Twist messages from the observation, mirroring
        //    `Mission::decide` by hand — and the whole digital-twin claim
        //    rested on the two staying identical with nothing checking it.
        //    Now the rig ships exactly what the simulator decided.
        let directive = mission.plan(&obs);
        wire.send(directive.into())?;

        // 2. Wait for its motor command. Anything else is the chip's
        //    belief (for the viewer) or human-facing noise.
        let mut duty = None;
        while duty.is_none() {
            let Some(line) = wire.recv_line()? else {
                eprintln!("[host] chip closed the connection");
                break;
            };
            match Message::parse(line.trim_end()) {
                Ok(Message::Motor { duty_l, duty_r }) => duty = Some((duty_l, duty_r)),
                Ok(Message::Pose { x, y, theta }) => {
                    belief_from_chip = sim_core::Pose::new(x, y, theta)
                }
                Ok(Message::Health { worst_us: us }) => worst_us = worst_us.max(us),
                Ok(_) => {}
                Err(_) => eprintln!("[host] {}", line.trim_end()),
            }
        }
        let Some((duty_l, duty_r)) = duty else { break };

        // 3. Physics — the SAME advance() sim-run calls. Duty scaled to
        //    commanded wheel speeds is the only translation.
        let tick = mission.advance(obs, duty_l as f64 * duty_scale, duty_r as f64 * duty_scale);

        // 4. Hand back what the encoders saw.
        wire.send(Message::Sensors {
            dl: tick.dticks.0,
            dr: tick.dticks.1,
        })?;

        // 5. Draw it — the same viewer sim-run uses.
        viz::draw(&rec, &mission, &tick, &mut trail_true, &mut trail_belief)?;
        ticks += 1;
        if ticks % 50 == 0 {
            eprintln!(
                "[{:5.1}s] true ({:.2}, {:.2})  chip ({:.2}, {:.2})  drift {:.3} m  duty {duty_l}/{duty_r}",
                tick.obs.t,
                tick.true_pose.x,
                tick.true_pose.y,
                belief_from_chip.x,
                belief_from_chip.y,
                tick.drift
            );
        }
    }

    if let Some(child) = &mut emulator {
        let _ = child.kill();
    }

    let outcome = mission.outcome();
    println!(
        "\n{} ticks ({:.1} s simulated).\n  waypoints:   {}/{}\n  true pose:   x={:.3} y={:.3}\n  chip belief: x={:.3} y={:.3}\n  drift: {:.3} m,  wall bumps: {}",
        ticks,
        ticks as f64 * dt,
        outcome.waypoints_reached,
        outcome.waypoints_total,
        outcome.final_pose.x,
        outcome.final_pose.y,
        belief_from_chip.x,
        belief_from_chip.y,
        outcome.drift,
        outcome.bumps
    );

    // ---- did today's code agree with the recording? ----
    let divergences = wire.divergences();
    if !divergences.is_empty() || wire.unconsumed() > 0 {
        eprintln!("\nREPLAY DIVERGED from the recorded session:");
        for d in divergences.iter().take(10) {
            eprintln!("  - {d}");
        }
        if divergences.len() > 10 {
            eprintln!("  ... and {} more", divergences.len() - 10);
        }
        if wire.unconsumed() > 0 {
            eprintln!(
                "  - {} recorded lines were never reached — this run ended \
                 earlier than the one captured",
                wire.unconsumed()
            );
        }
        eprintln!(
            "\nThe robot would behave differently from the recording. If that \
             is intended, re-record; if not, this is the regression."
        );
        std::process::exit(1);
    }
    if replay.is_some() {
        println!("  replay:      matched the recording exactly");
    }

    // The deadline, checked against the chip rather than remembered.
    let budget_us = (dt * 1e6) as u32;
    if worst_us == 0 {
        eprintln!("  timing:      chip reported none (old firmware?)");
    } else {
        let used = 100.0 * worst_us as f64 / budget_us as f64;
        println!(
            "  timing:      worst {worst_us} us of {budget_us} us ({used:.2}%), {:.0}x headroom",
            budget_us as f64 / worst_us as f64
        );
        if worst_us > budget_us {
            // Not a warning. A robot that misses its control period is
            // integrating stale sensor data and steering on it.
            eprintln!(
                "\nFAIL: the chip missed its {budget_us} us control deadline \
                 (worst {worst_us} us). The loop grew past what the chip can \
                 do at {:.0} Hz.",
                1.0 / dt
            );
            std::process::exit(1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    /// The bug this guards: `--record run.wire` was read as a positional
    /// UF2 path, so the emulator was asked to boot a file called
    /// `--record` and the run died after one line.
    #[test]
    fn a_flags_value_is_not_mistaken_for_the_uf2() {
        assert_eq!(positional(&argv("--record run.wire")), None);
        assert_eq!(positional(&argv("--serial /dev/cu.x --record r.wire")), None);
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
