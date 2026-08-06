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

use hil_protocol::Message;
use sim_core::RobotSpec;
use sim_run::{viz, Mission, MissionConfig};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

/// The robot the host simulates. Must be the SAME spec the firmware
/// believes in, or the rig tests the chip against a machine that does not
/// exist — hence `REAL_BOT`, matching `pico-robot`.
const SPEC: RobotSpec = RobotSpec::REAL_BOT;
/// Duty ±1000 maps to ±`max_wheel_rad_s` of commanded wheel speed.
/// Derived, not chosen: the same quantity the firmware scales by.
const DUTY_SCALE: f64 = SPEC.max_wheel_rad_s / 1000.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let serial_port = args
        .iter()
        .position(|a| a == "--serial")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let rec = rerun::RecordingStreamBuilder::new("robotiq_hil").spawn()?;

    // The SAME mission sim-run runs. Not a copy — the same type, the same
    // default config, the same world.
    let config = MissionConfig {
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
    viz::draw_world(&rec, &mission)?;

    // ---- connect to the brain ----
    let mut emulator: Option<Child> = None;
    let (mut to_chip, mut from_chip): (Box<dyn Write>, Box<dyn BufRead>) = match &serial_port {
        Some(port) => {
            eprintln!("[host] opening {port}");
            let sp = serialport::new(port, 115_200)
                .timeout(std::time::Duration::from_secs(5))
                .open()?;
            // Two handles: the read side blocks, and we must be able to
            // write while it does.
            let reader = sp.try_clone()?;
            (Box::new(sp), Box::new(BufReader::new(reader)))
        }
        None => {
            let uf2 = args
                .first()
                .cloned()
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
            (Box::new(w), Box::new(BufReader::new(r)))
        }
    };

    let mut trail_true: Vec<[f32; 2]> = Vec::new();
    let mut trail_belief: Vec<[f32; 2]> = Vec::new();
    let mut belief_from_chip = mission.config.start;
    let mut ticks = 0usize;

    // ---- the loop: plan here, control there ----
    while let Some(obs) = mission.observe() {
        // 1. Tell the chip where to aim. This is the planner's output —
        //    the chip never sees the map.
        // In Goto the chip steers; in Avoid the HOST decides, because the
        // reflex needs a depth scan the chip does not have. Same split as
        // the real robot, where the camera is on Tier 2.
        let command = match obs.mode {
            sim_core::Mode::Goto => Message::Goal {
                x: obs.target.0,
                y: obs.target.1,
                // The budget sim-run's own `decide()` would use:
                // proportional to distance to the REAL goal, not to the
                // lookahead point. Send it, or the chip crawls.
                budget: mission.config.gains.kp_dist * obs.dist_to_goal,
            },
            sim_core::Mode::Avoid => Message::Twist {
                v: mission.config.avoid_v,
                w: obs.summary.turn_direction() * mission.config.avoid_w,
            },
        };
        send(&mut to_chip, command)?;

        // 2. Wait for its motor command. Anything else is the chip's
        //    belief (for the viewer) or human-facing noise.
        let mut duty = None;
        while duty.is_none() {
            let mut line = String::new();
            if from_chip.read_line(&mut line)? == 0 {
                eprintln!("[host] chip closed the connection");
                break;
            }
            match Message::parse(line.trim_end()) {
                Ok(Message::Motor { duty_l, duty_r }) => duty = Some((duty_l, duty_r)),
                Ok(Message::Pose { x, y, theta }) => {
                    belief_from_chip = sim_core::Pose::new(x, y, theta)
                }
                Ok(_) => {}
                Err(_) => eprintln!("[host] {}", line.trim_end()),
            }
        }
        let Some((duty_l, duty_r)) = duty else { break };

        // 3. Physics — the SAME advance() sim-run calls. Duty scaled to
        //    commanded wheel speeds is the only translation.
        let tick = mission.advance(obs, duty_l as f64 * DUTY_SCALE, duty_r as f64 * DUTY_SCALE);

        // 4. Hand back what the encoders saw.
        send(
            &mut to_chip,
            Message::Sensors {
                dl: tick.dticks.0,
                dr: tick.dticks.1,
            },
        )?;

        // 5. Draw it — the same viewer sim-run uses.
        viz::draw(&rec, &mission, &tick, &mut trail_true, &mut trail_belief)?;
        ticks += 1;
        if ticks % 50 == 0 {
            eprintln!(
                "[{:5.1}s] true ({:.2}, {:.2})  chip ({:.2}, {:.2})  drift {:.3} m  duty {duty_l}/{duty_r}",
                tick.t,
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
    Ok(())
}

/// Every send goes through `hil-protocol` rather than being hand-formatted
/// on this side — that is the whole point of the crate existing.
fn send(w: &mut Box<dyn Write>, m: Message) -> std::io::Result<()> {
    let mut s = String::new();
    let _ = m.write_into(&mut s);
    w.write_all(s.as_bytes())?;
    w.flush()
}
