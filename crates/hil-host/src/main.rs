//! LEVEL 3 host — the robot's *body*, for a brain that lives on a chip.
//!
//! Spawns the emulator (which runs `firmware/pico-robot`), then plays the
//! physical world to it: motor lag, wheel slip, quantized encoders, solid
//! walls. Logs the true pose AND the chip's believed pose to Rerun, so you
//! can watch dead reckoning drift on hardware you can't see.
//!
//! The firmware does not know any of this exists. Protocol and rationale:
//! docs/10-hil-protocol.md
//!
//! Run with: tools/sim-hil.sh

use sim_core::{DiffDrive, Encoders, Motor, Pose, Rng, Robot, World};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

/// Physics step. Must match the firmware's DT.
const DT: f64 = 0.02;
/// Stop after this much simulated time.
const MAX_STEPS: usize = 3000; // 60 s
const SEED: u64 = 7;

/// The robot as it truly is: 1% worn tyres the firmware doesn't know about.
const TRUE_WHEEL_RADIUS: f64 = 0.0297;
const TRACK_WIDTH: f64 = 0.15;
const TICKS_PER_REV: f64 = 1024.0;
/// Motor: first-order lag + saturation (sim-core's model).
const MOTOR_TAU: f64 = 0.15;
const MOTOR_MAX: f64 = 30.0;
/// Duty ±1000 maps to ±MOTOR_MAX rad/s of commanded wheel speed.
const DUTY_SCALE: f64 = MOTOR_MAX / 1000.0;
const ROBOT_RADIUS: f64 = 0.09;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let uf2 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "firmware/pico-robot/pico-robot.uf2".into());

    let rec = rerun::RecordingStreamBuilder::new("robotiq_hil").spawn()?;

    // ---- the world the firmware will bump into ----
    let mut world = World::room(3.4, 4.6);
    world.add_box(1.5, 1.9, 1.9, 2.7); // a pillar mid-room

    let wall_strips: Vec<[[f32; 2]; 2]> = world
        .walls
        .iter()
        .map(|s| [[s.a.0 as f32, s.a.1 as f32], [s.b.0 as f32, s.b.1 as f32]])
        .collect();
    rec.log_static(
        "world/walls",
        &rerun::LineStrips2D::new(wall_strips).with_colors([rerun::Color::from_rgb(130, 130, 140)]),
    )?;

    // ---- physical state ----
    let start = Pose::new(1.0, 1.0, 0.0);
    let mut robot = Robot {
        model: DiffDrive {
            wheel_radius: TRUE_WHEEL_RADIUS,
            track_width: TRACK_WIDTH,
        },
        pose: start,
    };
    let mut motor_l = Motor::new(MOTOR_TAU, MOTOR_MAX);
    let mut motor_r = Motor::new(MOTOR_TAU, MOTOR_MAX);
    let mut encoders = Encoders::new(TICKS_PER_REV);
    let mut rng = Rng::new(SEED);

    // ---- spawn the emulator ----
    eprintln!("[host] spawning emulator for {uf2}");
    let mut child: Child = Command::new("npx")
        .args(["tsx", "demo/hil-bridge.ts", &format!("../../{uf2}")])
        .current_dir("tools/rp2040js")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut to_chip = child.stdin.take().expect("stdin");
    let from_chip = BufReader::new(child.stdout.take().expect("stdout"));

    let mut trail_true: Vec<[f32; 2]> = Vec::new();
    let mut trail_belief: Vec<[f32; 2]> = Vec::new();
    let mut belief = start;
    let mut steps = 0usize;
    let mut bumps = 0usize;
    let mut idle = 0usize;

    // ---- the loop: read the chip's decisions, answer with consequences ----
    for line in from_chip.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        match parts.next() {
            // The chip's believed pose — for display only.
            Some("P") => {
                let x: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let y: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let th: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                belief = Pose::new(x, y, th);
            }
            // Motor command: advance physics one step, reply with ticks.
            Some("M") => {
                let duty_l: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let duty_r: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);

                // duty -> commanded wheel speed -> what the motor ACTUALLY does
                let act_l = motor_l.step(duty_l * DUTY_SCALE, DT);
                let act_r = motor_r.step(duty_r * DUTY_SCALE, DT);

                // the ground steals a random 0-1% per wheel per step
                let slip_l = 1.0 - 0.01 * rng.uniform();
                let slip_r = 1.0 - 0.01 * rng.uniform();
                let before = robot.pose;
                robot.step(act_l * slip_l, act_r * slip_r, DT);
                if world.collides(robot.pose.x, robot.pose.y, ROBOT_RADIUS) {
                    // Walls are solid: refuse the translation, keep the turn.
                    robot.pose.x = before.x;
                    robot.pose.y = before.y;
                    bumps += 1;
                }

                // encoders watch the MOTOR shaft, so they never see slip
                let (dl, dr) = encoders.advance(act_l, act_r, DT);
                writeln!(to_chip, "S {dl} {dr}")?;
                to_chip.flush()?;

                // ---- telemetry ----
                let t = steps as f64 * DT;
                rec.set_duration_secs("sim_time", t);
                let (x, y) = (robot.pose.x as f32, robot.pose.y as f32);
                trail_true.push([x, y]);
                trail_belief.push([belief.x as f32, belief.y as f32]);
                rec.log(
                    "robot/trail",
                    &rerun::LineStrips2D::new([trail_true.clone()])
                        .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
                )?;
                rec.log(
                    "robot/body",
                    &rerun::Points2D::new([[x, y]])
                        .with_radii([ROBOT_RADIUS as f32])
                        .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
                )?;
                rec.log(
                    "robot/heading",
                    &rerun::Arrows2D::from_vectors([[
                        0.25 * robot.pose.theta.cos() as f32,
                        0.25 * robot.pose.theta.sin() as f32,
                    ]])
                    .with_origins([[x, y]])
                    .with_colors([rerun::Color::from_rgb(255, 90, 90)]),
                )?;
                rec.log(
                    "chip_belief/trail",
                    &rerun::LineStrips2D::new([trail_belief.clone()])
                        .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
                )?;
                rec.log(
                    "chip_belief/body",
                    &rerun::Points2D::new([[belief.x as f32, belief.y as f32]])
                        .with_radii([0.06])
                        .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
                )?;
                rec.log(
                    "telemetry/drift_m",
                    &rerun::Scalars::single(robot.pose.distance_to(&belief)),
                )?;
                rec.log("telemetry/duty_l", &rerun::Scalars::single(duty_l))?;
                rec.log("telemetry/duty_r", &rerun::Scalars::single(duty_r))?;

                // Progress narration: every simulated second, plus a note
                // when the chip stops commanding (tour finished).
                if steps % 50 == 0 {
                    eprintln!(
                        "[{:5.1}s] true ({:.2}, {:.2}) th={:+.2}   chip ({:.2}, {:.2})   drift {:.3} m   duty {:>5}/{:<5}",
                        t,
                        robot.pose.x,
                        robot.pose.y,
                        robot.pose.theta,
                        belief.x,
                        belief.y,
                        robot.pose.distance_to(&belief),
                        duty_l as i32,
                        duty_r as i32
                    );
                }
                if duty_l == 0.0 && duty_r == 0.0 && steps > 10 {
                    idle += 1;
                    if idle == 25 {
                        eprintln!("[{t:5.1}s] chip has stopped commanding — tour complete");
                    }
                } else {
                    idle = 0;
                }

                steps += 1;
                if steps >= MAX_STEPS || idle > 100 {
                    break;
                }
            }
            _ => eprintln!("[host] unknown line: {line}"),
        }
    }

    let _ = child.kill();
    println!(
        "\n{} steps ({:.1} s simulated).\n  true pose:   x={:.3} y={:.3} th={:.3}\n  chip belief: x={:.3} y={:.3} th={:.3}\n  drift: {:.3} m,  wall bumps: {}",
        steps,
        steps as f64 * DT,
        robot.pose.x,
        robot.pose.y,
        robot.pose.theta,
        belief.x,
        belief.y,
        belief.theta,
        robot.pose.distance_to(&belief),
        bumps
    );
    Ok(())
}
