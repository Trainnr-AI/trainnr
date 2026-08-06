//! Stage 0 runner, milestone M5: the robot REMEMBERS and PLANS.
//!
//! Full pipeline, every tick:
//!   SENSE  — depth camera scans the true world (exercise 5 rays)
//!   MAP    — every ray burned into the occupancy grid (exercise 6)
//!   PLAN   — A* over the map to the goal, twice a second (planner.rs)
//!   DECIDE — follow the plan (M3 controller) with Avoid as emergency reflex
//!   ACT    — laggy motors, slip, SOLID walls (collision)
//!   OBSERVE— odometry belief from encoder ticks (exercise 3)
//!
//! Same U-trap that defeated the reactive robot: now the map paints the
//! dead end and A* routes around the outside instead of entering it.
//!
//! **This file is now only the viewer.** The mission itself lives in
//! `sim_run::mission`, where it can run headless in a test — see
//! `mapping_and_planning_beat_the_u_trap`, which asserts the result this
//! window shows you. `main` sets up the world, draws each tick, and
//! paces the loop; it makes no decisions.

use sim_run::viz;
use sim_run::{Mission, MissionConfig};
use std::time::Duration;

/// 1.0 = watch live in real time; raise to fast-forward.
const SPEEDUP: f64 = 1.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec = rerun::RecordingStreamBuilder::new("robotiq_stage0").spawn()?;

    let config = MissionConfig::default();
    let dt = config.dt;
    let mut mission = Mission::new(config);

    viz::draw_world(&rec, &mission)?;

    let mut trail_true: Vec<[f32; 2]> = Vec::new();
    let mut trail_belief: Vec<[f32; 2]> = Vec::new();

    while let Some(tick) = mission.step() {
        viz::draw(&rec, &mission, &tick, &mut trail_true, &mut trail_belief)?;
        std::thread::sleep(Duration::from_secs_f64(dt / SPEEDUP));
    }

    let outcome = mission.outcome();
    if let Some(t) = outcome.completed_at {
        println!("Mission complete at t = {t:.1} s.");
    }
    println!(
        "Waypoints reached: {}/{}. Final drift truth-vs-belief: {:.3} m.",
        outcome.waypoints_reached, outcome.waypoints_total, outcome.drift
    );
    Ok(())
}
