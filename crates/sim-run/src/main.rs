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

use sim_core::{Cell, Mode};
use sim_run::{Mission, MissionConfig, Tick};
use std::time::Duration;

/// 1.0 = watch live in real time; raise to fast-forward.
const SPEEDUP: f64 = 1.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec = rerun::RecordingStreamBuilder::new("robotiq_stage0").spawn()?;

    let config = MissionConfig::default();
    let dt = config.dt;
    let mut mission = Mission::new(config);

    // ---- static scenery ----
    let wall_strips: Vec<[[f32; 2]; 2]> = mission
        .world
        .walls
        .iter()
        .map(|s| [[s.a.0 as f32, s.a.1 as f32], [s.b.0 as f32, s.b.1 as f32]])
        .collect();
    rec.log_static(
        "world/walls",
        &rerun::LineStrips2D::new(wall_strips).with_colors([rerun::Color::from_rgb(130, 130, 140)]),
    )?;
    rec.log_static(
        "world/waypoints",
        &rerun::Points2D::new(
            mission
                .config
                .waypoints
                .iter()
                .map(|&(x, y)| [x as f32, y as f32])
                .collect::<Vec<_>>(),
        )
        .with_radii([0.05])
        .with_colors([rerun::Color::from_rgb(120, 255, 120)]),
    )?;

    let mut trail_true: Vec<[f32; 2]> = Vec::new();
    let mut trail_belief: Vec<[f32; 2]> = Vec::new();

    while let Some(tick) = mission.step() {
        draw(&rec, &mission, &tick, &mut trail_true, &mut trail_belief)?;
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

/// Everything Rerun needs for one tick. No decisions here.
fn draw(
    rec: &rerun::RecordingStream,
    mission: &Mission,
    tick: &Tick,
    trail_true: &mut Vec<[f32; 2]>,
    trail_belief: &mut Vec<[f32; 2]>,
) -> Result<(), Box<dyn std::error::Error>> {
    rec.set_duration_secs("sim_time", tick.t);

    for (from, to) in &tick.mode_changes {
        rec.log(
            "events",
            &rerun::TextLog::new(format!("t={:.1}s: {from:?} -> {to:?}", tick.t)),
        )?;
    }

    // The robot's vision, drawn: one line per sight-line, out to its hit.
    let pose = tick.true_pose;
    let vision: Vec<[[f32; 2]; 2]> = tick
        .scan
        .iter()
        .enumerate()
        .map(|(idx, &d)| {
            let a = pose.theta + mission.camera.ray_angle(idx);
            [
                [pose.x as f32, pose.y as f32],
                [(pose.x + d * a.cos()) as f32, (pose.y + d * a.sin()) as f32],
            ]
        })
        .collect();
    let vision_color = match tick.mode {
        Mode::Goto => rerun::Color::from_rgb(80, 110, 140),
        Mode::Avoid => rerun::Color::from_rgb(255, 120, 40),
    };
    rec.log(
        "robot/vision",
        &rerun::LineStrips2D::new(vision).with_colors([vision_color]),
    )?;

    // The MAP as the robot remembers it, every 10th tick to keep the
    // stream light.
    if tick.index % 10 == 0 {
        let map = &mission.map;
        let mut occupied: Vec<[f32; 2]> = Vec::new();
        for cy in 0..map.height {
            for cx in 0..map.width {
                if map.get(cx, cy) == Cell::Occupied {
                    let (wx, wy) = map.cell_to_world(cx, cy);
                    occupied.push([wx as f32, wy as f32]);
                }
            }
        }
        rec.log(
            "map/occupied",
            &rerun::Points2D::new(occupied)
                .with_radii([0.045])
                .with_colors([rerun::Color::from_rgb(235, 100, 100)]),
        )?;
    }

    if let Some(p) = &tick.path {
        let pts: Vec<[f32; 2]> = p.iter().map(|&(px, py)| [px as f32, py as f32]).collect();
        rec.log(
            "plan/path",
            &rerun::LineStrips2D::new([pts]).with_colors([rerun::Color::from_rgb(120, 255, 120)]),
        )?;
    }
    rec.log(
        "plan/target",
        &rerun::Points2D::new([[tick.target.0 as f32, tick.target.1 as f32]])
            .with_radii([0.06])
            .with_colors([rerun::Color::from_rgb(255, 255, 120)]),
    )?;
    rec.log(
        "world/current_goal",
        &rerun::Points2D::new([[tick.goal.0 as f32, tick.goal.1 as f32]])
            .with_radii([0.09])
            .with_colors([rerun::Color::from_rgb(60, 255, 60)]),
    )?;

    let (x, y) = (pose.x as f32, pose.y as f32);
    trail_true.push([x, y]);
    rec.log(
        "robot/trail",
        &rerun::LineStrips2D::new([trail_true.clone()])
            .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
    )?;
    rec.log(
        "robot/body",
        &rerun::Points2D::new([[x, y]])
            .with_radii([0.09])
            .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
    )?;
    let (hx, hy) = (
        0.25 * pose.theta.cos() as f32,
        0.25 * pose.theta.sin() as f32,
    );
    rec.log(
        "robot/heading",
        &rerun::Arrows2D::from_vectors([[hx, hy]])
            .with_origins([[x, y]])
            .with_colors([rerun::Color::from_rgb(255, 90, 90)]),
    )?;

    let (bx, by) = (tick.belief_pose.x as f32, tick.belief_pose.y as f32);
    trail_belief.push([bx, by]);
    rec.log(
        "belief/trail",
        &rerun::LineStrips2D::new([trail_belief.clone()])
            .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
    )?;
    rec.log(
        "belief/body",
        &rerun::Points2D::new([[bx, by]])
            .with_radii([0.07])
            .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
    )?;

    rec.log(
        "telemetry/min_obstacle_m",
        &rerun::Scalars::single(tick.min_dist),
    )?;
    rec.log(
        "telemetry/bumped",
        &rerun::Scalars::single(if tick.bumped { 1.0 } else { 0.0 }),
    )?;
    rec.log(
        "telemetry/mode",
        &rerun::Scalars::single(match tick.mode {
            Mode::Goto => 0.0,
            Mode::Avoid => 1.0,
        }),
    )?;
    rec.log("telemetry/drift_m", &rerun::Scalars::single(tick.drift))?;
    Ok(())
}
