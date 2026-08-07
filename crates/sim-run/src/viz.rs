//! Drawing a mission into Rerun — shared by every runner.
//!
//! `sim-run` and `hil-host` are meant to be **digital twins**: same world,
//! same physics, same maths, differing only in whether the decision comes
//! from a function call or from a chip on a serial cable. That only holds
//! if they also *look* the same, so the viewer lives here rather than in
//! either binary.

use crate::Mission;
use crate::Tick;
use sim_core::{Cell, Mode};

/// Draw the static scenery once: walls and waypoints.
pub fn draw_world(
    rec: &rerun::RecordingStream,
    mission: &Mission,
) -> Result<(), Box<dyn std::error::Error>> {
    let wall_strips: Vec<[[f32; 2]; 2]> = mission
        .world
        .walls
        .iter()
        .map(|s| {
            [
                [s.start.x as f32, s.start.y as f32],
                [s.end.x as f32, s.end.y as f32],
            ]
        })
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
                .map(|p| [p.x as f32, p.y as f32])
                .collect::<Vec<_>>(),
        )
        .with_radii([0.05])
        .with_colors([rerun::Color::from_rgb(120, 255, 120)]),
    )?;
    Ok(())
}

/// Everything Rerun needs for one tick. No decisions here.
pub fn draw(
    rec: &rerun::RecordingStream,
    mission: &Mission,
    tick: &Tick,
    trail_true: &mut Vec<[f32; 2]>,
    trail_belief: &mut Vec<[f32; 2]>,
) -> Result<(), Box<dyn std::error::Error>> {
    rec.set_duration_secs("sim_time", tick.obs.elapsed_seconds);

    // A trail is CUMULATIVE: to draw it you must re-send every point it
    // has ever had. Doing that 50 times a second is quadratic — a 22.5 s
    // run pushed **1.27 million points** where 2,252 would do, 564x more
    // data than necessary, and `sim-run` emits the whole run in about a
    // second. Rerun then ingests for several seconds afterwards, so
    // entities surface progressively and the trails — the heaviest
    // payload — arrive last. That is not a late trail; it is a saturated
    // viewer.
    //
    // `chase.rs` hit this and solved it with a bounded window. Here the
    // full path is worth keeping, so the trails redraw at the same 5 Hz
    // cadence the map already uses. Ten times less data, and 0.2 s of
    // trail lag that nobody can see.
    let redraw_trails = tick.obs.index.is_multiple_of(10);

    for (from, to) in &tick.obs.mode_changes {
        rec.log(
            "events",
            &rerun::TextLog::new(format!(
                "t={:.1}s: {from:?} -> {to:?}",
                tick.obs.elapsed_seconds
            )),
        )?;
    }

    // The robot's vision, drawn: one line per sight-line, out to its hit.
    let pose = tick.true_pose;
    let vision: Vec<[[f32; 2]; 2]> = tick
        .obs
        .scan
        .iter()
        .enumerate()
        .map(|(idx, &d)| {
            let a = pose.heading + mission.camera.ray_angle(idx);
            [
                [pose.x as f32, pose.y as f32],
                [(pose.x + d * a.cos()) as f32, (pose.y + d * a.sin()) as f32],
            ]
        })
        .collect();
    let vision_color = match tick.obs.mode {
        Mode::Goto => rerun::Color::from_rgb(80, 110, 140),
        Mode::Avoid => rerun::Color::from_rgb(255, 120, 40),
    };
    rec.log(
        "robot/vision",
        &rerun::LineStrips2D::new(vision).with_colors([vision_color]),
    )?;

    // The MAP as the robot remembers it, every 10th tick to keep the
    // stream light.
    if redraw_trails {
        let map = &mission.map;
        let mut occupied: Vec<[f32; 2]> = Vec::new();
        for cy in 0..map.height {
            for cx in 0..map.width {
                if map.get(cx, cy) == Cell::Occupied {
                    let centre = map.cell_to_world(cx, cy);
                    occupied.push([centre.x as f32, centre.y as f32]);
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

    if let Some(p) = &tick.obs.path {
        let pts: Vec<[f32; 2]> = p.iter().map(|p| [p.x as f32, p.y as f32]).collect();
        rec.log(
            "plan/path",
            &rerun::LineStrips2D::new([pts]).with_colors([rerun::Color::from_rgb(120, 255, 120)]),
        )?;
    }
    rec.log(
        "plan/target",
        &rerun::Points2D::new([[tick.obs.target.x as f32, tick.obs.target.y as f32]])
            .with_radii([0.06])
            .with_colors([rerun::Color::from_rgb(255, 255, 120)]),
    )?;
    rec.log(
        "world/current_goal",
        &rerun::Points2D::new([[tick.obs.goal.x as f32, tick.obs.goal.y as f32]])
            .with_radii([0.09])
            .with_colors([rerun::Color::from_rgb(60, 255, 60)]),
    )?;

    let (x, y) = (pose.x as f32, pose.y as f32);
    trail_true.push([x, y]);
    if redraw_trails {
        rec.log(
            "robot/trail",
            &rerun::LineStrips2D::new([trail_true.clone()])
                .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
        )?;
    }
    rec.log(
        "robot/body",
        &rerun::Points2D::new([[x, y]])
            .with_radii([0.09])
            .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
    )?;
    let (hx, hy) = (
        0.25 * pose.heading.cos() as f32,
        0.25 * pose.heading.sin() as f32,
    );
    rec.log(
        "robot/heading",
        &rerun::Arrows2D::from_vectors([[hx, hy]])
            .with_origins([[x, y]])
            .with_colors([rerun::Color::from_rgb(255, 90, 90)]),
    )?;

    let (bx, by) = (tick.belief_pose.x as f32, tick.belief_pose.y as f32);
    trail_belief.push([bx, by]);
    if redraw_trails {
        rec.log(
            "belief/trail",
            &rerun::LineStrips2D::new([trail_belief.clone()])
                .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
        )?;
    }
    // Smaller than the true body (0.09) so it reads as a marker *inside*
    // the robot rather than a second robot.
    //
    // Worth knowing when watching: for the first several seconds the blue
    // belief and the yellow truth are **on top of each other**. Drift is
    // 0.0001 m at t=0 and still only 0.026 m at t=4 — well inside these
    // markers — so the blue trail does not become distinguishable until
    // around t=5. That is the odometry being good, not the viewer being
    // late.
    rec.log(
        "belief/body",
        &rerun::Points2D::new([[bx, by]])
            .with_radii([0.07])
            .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
    )?;

    rec.log(
        "telemetry/min_obstacle_m",
        &rerun::Scalars::single(tick.obs.summary.nearest),
    )?;
    rec.log(
        "telemetry/bumped",
        &rerun::Scalars::single(if tick.bumped { 1.0 } else { 0.0 }),
    )?;
    rec.log(
        "telemetry/mode",
        &rerun::Scalars::single(match tick.obs.mode {
            Mode::Goto => 0.0,
            Mode::Avoid => 1.0,
        }),
    )?;
    rec.log("telemetry/drift_m", &rerun::Scalars::single(tick.drift))?;
    Ok(())
}
