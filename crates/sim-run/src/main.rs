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

use sim_core::exercises::shortest_turn;
use sim_core::{
    plan, Cell, DepthCamera, DiffDrive, Encoders, Motor, OccupancyGrid, Odometry, Pid, Pose, Rng,
    Robot, World,
};
use std::time::Duration;

/// Control-loop period: 50 Hz.
const DT: f64 = 0.02;
/// Give up after this much sim time even if waypoints remain.
const DURATION: f64 = 60.0;
/// 1.0 = watch live in real time; raise to fast-forward.
const SPEEDUP: f64 = 1.0;
const SEED: u64 = 7;

// ---- M3 tuning knobs ----
const HEADING_KP: f64 = 6.0;
const HEADING_KI: f64 = 0.0;
const HEADING_KD: f64 = 0.6;
const MOTOR_TAU: f64 = 0.15;
const MOTOR_MAX: f64 = 30.0;
const KP_DIST: f64 = 0.8;
const V_MAX: f64 = 0.45;
const ARRIVE_RADIUS: f64 = 0.15;
const CONTROL_ON_BELIEF: bool = false;

// ---- M4: the camera and the avoid behavior ----
/// 70° forward fan, RealSense-ish, 21 sight-lines, 2.5 m range.
const CAM_RAYS: usize = 21;
const CAM_FOV: f64 = 1.22;
const CAM_RANGE: f64 = 2.5;
/// Enter Avoid when anything is closer than this... With a planner keeping
/// clearance for us, Avoid demotes to a pure emergency reflex (tighter
/// thresholds than M4, so it doesn't fight the plan in narrow gaps).
const AVOID_ENTER: f64 = 0.35;
/// ...and only go back to Goto once everything is farther than this.
/// The gap (hysteresis) prevents rapid flip-flopping at the boundary.
const AVOID_EXIT: f64 = 0.6;
// ---- M5 act 2: memory + planning ----
/// Map resolution (10 cm cells).
const MAP_RES: f64 = 0.1;
/// Re-run A* every this many ticks (0.5 s) — the map grows constantly.
const REPLAN_TICKS: usize = 25;
/// Keep this many cells of clearance from mapped walls (robot has a body).
const INFLATION_CELLS: usize = 2;
/// Steer at the first path point at least this far ahead (pure pursuit-ish).
const LOOKAHEAD: f64 = 0.35;
/// Avoid-mode motion: creep forward, turn hard toward the open side.
const AVOID_V: f64 = 0.12;
const AVOID_W: f64 = 1.8;
/// Body radius for collision — walls are SOLID now (Prakhar caught the
/// robot ghosting through them in the viewer; sim bug, fixed 2026-07-29).
const ROBOT_RADIUS: f64 = 0.09;

/// The robot's behavior state. An `enum`: a type whose value is exactly
/// one of these variants. The `match` in the loop is forced by the
/// compiler to handle every variant — add a state, and every match must
/// be updated or the build fails. That's why robot FSMs love enums.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Goto,
    Avoid,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec = rerun::RecordingStreamBuilder::new("robotiq_stage0").spawn()?;

    // M5 act 1: THE U-TRAP. Three walls forming a U with its mouth facing
    // the robot; the goal sits right behind the back wall. A purely
    // reactive avoider cannot solve this — watch it try forever.
    let mut world = World::room(8.0, 6.0);
    world.walls.push(sim_core::Segment {
        a: (5.0, 2.2),
        b: (5.0, 3.8), // back wall
    });
    world.walls.push(sim_core::Segment {
        a: (2.6, 3.8),
        b: (5.0, 3.8), // top arm (deep!)
    });
    world.walls.push(sim_core::Segment {
        a: (2.6, 2.2),
        b: (5.0, 2.2), // bottom arm (deep!)
    });
    // Flanking spurs at the mouth so wall-following around an arm tip
    // shoves the robot back toward the entrance.
    world.walls.push(sim_core::Segment {
        a: (2.6, 3.8),
        b: (2.6, 5.2), // spur up from the top arm tip
    });
    world.walls.push(sim_core::Segment {
        a: (2.6, 0.8),
        b: (2.6, 2.2), // spur down from the bottom arm tip
    });

    let wall_strips: Vec<[[f32; 2]; 2]> = world
        .walls
        .iter()
        .map(|s| [[s.a.0 as f32, s.a.1 as f32], [s.b.0 as f32, s.b.1 as f32]])
        .collect();
    rec.log_static(
        "world/walls",
        &rerun::LineStrips2D::new(wall_strips).with_colors([rerun::Color::from_rgb(130, 130, 140)]),
    )?;

    // One goal, dead center behind the U's back wall.
    let waypoints: [(f64, f64); 1] = [(6.5, 3.0)];
    rec.log_static(
        "world/waypoints",
        &rerun::Points2D::new(waypoints.map(|(x, y)| [x as f32, y as f32]))
            .with_radii([0.05])
            .with_colors([rerun::Color::from_rgb(120, 255, 120)]),
    )?;

    let nominal = DiffDrive {
        wheel_radius: 0.03,
        track_width: 0.15,
    };
    let true_model = DiffDrive {
        wheel_radius: 0.0297,
        track_width: 0.15,
    };

    let start = Pose::new(1.0, 3.0, 0.0); // facing straight into the U's mouth
    let mut robot = Robot {
        model: true_model,
        pose: start,
    };
    let mut encoders = Encoders::new(1024.0);
    let mut odometry = Odometry {
        model: nominal,
        ticks_per_rev: 1024.0,
        pose: start,
    };
    let mut rng = Rng::new(SEED);
    let mut motor_l = Motor::new(MOTOR_TAU, MOTOR_MAX);
    let mut motor_r = Motor::new(MOTOR_TAU, MOTOR_MAX);
    let mut heading_pid = Pid::new(HEADING_KP, HEADING_KI, HEADING_KD, 1.0);

    let camera = DepthCamera {
        n_rays: CAM_RAYS,
        fov: CAM_FOV,
        max_range: CAM_RANGE,
    };

    // M5: the robot's memory (starts all-Unknown) and its current plan.
    let mut map = OccupancyGrid::new(8.0, 6.0, MAP_RES);
    let mut path: Option<Vec<(f64, f64)>> = None;

    let mut mode = Mode::Goto;
    let mut wp_index = 0;
    let mut trail_true: Vec<[f32; 2]> = Vec::new();
    let mut trail_belief: Vec<[f32; 2]> = Vec::new();
    let steps = (DURATION / DT) as usize;

    for i in 0..steps {
        let t = i as f64 * DT;

        if wp_index >= waypoints.len() {
            println!("Mission complete at t = {:.1} s.", t);
            break;
        }
        let (gx, gy) = waypoints[wp_index];

        // ---- SENSE: the camera sees the PHYSICAL world (true pose). ----
        let scan = camera.scan(&robot.pose, &world);
        let min_dist = scan.iter().cloned().fold(f64::INFINITY, f64::min);
        // Split the fan: indices below center look right, above look left
        // (ray 0 is the rightmost — most negative angle).
        let center = CAM_RAYS / 2;
        let right_min = scan[..center].iter().cloned().fold(f64::INFINITY, f64::min);
        let left_min = scan[center + 1..]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        // ---- The state machine: transitions with hysteresis. ----
        let prev_mode = mode;
        mode = match mode {
            Mode::Goto if min_dist < AVOID_ENTER => Mode::Avoid,
            Mode::Avoid if min_dist > AVOID_EXIT => Mode::Goto,
            m => m, // no transition
        };
        if mode != prev_mode {
            heading_pid.reset(); // stale momentum doesn't cross modes
            rec.log(
                "events",
                &rerun::TextLog::new(format!("t={t:.1}s: {prev_mode:?} -> {mode:?}")),
            )?;
        }

        // The controller's own world-view (truth, or belief if enabled).
        let pose = if CONTROL_ON_BELIEF {
            odometry.pose
        } else {
            robot.pose
        };

        // ---- MAP: burn every camera ray into memory (exercise 6b). ----
        for (idx, &d) in scan.iter().enumerate() {
            let a = pose.theta + camera.ray_angle(idx);
            map.mark_ray(pose.x, pose.y, a, d, CAM_RANGE);
        }

        // ---- PLAN: re-run A* over the growing map twice a second. ----
        if i % REPLAN_TICKS == 0 || path.is_none() {
            path = plan(&map, (pose.x, pose.y), (gx, gy), INFLATION_CELLS);
        }

        // ---- DECIDE, per mode. ----
        let dist_to_goal = (gx - pose.x).hypot(gy - pose.y);
        if dist_to_goal < ARRIVE_RADIUS {
            wp_index += 1;
            heading_pid.reset();
            path = None;
            continue;
        }
        // Follow the plan: steer at the first path point beyond the
        // lookahead. No plan (empty map / goal unreachable)? Head straight
        // for the goal and let mapping + the reflex sort it out.
        let (tx, ty) = match &path {
            Some(p) => {
                let mut target = (gx, gy);
                for &(px, py) in p {
                    if (px - pose.x).hypot(py - pose.y) > LOOKAHEAD {
                        target = (px, py);
                        break;
                    }
                }
                target
            }
            None => (gx, gy),
        };
        let (v_cmd, w_cmd) = match mode {
            Mode::Goto => {
                let bearing = (ty - pose.y).atan2(tx - pose.x);
                let heading_error = shortest_turn(pose.theta, bearing);
                let w = heading_pid.update(heading_error, DT);
                let alignment = (1.0 - heading_error.abs() / std::f64::consts::FRAC_PI_2).max(0.0);
                let v = (KP_DIST * dist_to_goal).min(V_MAX) * alignment;
                (v, w)
            }
            Mode::Avoid => {
                // Turn toward whichever side has more room; creep forward.
                let w = if left_min > right_min {
                    AVOID_W
                } else {
                    -AVOID_W
                };
                (AVOID_V, w)
            }
        };

        // ---- ACT through the imperfect world (M3 + M2 physics). ----
        let (cmd_l, cmd_r) = nominal.inverse(v_cmd, w_cmd);
        let act_l = motor_l.step(cmd_l, DT);
        let act_r = motor_r.step(cmd_r, DT);
        let slip_l = 1.0 - 0.01 * rng.uniform();
        let slip_r = 1.0 - 0.01 * rng.uniform();
        let before = robot.pose;
        robot.step(act_l * slip_l, act_r * slip_r, DT);
        // Walls are solid: if the body would overlap one, the translation
        // is refused (rotation survives — a bumped robot can still pivot).
        // NOTE the honest consequence: the wheels DID spin (encoders count
        // them), so while grinding against a wall, odometry drifts fast —
        // exactly like a real robot pushing a wall.
        let bumped = world.collides(robot.pose.x, robot.pose.y, ROBOT_RADIUS);
        if bumped {
            robot.pose.x = before.x;
            robot.pose.y = before.y;
        }

        // ---- OBSERVE: belief from ticks alone. ----
        let (dticks_l, dticks_r) = encoders.advance(act_l, act_r, DT);
        odometry.update(dticks_l, dticks_r);

        // ---- telemetry ----
        rec.set_duration_secs("sim_time", t);

        // The robot's vision, drawn: one line per sight-line, out to its hit.
        let vision: Vec<[[f32; 2]; 2]> = scan
            .iter()
            .enumerate()
            .map(|(idx, &d)| {
                let a = robot.pose.theta + camera.ray_angle(idx);
                [
                    [robot.pose.x as f32, robot.pose.y as f32],
                    [
                        (robot.pose.x + d * a.cos()) as f32,
                        (robot.pose.y + d * a.sin()) as f32,
                    ],
                ]
            })
            .collect();
        let vision_color = match mode {
            Mode::Goto => rerun::Color::from_rgb(80, 110, 140),
            Mode::Avoid => rerun::Color::from_rgb(255, 120, 40),
        };
        rec.log(
            "robot/vision",
            &rerun::LineStrips2D::new(vision).with_colors([vision_color]),
        )?;

        // The MAP, as the robot remembers it (occupied cells only), every
        // 10th tick to keep the stream light.
        if i % 10 == 0 {
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
        // The current plan and the point being steered at.
        if let Some(p) = &path {
            let pts: Vec<[f32; 2]> = p.iter().map(|&(px, py)| [px as f32, py as f32]).collect();
            rec.log(
                "plan/path",
                &rerun::LineStrips2D::new([pts])
                    .with_colors([rerun::Color::from_rgb(120, 255, 120)]),
            )?;
        }
        rec.log(
            "plan/target",
            &rerun::Points2D::new([[tx as f32, ty as f32]])
                .with_radii([0.06])
                .with_colors([rerun::Color::from_rgb(255, 255, 120)]),
        )?;

        rec.log(
            "world/current_goal",
            &rerun::Points2D::new([[gx as f32, gy as f32]])
                .with_radii([0.09])
                .with_colors([rerun::Color::from_rgb(60, 255, 60)]),
        )?;

        let (x, y) = (robot.pose.x as f32, robot.pose.y as f32);
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
            0.25 * robot.pose.theta.cos() as f32,
            0.25 * robot.pose.theta.sin() as f32,
        );
        rec.log(
            "robot/heading",
            &rerun::Arrows2D::from_vectors([[hx, hy]])
                .with_origins([[x, y]])
                .with_colors([rerun::Color::from_rgb(255, 90, 90)]),
        )?;

        let (bx, by) = (odometry.pose.x as f32, odometry.pose.y as f32);
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
            &rerun::Scalars::single(min_dist),
        )?;
        rec.log(
            "telemetry/bumped",
            &rerun::Scalars::single(if bumped { 1.0 } else { 0.0 }),
        )?;
        rec.log(
            "telemetry/mode",
            &rerun::Scalars::single(match mode {
                Mode::Goto => 0.0,
                Mode::Avoid => 1.0,
            }),
        )?;
        rec.log(
            "telemetry/drift_m",
            &rerun::Scalars::single(robot.pose.distance_to(&odometry.pose)),
        )?;

        std::thread::sleep(Duration::from_secs_f64(DT / SPEEDUP));
    }

    println!(
        "Waypoints reached: {}/{}. Final drift truth-vs-belief: {:.3} m.",
        wp_index,
        waypoints.len(),
        robot.pose.distance_to(&odometry.pose)
    );
    Ok(())
}
