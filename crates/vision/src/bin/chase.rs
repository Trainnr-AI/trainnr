//! P2 — perception drives control.
//!
//! Your webcam sees an object; the robot from Stage 0 turns to face it.
//! The entire steering path is code written weeks ago for a simulated
//! world, now fed by a real camera:
//!
//! ```text
//!   camera → D-FINE → bearing → shortest_turn → Pid → DiffDrive → Robot
//!            (P1)     (P1)      (exercise 2)  (ex. 4)  (M0)      (M0)
//! ```
//!
//! # What is and isn't a closed loop here — read this
//!
//! **The rotation IS a genuine closed-loop visual servo.** The robot's
//! target heading is the object's bearing in camera coordinates; the PID
//! drives the heading error to zero; move the object and the robot tracks
//! it. Hold the object still and the error converges. That is real.
//!
//! **The forward motion is open-loop illustration.** A real robot's camera
//! is bolted to the robot, so driving changes what it sees. Ours is bolted
//! to a laptop that doesn't move. Distance is estimated from box height —
//! a crude proxy the research warned about — and nothing corrects it.
//! Stage 3 closes that loop, when the camera physically rides the robot.
//!
//! Run from Terminal.app (not an editor terminal):
//!
//! ```sh
//! cargo run --release -p vision --bin chase
//! ```
//!
//! Then hold something the model knows — a cup, a bottle, a phone, a book,
//! or just yourself — and move it left and right.

use anyhow::Result;
use sim_core::{wrap_angle, DiffDrive, Pid, Pose, Robot};
use std::sync::mpsc;
use std::time::Instant;
use vision::{CameraSource, DFineDetector, Detection, Detector, Frame, NokhwaCamera};

const DESIRED: (u32, u32) = (640, 480);
const FPS: u32 = 30;
const MIN_CONFIDENCE: f32 = 0.40;
/// Rough webcam horizontal field of view (~60°).
const HORIZONTAL_FOV: f32 = 1.05;

/// Heading PID — the same gains tuned by eye in the Rerun viewer during
/// Stage 0's M3, now steering on camera input instead of a goal position.
const HEADING_KP: f64 = 3.0;
const HEADING_KI: f64 = 0.0;
const HEADING_KD: f64 = 0.3;

/// Robot geometry, unchanged from the simulator.
const WHEEL_RADIUS: f64 = 0.03;
const TRACK_WIDTH: f64 = 0.15;

/// Fraction of frame height a box occupies when the robot should stop
/// approaching. Bigger box = closer object.
const STOP_AT_HEIGHT_FRACTION: f32 = 0.55;
const V_MAX: f64 = 0.35;

fn main() -> Result<()> {
    let rec = rerun::RecordingStreamBuilder::new("robotiq_chase").spawn()?;
    rec.log_static(
        "/",
        &rerun::AnnotationContext::new([
            (0, "person", rerun::Rgba32::from_rgb(255, 90, 90)),
            (39, "bottle", rerun::Rgba32::from_rgb(90, 200, 255)),
            (41, "cup", rerun::Rgba32::from_rgb(120, 255, 120)),
            (67, "cell phone", rerun::Rgba32::from_rgb(255, 200, 60)),
            (73, "book", rerun::Rgba32::from_rgb(200, 140, 255)),
        ]),
    )?;

    let (tx, rx) = mpsc::channel();
    eprintln!("requesting camera access...");
    nokhwa::nokhwa_initialize(move |g| {
        let _ = tx.send(g);
    });
    if !rx.recv().unwrap_or(false) {
        anyhow::bail!("camera permission denied — run from Terminal.app, not an editor terminal");
    }

    let (frame_tx, frame_rx) = mpsc::sync_channel::<Frame>(1);
    let (res_tx, res_rx) = mpsc::channel::<(u32, u32)>();
    std::thread::spawn(move || {
        let mut cam = match NokhwaCamera::open(0, DESIRED, FPS) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("camera: {e:#}");
                return;
            }
        };
        let _ = res_tx.send(cam.resolution());
        loop {
            match cam.next_frame() {
                Ok(f) => match frame_tx.try_send(f) {
                    Ok(()) | Err(mpsc::TrySendError::Full(_)) => {}
                    Err(_) => return,
                },
                Err(e) => {
                    eprintln!("capture: {e:#}");
                    return;
                }
            }
        }
    });

    let (w, h) = res_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("camera failed"))?;
    println!("camera {w}x{h}; loading detector...");
    let mut detector = DFineDetector::new(MIN_CONFIDENCE)?;

    // The Stage 0 robot, unchanged.
    let model = DiffDrive {
        wheel_radius: WHEEL_RADIUS,
        track_width: TRACK_WIDTH,
    };
    let mut robot = Robot {
        model,
        pose: Pose::ORIGIN,
    };
    let mut heading_pid = Pid::new(HEADING_KP, HEADING_KI, HEADING_KD, 1.0);

    println!("hold up a cup, bottle, phone, book — or yourself — and move it around\n");

    let start = Instant::now();
    let mut trail: Vec<[f32; 2]> = Vec::new();
    let mut last_tick = Instant::now();
    let mut last_print = Instant::now();

    for frame in frame_rx {
        let dt = last_tick.elapsed().as_secs_f64().clamp(0.001, 0.2);
        last_tick = Instant::now();

        let detections = detector.detect(&frame)?;
        let target = detections
            .iter()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence));

        let (v_cmd, w_cmd, heading_error) = match target {
            Some(d) => {
                // ---- the closed loop ----
                // The object's bearing IS the heading we want, expressed in
                // camera coordinates. shortest_turn gives the error from
                // where the robot currently points — the same function that
                // steered it toward waypoints in Stage 0.
                let target_heading = d.bearing(frame.width, HORIZONTAL_FOV) as f64;
                let error = wrap_angle(shortest(robot.pose.theta, target_heading));
                let w = heading_pid.update(error, dt);

                // ---- the open-loop part (see module docs) ----
                // Box height as a distance proxy: taller box = closer.
                let height_fraction = d.height / frame.height as f32;
                let approach = (1.0 - height_fraction / STOP_AT_HEIGHT_FRACTION).clamp(0.0, 1.0);
                // Turn first, drive second — same alignment throttle as M3.
                let alignment = (1.0 - error.abs() / std::f64::consts::FRAC_PI_2).max(0.0);
                let v = V_MAX * approach as f64 * alignment;
                (v, w, error)
            }
            None => {
                // Nothing seen: stop, and forget accumulated PID state so a
                // reappearing object doesn't inherit a stale integral.
                heading_pid.reset();
                (0.0, 0.0, 0.0)
            }
        };

        let (omega_l, omega_r) = model.inverse(v_cmd, w_cmd);
        robot.step(omega_l, omega_r, dt);

        // ---- telemetry ----
        let t = start.elapsed().as_secs_f64();
        rec.set_duration_secs("time", t);
        rec.log(
            "camera/image",
            &rerun::Image::from_rgb24(frame.rgb, [frame.width, frame.height]),
        )?;
        log_boxes(&rec, &detections)?;

        let (rx_, ry_) = (robot.pose.x as f32, robot.pose.y as f32);
        trail.push([rx_, ry_]);
        rec.log(
            "robot/trail",
            &rerun::LineStrips2D::new([trail.clone()])
                .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
        )?;
        rec.log(
            "robot/body",
            &rerun::Points2D::new([[rx_, ry_]])
                .with_radii([0.09])
                .with_colors([rerun::Color::from_rgb(255, 200, 60)]),
        )?;
        rec.log(
            "robot/heading",
            &rerun::Arrows2D::from_vectors([[
                0.3 * robot.pose.theta.cos() as f32,
                0.3 * robot.pose.theta.sin() as f32,
            ]])
            .with_origins([[rx_, ry_]])
            .with_colors([rerun::Color::from_rgb(255, 90, 90)]),
        )?;
        rec.log(
            "control/heading_error_rad",
            &rerun::Scalars::single(heading_error),
        )?;
        rec.log("control/turn_rate", &rerun::Scalars::single(w_cmd))?;
        rec.log("control/forward_speed", &rerun::Scalars::single(v_cmd))?;

        if last_print.elapsed().as_secs_f64() >= 1.0 {
            match target {
                Some(d) => println!(
                    "{:>11} {:.0}%  err={:+.3} rad  ->  v={:.2} w={:+.2}  robot θ={:+.2}",
                    d.label,
                    d.confidence * 100.0,
                    heading_error,
                    v_cmd,
                    w_cmd,
                    robot.pose.theta
                ),
                None => println!("(nothing detected — robot stopped)"),
            }
            last_print = Instant::now();
        }
    }
    Ok(())
}

/// Shortest signed rotation from `from` to `to` — the exercise-2 function,
/// re-derived here because `sim_core::exercises` is where it lives.
fn shortest(from: f64, to: f64) -> f64 {
    sim_core::exercises::shortest_turn(from, to)
}

fn log_boxes(rec: &rerun::RecordingStream, dets: &[Detection]) -> Result<()> {
    let mins: Vec<(f32, f32)> = dets.iter().map(|d| (d.x, d.y)).collect();
    let sizes: Vec<(f32, f32)> = dets.iter().map(|d| (d.width, d.height)).collect();
    let labels: Vec<String> = dets
        .iter()
        .map(|d| format!("{} {:.0}%", d.label, d.confidence * 100.0))
        .collect();
    let ids: Vec<u16> = dets.iter().map(|d| d.class_id).collect();
    rec.log(
        "camera/image/detections",
        &rerun::Boxes2D::from_mins_and_sizes(mins, sizes)
            .with_labels(labels)
            .with_class_ids(ids),
    )?;
    Ok(())
}
