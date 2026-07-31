//! P2 — perception drives control.
//!
//! Your webcam sees an object; the robot from Stage 0 turns to face it.
//! The entire steering path is code written weeks ago for a simulated
//! world, now fed by a real camera:
//!
//! ```text
//!   camera → Detector → bearing → GotoController → DiffDrive → Robot
//!            (P1)       (P1)      (ex. 2 + 4, M3)   (M0)       (M0)
//! ```
//!
//! `Detector` is a trait, and which model sits behind it is a CLI flag —
//! so the same control loop runs on any of the seven detectors in
//! `DetectorModel`, or on the open-vocabulary detector from `find`.
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
//! cargo run --release -p vision --bin chase -- --model deimv2-s
//! cargo run --release -p vision --bin chase -- --find "red mug"   # open-vocab
//! ```
//!
//! Then hold something the model knows — a cup, a bottle, a phone, a book,
//! or just yourself — and move it left and right.

use anyhow::Result;
use sim_core::{wrap_angle, ControlGains, GotoController, Pid, Pose, Robot, RobotSpec};
use std::time::Instant;
use vision::{
    approach_factor, deadband, pick_target, Args, Detection, Detector, LowPass, ObjectDetector,
    OpenVocabDetector, Source, Stream,
};

const DESIRED: (u32, u32) = (640, 480);
const FPS: u32 = 30;
const MIN_CONFIDENCE: f32 = 0.40;
/// Rough webcam horizontal field of view (~60°).
const HORIZONTAL_FOV: f32 = 1.05;

/// The robot and its steering profile — the same definitions the simulator
/// and the firmware compile against. VISUAL_SERVO is deliberately gentler
/// than WAYPOINT; the reason is documented on the constant itself.
const SPEC: RobotSpec = RobotSpec::SIM_BOT;
const GAINS: ControlGains = ControlGains::VISUAL_SERVO;

/// Fraction of frame height a box occupies when the robot should stop
/// approaching. Bigger box = closer object.
const STOP_AT_HEIGHT_FRACTION: f32 = 0.55;

// ---- Noise handling. Tune these and watch the two turn_rate plots. ----
/// Low-pass on the measured bearing. 1.0 = off, 0.05 = very smooth/laggy.
const BEARING_ALPHA: f64 = 0.25;
/// Heading errors below this (radians) are treated as zero. ~0.02 rad is
/// about 1° — finer than the detector can honestly resolve.
const HEADING_DEADBAND: f64 = 0.02;

fn main() -> Result<()> {
    vision::logging::init();
    let args = Args::parse_from(std::env::args().skip(1))?;
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

    // Permission + capture thread + latest-wins channel, in one call.
    let source = match args.camera {
        Some(i) => Source::Index(i),
        None => Source::Name("Brio"),
    };
    let stream = match Stream::start(source, DESIRED, FPS) {
        Ok(s) => s,
        // Named lookup can miss if no Brio is attached; fall back to any.
        Err(_) if args.camera.is_none() => Stream::open(Source::Index(0), DESIRED, FPS)?,
        Err(e) => return Err(e),
    };
    let (w, h) = stream.resolution;

    // The detector is chosen here and never mentioned again — everything
    // downstream talks to `dyn Detector`. This is what makes `--find`
    // cost one line instead of a second binary.
    let mut detector: Box<dyn Detector> = match &args.find {
        Some(phrase) => {
            println!("camera {w}x{h}; loading open-vocabulary detector for {phrase:?}...");
            Box::new(OpenVocabDetector::new(&args.phrases(), MIN_CONFIDENCE)?)
        }
        None => {
            println!(
                "camera {w}x{h}; loading {} ({})...",
                args.model.name(),
                args.model.describe()
            );
            Box::new(ObjectDetector::load(
                args.model,
                MIN_CONFIDENCE,
                vision::Backend::Cpu,
            )?)
        }
    };

    // The Stage 0 robot, unchanged.
    let mut robot = Robot {
        model: SPEC.drive(),
        pose: Pose::ORIGIN,
    };
    let mut controller = GotoController::new(GAINS);
    // A second, identical PID fed the UNFILTERED signal. It steers
    // nothing — it exists so the viewer can plot what we avoided.
    let mut raw_pid = Pid::new(
        GAINS.heading_kp,
        GAINS.heading_ki,
        GAINS.heading_kd,
        GAINS.heading_i_limit,
    );
    let mut bearing_filter = LowPass::new(BEARING_ALPHA);

    println!("hold up a cup, bottle, phone, book — or yourself — and move it around\n");

    let start = Instant::now();
    let mut trail: Vec<[f32; 2]> = Vec::new();
    let mut last_tick = Instant::now();
    let mut last_print = Instant::now();

    for frame in stream.frames {
        let dt = last_tick.elapsed().as_secs_f64().clamp(0.001, 0.2);
        last_tick = Instant::now();

        let detections = detector.detect(&frame)?;
        let target = pick_target(&detections);

        let (v_cmd, w_cmd, heading_error, w_raw) = match target {
            Some(d) => {
                // ---- the closed loop ----
                // The object's bearing IS the heading we want, expressed in
                // camera coordinates. shortest_turn gives the error from
                // where the robot currently points — the same function that
                // steered it toward waypoints in Stage 0.
                let measured = d.bearing(frame.width, HORIZONTAL_FOV) as f64;

                // What the controller WOULD do on the raw signal — computed
                // only so the viewer can show both curves at once.
                let raw_error = wrap_angle(shortest(robot.pose.theta, measured));
                let w_raw = raw_pid.update(raw_error, dt);

                // ---- noise handling ----
                // 1. Smooth the measurement before it reaches the PID, because
                //    the D term differentiates whatever jitter survives.
                // 2. Deadband the error, so sub-degree wobble commands nothing.
                let target_heading = bearing_filter.update(measured);
                let error = deadband(
                    wrap_angle(shortest(robot.pose.theta, target_heading)),
                    HEADING_DEADBAND,
                );
                // ---- the open-loop part (see module docs) ----
                // Box height as a distance proxy: taller box = closer.
                let approach = approach_factor(d.height, frame.height, STOP_AT_HEIGHT_FRACTION);

                // The shared steering law. Identical to the simulator's and
                // the firmware's — only the speed budget differs, because
                // here "how far away" comes from box size, not a map.
                let (v, w) = controller.steer(error, GAINS.v_max * approach as f64, dt);
                (v, w, error, w_raw)
            }
            None => {
                // Nothing seen: stop, and forget accumulated PID state so a
                // reappearing object doesn't inherit a stale integral.
                controller.reset();
                raw_pid.reset();
                bearing_filter.reset();
                (0.0, 0.0, 0.0, 0.0)
            }
        };

        let (omega_l, omega_r) = SPEC.drive().inverse(v_cmd, w_cmd);
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
        rec.log("control/turn_rate_raw", &rerun::Scalars::single(w_raw))?;
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
