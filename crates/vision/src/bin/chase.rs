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
//! `--find` is the **fast/slow split**: Grounding DINO interprets the
//! phrase once at 0.3 fps, hands over a `TargetLock`, and D-FINE tracks
//! that class-plus-colour at 20 fps. The control loop never waits on the
//! slow model. See `vision::lock`.
//!
//! Then hold something the model knows — a cup, a bottle, a phone, a book,
//! or just yourself — and move it left and right.

use anyhow::Result;
use sim_core::{wrap_angle, ControlGains, GotoController, Pid, Pose, Robot, RobotSpec};
use std::collections::VecDeque;
use std::time::Instant;
use vision::session::Perceived;
use vision::{
    approach_factor, deadband, pick_target, Args, Detection, Detector, Frame, LowPass,
    ObjectDetector, OpenVocabDetector, Source, Stream, TargetLock,
};

const DESIRED: (u32, u32) = (640, 480);
const FPS: u32 = 30;
const MIN_CONFIDENCE: f32 = 0.40;
/// Acquisition threshold for the open-vocabulary pass — **lower than
/// [`MIN_CONFIDENCE`] on purpose.**
///
/// Grounding DINO scores on a different scale from a closed-set DETR: its
/// confidences are text-image similarities, not class posteriors, and a
/// correct hit routinely lands around 0.3. Reusing D-FINE's 0.40 here
/// found nothing at all on the first real run. `find` has always used
/// 0.30; this matches it and goes slightly lower, because a false hit at
/// acquisition is cheap (you see the wrong lock printed and rerun) while
/// a missed hit costs a 3-second model pass.
const ACQUIRE_CONFIDENCE: f32 = 0.25;
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
/// How much of the robot's path to keep on screen. 600 points is ~30 s at
/// 20 fps — enough to see where it came from, bounded so a long session
/// does not slow down.
const TRAIL_POINTS: usize = 600;

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

    // ---- replay: before any hardware is touched ----
    //
    // Deliberately ahead of the camera and the detector. Placed after
    // them, replay still asked macOS for camera permission and spent
    // seconds loading a model it would never call — which makes the one
    // thing this mode is for, re-running a moment instantly with nothing
    // attached, quietly untrue.
    if let Some(path) = &args.replay {
        let frames = vision::session::read(path).map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "replaying {} frames from {}\n",
            frames.len(),
            path.display()
        );
        let mut chase = Chase::new();
        rec.log("camera/image/named", &rerun::Clear::flat())?;
        rec.log("camera/image/candidates", &rerun::Clear::flat())?;
        // Footage if `--video` captured it; boxes on black otherwise.
        let footage = vision::session::frames_dir(path).is_dir();
        if !footage {
            println!("(no footage: this log was recorded without --video)");
        }
        // `--replay X --record Y` re-records: same perception, freshly
        // computed commands. Without it the advice this mode prints on a
        // divergence — "if that is intended, re-record" — would mean
        // "go and find a camera and stage the scene again", which for a
        // moment that has already happened is no advice at all.
        let mut rerecord = vision::session::Recorder::create(args.record.as_deref(), false)?;
        if let Some(out) = &args.record {
            println!("re-recording commands to {}", out.display());
            if args.video {
                eprintln!("  (--video ignored: re-recording keeps the original footage)");
            }
        }

        let mut divergences: Vec<String> = Vec::new();
        for (i, p) in frames.iter().enumerate() {
            let img = footage
                .then(|| vision::session::load_frame(path, i))
                .flatten();
            let got = chase.step(p, img.as_ref(), &rec)?;
            if let Err(d) = vision::session::check_command(i, p, got) {
                divergences.push(d);
            }
            rerecord.write(
                &vision::session::Perceived {
                    command: Some(got),
                    ..p.clone()
                },
                None,
            )?;
        }

        // Re-recording is a deliberate act of saying "the new behaviour is
        // correct", so it must not also fail on the difference it just
        // captured.
        if args.record.is_some() {
            println!(
                "re-recorded {} frames ({} command(s) changed)",
                frames.len(),
                divergences.len()
            );
            return Ok(());
        }

        let checked = frames.iter().filter(|p| p.command.is_some()).count();
        if !divergences.is_empty() {
            eprintln!("\nREPLAY DIVERGED from the recorded session:");
            for d in divergences.iter().take(10) {
                eprintln!("  - {d}");
            }
            if divergences.len() > 10 {
                eprintln!("  ... and {} more", divergences.len() - 10);
            }
            eprintln!(
                "\n{} of {checked} checked frames command differently. If that \
                 is intended, re-record; if not, this is the regression.",
                divergences.len()
            );
            std::process::exit(1);
        }
        match checked {
            0 => println!(
                "\nreplayed {} frames — NOT verified: this log predates \
                 recorded commands, so there was nothing to check against",
                frames.len()
            ),
            n => println!("\nreplay complete: {n} frames, every command matched"),
        }
        return Ok(());
    }

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
    println!(
        "camera {w}x{h}; loading {} ({})...",
        args.model.name(),
        args.model.describe()
    );
    // Concrete type, not `Box<dyn Detector>`: acquisition needs to retune
    // the confidence floor, which is not part of the trait. It is coerced
    // to `&mut dyn Detector` at the one place polymorphism is needed.
    let mut detector = ObjectDetector::load(args.model, MIN_CONFIDENCE, vision::Backend::Cpu)?;

    // ---- the fast/slow handoff (docs/14, vision::lock) ----
    //
    // `--find` does NOT swap the control loop onto the open-vocabulary
    // detector. That would run the loop at 0.3 fps. Instead the slow model
    // runs ONCE to interpret the phrase, hands over a TargetLock, and the
    // fast model does the tracking at 20 fps.
    let lock = match &args.find {
        None => None,
        Some(phrase) => {
            println!("loading open-vocabulary detector to find {phrase:?} (slow, once)...");
            let mut slow = OpenVocabDetector::new(&args.phrases(), ACQUIRE_CONFIDENCE)?;

            let frame = stream
                .frames
                .recv()
                .map_err(|_| anyhow::anyhow!("camera stopped before acquisition"))?;

            let t = Instant::now();
            let named = slow.detect(&frame)?;
            println!(
                "  open-vocab pass: {:.0} ms, {} hit(s)",
                t.elapsed().as_secs_f64() * 1000.0,
                named.len()
            );
            // Print every hit with its score. A failed acquisition is
            // otherwise undiagnosable: "0 hits" cannot distinguish "not in
            // frame" from "scored 0.24 against a 0.25 bar".
            for d in &named {
                println!("    {:>6.0}%  {}", d.confidence * 100.0, d.label);
            }
            let hit = pick_target(&named).ok_or_else(|| {
                anyhow::anyhow!(
                    "could not find {phrase:?} in view (nothing scored above \
                     {:.0}%). Try a plainer noun, or move the object closer.",
                    ACQUIRE_CONFIDENCE * 100.0
                )
            })?;

            // Both detectors on the SAME frame: the bridge is spatial.
            //
            // Drop the fast detector's floor for this one pass. Acquisition
            // wants RECALL — every plausible box is a candidate for spatial
            // matching — while the control loop wants precision. At the
            // tracking threshold a small mug never becomes a candidate and
            // the handoff fails against a frame full of furniture.
            detector.set_min_confidence(ACQUIRE_CONFIDENCE);
            let candidates = detector.detect(&frame)?;
            detector.set_min_confidence(MIN_CONFIDENCE);

            // Show the overlap, not just the labels: a failed handoff needs
            // to distinguish "the object was never detected" from "it was,
            // and the overlap was 0.28 against a 0.30 bar".
            for c in &candidates {
                println!(
                    "    {:>6.0}%  {:<16} overlap {:.2}",
                    c.confidence * 100.0,
                    c.label,
                    vision::iou(hit, c)
                );
            }

            // Put the acquisition frame on the timeline BEFORE deciding.
            // A failed handoff used to leave an empty viewer, which is the
            // worst possible moment to have nothing on screen: you cannot
            // tell whether the object was out of frame, boxed differently
            // by the two models, or simply missed by the fast one.
            log_acquisition(&rec, &frame, hit, &candidates)?;
            match TargetLock::acquire(phrase, hit, &candidates, &frame) {
                Some(lock) => {
                    println!("  {}\n", lock.describe());
                    Some(lock)
                }
                None => {
                    // Not fatal any more: fall back to chasing the most
                    // confident thing, and leave the acquisition frame on
                    // the timeline so the disagreement is visible.
                    eprintln!(
                        "  no COCO-80 class overlaps {phrase:?} (best overlap \
                         {:.2}, need {:.2}).\n  \
                         Falling back to untargeted chase — the viewer shows \
                         what each model saw.\n",
                        candidates
                            .iter()
                            .map(|c| vision::iou(hit, c))
                            .fold(0.0f32, f32::max),
                        vision::lock::MIN_OVERLAP
                    );
                    None
                }
            }
        }
    };

    let mut chase = Chase::new();

    // Clear the acquisition overlay before the loop starts.
    //
    // Rerun keeps a logged entity visible until it is overwritten, and
    // these were logged ONCE at t=0. Without this they hang over every
    // live frame for the rest of the session — a stale box from a frame
    // three minutes ago, indistinguishable from a live detection. Caught
    // on screen, which is the only place it was visible.
    rec.log("camera/image/named", &rerun::Clear::flat())?;
    rec.log("camera/image/candidates", &rerun::Clear::flat())?;

    let mut recorder = vision::session::Recorder::create(args.record.as_deref(), args.video)?;
    if let Some(p) = &args.record {
        match args.video {
            true => println!(
                "recording perception to {} and footage to {}\n",
                p.display(),
                vision::session::frames_dir(p).display()
            ),
            false => println!(
                "recording perception to {} (add --video for footage)\n",
                p.display()
            ),
        }
    }
    println!("hold up a cup, bottle, phone, book — or yourself — and move it around\n");

    let mut last_tick = Instant::now();
    for frame in stream.frames {
        let dt = last_tick.elapsed().as_secs_f64().clamp(0.001, 0.2);
        last_tick = Instant::now();

        let detections = detector.detect(&frame)?;
        // Locked: track that class (and colour). Otherwise: most confident.
        // Selection happens HERE, on the live frame, because the lock
        // matches on hue and needs pixels. Its *result* is recorded — see
        // `vision::session`.
        let chosen = match &lock {
            Some(l) => l.pick(&detections, &frame),
            None => pick_target(&detections),
        };
        let target = chosen.and_then(|c| detections.iter().position(|d| std::ptr::eq(d, c)));

        let mut perceived = Perceived {
            dt,
            frame_w: frame.width,
            frame_h: frame.height,
            detections,
            target,
            command: None,
        };
        // Step FIRST: the command is part of the record, so there is
        // something for a replay to check against.
        perceived.command = Some(chase.step(&perceived, Some(&frame), &rec)?);
        recorder.write(&perceived, Some(&frame))?;
    }
    Ok(())
}

/// Everything that persists between frames — and the one place a frame's
/// perception becomes motion.
///
/// Split out so that a live camera and a recording drive **the same code**.
/// That is the same seam the HIL rig has between `observe` and `advance`,
/// and it exists for the same reason: two loops that "do the same thing"
/// drift, and nothing notices until the robot behaves differently from the
/// session you thought you were reproducing.
struct Chase {
    robot: Robot,
    controller: GotoController,
    /// A second, identical PID fed the UNFILTERED signal. It steers
    /// nothing — it exists so the viewer can plot what we avoided.
    raw_pid: Pid,
    bearing_filter: LowPass,
    /// Bounded, because this loop never ends. An unbounded trail grows at
    /// 20 points/s AND is cloned into Rerun every frame, so the per-frame
    /// cost climbs with runtime — after an hour it is cloning 72k points
    /// 20x a second. A fixed window keeps that constant.
    trail: VecDeque<[f32; 2]>,
    /// Session time, accumulated from the per-frame `dt` rather than read
    /// off a clock.
    ///
    /// It has to be, or replay lies: 140 recorded frames worth 7 seconds
    /// replay in well under one, so a wall-clock timeline would squash the
    /// whole session into the first instant of the Rerun scrubber and
    /// print nothing to the console. Accumulating `dt` makes the live and
    /// replayed timelines the same timeline — which is the entire point of
    /// recording it.
    t: f64,
    /// Session time of the last console line.
    last_print_t: f64,
}

impl Chase {
    fn new() -> Chase {
        Chase {
            // The Stage 0 robot, unchanged.
            robot: Robot {
                model: SPEC.drive(),
                pose: Pose::ORIGIN,
            },
            controller: GotoController::new(GAINS),
            raw_pid: Pid::new(
                GAINS.heading_kp,
                GAINS.heading_ki,
                GAINS.heading_kd,
                GAINS.heading_i_limit,
            ),
            bearing_filter: LowPass::new(BEARING_ALPHA),
            trail: VecDeque::with_capacity(TRAIL_POINTS),
            t: 0.0,
            last_print_t: 0.0,
        }
    }

    /// One frame: perception in, motion out. `frame` is `None` on replay,
    /// where there are no pixels — everything except the camera image is
    /// identical.
    ///
    /// Returns the commanded `(v, w)`, which the live loop records and the
    /// replay loop checks. That return value is the whole difference
    /// between a replay you watch and a replay that can fail.
    fn step(
        &mut self,
        p: &Perceived,
        frame: Option<&Frame>,
        rec: &rerun::RecordingStream,
    ) -> Result<(f64, f64)> {
        let dt = p.dt;
        self.t += dt;
        let (v_cmd, w_cmd, heading_error, w_raw) = match p.target() {
            Some(d) => {
                // ---- the closed loop ----
                // The object's bearing IS the heading we want, expressed in
                // camera coordinates. shortest_turn gives the error from
                // where the robot currently points — the same function that
                // steered it toward waypoints in Stage 0.
                let measured = d.bearing(p.frame_w, HORIZONTAL_FOV) as f64;

                // What the controller WOULD do on the raw signal — computed
                // only so the viewer can show both curves at once.
                let raw_error = wrap_angle(shortest(self.robot.pose.theta, measured));
                let w_raw = self.raw_pid.update(raw_error, dt);

                // ---- noise handling ----
                // 1. Smooth the measurement before it reaches the PID, because
                //    the D term differentiates whatever jitter survives.
                // 2. Deadband the error, so sub-degree wobble commands nothing.
                let target_heading = self.bearing_filter.update(measured);
                let error = deadband(
                    wrap_angle(shortest(self.robot.pose.theta, target_heading)),
                    HEADING_DEADBAND,
                );
                // ---- the open-loop part (see module docs) ----
                // Box height as a distance proxy: taller box = closer.
                let approach = approach_factor(d.height, p.frame_h, STOP_AT_HEIGHT_FRACTION);

                // The shared steering law. Identical to the simulator's and
                // the firmware's — only the speed budget differs, because
                // here "how far away" comes from box size, not a map.
                let (v, w) = self
                    .controller
                    .steer(error, GAINS.v_max * approach as f64, dt);
                (v, w, error, w_raw)
            }
            None => {
                // Nothing seen: stop, and forget accumulated PID state so a
                // reappearing object doesn't inherit a stale integral.
                self.controller.reset();
                self.raw_pid.reset();
                self.bearing_filter.reset();
                (0.0, 0.0, 0.0, 0.0)
            }
        };

        let (omega_l, omega_r) = SPEC.drive().inverse(v_cmd, w_cmd);
        self.robot.step(omega_l, omega_r, dt);

        // ---- telemetry ----
        rec.set_duration_secs("time", self.t);
        if let Some(f) = frame {
            rec.log(
                "camera/image",
                &rerun::Image::from_rgb24(f.rgb.clone(), [f.width, f.height]),
            )?;
        }
        log_boxes(rec, &p.detections)?;

        let (rx_, ry_) = (self.robot.pose.x as f32, self.robot.pose.y as f32);
        if self.trail.len() == TRAIL_POINTS {
            self.trail.pop_front();
        }
        self.trail.push_back([rx_, ry_]);
        rec.log(
            "robot/trail",
            &rerun::LineStrips2D::new([self.trail.iter().copied().collect::<Vec<_>>()])
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
                0.3 * self.robot.pose.theta.cos() as f32,
                0.3 * self.robot.pose.theta.sin() as f32,
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

        if self.t - self.last_print_t >= 1.0 {
            match p.target() {
                Some(d) => println!(
                    "{:>11} {:.0}%  err={:+.3} rad  ->  v={:.2} w={:+.2}  robot θ={:+.2}",
                    d.label,
                    d.confidence * 100.0,
                    heading_error,
                    v_cmd,
                    w_cmd,
                    self.robot.pose.theta
                ),
                None => println!("(nothing detected — robot stopped)"),
            }
            self.last_print_t = self.t;
        }
        Ok((v_cmd, w_cmd))
    }
}

/// Shortest signed rotation from `from` to `to` — the exercise-2 function,
/// re-derived here because `sim_core::exercises` is where it lives.
fn shortest(from: f64, to: f64) -> f64 {
    sim_core::exercises::shortest_turn(from, to)
}

/// Draw what each model saw during the handoff: the named object in
/// yellow, the fast detector's candidates in blue. Their disagreement is
/// the whole story when acquisition fails.
fn log_acquisition(
    rec: &rerun::RecordingStream,
    frame: &Frame,
    hit: &Detection,
    candidates: &[Detection],
) -> Result<()> {
    rec.set_duration_secs("time", 0.0);
    rec.log(
        "camera/image",
        &rerun::Image::from_rgb24(frame.rgb.clone(), [frame.width, frame.height]),
    )?;
    rec.log(
        "camera/image/named",
        &rerun::Boxes2D::from_mins_and_sizes([(hit.x, hit.y)], [(hit.width, hit.height)])
            .with_labels([format!("{} {:.0}%", hit.label, hit.confidence * 100.0)])
            .with_colors([rerun::Color::from_rgb(255, 230, 60)]),
    )?;
    let mins: Vec<(f32, f32)> = candidates.iter().map(|d| (d.x, d.y)).collect();
    let sizes: Vec<(f32, f32)> = candidates.iter().map(|d| (d.width, d.height)).collect();
    let labels: Vec<String> = candidates
        .iter()
        .map(|d| {
            format!(
                "{} {:.0}% iou {:.2}",
                d.label,
                d.confidence * 100.0,
                vision::iou(hit, d)
            )
        })
        .collect();
    rec.log(
        "camera/image/candidates",
        &rerun::Boxes2D::from_mins_and_sizes(mins, sizes)
            .with_labels(labels)
            .with_colors([rerun::Color::from_rgb(90, 200, 255)]),
    )?;
    Ok(())
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
