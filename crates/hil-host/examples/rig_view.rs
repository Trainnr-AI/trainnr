//! The whole rig in one window: what the camera sees, what the robot
//! believes, and what the motors are doing about it.
//!
//! ```sh
//! cargo run -p hil-host --example rig_view -- /dev/cu.usbmodem11
//! cargo run -p hil-host --example rig_view -- /dev/cu.usbmodem11 --record run.wire
//! cargo run -p hil-host --example rig_view -- --replay run.wire
//! ```
//!
//! Needs `firmware/pico-odom` built with `camera` (typically `chase`).
//!
//! # Why one viewer and not two
//!
//! The serial port has one reader. `odom_view` drew the pose and never
//! the camera; `camera_view` drew the camera and never the pose — so the
//! question the chase loop actually raises, *"did the wheels do that
//! because of what the camera saw?"*, could not be answered by either.
//! Cause and effect belong on one screen, on one clock.
//!
//! (`odom_view` remains the transport-comparison instrument — USB against
//! radio — a job this viewer does not attempt, and a build this viewer's
//! firmware cannot even carry: `camera,wifi` is refused at compile time.)
//!
//! # Replay is the point
//!
//! `--record` captures the raw lines while drawing them, so the session
//! that gets watched is byte-for-byte the session that can be re-watched
//! without the hardware — how one bench afternoon became the regression
//! fixture `rp2350-utrap.wire`, and the treatment a chase run deserves.
//!
//! # Everything here is the shared vocabulary
//!
//! `hil_protocol::Status` for pose lines, `hil_protocol::thumbnail` for
//! the picture, `belief_viz::Trail` for the drawing, `blob` for the
//! pixel conversion. This file adds no parsing of its own, so it cannot
//! drift from what the chip writes.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use hil_protocol::{thumbnail, Status};

/// Status lines per second — `1000 / REPORT_MS` in `firmware/pico-odom`.
/// Live runs use the wall clock; a replayed file carries no timestamps,
/// so its clock is reconstructed from this instead. Saying which is
/// which matters — a replay on a wall clock would run at whatever speed
/// the disk felt like.
const REPORTS_PER_SECOND: u64 = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ⚠️ Parsed by NAME, not by position — `joint_viz` learned this when
    // `args.last()` silently made `--record out.wire` the thing it tried
    // to open.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let replay = flag("--replay");
    let capture_to = flag("--record");
    let source = replay.clone().or_else(|| {
        args.iter()
            .find(|a| !a.starts_with("--"))
            .filter(|a| Some(*a) != capture_to.as_ref())
            .cloned()
    });
    let Some(source) = source else {
        eprintln!("usage: rig_view <serial-port> [--record <file>] | --replay <file.wire>");
        std::process::exit(2);
    };
    let replay = replay.is_some();

    let rec = rerun::RecordingStreamBuilder::new("robotiq_rig")
        .with_blueprint(layout())
        .spawn()?;

    let lines: Box<dyn BufRead> = if replay {
        Box::new(BufReader::new(std::fs::File::open(&source)?))
    } else {
        // ⚠️ Opening raises DTR, which un-gates the chip — on a `chase`
        // build, the robot is ARMED from this line on.
        println!("⚠️  a chase build is armed the moment the port opens");
        Box::new(BufReader::new(hil_protocol::link::open(
            &source,
            Duration::from_millis(500),
        )?))
    };
    let mut capture = capture_to.map(std::fs::File::create).transpose()?;
    println!("watching {source} — pick `robotiq_rig` in the viewer");

    let started = Instant::now();
    let mut trail = belief_viz::Trail::new(
        belief_viz::belief(),
        belief_viz::Size::BENCH,
        REPORTS_PER_SECOND,
    );
    let mut expecting: Option<(u32, u32)> = None;
    let mut pixels: Vec<u8> = Vec::new();
    let mut frames = 0u64;
    let mut reports = 0u64;
    let (mut last_errors, mut announced_stall) = ((0u64, 0u64), false);

    for line in lines.lines() {
        let Ok(line) = line else { break };
        if let Some(file) = &mut capture {
            use std::io::Write as _;
            writeln!(file, "{}", line.trim_end())?;
        }
        let line = line.trim();
        rec.set_duration_secs(
            if replay { "run" } else { "wall_time" },
            if replay {
                reports as f64 / REPORTS_PER_SECOND as f64
            } else {
                started.elapsed().as_secs_f64()
            },
        );

        // ---- the picture ----
        if let Some(dims) = thumbnail::parse_header(line) {
            expecting = Some(dims);
            pixels.clear();
            continue;
        }
        if let Some((width, height)) = expecting {
            if let Some(row) = thumbnail::parse_row(line, width) {
                pixels.extend(row.flat_map(blob::rgb565_to_rgb888));
                if pixels.len() >= (width * height * 3) as usize {
                    frames += 1;
                    rec.log(
                        "camera/image",
                        &rerun::Image::from_rgb24(std::mem::take(&mut pixels), [width, height]),
                    )?;
                    expecting = None;
                }
                continue;
            }
        }

        // ---- the words: camera identity, blob, anything else the chip
        // wants read ----
        if let Some(note) = line.strip_prefix("# ") {
            // A pixel row whose header was lost in transit is pure hex
            // and belongs to no panel — without this check, one dropped
            // line spams thirty rows of hex into the event log.
            if note.len() >= 8 && note.bytes().all(|b| b.is_ascii_hexdigit()) {
                continue;
            }
            rec.log("events", &rerun::TextLog::new(note.to_string()))?;
            continue;
        }

        // ---- the belief and the motors ----
        let Some(report) = Status::parse(line) else {
            continue;
        };
        reports += 1;
        trail.draw(
            &rec,
            "belief",
            sim_core::Pose::new(report.x, report.y, report.heading),
        )?;
        // The stall-revealing pair: commanded duty rising while the tick
        // lines stay flat. docs/07 records the run where exactly this
        // panel made a bug visible that the tests could not see.
        rec.log(
            "motor/duty_percent",
            &rerun::Scalars::single(report.duty_percent as f64),
        )?;
        rec.log(
            "ticks/left",
            &rerun::Scalars::single(report.ticks_left as f64),
        )?;
        rec.log(
            "ticks/right",
            &rerun::Scalars::single(report.ticks_right as f64),
        )?;
        // Plotted, not just printed: a rising decode-error count means the
        // tick stream is undercounted, so every pose after it is wrong by
        // an amount nothing else on screen reveals. (The same signal
        // `odom_view` draws; each viewer announces to its own screen.)
        rec.log(
            "errors/left",
            &rerun::Scalars::single(report.errors_left as f64),
        )?;
        rec.log(
            "errors/right",
            &rerun::Scalars::single(report.errors_right as f64),
        )?;
        let errors_now = (report.errors_left, report.errors_right);
        if errors_now > last_errors {
            rec.log(
                "events",
                &rerun::TextLog::new(format!(
                    "decode errors L{} R{} — pose is now an UNDERCOUNT",
                    errors_now.0, errors_now.1
                )),
            )?;
            last_errors = errors_now;
        }
        // The firmware stops driving when this latches, so it is said
        // once rather than fifty times a second for as long as it parks.
        if report.stalled && !announced_stall {
            rec.log(
                "events",
                &rerun::TextLog::new("STALLED — commanded but not moving. Check the battery."),
            )?;
            announced_stall = true;
        }
    }

    println!("{frames} camera frames, {reports} status reports drawn");
    Ok(())
}

/// Cause on the left, effect on the right: the camera above the pose it
/// steers, the motor evidence beside them.
fn layout() -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{
        Blueprint, Horizontal, Spatial2DView, TextLogView, TimeSeriesView, Vertical,
    };
    Blueprint::new(
        Horizontal::new([
            Vertical::new([
                Spatial2DView::new("what the camera sees")
                    .with_origin("/camera")
                    .into(),
                Spatial2DView::new("where the robot believes it is")
                    .with_origin("/belief")
                    .into(),
            ])
            .into(),
            Vertical::new([
                TimeSeriesView::new("duty (%)").with_origin("/motor").into(),
                TimeSeriesView::new("encoder ticks")
                    .with_origin("/ticks")
                    .into(),
                TimeSeriesView::new("decode errors — undercount alarm")
                    .with_origin("/errors")
                    .into(),
                TextLogView::new("what the chip says")
                    .with_origin("/events")
                    .into(),
            ])
            .into(),
        ])
        .with_column_shares([0.5, 0.5]),
    )
    .with_auto_views(false)
}
