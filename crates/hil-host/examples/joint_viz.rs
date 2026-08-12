//! Watch a real arm joint in Rerun.
//!
//! ```sh
//! cargo run -p hil-host --example joint_viz -- /dev/cu.usbmodem11
//! cargo run -p hil-host --example joint_viz -- --replay run.wire
//! ```
//!
//! # The same picture, from a motor instead of a simulator
//!
//! `cargo run -p arm --example watch` draws a simulated arm. This draws
//! the *measured* one, through `belief_viz::arm` — the same routine — so
//! the two are directly comparable. A second drawing routine would
//! disagree first about which way a joint bends, which is the one thing a
//! picture exists to catch.
//!
//! ⚠️ The figure is SO-ARM101's geometry driven by whatever angles
//! arrive. On the bench there are no links on those shafts, so it is not
//! a picture of the bench: it is the arm those angles *would* command.
//!
//! # Replay is the point
//!
//! `--replay` reads a `.wire` capture instead of a port, so a hardware
//! session becomes something you can re-watch without the hardware —
//! which is how `rp2350-utrap.wire` turned one afternoon on a bench into
//! a regression fixture that still runs.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use hil_protocol::{JointPhase, Message};

/// Joint names, in wire-index order, matching `ArmSpec::bench_two_joint`.
/// Named rather than numbered so a plot legend says what it is.
const NAMES: [&str; 2] = ["shoulder_lift", "elbow_flex"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ⚠️ Parsed by NAME, not by position. The first version took
    // `args.last()` as the source, which silently made `--record out.wire`
    // the thing it tried to *open* — a flag added later changing the
    // meaning of an argument added earlier.
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
        eprintln!("usage: joint_viz <serial-port> [--record <file>] | --replay <file.wire>");
        std::process::exit(2);
    };
    let replay = replay.is_some();

    let rec = rerun::RecordingStreamBuilder::new("robotiq_joint")
        .with_blueprint(layout())
        .spawn()?;
    belief_viz::arm::draw_stand(&rec, "arm")?;

    let lines: Box<dyn BufRead> = if replay {
        Box::new(BufReader::new(std::fs::File::open(&source)?))
    } else {
        // ⚠️ Opening raises DTR, which un-gates the firmware's reports.
        Box::new(BufReader::new(hil_protocol::link::open(
            &source,
            Duration::from_millis(500),
        )?))
    };
    // `--record <file>` alongside a port captures the raw lines while
    // drawing them, so the session that gets watched is byte-for-byte the
    // session that can be replayed later.
    let mut capture = capture_to.map(std::fs::File::create).transpose()?;
    println!("watching {source} — pick `robotiq_joint` in the viewer");

    let started = Instant::now();
    let mut angles = [0.0f32; 2];
    let mut frames: u64 = 0;
    let mut phase_now: Option<JointPhase> = None;

    for line in lines.lines() {
        let Ok(line) = line else { break };
        if let Some(file) = &mut capture {
            use std::io::Write as _;
            writeln!(file, "{}", line.trim_end())?;
        }
        let Ok(Message::Joint {
            index,
            raw_ticks,
            milliradians,
            duty,
            phase,
        }) = Message::parse(line.trim())
        else {
            continue;
        };
        let Some(name) = NAMES.get(usize::from(index)) else {
            continue;
        };

        // Wall time on a live port; for a replay the file carries no
        // timestamps, so frames are counted instead. Saying which is
        // which matters — a replay drawn on a wall clock would look like
        // a machine running at whatever speed the disk felt like.
        rec.set_duration_secs(
            if replay { "frame" } else { "wall_time" },
            if replay {
                frames as f64 / (2.0 * 5.0)
            } else {
                started.elapsed().as_secs_f64()
            },
        );
        frames += 1;

        let radians = f64::from(milliradians) / 1000.0;
        angles[usize::from(index)] = radians as f32;
        rec.log(format!("angles/{name}"), &rerun::Scalars::single(radians))?;
        rec.log(
            format!("duty/{name}"),
            &rerun::Scalars::single(f64::from(duty)),
        )?;
        rec.log(
            format!("encoder/{name}"),
            &rerun::Scalars::single(raw_ticks as f64),
        )?;
        // 1 while the guard authorises motion, 0 while it refuses — the
        // shape of the failsafe, the same plot the simulated viewer draws.
        rec.log(
            "guard/authorised",
            &rerun::Scalars::single(f64::from(u8::from(phase == JointPhase::Holding))),
        )?;

        if phase_now != Some(phase) {
            rec.log("events", &rerun::TextLog::new(describe(phase)))?;
            println!("  → {phase}");
            phase_now = Some(phase);
        }

        let colour = if phase == JointPhase::Holding {
            belief_viz::belief()
        } else {
            belief_viz::arm::holding()
        };
        belief_viz::arm::draw(&rec, "arm", angles[0], angles[1], colour)?;
    }

    println!("{frames} joint reports drawn");
    Ok(())
}

/// What a phase means, in the words you would want in a log at 2 a.m.
fn describe(phase: JointPhase) -> String {
    match phase {
        JointPhase::Homing => "homing — creeping toward a hard stop".into(),
        JointPhase::Homed => "homed — this position is now zero".into(),
        JointPhase::Holding => "holding — the guard is authorising motion".into(),
        JointPhase::Held => "HELD — the guard refused; joints keep their last target. \
             On a base the same silence would cut the outputs and coast."
            .into(),
        JointPhase::NoZero => "no zero was ever adopted — torque is off".into(),
    }
}

/// Panels grouped by unit, as in the simulated viewer: radians, duty and
/// raw counts never share an axis, because one outlier flattens the rest.
fn layout() -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{
        Blueprint, Horizontal, Spatial2DView, TextLogView, TimeSeriesView, Vertical,
    };
    Blueprint::new(
        Horizontal::new([
            Vertical::new([
                Spatial2DView::new("the arm those angles command")
                    .with_origin("/arm")
                    .into(),
                TimeSeriesView::new("authorised — 1 move, 0 HELD")
                    .with_origin("/guard")
                    .into(),
            ])
            .into(),
            Vertical::new([
                TimeSeriesView::new("measured angle (rad)")
                    .with_origin("/angles")
                    .into(),
                TimeSeriesView::new("duty (±1000)")
                    .with_origin("/duty")
                    .into(),
                TimeSeriesView::new("encoder (raw ticks)")
                    .with_origin("/encoder")
                    .into(),
                TextLogView::new("what the joint said")
                    .with_origin("/events")
                    .into(),
            ])
            .into(),
        ])
        .with_column_shares([0.45, 0.55]),
    )
    .with_auto_views(false)
}
