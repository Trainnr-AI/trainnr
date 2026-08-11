//! Watch the arm's safety layer decide, in Rerun.
//!
//! ```sh
//! cargo run -p arm --example watch
//! ```
//!
//! # Why a picture and not a printout
//!
//! Reading `HoldStale { age_ms: 520 }` tells you a number changed.
//! Watching the arm keep its pose while the plan runs on without it tells
//! you the failsafe **inverted correctly** — that silence produced a hold
//! and not a collapse. This project has learned twice that a bug can be
//! invisible in a passing test and obvious on a screen.
//!
//! # The link lengths are real now, and they still live here
//!
//! They come from the SO-ARM101 vendor outline drawing (2026-08-12) —
//! see `LINK_METRES` — which beats the guesses that were here before.
//! They stay in the example anyway, for two reasons that outlived the
//! guessing:
//!
//! 1. The drawing says in red that its dimensions *"are for reference
//!    only and may differ from the actual size."* Vendor CAD is fine for
//!    pixels and not fine for kinematics.
//! 2. Nothing in `crates/arm` does kinematics. Geometry moves into
//!    `ArmSpec` when something needs it and it has been measured off the
//!    arm in the room — not before, or the library ships a number its
//!    tests will quietly start depending on.
//!
//! A stick figure earns its keep regardless: it is the only thing that
//! catches a joint bending the WRONG WAY, which no scalar plot shows.
//!
//! # What the four acts show
//!
//! ```text
//!   1  follow      a plan is tracked, with the servo's lag visible
//!   2  overshoot   a huge command becomes a slow ramp — the step limit
//!   3  abandoned   the commander dies; grace period, then HOLD forever
//!   4  overheat    a joint gets hot and outranks a perfectly fresh plan
//! ```

use arm::sim::SimJoint;
use arm::{ArmSpec, Guard, Joint, Plan, SensingJoint, Verdict};

/// Control period, seconds. Matches the rest of the project's 50 Hz.
const PERIOD_SECONDS: f64 = 0.02;
const PERIOD_MS: u64 = 20;
/// How long the arm may hear nothing before it stops accepting plans.
const SOURCE_TIMEOUT_MS: u64 = 500;

/// Upper arm (shoulder axis → elbow axis) and forearm (elbow axis →
/// gripper fingertip), metres, from the SO-ARM101 vendor outline drawing
/// (retrieved 2026-08-12): **111.67 mm** and **316.62 mm**.
///
/// The second number swallows the wrist. This spec is four DOF — pan,
/// lift, elbow, gripper — so `wrist_flex` and `wrist_roll` are not
/// modelled, and everything past the elbow is one rigid segment. That is
/// the honest drawing of what is being controlled, not a simplification
/// of it.
///
/// ⚠️ Vendor CAD, not a measured arm. The drawing itself says *"the
/// dimensions above are for reference only and may differ from the actual
/// size."* Good enough for pixels; not good enough for kinematics, which
/// is the other reason these stay out of `ArmSpec`.
const LINK_METRES: [f32; 2] = [0.111_67, 0.316_62];

/// Base plate to the shoulder-pitch axis: 74.80 + 44.20 mm. Drawn as a
/// post so the arm stands on a table rather than floating, which is what
/// makes a wrong-way bend read as wrong at a glance.
const SHOULDER_HEIGHT_METRES: f32 = 0.119;

/// World coordinates (+Y up, the way an arm stands) into Rerun's 2D view
/// (+Y **down**, the image convention it inherits from pixel rasters).
///
/// Without this the stand rises above the shoulder and the arm hangs off
/// the ceiling — which is what it did, and which no test would have
/// caught, because every length and angle was already correct.
///
/// Every point goes through this one function, so the flip cannot be
/// applied to the links and forgotten on the stand.
fn to_view(world: [f32; 2]) -> [f32; 2] {
    [world[0], -world[1]]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = ArmSpec::so101_four_dof();
    let mut guard = Guard::new(spec.clone(), PERIOD_SECONDS, SOURCE_TIMEOUT_MS);
    let mut joints: Vec<SimJoint> = (0..spec.joints()).map(|_| SimJoint::at(0.0)).collect();

    let rec = rerun::RecordingStreamBuilder::new("robotiq_arm")
        .with_blueprint(layout())
        .spawn()?;
    println!("watching the arm — scrub `wall_time` to these windows:\n");

    // The stand never moves, so it is logged once as static rather than
    // re-sent 750 times saying the same thing.
    rec.log_static(
        "arm/stand",
        &rerun::LineStrips2D::new([vec![
            to_view([0.0, 0.0]),
            to_view([0.0, SHOULDER_HEIGHT_METRES]),
        ]])
        .with_colors([rerun::Color::from_rgb(110, 110, 120)]),
    )?;

    let mut tick: u64 = 0;
    let mut last_reported: Option<(&'static str, &'static str)> = None;
    let act = |rec: &rerun::RecordingStream,
               last_reported: &mut Option<(&'static str, &'static str)>,
               guard: &mut Guard,
               joints: &mut Vec<SimJoint>,
               tick: &mut u64,
               name: &str,
               steps: u64,
               plan: &mut Plan,
               feeding: bool|
     -> Result<(), Box<dyn std::error::Error>> {
        rec.set_duration_secs("wall_time", *tick as f64 * PERIOD_SECONDS);
        rec.log("events", &rerun::TextLog::new(format!("── {name} ──")))?;
        let opened_at = *tick as f64 * PERIOD_SECONDS;
        let mut authorised_ticks = 0u64;
        for _ in 0..steps {
            let now_ms = *tick * PERIOD_MS;
            rec.set_duration_secs("wall_time", *tick as f64 * PERIOD_SECONDS);

            if feeding {
                guard.fed(now_ms);
            }

            // Temperatures come from the joints themselves, exactly as
            // they will off register 63 on a real STS3215.
            for (index, joint) in joints.iter_mut().enumerate() {
                let celsius = joint.temperature_celsius()?;
                guard.observe_temperature(index, celsius);
            }

            let measured: Vec<f64> = joints
                .iter_mut()
                .map(|j| j.measured())
                .collect::<Result<_, _>>()?;
            let verdict = guard.authorise(now_ms, &measured, plan.current());

            if let Verdict::Move { radians } = &verdict {
                authorised_ticks += 1;
                for (joint, angle) in joints.iter_mut().zip(radians) {
                    joint.command(*angle)?;
                }
            }
            draw(
                rec,
                &spec,
                plan.current(),
                &measured,
                &verdict,
                joints,
                last_reported,
            )?;

            joints.iter_mut().for_each(SimJoint::step);
            plan.advance();
            *tick += 1;
        }
        // Say in the terminal WHEN to look and WHAT the `authorised` panel
        // should show there, so the two can be COMPARED. A screen you have
        // to take on trust is the same failure as a log you have to take
        // on trust — and both numbers here are derived, never asserted,
        // so neither can drift away from what was actually logged.
        let closed_at = *tick as f64 * PERIOD_SECONDS;
        println!(
            "  {opened_at:5.1}–{closed_at:4.1} s  authorised {authorised_ticks:3}/{steps:<3}  {name}"
        );
        Ok(())
    };

    // 1 — a plan is followed, and the servo's lag is visible behind it.
    let mut plan = Plan::interpolate(&vec![0.0; 4], &[0.9, 0.7, -0.6, 0.5], 100).unwrap();
    act(
        &rec,
        &mut last_reported,
        &mut guard,
        &mut joints,
        &mut tick,
        "follow a plan",
        150,
        &mut plan,
        true,
    )?;

    // 2 — an absurd command. Travel clamps it, then the step limit turns
    // what is left into a ramp instead of a lurch.
    let mut plan = Plan::hold(vec![99.0; 4]);
    act(
        &rec,
        &mut last_reported,
        &mut guard,
        &mut joints,
        &mut tick,
        "step limit vs an absurd command",
        150,
        &mut plan,
        true,
    )?;

    // 3 — the commander dies mid-plan. THE ACT THAT MATTERS: the arm
    // finishes its grace period and then holds, rather than releasing.
    let here: Vec<f64> = joints.iter_mut().map(|j| j.measured().unwrap()).collect();
    let mut plan = Plan::interpolate(&here, &vec![0.0; 4], 400).unwrap();
    act(
        &rec,
        &mut last_reported,
        &mut guard,
        &mut joints,
        &mut tick,
        "commander dies — grace, then HOLD",
        300,
        &mut plan,
        false,
    )?;

    // 4 — a hot joint outranks a perfectly fresh plan.
    joints[1].heat_to(70.0);
    let mut plan = Plan::hold(vec![0.0; 4]);
    act(
        &rec,
        &mut last_reported,
        &mut guard,
        &mut joints,
        &mut tick,
        "shoulder overheats",
        150,
        &mut plan,
        true,
    )?;

    println!(
        "\n{:.1} s logged. In the viewer pick the newest source, `robotiq_arm`.",
        tick as f64 * PERIOD_SECONDS
    );
    Ok(())
}

/// Where the panels go.
///
/// # Why the example ships its own layout
///
/// Rerun keeps one blueprint per *application id*, and a viewer that is
/// already open on another recording keeps showing that one — so a run
/// with no blueprint arrives as a nameless entry in the Sources list,
/// under somebody else's panel arrangement, and looks like it failed.
/// Sending a layout makes the run present itself.
///
/// The split is by **unit**, not by joint: radians and degrees Celsius on
/// one axis would put a 70 °C reading three orders of magnitude above a
/// 0.9 rad one and flatten every angle into a line at zero.
fn layout() -> rerun::blueprint::Blueprint {
    use rerun::blueprint::{
        Blueprint, Horizontal, Spatial2DView, TextLogView, TimeSeriesView, Vertical,
    };

    Blueprint::new(
        Horizontal::new([
            Vertical::new([
                Spatial2DView::new("side view — blue moving, amber holding")
                    .with_origin("/arm")
                    .into(),
                // The failsafe as a square wave. This is the panel to
                // watch: it must sit at 0 for all of acts 3 and 4.
                TimeSeriesView::new("authorised — 1 move, 0 HOLD")
                    .with_origin("/guard")
                    .into(),
                // Quarantined on its own axis: act 2 asks for 99 rad.
                TimeSeriesView::new("what the plan ASKED (rad)")
                    .with_origin("/asked")
                    .into(),
            ])
            .into(),
            Vertical::new([
                TimeSeriesView::new("joint angles (rad) — commanded vs measured")
                    .with_origin("/angles")
                    .into(),
                TimeSeriesView::new("temperature (°C)")
                    .with_origin("/temperature")
                    .into(),
                TextLogView::new("why the arm decided that")
                    .with_origin("/events")
                    .into(),
            ])
            .into(),
        ])
        .with_column_shares([0.45, 0.55]),
    )
    // Nothing outside these five panels is worth a pane; an auto-added
    // view for every entity would bury the one that matters.
    .with_auto_views(false)
}

/// One frame: the plots that show behaviour over time, and the stick
/// figure that shows a joint bending the wrong way.
///
/// `last_reported` is the decision most recently announced to the event
/// log, so the same one is not announced 150 times running. It belongs to
/// the caller because it must survive **across** acts: a hold that runs
/// from act 3 into act 4 for a *different* reason is a transition, and
/// state reset per act would miss it.
fn draw(
    rec: &rerun::RecordingStream,
    spec: &ArmSpec,
    wanted: &[f64],
    measured: &[f64],
    verdict: &Verdict,
    joints: &mut [SimJoint],
    last_reported: &mut Option<(&'static str, &'static str)>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Grouped by UNIT, not by joint — radians under `angles/`, Celsius
    // under `temperature/`. Not cosmetic: a view is selected by entity
    // path prefix, and Rerun's `**` is a trailing wildcard only, so
    // `angles/<joint>/measured` can be picked out by prefix while
    // `joints/**/measured` matches nothing at all.
    //
    // What the plan ASKED for lives outside `angles/`, on its own panel,
    // because a plan may ask for 99 rad and an axis stretched to 99 draws
    // every real angle as a line at zero. Same reason radians and Celsius
    // are not on one axis: one outlier erases the signal.
    for (index, joint) in spec.joints.iter().enumerate() {
        rec.log(
            format!("asked/{}", joint.name),
            &rerun::Scalars::single(wanted[index]),
        )?;
        rec.log(
            format!("angles/{}/measured", joint.name),
            &rerun::Scalars::single(measured[index]),
        )?;
        // Only when the guard actually authorised something. A gap in
        // this series is not missing data — it is the arm being told
        // nothing, which is what a hold IS.
        if let Verdict::Move { radians } = verdict {
            rec.log(
                format!("angles/{}/commanded", joint.name),
                &rerun::Scalars::single(radians[index]),
            )?;
        }
        if let Some(celsius) = joints[index].temperature_celsius()? {
            rec.log(
                format!("temperature/{}", joint.name),
                &rerun::Scalars::single(celsius),
            )?;
        }
    }

    // A step plot of the decision itself. 1 while motion is authorised,
    // 0 while the arm is holding — the shape of the failsafe, visible at
    // a glance rather than read out of a log.
    let authorised = f64::from(u8::from(matches!(verdict, Verdict::Move { .. })));
    rec.log("guard/authorised", &rerun::Scalars::single(authorised))?;

    // Say WHY, but only when the DECISION changes — at 50 Hz an
    // unfiltered line per tick is 3,000 a minute saying nothing happened,
    // and the one transition that matters scrolls off the top.
    //
    // Keyed on the kind of verdict, never on the message: `HoldStale`
    // carries an age that climbs every tick, so deduplicating on the text
    // would filter nothing at all. The joint rides along so a second
    // joint overheating is a new line rather than a continuation.
    let key: (&'static str, &'static str) = match verdict {
        Verdict::Move { .. } => ("move", ""),
        Verdict::HoldUncommanded => ("uncommanded", ""),
        Verdict::HoldStale { .. } => ("stale", ""),
        Verdict::HoldClockFault { .. } => ("clock", ""),
        Verdict::HoldOverheated { joint, .. } => ("hot", joint),
        Verdict::Refused { .. } => ("refused", ""),
    };
    let changed = *last_reported != Some(key);
    *last_reported = Some(key);
    let reason = match verdict {
        Verdict::Move { .. } => None,
        Verdict::HoldUncommanded => Some("HOLD — nothing has commanded this arm yet".to_string()),
        Verdict::HoldStale { age_ms } => Some(format!("HOLD — source quiet for {age_ms} ms")),
        Verdict::HoldClockFault { age_ms } => {
            Some(format!("HOLD — clock fault, age jumped {age_ms} ms"))
        }
        Verdict::HoldOverheated { joint, celsius } => {
            Some(format!("HOLD — {joint} at {celsius:.0} °C"))
        }
        Verdict::Refused { wanted, joints } => {
            Some(format!("REFUSED — {wanted} angles for {joints} joints"))
        }
    };
    if let (true, Some(reason)) = (changed, reason) {
        rec.log("events", &rerun::TextLog::new(reason))?;
    }

    // Side view. shoulder_lift and elbow_flex are the joints that move
    // the arm in this plane; shoulder_pan rotates the plane itself and so
    // is honestly not drawable here — read it off its plot instead.
    //
    // Zero is arm-straight-out-to-the-right. The vendor drawing's pose is
    // lift = +90°, elbow = −90°, which is a fact about the drawing, not
    // about where the servos read zero.
    let lift = measured[spec.index_of("shoulder_lift").unwrap()] as f32;
    let elbow = measured[spec.index_of("elbow_flex").unwrap()] as f32;
    let shoulder = [0.0f32, SHOULDER_HEIGHT_METRES];
    let elbow_at = [
        shoulder[0] + LINK_METRES[0] * lift.cos(),
        shoulder[1] + LINK_METRES[0] * lift.sin(),
    ];
    let tip = [
        elbow_at[0] + LINK_METRES[1] * (lift + elbow).cos(),
        elbow_at[1] + LINK_METRES[1] * (lift + elbow).sin(),
    ];
    // Amber while holding, blue while moving — the same "belief" blue the
    // rest of the repo uses for a robot acting on what it believes.
    let colour = if authorised > 0.5 {
        belief_viz::belief()
    } else {
        rerun::Color::from_rgb(255, 176, 60)
    };
    let drawn = [to_view(shoulder), to_view(elbow_at), to_view(tip)];
    rec.log(
        "arm/links",
        &rerun::LineStrips2D::new([drawn.to_vec()]).with_colors([colour]),
    )?;
    rec.log(
        "arm/joints",
        &rerun::Points2D::new(drawn)
            .with_radii([0.008])
            .with_colors([colour]),
    )?;
    Ok(())
}
