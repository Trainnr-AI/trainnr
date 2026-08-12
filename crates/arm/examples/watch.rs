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

// Binaries and tests may panic — see the workspace lint block: "these are
// `warn` rather than `deny` because the BINARIES legitimately panic at
// startup". The LIBRARY denies both; an example that cannot find a joint
// by name should stop loudly rather than carry on with a wrong index.
#![allow(clippy::unwrap_used)]

use arm::sim::SimJoint;
use arm::{ArmSpec, Guard, Joint, Plan, SensingJoint, Verdict};

/// Control period, seconds. Matches the rest of the project's 50 Hz.
const PERIOD_SECONDS: f64 = 0.02;
const PERIOD_MS: u64 = 20;
/// How long the arm may hear nothing before it stops accepting plans.
const SOURCE_TIMEOUT_MS: u64 = 500;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bench = Bench::new(ArmSpec::so101_four_dof());

    let rec = rerun::RecordingStreamBuilder::new("robotiq_arm")
        .with_blueprint(layout())
        .spawn()?;
    println!("watching the arm — scrub `wall_time` to these windows:\n");

    // The stand never moves, so it is logged once as static rather than
    // re-sent 750 times saying the same thing.
    belief_viz::arm::draw_stand(&rec, "arm")?;

    // 1 — a plan is followed, and the servo's lag is visible behind it.
    let mut plan = Plan::interpolate(&vec![0.0; 4], &[0.9, 0.7, -0.6, 0.5], 100).unwrap();
    bench.act(&rec, "follow a plan", 150, &mut plan, Commander::Alive)?;

    // 2 — an absurd command. Travel clamps it, then the step limit turns
    // what is left into a ramp instead of a lurch.
    let mut plan = Plan::hold(vec![99.0; 4]);
    bench.act(
        &rec,
        "step limit vs an absurd command",
        150,
        &mut plan,
        Commander::Alive,
    )?;

    // 3 — the commander dies mid-plan. THE ACT THAT MATTERS: the arm
    // finishes its grace period and then holds, rather than releasing.
    let mut plan = Plan::interpolate(&bench.here()?, &vec![0.0; 4], 400).unwrap();
    bench.act(
        &rec,
        "commander dies — grace, then HOLD",
        300,
        &mut plan,
        Commander::Dead,
    )?;

    // 4 — a hot joint outranks a perfectly fresh plan.
    bench.joints[1].heat_to(70.0);
    let mut plan = Plan::hold(vec![0.0; 4]);
    bench.act(&rec, "shoulder overheats", 150, &mut plan, Commander::Alive)?;

    println!(
        "\n{:.1} s logged. In the viewer pick the newest source, `robotiq_arm`.",
        bench.elapsed_seconds()
    );
    Ok(())
}

/// Whether anything is still feeding the guard during an act.
///
/// A named pair rather than a `bool` argument: `act(…, 150, &mut plan,
/// false)` at the call site says nothing about *what* is false, and this
/// is the flag that decides whether the arm keeps moving or holds.
#[derive(Clone, Copy, PartialEq)]
enum Commander {
    Alive,
    Dead,
}

/// Everything that persists from one act to the next.
///
/// This began as a closure with nine parameters, six of them `&mut` to
/// state the caller was only holding on its behalf. The state belongs
/// here; an act is then a method with five arguments, all of which are
/// about the act rather than about plumbing.
struct Bench {
    spec: ArmSpec,
    guard: Guard,
    joints: Vec<SimJoint>,
    tick: u64,
    /// The decision most recently announced to the event log, so the same
    /// one is not announced 150 times running. It lives at bench scope
    /// because it must survive **across** acts: a hold that runs from act
    /// 3 into act 4 for a *different* reason is a transition worth a line,
    /// and per-act state would miss it.
    last_reported: Option<(&'static str, &'static str)>,
}

impl Bench {
    fn new(spec: ArmSpec) -> Self {
        Bench {
            guard: Guard::new(spec.clone(), PERIOD_SECONDS, SOURCE_TIMEOUT_MS),
            joints: (0..spec.joints()).map(|_| SimJoint::at(0.0)).collect(),
            spec,
            tick: 0,
            last_reported: None,
        }
    }

    fn elapsed_seconds(&self) -> f64 {
        self.tick as f64 * PERIOD_SECONDS
    }

    /// Where the joints are now — the starting point for a plan that has
    /// to begin from wherever the last act left the arm.
    fn here(&mut self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(self
            .joints
            .iter_mut()
            .map(|joint| joint.measured())
            .collect::<Result<_, _>>()?)
    }

    /// Run one act: `steps` control periods against `plan`.
    fn act(
        &mut self,
        rec: &rerun::RecordingStream,
        name: &str,
        steps: u64,
        plan: &mut Plan,
        commander: Commander,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let opened_at = self.elapsed_seconds();
        rec.set_duration_secs("wall_time", opened_at);
        rec.log("events", &rerun::TextLog::new(format!("── {name} ──")))?;

        let mut authorised_ticks = 0u64;
        for _ in 0..steps {
            let now_ms = self.tick * PERIOD_MS;
            rec.set_duration_secs("wall_time", self.elapsed_seconds());

            if commander == Commander::Alive {
                self.guard.fed(now_ms);
            }

            // Temperatures come from the joints themselves, exactly as
            // they will off register 63 on a real STS3215.
            for (index, joint) in self.joints.iter_mut().enumerate() {
                let celsius = joint.temperature_celsius()?;
                self.guard.observe_temperature(index, celsius);
            }

            let measured = self.here()?;
            let verdict = self.guard.authorise(now_ms, &measured, plan.current());

            if let Verdict::Move { radians } = &verdict {
                authorised_ticks += 1;
                for (joint, angle) in self.joints.iter_mut().zip(radians) {
                    joint.command(*angle)?;
                }
            }
            self.draw(rec, plan.current(), &measured, &verdict)?;

            self.joints.iter_mut().for_each(SimJoint::step);
            plan.advance();
            self.tick += 1;
        }

        // Say in the terminal WHEN to look and WHAT the `authorised` panel
        // should show there, so the two can be COMPARED. A screen you have
        // to take on trust is the same failure as a log you have to take
        // on trust — and both numbers here are derived, never asserted,
        // so neither can drift away from what was actually logged.
        println!(
            "  {opened_at:5.1}–{:4.1} s  authorised {authorised_ticks:3}/{steps:<3}  {name}",
            self.elapsed_seconds()
        );
        Ok(())
    }
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

impl Bench {
    /// One frame: the plots that show behaviour over time, and the stick
    /// figure that shows a joint bending the wrong way.
    fn draw(
        &mut self,
        rec: &rerun::RecordingStream,
        wanted: &[f64],
        measured: &[f64],
        verdict: &Verdict,
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
        for index in 0..self.spec.joints() {
            // Copied out rather than borrowed, so reading the joint's own
            // temperature below is not a second borrow of `self`.
            let name = self.spec.joints[index].name;
            rec.log(
                format!("asked/{name}"),
                &rerun::Scalars::single(wanted[index]),
            )?;
            rec.log(
                format!("angles/{name}/measured"),
                &rerun::Scalars::single(measured[index]),
            )?;
            // Only when the guard actually authorised something. A gap in
            // this series is not missing data — it is the arm being told
            // nothing, which is what a hold IS.
            if let Verdict::Move { radians } = verdict {
                rec.log(
                    format!("angles/{name}/commanded"),
                    &rerun::Scalars::single(radians[index]),
                )?;
            }
            if let Some(celsius) = self.joints[index].temperature_celsius()? {
                rec.log(
                    format!("temperature/{name}"),
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
            Verdict::HoldUnmeasurable { joint } => ("unmeasurable", joint),
            Verdict::Refused { .. } => ("refused", ""),
        };
        let changed = self.last_reported != Some(key);
        self.last_reported = Some(key);
        let reason = match verdict {
            Verdict::Move { .. } => None,
            Verdict::HoldUncommanded => {
                Some("HOLD — nothing has commanded this arm yet".to_string())
            }
            Verdict::HoldStale { age_ms } => Some(format!("HOLD — source quiet for {age_ms} ms")),
            Verdict::HoldClockFault { age_ms } => {
                Some(format!("HOLD — clock fault, age jumped {age_ms} ms"))
            }
            Verdict::HoldOverheated { joint, celsius } => {
                Some(format!("HOLD — {joint} at {celsius:.0} °C"))
            }
            Verdict::HoldUnmeasurable { joint } => {
                Some(format!("HOLD — {joint} cannot report its temperature"))
            }
            Verdict::Refused { wanted, joints } => {
                Some(format!("REFUSED — {wanted} angles for {joints} joints"))
            }
        };
        if let (true, Some(reason)) = (changed, reason) {
            rec.log("events", &rerun::TextLog::new(reason))?;
        }

        // Side view — drawn by `belief_viz::arm`, the same routine the
        // hardware viewer uses, so a simulated angle and a measured one
        // make the SAME picture. Two copies would disagree first about
        // which way a joint bends, which is the one thing the picture is
        // here to catch.
        let lift = measured[self.spec.index_of("shoulder_lift").unwrap()] as f32;
        let elbow = measured[self.spec.index_of("elbow_flex").unwrap()] as f32;
        let colour = if authorised > 0.5 {
            belief_viz::belief()
        } else {
            belief_viz::arm::holding()
        };
        belief_viz::arm::draw(rec, "arm", lift, elbow, colour)?;
        Ok(())
    }
}
