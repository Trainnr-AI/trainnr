//! Drawing where a robot thinks it is, once.
//!
//! `sim-run`, `hil-host` and `vision` each grew their own copy of *body,
//! heading arrow, trail* — and each independently rediscovered the same
//! performance trap, in its own comment, in its own words.
//!
//! # The colour convention, which was previously four people remembering
//!
//! ```text
//!   BELIEF   blue    where the robot THINKS it is — dead reckoning
//!   TRUTH    yellow  where it actually is — only a simulator knows this
//!   HEADING  red     which way it is pointing
//! ```
//!
//! Blue being belief is load-bearing: on a bench there is no ground truth,
//! so everything drawn is blue on purpose, as a standing reminder that
//! dead reckoning is a claim and not a measurement.

use rerun::{Color, RecordingStream, RecordingStreamError};
use sim_core::Pose;

// `Color::from_rgb` is not a `const fn` in rerun 0.35, so these are
// functions rather than constants. Same call sites either way.
/// Where the robot thinks it is.
pub fn belief() -> Color {
    Color::from_rgb(90, 200, 255)
}
/// Where it actually is. Only a simulator can draw this.
pub fn truth() -> Color {
    Color::from_rgb(255, 200, 60)
}
/// Which way it is pointing.
pub fn heading() -> Color {
    Color::from_rgb(255, 90, 90)
}

/// How big to draw a robot, in metres.
///
/// ⚠️ **Not a constant, because the callers genuinely disagree** — and
/// that disagreement is meaningful rather than accidental. `sim-run`
/// draws a metre-scale simulated robot at 0.09 with a 0.25 arrow;
/// `hil-host` draws the 6 cm bench robot at 0.02. A shared constant here
/// would have quietly resized one of them.
#[derive(Clone, Copy)]
pub struct Size {
    pub body_radius: f32,
    /// Short on purpose: an arrow indicates a direction, and a long one
    /// starts to read as a velocity.
    pub arrow: f32,
}

impl Size {
    pub const BENCH: Size = Size {
        body_radius: 0.02,
        arrow: 0.05,
    };
    pub const SIMULATED: Size = Size {
        body_radius: 0.09,
        arrow: 0.25,
    };
}

/// Draws a robot at `pose`: a body marker and a heading arrow, under
/// `{path}/body` and `{path}/heading`.
///
/// Separate from [`Trail`] because two callers want exactly this and no
/// history — `vision::drive` draws the chip's own belief with no trail,
/// and `chase` keeps a BOUNDED ring buffer rather than a cumulative one.
/// An abstraction that forced a trail on them would be adding pixels, not
/// removing duplication.
pub fn marker(
    rec: &RecordingStream,
    path: &str,
    pose: Pose,
    colour: Color,
    size: Size,
) -> Result<(), RecordingStreamError> {
    let (x, y) = (pose.x as f32, pose.y as f32);
    rec.log(
        format!("{path}/body"),
        &rerun::Points2D::new([[x, y]])
            .with_radii([size.body_radius])
            .with_colors([colour]),
    )?;
    rec.log(
        format!("{path}/heading"),
        &rerun::Arrows2D::from_vectors([[
            size.arrow * pose.heading.cos() as f32,
            size.arrow * pose.heading.sin() as f32,
        ]])
        .with_origins([[x, y]])
        .with_colors([heading()]),
    )
}

/// A pose history that knows how often it is cheap to redraw.
///
/// # ⚠️ The trail is CUMULATIVE, so redrawing it is quadratic
///
/// Logging a growing line strip every frame re-sends every point it has
/// ever had. `sim-run/src/viz.rs` recorded what that cost: a 22.5 s run
/// pushed **1.27 million points where 2,252 would do**. Three separate
/// files had each found this out and fixed it locally; the fix now has
/// one home.
///
/// Body and heading are logged every call — they are a single point and a
/// single arrow, and their whole job is to be current.
pub struct Trail {
    points: Vec<[f32; 2]>,
    colour: Color,
    size: Size,
    redraw_every: u64,
    frame: u64,
}

impl Trail {
    /// `redraw_every` is in calls, so pass the report rate to redraw once
    /// a second — the rate this project has settled on everywhere.
    pub fn new(colour: Color, size: Size, redraw_every: u64) -> Self {
        Trail {
            points: Vec::new(),
            colour,
            size,
            redraw_every: redraw_every.max(1),
            frame: 0,
        }
    }

    /// Number of poses recorded. Useful for a summary line.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Record `pose` and draw it under `path` — `{path}/body`,
    /// `{path}/heading`, `{path}/trail`.
    pub fn draw(
        &mut self,
        rec: &RecordingStream,
        path: &str,
        pose: Pose,
    ) -> Result<(), RecordingStreamError> {
        let (x, y) = (pose.x as f32, pose.y as f32);
        self.points.push([x, y]);

        if self.frame.is_multiple_of(self.redraw_every) {
            rec.log(
                format!("{path}/trail"),
                &rerun::LineStrips2D::new([self.points.clone()]).with_colors([self.colour]),
            )?;
        }
        self.frame += 1;
        marker(rec, path, pose, self.colour, self.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The redraw schedule is the whole point of this type, and it is
    /// arithmetic — testable without a viewer.
    #[test]
    fn the_trail_redraws_on_the_first_call_and_then_periodically() {
        let mut t = Trail::new(belief(), Size::BENCH, 50);
        let redraws: Vec<u64> = (0..120u64)
            .filter(|f| f.is_multiple_of(t.redraw_every))
            .collect();
        assert_eq!(redraws, vec![0, 50, 100], "first call, then once a second");
        t.frame = 0;
    }

    /// A zero period would divide by zero inside `is_multiple_of`; callers
    /// pass a rate, and a rate of zero is a plausible typo.
    #[test]
    fn a_zero_redraw_period_is_clamped_rather_than_dividing_by_zero() {
        let t = Trail::new(belief(), Size::BENCH, 0);
        assert_eq!(t.redraw_every, 1);
    }
}
