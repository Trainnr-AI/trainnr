//! One definition of how an arm is drawn.
//!
//! Used by `arm`'s simulated viewer and by the hardware viewer in
//! `hil-host`, so a joint angle from a simulator and a joint angle from a
//! motor produce the *same picture*. Two drawing routines would diverge —
//! and the first thing they would disagree about is which way a joint
//! bends, which is the one thing a picture is here to catch.
//!
//! # ⚠️ The geometry is SO-ARM101's, whatever is driving it
//!
//! Link lengths come from the vendor outline drawing (2026-08-12):
//! **111.67 mm** shoulder→elbow, **316.62 mm** elbow→fingertip,
//! **119.00 mm** base→shoulder. The drawing itself says its dimensions
//! *"are for reference only and may differ from the actual size."*
//!
//! So when bench gearmotors drive this, the figure is **not a picture of
//! the bench** — there are no links on those shafts. It is the arm those
//! angles *would* command, which is exactly what makes a wrong-way bend
//! visible before any aluminium exists.

use rerun::{Color, LineStrips2D, Points2D, RecordingStream};

/// Upper arm and forearm, metres. The forearm swallows the wrist: a
/// 4-DOF spec does not model `wrist_flex`/`wrist_roll`, so everything
/// past the elbow is one rigid segment — an honest drawing of what is
/// being controlled rather than a simplification of it.
pub const LINK_METRES: [f32; 2] = [0.111_67, 0.316_62];

/// Base plate to the shoulder-pitch axis: 74.80 + 44.20 mm.
pub const SHOULDER_HEIGHT_METRES: f32 = 0.119;

/// Amber. Used when the arm is holding rather than moving — the same
/// distinction `Verdict` draws, in a colour.
pub fn holding() -> Color {
    Color::from_rgb(255, 176, 60)
}

/// World coordinates (+Y up, the way an arm stands) into Rerun's 2D view
/// (+Y **down**, the image convention it inherits from pixel rasters).
///
/// Without this the stand rises above the shoulder and the arm hangs off
/// the ceiling — which it did, and which no test caught, because every
/// length and angle was already correct.
fn to_view(world: [f32; 2]) -> [f32; 2] {
    [world[0], -world[1]]
}

/// The post the arm stands on. Static — it never moves, so it is logged
/// once rather than re-sent every tick.
pub fn draw_stand(rec: &RecordingStream, entity: &str) -> Result<(), rerun::RecordingStreamError> {
    rec.log_static(
        format!("{entity}/stand"),
        &LineStrips2D::new([vec![
            to_view([0.0, 0.0]),
            to_view([0.0, SHOULDER_HEIGHT_METRES]),
        ]])
        .with_colors([Color::from_rgb(110, 110, 120)]),
    )
}

/// A side-on stick figure from two joint angles, in radians.
///
/// `lift` and `elbow` are the joints that move the arm in this plane; a
/// base rotation turns the plane itself and is honestly not drawable
/// here. Zero is arm-straight-out; the vendor drawing's pose is
/// `lift = +90°, elbow = −90°`, which is a fact about the drawing rather
/// than about where any servo reads zero.
pub fn draw(
    rec: &RecordingStream,
    entity: &str,
    lift: f32,
    elbow: f32,
    colour: Color,
) -> Result<(), rerun::RecordingStreamError> {
    let shoulder = [0.0f32, SHOULDER_HEIGHT_METRES];
    let elbow_at = [
        shoulder[0] + LINK_METRES[0] * lift.cos(),
        shoulder[1] + LINK_METRES[0] * lift.sin(),
    ];
    let tip = [
        elbow_at[0] + LINK_METRES[1] * (lift + elbow).cos(),
        elbow_at[1] + LINK_METRES[1] * (lift + elbow).sin(),
    ];
    // Every point through `to_view`, so the flip cannot be applied to the
    // links and forgotten on the joints.
    let drawn = [to_view(shoulder), to_view(elbow_at), to_view(tip)];
    rec.log(
        format!("{entity}/links"),
        &LineStrips2D::new([drawn.to_vec()]).with_colors([colour]),
    )?;
    rec.log(
        format!("{entity}/joints"),
        &Points2D::new(drawn)
            .with_radii([0.008])
            .with_colors([colour]),
    )
}
