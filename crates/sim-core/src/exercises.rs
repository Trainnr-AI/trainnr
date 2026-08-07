//! YOUR code goes here. Each exercise has tests already written below —
//! implement the function, then prove it works:
//!
//! ```text
//! cargo test -p sim-core exercises -- --ignored
//! ```
//!
//! Tests are `#[ignore]`d so the main suite stays green until you start.
//! Replace the `todo!()` with your implementation. `todo!()` is a macro that
//! compiles fine but panics if the code actually runs — Rust's way of letting
//! you sketch APIs before filling them in.

// Float math (`cos`, `sqrt`, `exp`, ...) lives in `std`. On bare metal it
// comes from libm through this trait, with identical method syntax.
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use crate::pose::wrap_angle;
use crate::world::Segment;

/// EXERCISE 1 — warm-up with ownership & floats.
///
/// Return the total length of a path given as a slice of (x, y) points.
/// A slice `&[(f64, f64)]` is a *borrowed view* into someone else's Vec or
/// array — you can read it, but you don't own it and can't modify it.
///
/// Hints: `points.windows(2)` yields overlapping pairs `[a, b]`;
/// `f64::hypot(dx, dy)` gives `sqrt(dx² + dy²)`. An iterator-chain solution
/// is ~3 lines; a for-loop solution is equally fine — write whichever reads
/// clearest to you.
pub fn path_length(points: &[(f64, f64)]) -> f64 {
    let mut total = 0.0;
    for pair in points.windows(2) {
        let (x1, y1) = pair[0];
        let (x2, y2) = pair[1];
        total += (x2 - x1).hypot(y2 - y1);
    }
    total
}

/// EXERCISE 2 — the angle-wrap bug, defeated by you personally.
///
/// Return the SHORTEST signed rotation (radians) that takes heading `from`
/// to heading `to`. The result must always be in (-π, π]:
/// from +170° to -170° the answer is +20° (in radians), NOT -340°.
///
/// This function becomes the heart of the heading PID controller in
/// milestone M3 — a robot with a naive version spins the long way around.
/// Hint: one line; `wrap_angle` (already imported) does the heavy lifting.
pub fn shortest_turn(from: f64, to: f64) -> f64 {
    wrap_angle(to - from)
}

/// EXERCISE 5 — the eye: does this ray hit this wall, and how far away?
///
/// A ray starts at (ox, oy) and points along `angle`. Return `Some(distance)`
/// if it hits the segment, `None` if it misses. Your first function that
/// PRODUCES an Option (you've consumed one in the PID).
///
/// Full derivation in docs/learning/math-06-raycasting.md. The recipe —
/// pure translation of the solved equations:
///
/// ```text
/// dx = angle.cos()      dy = angle.sin()        // ray direction
/// ex = seg.end.0 - seg.start.0  ey = seg.end.1 - seg.start.1  // segment direction
/// rx = seg.start.0 - ox     ry = seg.start.1 - oy       // origin → segment start
///
/// det = ex * dy - ey * dx
/// if det ≈ 0 (|det| < 1e-12): parallel → None
///
/// t = (ex * ry - ey * rx) / det   // distance along the RAY
/// u = (dx * ry - dy * rx) / det   // position along the SEGMENT (0..1)
///
/// hit iff t >= 0  AND  0 <= u <= 1  → Some(t), else None
/// ```
///
/// Rust notes: return values are written `Some(t)` / `None` (both are just
/// expressions — an `if/else` chain returning them works). `(0.0..=1.0)
/// .contains(&u)` is a tidy way to write the u check, or use two
/// comparisons — your choice.
pub fn ray_segment_hit(ox: f64, oy: f64, angle: f64, seg: &Segment) -> Option<f64> {
    let dx = angle.cos();
    let dy = angle.sin();
    let ex = seg.end.0 - seg.start.0;
    let ey = seg.end.1 - seg.start.1;
    let rx = seg.start.0 - ox;
    let ry = seg.start.1 - oy;

    let det = ex * dy - ey * dx;
    if det.abs() < 1e-12 {
        return None;
    }
    let t = (ex * ry - ey * rx) / det;
    let u = (dx * ry - dy * rx) / det;
    if t >= 0.0 && (0.0..=1.0).contains(&u) {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn path_length_of_square() {
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        assert!((path_length(&square) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn path_length_edge_cases() {
        assert_eq!(path_length(&[]), 0.0);
        assert_eq!(path_length(&[(3.0, 4.0)]), 0.0);
        assert!((path_length(&[(0.0, 0.0), (3.0, 4.0)]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn shortest_turn_simple() {
        assert!((shortest_turn(0.0, PI / 2.0) - PI / 2.0).abs() < 1e-12);
        assert!((shortest_turn(PI / 2.0, 0.0) + PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn shortest_turn_crosses_the_seam() {
        let from = (170.0_f64).to_radians();
        let to = (-170.0_f64).to_radians();
        // +20°, not -340°.
        assert!((shortest_turn(from, to) - (20.0_f64).to_radians()).abs() < 1e-9);
        // And the reverse: -20°.
        assert!((shortest_turn(to, from) + (20.0_f64).to_radians()).abs() < 1e-9);
    }

    fn vertical_wall_at_x2() -> Segment {
        Segment {
            start: (2.0, -1.0),
            end: (2.0, 1.0),
        }
    }

    #[test]
    fn ray_hits_wall_straight_ahead() {
        // From origin, looking along +x, wall at x = 2: hit at distance 2.
        let d = ray_segment_hit(0.0, 0.0, 0.0, &vertical_wall_at_x2());
        assert!((d.unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ray_facing_away_misses() {
        // Same wall, but looking along -x: behind us → None.
        assert_eq!(ray_segment_hit(0.0, 0.0, PI, &vertical_wall_at_x2()), None);
    }

    #[test]
    fn parallel_ray_misses() {
        // Looking along +y, parallel to the vertical wall → None.
        assert_eq!(
            ray_segment_hit(0.0, 0.0, PI / 2.0, &vertical_wall_at_x2()),
            None
        );
    }

    #[test]
    fn ray_passing_beyond_segment_end_misses() {
        // Aim 45° up: would cross x=2 at y=2, but the wall ends at y=1.
        assert_eq!(
            ray_segment_hit(0.0, 0.0, PI / 4.0, &vertical_wall_at_x2()),
            None
        );
    }

    #[test]
    fn diagonal_hit_at_sqrt2() {
        // Wall from (1,0) to (1,2); aim 45°: hit at (1,1), distance √2.
        let wall = Segment {
            start: (1.0, 0.0),
            end: (1.0, 2.0),
        };
        let d = ray_segment_hit(0.0, 0.0, PI / 4.0, &wall);
        assert!((d.unwrap() - 2.0_f64.sqrt()).abs() < 1e-12);
    }
}
