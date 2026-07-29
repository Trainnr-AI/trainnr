//! The world: a flat 2D floor with walls, described as line segments.
//!
//! Segments are all we need for a long time: rooms, corridors, and box
//! obstacles are just segments, and later the "lidar" will be rays
//! intersected against them.

// Float math (`cos`, `sqrt`, `exp`, ...) lives in `std`. On bare metal it
// comes from libm through this trait, with identical method syntax.
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// A wall piece from point `a` to point `b` (world frame, meters).
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub a: (f64, f64),
    pub b: (f64, f64),
}

impl Segment {
    /// Shortest distance from point (px, py) to any point on this segment.
    /// Project the point onto the segment's line, clamp to the ends, then
    /// measure. Used for collision: "is the robot's body touching a wall?"
    pub fn distance_to_point(&self, px: f64, py: f64) -> f64 {
        let (ax, ay) = self.a;
        let (bx, by) = self.b;
        let ex = bx - ax;
        let ey = by - ay;
        let len2 = ex * ex + ey * ey;
        let t = if len2 < 1e-12 {
            0.0 // degenerate zero-length segment: just measure to point a
        } else {
            (((px - ax) * ex + (py - ay) * ey) / len2).clamp(0.0, 1.0)
        };
        let cx = ax + t * ex;
        let cy = ay + t * ey;
        (px - cx).hypot(py - cy)
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Default)]
pub struct World {
    pub walls: Vec<Segment>,
}

#[cfg(feature = "std")]
impl World {
    /// An empty rectangular room with corners (0,0) and (width, height).
    pub fn room(width: f64, height: f64) -> Self {
        let mut w = World::default();
        w.add_box(0.0, 0.0, width, height);
        w
    }

    /// Add an axis-aligned box outline (also used for obstacles).
    pub fn add_box(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        for i in 0..4 {
            self.walls.push(Segment {
                a: corners[i],
                b: corners[(i + 1) % 4],
            });
        }
    }

    /// Would a robot body (circle of `radius` at x, y) overlap any wall?
    /// Walls are physical: the simulator must refuse motion into them —
    /// cameras are for *avoiding* walls, collision is for *enforcing* them.
    pub fn collides(&self, x: f64, y: f64, radius: f64) -> bool {
        self.walls
            .iter()
            .any(|s| s.distance_to_point(x, y) < radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn room_has_four_walls() {
        assert_eq!(World::room(8.0, 6.0).walls.len(), 4);
    }

    #[test]
    fn obstacle_adds_four_more() {
        let mut w = World::room(8.0, 6.0);
        w.add_box(2.0, 2.0, 3.0, 3.0);
        assert_eq!(w.walls.len(), 8);
    }

    #[test]
    fn distance_to_segment_cases() {
        let s = Segment {
            a: (0.0, 0.0),
            b: (2.0, 0.0),
        };
        // Straight above the middle: perpendicular distance.
        assert!((s.distance_to_point(1.0, 0.5) - 0.5).abs() < 1e-12);
        // Beyond the end: distance to the endpoint, not the infinite line.
        assert!((s.distance_to_point(3.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((s.distance_to_point(-1.0, 1.0) - 2.0_f64.sqrt()).abs() < 1e-12);
        // On the segment: zero.
        assert!(s.distance_to_point(0.7, 0.0).abs() < 1e-12);
    }

    #[test]
    fn collision_against_room_walls() {
        let w = World::room(8.0, 6.0);
        assert!(!w.collides(4.0, 3.0, 0.09)); // mid-room: clear
        assert!(w.collides(0.05, 3.0, 0.09)); // hugging the west wall
        assert!(w.collides(4.0, 5.95, 0.09)); // hugging the north wall
        assert!(!w.collides(0.2, 3.0, 0.09)); // 20 cm off the wall: clear
    }
}
