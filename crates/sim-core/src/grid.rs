//! Occupancy grid — the robot's MEMORY of the world.
//!
//! The reactive avoider fails in the U-trap because it remembers nothing:
//! every glance is its first. The fix is a map built from what the camera
//! has seen: the floor divided into small cells, each marked Unknown (never
//! looked there), Free (a ray passed through it), or Occupied (a ray ended
//! there — something solid).
//!
//! Every camera ray teaches us twice: the space the ray CROSSED is free
//! (light doesn't pass through walls), and the spot where it STOPPED is
//! occupied. Exercise 6 implements exactly that.
//!
//! (Real systems store probabilities per cell and update them with log-odds
//! — cells accumulate evidence instead of flipping hard. Our three-state
//! version is that idea with the training wheels on.)

use crate::pose::Point;

/// What the robot believes about one patch of floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    Unknown,
    Free,
    Occupied,
}

pub struct OccupancyGrid {
    /// Cell size in meters (0.1 = 10 cm patches).
    pub resolution: f64,
    /// Grid dimensions in cells.
    pub width: usize,
    pub height: usize,
    // Row-major: index = cy * width + cx.
    cells: Vec<Cell>,
}

impl OccupancyGrid {
    /// Cover a world_w × world_h world (meters) at the given resolution.
    pub fn new(world_w: f64, world_h: f64, resolution: f64) -> Self {
        let width = (world_w / resolution).ceil() as usize;
        let height = (world_h / resolution).ceil() as usize;
        OccupancyGrid {
            resolution,
            width,
            height,
            cells: vec![Cell::Unknown; width * height],
        }
    }

    pub fn get(&self, cx: usize, cy: usize) -> Cell {
        self.cells[cy * self.width + cx]
    }

    pub fn set(&mut self, cx: usize, cy: usize, c: Cell) {
        self.cells[cy * self.width + cx] = c;
    }

    /// EXERCISE 6a — which cell is the world point (x, y) in?
    ///
    /// `Some((cx, cy))` if inside the grid, `None` if outside. Recipe:
    /// 1. If `x < 0.0 || y < 0.0`, return None (off the map's corner).
    /// 2. `cx = (x / resolution).floor() as usize` — same for cy with y.
    ///    (floor: a point at 0.37 m with 0.1 m cells is in cell 3.)
    /// 3. If `cx >= width || cy >= height`, return None (off the far edge).
    /// 4. Otherwise `Some((cx, cy))`.
    pub fn world_to_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        if x < 0.0 || y < 0.0 {
            return None;
        }
        let cx = (x / self.resolution).floor() as usize;
        let cy = (y / self.resolution).floor() as usize;
        if cx >= self.width || cy >= self.height {
            return None;
        }
        Some((cx, cy))
    }

    /// Center of a cell, in world coordinates (for the planner).
    pub fn cell_to_world(&self, cx: usize, cy: usize) -> Point {
        Point::new(
            (cx as f64 + 0.5) * self.resolution,
            (cy as f64 + 0.5) * self.resolution,
        )
    }

    /// EXERCISE 6b — burn one camera ray into memory.
    ///
    /// A ray from (ox, oy) along `angle` measured distance `dist` (capped
    /// at `max_range`). Two lessons per ray:
    ///
    /// 1. WALK the ray from s = 0 up to (but not including) `dist` in steps
    ///    of `self.resolution / 2.0` (half-cell steps so no cell is jumped
    ///    over). At each s, the point is
    ///    `(ox + s * angle.cos(), oy + s * angle.sin())` — mark its cell
    ///    `Cell::Free`, EXCEPT never overwrite an `Occupied` cell (a wall
    ///    seen once stays seen; a grazing ray must not erase it).
    /// 2. If `dist < max_range - 1e-9` the ray actually HIT something
    ///    (didn't just run out of range): mark the cell at the endpoint
    ///    (the point at s = dist) `Cell::Occupied`.
    ///
    /// Rust you'll want: a `while s < dist` loop with `s += step;`, and
    /// `if let Some((cx, cy)) = self.world_to_cell(px, py) { ... }` —
    /// "if this Option is Some, unpack it and run the block; if None, skip"
    /// (a one-armed match; points outside the grid are simply ignored).
    pub fn mark_ray(&mut self, ox: f64, oy: f64, angle: f64, dist: f64, max_range: f64) {
        let dx = angle.cos();
        let dy = angle.sin();
        let step = self.resolution / 2.0;
        let mut s = 0.0;
        while s < dist {
            if let Some((cx, cy)) = self.world_to_cell(ox + s * dx, oy + s * dy) {
                if self.get(cx, cy) != Cell::Occupied {
                    self.set(cx, cy, Cell::Free);
                }
            }
            s += step;
        }
        if dist < max_range - 1e-9 {
            if let Some((cx, cy)) = self.world_to_cell(ox + dist * dx, oy + dist * dy) {
                self.set(cx, cy, Cell::Occupied);
            }
        }
    }

    /// Is any cell within `radius` cells of (cx, cy) Occupied?
    /// Planning uses this to keep the robot's body-width away from walls
    /// ("obstacle inflation") — the path is for the robot's CENTER.
    pub fn blocked_near(&self, cx: usize, cy: usize, radius: usize) -> bool {
        let x0 = cx.saturating_sub(radius);
        let y0 = cy.saturating_sub(radius);
        let x1 = (cx + radius).min(self.width - 1);
        let y1 = (cy + radius).min(self.height - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if self.get(x, y) == Cell::Occupied {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> OccupancyGrid {
        OccupancyGrid::new(2.0, 2.0, 0.1) // 20 x 20 cells
    }

    #[test]
    fn new_grid_is_unknown() {
        let g = grid();
        assert_eq!(g.width, 20);
        assert_eq!(g.height, 20);
        assert_eq!(g.get(0, 0), Cell::Unknown);
        assert_eq!(g.get(19, 19), Cell::Unknown);
    }

    #[test]
    fn world_to_cell_basics() {
        let g = grid();
        assert_eq!(g.world_to_cell(0.05, 0.05), Some((0, 0)));
        assert_eq!(g.world_to_cell(0.95, 0.55), Some((9, 5)));
        assert_eq!(g.world_to_cell(1.99, 1.99), Some((19, 19)));
        assert_eq!(g.world_to_cell(-0.1, 0.5), None); // off the near corner
        assert_eq!(g.world_to_cell(2.5, 0.5), None); // off the far edge
    }

    #[test]
    fn ray_marks_free_along_and_occupied_at_end() {
        let mut g = grid();
        // Ray along +x from (0.05, 0.05), hit at 1.0 m (range 2.0).
        g.mark_ray(0.05, 0.05, 0.0, 1.0, 2.0);
        assert_eq!(g.get(3, 0), Cell::Free); // crossed
        assert_eq!(g.get(9, 0), Cell::Free); // crossed
        assert_eq!(g.get(10, 0), Cell::Occupied); // endpoint: 1.05 m → cell 10
        assert_eq!(g.get(15, 0), Cell::Unknown); // beyond the hit: no idea
        assert_eq!(g.get(5, 5), Cell::Unknown); // off-ray: no idea
    }

    #[test]
    fn max_range_ray_marks_no_obstacle() {
        let mut g = grid();
        // Saw nothing within range: free corridor, NO occupied endpoint.
        g.mark_ray(0.05, 0.05, 0.0, 2.0, 2.0);
        assert_eq!(g.get(9, 0), Cell::Free);
        assert!(!(0..20).any(|cx| g.get(cx, 0) == Cell::Occupied));
    }

    #[test]
    fn free_never_overwrites_occupied() {
        let mut g = grid();
        g.set(5, 0, Cell::Occupied); // a wall we've already seen
                                     // A later ray whose walk passes that cell (hit far beyond it):
        g.mark_ray(0.05, 0.05, 0.0, 1.5, 2.0);
        assert_eq!(g.get(5, 0), Cell::Occupied); // still there
    }

    #[test]
    fn blocked_near_sees_inflation_radius() {
        let mut g = grid();
        g.set(10, 10, Cell::Occupied);
        assert!(g.blocked_near(10, 10, 0));
        assert!(g.blocked_near(8, 10, 2)); // 2 cells away, radius 2
        assert!(!g.blocked_near(7, 10, 2)); // 3 cells away, radius 2
    }
}
