//! A* path planning over the occupancy grid.
//!
//! Where the reactive avoider asks "what do I see right now?", the planner
//! asks "given everything I REMEMBER, what's the shortest safe route?" —
//! and that difference is exactly what escapes the U-trap.
//!
//! A* in one breath, for a software engineer: Dijkstra's shortest-path,
//! plus a per-node "how far to the goal as the crow flies" bonus that makes
//! the search rush toward the goal instead of flooding in all directions.
//! The bonus (heuristic) never overestimates, so the result is still the
//! true shortest path. Full intuition: docs/learning/math-07-maps-and-astar.md.
//!
//! Rust showcase (worth reading slowly — new types in here):
//! `BinaryHeap` (priority queue), `Reverse` (flips max-heap to min-heap),
//! and costs kept as integer millimeters so they're orderable (f64 isn't
//! `Ord` in Rust — NaN ruins total ordering — a famous ergonomic speed bump).

use crate::pose::Point;

use crate::grid::OccupancyGrid;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// 8-connected neighbors: straight moves cost 1000 "milli-cells",
/// diagonal moves cost 1414 (√2). Integer costs → orderable.
const MOVES: [(i64, i64, i64); 8] = [
    (1, 0, 1000),
    (-1, 0, 1000),
    (0, 1, 1000),
    (0, -1, 1000),
    (1, 1, 1414),
    (1, -1, 1414),
    (-1, 1, 1414),
    (-1, -1, 1414),
];

/// Straight-line distance heuristic, in milli-cells. Never overestimates
/// the real cost — the property that keeps A* exact.
fn heuristic(a: (usize, usize), b: (usize, usize)) -> i64 {
    let dx = a.0 as f64 - b.0 as f64;
    let dy = a.1 as f64 - b.1 as f64;
    (dx.hypot(dy) * 1000.0) as i64
}

/// Plan a path from `start_w` to `goal_w` (world coords, meters).
///
/// Treats Unknown cells as traversable (optimism — plan through the fog,
/// correct when the camera says otherwise) and keeps `inflation` cells of
/// clearance from anything Occupied. Returns world-coordinate waypoints
/// (every 3rd cell + the goal), or None if the goal is unreachable in the
/// current map.
pub fn plan(
    grid: &OccupancyGrid,
    start_w: Point,
    goal_w: Point,
    inflation: usize,
) -> Option<Vec<Point>> {
    let start = grid.world_to_cell(start_w.x, start_w.y)?;
    let goal = grid.world_to_cell(goal_w.x, goal_w.y)?;

    let idx = |c: (usize, usize)| c.1 * grid.width + c.0;

    // g-score: best known cost from start to each cell (i64::MAX = untouched).
    let mut g = vec![i64::MAX; grid.width * grid.height];
    // Breadcrumbs for rebuilding the path once the goal is reached.
    let mut came_from: Vec<Option<(usize, usize)>> = vec![None; grid.width * grid.height];

    // Frontier, ordered by f = g + heuristic. BinaryHeap is a MAX-heap;
    // wrapping the key in `Reverse` makes the smallest f pop first.
    let mut open: BinaryHeap<Reverse<(i64, (usize, usize))>> = BinaryHeap::new();
    g[idx(start)] = 0;
    open.push(Reverse((heuristic(start, goal), start)));

    while let Some(Reverse((_, current))) = open.pop() {
        if current == goal {
            // Walk the breadcrumbs backwards, then flip.
            let mut cells = vec![current];
            let mut c = current;
            while let Some(prev) = came_from[idx(c)] {
                cells.push(prev);
                c = prev;
            }
            cells.reverse();
            // Decimate: every 3rd cell + always the final one, as world pts.
            let mut path: Vec<Point> = cells
                .iter()
                .step_by(3)
                .map(|&(cx, cy)| grid.cell_to_world(cx, cy))
                .collect();
            path.push(goal_w);
            return Some(path);
        }

        for (dx, dy, cost) in MOVES {
            let nx = current.0 as i64 + dx;
            let ny = current.1 as i64 + dy;
            if nx < 0 || ny < 0 || nx >= grid.width as i64 || ny >= grid.height as i64 {
                continue;
            }
            let next = (nx as usize, ny as usize);
            if grid.blocked_near(next.0, next.1, inflation) {
                continue;
            }
            let candidate = g[idx(current)].saturating_add(cost);
            if candidate < g[idx(next)] {
                g[idx(next)] = candidate;
                came_from[idx(next)] = Some(current);
                open.push(Reverse((candidate + heuristic(next, goal), next)));
            }
        }
    }
    None // frontier exhausted: no route exists in the current map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Cell;

    // These construct grids with set() directly, but plan() needs
    // world_to_cell (exercise 6a) for its endpoints — hence ignored.

    #[test]
    fn open_room_finds_a_path() {
        let g = OccupancyGrid::new(2.0, 2.0, 0.1);
        let path = plan(&g, Point::new(0.2, 0.2), Point::new(1.8, 1.8), 0).unwrap();
        assert!(path.len() >= 2);
        let end = *path.last().unwrap();
        assert!((end.x - 1.8).abs() < 1e-9 && (end.y - 1.8).abs() < 1e-9);
    }

    #[test]
    fn path_routes_around_a_wall() {
        let mut g = OccupancyGrid::new(2.0, 2.0, 0.1);
        // Vertical wall at cx = 10 with a gap at the top (cy 16..20 open).
        for cy in 0..16 {
            g.set(10, cy, Cell::Occupied);
        }
        let path = plan(&g, Point::new(0.2, 0.2), Point::new(1.8, 0.2), 0).unwrap();
        // Must detour up through the gap: some waypoint has y near the top.
        assert!(path.iter().any(|p| p.y > 1.4));
    }

    #[test]
    fn walled_off_goal_is_unreachable() {
        let mut g = OccupancyGrid::new(2.0, 2.0, 0.1);
        for cy in 0..20 {
            g.set(10, cy, Cell::Occupied); // full wall, no gap
        }
        assert_eq!(
            plan(&g, Point::new(0.2, 0.2), Point::new(1.8, 0.2), 0),
            None
        );
    }
}
