//! Navigation decisions: reading a depth scan, switching behaviours, and
//! choosing what point to steer at.
//!
//! All of it is pure arithmetic over slices, and all of it used to live
//! inside `sim-run`'s `main()` where it could not be tested. A reactive
//! obstacle reflex is exactly the kind of code that *looks* obviously
//! right and then drives into a wall at the one geometry nobody pictured
//! — see the wall-collision bug in docs/07.
//!
//! No allocation, so the firmware can use this too when it grows a
//! forward-facing sensor.

#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use crate::pose::{Point, Pose};

/// What the robot is currently trying to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Drive toward the goal.
    Goto,
    /// Something is close — turn away and creep.
    Avoid,
}

/// Distances at which the robot switches between [`Mode`]s.
///
/// # Why two thresholds and not one
///
/// With a single threshold, a robot sitting exactly at that distance
/// flips between modes every tick — it turns, the reading changes, it
/// drives, the reading changes back. The controller is reset on every
/// transition, so it never builds up enough correction to escape, and the
/// robot judders in place forever.
///
/// Two thresholds (`exit` strictly greater than `enter`) create a dead
/// zone: once avoiding, the robot must get *meaningfully* clearer before
/// it resumes driving. That is hysteresis, and it is the standard fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AvoidHysteresis {
    /// Enter [`Mode::Avoid`] when the nearest reading is below this.
    pub enter: f64,
    /// Return to [`Mode::Goto`] only once the nearest reading exceeds this.
    pub exit: f64,
}

impl AvoidHysteresis {
    /// The next mode, given the current one and the nearest obstacle.
    pub fn next(&self, current: Mode, min_distance: f64) -> Mode {
        match current {
            Mode::Goto if min_distance < self.enter => Mode::Avoid,
            Mode::Avoid if min_distance > self.exit => Mode::Goto,
            unchanged => unchanged,
        }
    }

    /// Is this configuration actually hysteretic?
    ///
    /// `exit <= enter` collapses the dead zone and reintroduces the
    /// judder. Callers can assert on this rather than discovering it in
    /// the viewer.
    pub fn is_valid(&self) -> bool {
        self.exit > self.enter
    }
}

/// What a forward depth scan says, reduced to the three numbers the
/// reflex actually uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanSummary {
    /// Nearest reading anywhere in the fan.
    pub nearest: f64,
    /// Nearest reading in the left half (later indices — see below).
    pub left_min: f64,
    /// Nearest reading in the right half (earlier indices).
    pub right_min: f64,
}

/// Reduce a depth scan to [`ScanSummary`].
///
/// # Index convention
///
/// Rays are ordered right-to-left across the fan, matching
/// `DepthCamera`: index 0 is the rightmost ray, the last index is the
/// leftmost. The centre ray of an odd-length scan belongs to **neither**
/// half — it points straight ahead, so it cannot argue for turning either
/// way, and including it in both halves would bias every comparison.
///
/// An empty scan reports infinity everywhere: no data means no obstacle
/// detected, which keeps the robot in [`Mode::Goto`] rather than having it
/// freeze because a sensor dropped out.
pub fn summarize_scan(scan: &[f64]) -> ScanSummary {
    if scan.is_empty() {
        return ScanSummary {
            nearest: f64::INFINITY,
            left_min: f64::INFINITY,
            right_min: f64::INFINITY,
        };
    }
    let centre = scan.len() / 2;
    let fold = |s: &[f64]| s.iter().copied().fold(f64::INFINITY, f64::min);
    ScanSummary {
        nearest: fold(scan),
        right_min: fold(&scan[..centre]),
        left_min: fold(&scan[centre + 1..]),
    }
}

impl ScanSummary {
    /// Which way to turn to escape: `+1.0` for left, `-1.0` for right.
    ///
    /// Turn toward whichever side has more room. Ties go right, which is
    /// arbitrary but must be *deterministic* — a tie-break that varies
    /// run to run makes a stuck robot impossible to debug.
    pub fn turn_direction(&self) -> f64 {
        if self.left_min > self.right_min {
            1.0
        } else {
            -1.0
        }
    }
}

/// Pick the point on `path` to steer at.
///
/// Returns the first path point more than `lookahead` metres away, or
/// `fallback` (normally the goal) if the path is empty or entirely within
/// the lookahead radius.
///
/// # Why not steer at the next path node
///
/// A grid planner emits points a few centimetres apart. Steering at the
/// nearest one means the heading error is dominated by grid quantisation,
/// so the robot weaves along the path instead of following it. Looking
/// ahead a fixed distance smooths that out — it is the standard
/// pure-pursuit trick.
pub fn lookahead_point(path: &[Point], from: &Pose, lookahead: f64, fallback: Point) -> Point {
    for &node in path {
        if (node.x - from.x).hypot(node.y - from.y) > lookahead {
            return node;
        }
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- hysteresis ----

    const H: AvoidHysteresis = AvoidHysteresis {
        enter: 0.5,
        exit: 0.8,
    };

    #[test]
    fn a_close_obstacle_triggers_avoidance() {
        assert_eq!(H.next(Mode::Goto, 0.4), Mode::Avoid);
    }

    #[test]
    fn a_distant_obstacle_leaves_goto_alone() {
        assert_eq!(H.next(Mode::Goto, 2.0), Mode::Goto);
    }

    #[test]
    fn avoidance_persists_through_the_dead_zone() {
        // THE point of hysteresis: between enter and exit, whatever mode
        // we are in is the mode we stay in.
        for d in [0.55, 0.6, 0.7, 0.79] {
            assert_eq!(H.next(Mode::Avoid, d), Mode::Avoid, "at {d}");
            assert_eq!(H.next(Mode::Goto, d), Mode::Goto, "at {d}");
        }
    }

    #[test]
    fn clearing_the_exit_threshold_resumes_driving() {
        assert_eq!(H.next(Mode::Avoid, 0.9), Mode::Goto);
    }

    #[test]
    fn a_robot_parked_on_the_threshold_does_not_oscillate() {
        // Regression for judder: feed the SAME borderline reading many
        // times and the mode must settle, not alternate.
        let mut mode = Mode::Goto;
        let mut flips = 0;
        for _ in 0..100 {
            let next = H.next(mode, 0.5);
            if next != mode {
                flips += 1;
            }
            mode = next;
        }
        assert!(
            flips <= 1,
            "mode flipped {flips} times on a constant reading"
        );
    }

    #[test]
    fn our_configuration_is_actually_hysteretic() {
        assert!(H.is_valid());
        assert!(!AvoidHysteresis {
            enter: 0.8,
            exit: 0.5
        }
        .is_valid());
        assert!(!AvoidHysteresis {
            enter: 0.5,
            exit: 0.5
        }
        .is_valid());
    }

    // ---- scan summary ----

    #[test]
    fn an_empty_scan_reports_no_obstacles() {
        let s = summarize_scan(&[]);
        assert!(s.nearest.is_infinite());
        assert!(s.left_min.is_infinite());
        assert!(s.right_min.is_infinite());
        // And must NOT trigger avoidance.
        assert_eq!(H.next(Mode::Goto, s.nearest), Mode::Goto);
    }

    #[test]
    fn the_minimum_is_the_nearest_reading_anywhere() {
        let s = summarize_scan(&[3.0, 1.2, 2.5, 0.7, 4.0]);
        assert!((s.nearest - 0.7).abs() < 1e-12);
    }

    #[test]
    fn halves_are_split_around_the_centre_ray() {
        // 5 rays: indices 0,1 = right, 2 = centre, 3,4 = left.
        let s = summarize_scan(&[9.0, 8.0, 0.1, 2.0, 3.0]);
        assert!((s.right_min - 8.0).abs() < 1e-12, "right = {}", s.right_min);
        assert!((s.left_min - 2.0).abs() < 1e-12, "left = {}", s.left_min);
        // The centre ray is the nearest overall but belongs to neither half.
        assert!((s.nearest - 0.1).abs() < 1e-12);
    }

    #[test]
    fn the_robot_turns_toward_open_space() {
        let clear_left = summarize_scan(&[0.3, 0.3, 1.0, 5.0, 5.0]);
        assert_eq!(clear_left.turn_direction(), 1.0, "should turn left");

        let clear_right = summarize_scan(&[5.0, 5.0, 1.0, 0.3, 0.3]);
        assert_eq!(clear_right.turn_direction(), -1.0, "should turn right");
    }

    #[test]
    fn a_symmetric_obstacle_breaks_the_tie_deterministically() {
        let s = summarize_scan(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let first = s.turn_direction();
        for _ in 0..10 {
            assert_eq!(s.turn_direction(), first, "tie-break is not stable");
        }
    }

    #[test]
    fn a_single_ray_scan_has_no_halves() {
        let s = summarize_scan(&[1.5]);
        assert!((s.nearest - 1.5).abs() < 1e-12);
        assert!(s.left_min.is_infinite());
        assert!(s.right_min.is_infinite());
    }

    // ---- lookahead ----

    fn origin() -> Pose {
        Pose {
            x: 0.0,
            y: 0.0,
            heading: 0.0,
        }
    }

    #[test]
    fn an_empty_path_falls_back_to_the_goal() {
        assert_eq!(
            lookahead_point(&[], &origin(), 0.5, Point::new(9.0, 9.0)),
            Point::new(9.0, 9.0)
        );
    }

    #[test]
    fn the_first_point_beyond_the_lookahead_is_chosen() {
        let path = [
            Point::new(0.1, 0.0),
            Point::new(0.2, 0.0),
            Point::new(0.9, 0.0),
            Point::new(1.5, 0.0),
        ];
        assert_eq!(
            lookahead_point(&path, &origin(), 0.5, Point::new(9.0, 9.0)),
            Point::new(0.9, 0.0)
        );
    }

    #[test]
    fn a_path_entirely_within_the_lookahead_falls_back() {
        // Near the end of a path every node is close; steering at the
        // goal is right, and returning the last node would stall.
        let path = [
            Point::new(0.1, 0.0),
            Point::new(0.2, 0.0),
            Point::new(0.3, 0.0),
        ];
        assert_eq!(
            lookahead_point(&path, &origin(), 0.5, Point::new(9.0, 9.0)),
            Point::new(9.0, 9.0)
        );
    }

    #[test]
    fn lookahead_is_measured_from_the_robot_not_the_origin() {
        let pose = Pose {
            x: 5.0,
            y: 5.0,
            heading: 0.0,
        };
        let path = [Point::new(5.1, 5.0), Point::new(6.0, 5.0)];
        assert_eq!(
            lookahead_point(&path, &pose, 0.5, Point::new(9.0, 9.0)),
            Point::new(6.0, 5.0)
        );
    }

    #[test]
    fn distance_is_euclidean_not_axis_aligned() {
        // (0.4, 0.4) is 0.566 away — beyond a 0.5 lookahead, even though
        // neither coordinate alone exceeds it.
        let path = [Point::new(0.4, 0.4)];
        assert_eq!(
            lookahead_point(&path, &origin(), 0.5, Point::new(9.0, 9.0)),
            Point::new(0.4, 0.4)
        );
    }
}
