//! A simulated depth camera: a forward-facing fan of sight-lines, like the
//! RealSense / OAK-D depth cameras real hobby robots carry (~70° field of
//! view), reduced to 2D.
//!
//! In a 2D world, a pinhole camera IS a fan of rays: each "pixel column"
//! is one sight-line at one angle. A depth camera reports the distance
//! along each. (A lidar is the same geometry with a 360° fan — the math
//! doesn't care; the FOV and mounting do.)
//!
//! Each ray asks exercise 5's `ray_segment_hit` against every wall and
//! keeps the NEAREST hit — a camera sees the first surface in the way,
//! not what's behind it.

use crate::exercises::ray_segment_hit;
use crate::pose::Pose;
use crate::world::World;

pub struct DepthCamera {
    /// Number of sight-lines ("pixel columns") across the image.
    pub n_rays: usize,
    /// Field of view (radians), centered on the robot's heading.
    /// Real depth cams: ~1.2 rad (70°). Set 2π·(n-1)/n and it's a lidar.
    pub fov: f64,
    /// Beyond this, report max_range (real depth cameras cut off too).
    pub max_range: f64,
}

impl DepthCamera {
    /// The angle of sight-line `i` RELATIVE to the robot's heading.
    pub fn ray_angle(&self, i: usize) -> f64 {
        if self.n_rays == 1 {
            return 0.0;
        }
        // Spread evenly across [-fov/2, +fov/2]; index 0 = leftmost.
        -self.fov / 2.0 + self.fov * (i as f64) / ((self.n_rays - 1) as f64)
    }

    /// One depth frame from the given pose: a distance per sight-line.
    pub fn scan(&self, pose: &Pose, world: &World) -> Vec<f64> {
        (0..self.n_rays)
            .map(|i| {
                let angle = pose.theta + self.ray_angle(i);
                world
                    .walls
                    .iter()
                    // Every wall this sight-line hits...
                    .filter_map(|seg| ray_segment_hit(pose.x, pose.y, angle, seg))
                    // ...keep the nearest (first surface wins).
                    .fold(self.max_range, f64::min)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// Wide test camera (180°) so the geometry below is easy to reason about.
    fn cam() -> DepthCamera {
        DepthCamera {
            n_rays: 9,
            fov: PI,
            max_range: 5.0,
        }
    }

    #[test]
    fn ray_angles_span_the_fov() {
        let c = cam();
        assert!((c.ray_angle(0) + PI / 2.0).abs() < 1e-12); // leftmost -90°
        assert!((c.ray_angle(4)).abs() < 1e-12); // center 0°
        assert!((c.ray_angle(8) - PI / 2.0).abs() < 1e-12); // rightmost +90°
    }

    #[test]
    fn center_ray_sees_the_wall_ahead() {
        let world = World::room(8.0, 6.0);
        // Robot mid-room at (4,3) facing +x: east wall at x=8 → 4 m ahead.
        let scan = cam().scan(&Pose::new(4.0, 3.0, 0.0), &world);
        assert!((scan[4] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn open_directions_report_max_range() {
        let mut world = World::room(80.0, 60.0);
        world.walls.clear(); // empty void: nothing to see anywhere
        let scan = cam().scan(&Pose::new(4.0, 3.0, 0.0), &world);
        assert!(scan.iter().all(|&d| (d - 5.0).abs() < 1e-12));
    }

    #[test]
    fn nearer_obstacle_wins() {
        let mut world = World::room(8.0, 6.0);
        world.add_box(5.0, 2.5, 5.5, 3.5); // box between robot and east wall
        let scan = cam().scan(&Pose::new(4.0, 3.0, 0.0), &world);
        // Center ray must see the box face at x=5 (1 m), not the wall (4 m).
        assert!((scan[4] - 1.0).abs() < 1e-9);
    }
}
