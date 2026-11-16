"""Video to a G1 reference (docs/86 §5.1): the floor a tilted estimate
stands on is levelled, the retargeter's CSV units become mjlab's, and
smoothing keeps the movement while it takes the jitter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from trainnr_mjlab.gem_to_bvh import level
from trainnr_mjlab.tracking_motion import robot_csv, smooth


def _walk_on(tilt_deg: float, frames: int = 300, fps: float = 30.0) -> np.ndarray:
    """Two feet stepping forward 2 m over a floor tilted `tilt_deg` about x,
    as (frames, joints, 4, 4) world transforms."""
    world = np.tile(np.eye(4), (frames, 2, 1, 1))
    t = np.arange(frames) / fps
    for foot, phase in ((0, 0.0), (1, np.pi)):
        world[:, foot, 2, 3] = 2.0 * t / t[-1]  # forward along z
        world[:, foot, 1, 3] = 0.05 * np.clip(np.sin(2 * np.pi * t + phase), 0, None)
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_euler("x", tilt_deg, degrees=True).as_matrix()
    return turn @ world


class TheFloor(unittest.TestCase):
    def test_a_tilted_floor_is_turned_level_and_the_feet_stay_on_it(self) -> None:
        world, floor = level(_walk_on(12.8), [0, 1], 30.0)
        self.assertAlmostEqual(floor["tilt_deg"], 12.8, delta=0.3)
        lowest = world[:, :, 1, 3].min(1)
        self.assertLess(np.abs(lowest).max(), 0.01)  # rising 45 cm before
        self.assertGreater(world[:, :, 1, 3].max(), 0.04)  # the step stays


class TheRetargetersCsv(unittest.TestCase):
    def test_centimetres_degrees_and_euler_become_metres_radians_quaternions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g1.csv"
            path.write_text(
                "Frame,root_translateX,root_translateY,root_translateZ,"
                "root_rotateX,root_rotateY,root_rotateZ,left_knee_joint\n"
                "0,10,20,78,0,0,90,30\n"
            )
            pos, quat, q, joints = robot_csv(path)
        np.testing.assert_allclose(pos[0], [0.1, 0.2, 0.78])
        np.testing.assert_allclose(  # x y z w: 90 degrees about z
            quat[0], [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-9
        )
        np.testing.assert_allclose(q[0], [np.pi / 6])
        self.assertEqual(joints, ["left_knee_joint"])


class Smoothing(unittest.TestCase):
    def test_it_takes_jitter_keeps_the_movement_and_unit_quaternions(self) -> None:
        fps, n = 30.0, 300
        t = np.arange(n) / fps
        rng = np.random.default_rng(0)
        move = np.sin(2 * np.pi * 1.0 * t)[:, None]  # a 1 Hz dance step
        q = move + 0.05 * rng.standard_normal((n, 1))
        quat = Rotation.from_euler("z", 30 * move, degrees=True).as_quat()
        quat[::7] *= -1  # the same rotations, the other hemisphere
        pos = np.zeros((n, 3))
        _, quat_s, q_s = smooth(pos, quat, q, fps, 6.0)
        # white jitter keeps 6/15 of its power below a 6 Hz cutoff at 30 fps
        self.assertLess(np.abs(q_s - move).mean(), 0.75 * np.abs(q - move).mean())
        np.testing.assert_allclose(np.linalg.norm(quat_s, axis=1), 1.0)
        self.assertLess(np.abs(np.diff(quat_s, axis=0)).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
