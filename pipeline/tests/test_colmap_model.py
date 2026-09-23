"""COLMAP's text model read into arrays: the intrinsics by model, the
pose as the world-to-camera matrix COLMAP's quaternion spells, the
points with their colours."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rq_pipeline.scenes.colmap_model import read_text_model


def write_model(sparse: Path) -> None:
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "# cameras\n1 OPENCV 640 480 500 510 320 240 0.1 -0.01 0.001 0.002\n"
    )
    # a quarter turn about z: (w, x, y, z) = (cos 45, 0, 0, sin 45)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    (sparse / "images.txt").write_text(
        f"# images\n1 {c} 0 0 {s} 1 2 3 1 frame_0001.jpg\n10 20 -1\n"
        "2 1 0 0 0 0 0 0 1 frame_0002.jpg\n\n"
    )
    (sparse / "points3D.txt").write_text(
        "# points\n1 0.5 0.25 2 255 128 0 0.1 1 1\n2 -1 0 0 0 0 255 0.2 2 1\n"
    )


class TheTextModel(unittest.TestCase):
    def test_cameras_images_and_points_come_back_as_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp) / "sparse" / "0"
            write_model(sparse)
            model = read_text_model(sparse)
        cam = model.cameras[1]
        self.assertEqual(model.camera_model, "OPENCV")
        np.testing.assert_allclose(cam.K, [[500, 0, 320], [0, 510, 240], [0, 0, 1]])
        self.assertEqual(cam.radial[:2], (0.1, -0.01))
        self.assertEqual(cam.tangential, (0.001, 0.002))
        self.assertTrue(cam.distorted)
        self.assertEqual(
            [i.name for i in model.images], ["frame_0001.jpg", "frame_0002.jpg"]
        )
        w2c = model.images[0].world_to_camera
        # a quarter turn about z takes +x to +y, then the translation
        np.testing.assert_allclose(w2c @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-9)
        np.testing.assert_allclose(model.images[1].world_to_camera, np.eye(4))
        self.assertEqual(model.points.shape, (2, 3))
        self.assertEqual(model.colors.tolist(), [[255, 128, 0], [0, 0, 255]])

    def test_an_image_with_no_points_does_not_shift_the_next(self) -> None:
        """COLMAP leaves the 2D-point line EMPTY for an image with none;
        the image after it must still be read as a pose (skipping blank
        lines shifted every later image by a line, 2026-09-24)."""
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp)
            (sparse / "cameras.txt").write_text("1 PINHOLE 4 4 1 1 2 2\n")
            (sparse / "images.txt").write_text(
                "# images\n1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 5 0 0 1 b.jpg\n1 2 -1\n"
            )
            (sparse / "points3D.txt").write_text("")
            model = read_text_model(sparse)
        self.assertEqual([i.name for i in model.images], ["a.jpg", "b.jpg"])
        self.assertEqual(model.images[1].translation.tolist(), [5.0, 0.0, 0.0])

    def test_a_camera_row_short_of_its_model_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp)
            (sparse / "cameras.txt").write_text("1 OPENCV 4 4 1 1 2\n")
            (sparse / "images.txt").write_text("")
            (sparse / "points3D.txt").write_text("")
            with self.assertRaises(ValueError) as caught:
                read_text_model(sparse)
        self.assertIn("OPENCV", str(caught.exception))

    def test_an_unknown_camera_model_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp)
            (sparse / "cameras.txt").write_text("1 THIN_PRISM_FISHEYE 1 1 1 1 1\n")
            (sparse / "images.txt").write_text("")
            (sparse / "points3D.txt").write_text("")
            with self.assertRaises(ValueError) as caught:
                read_text_model(sparse)
        self.assertIn("THIN_PRISM_FISHEYE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
