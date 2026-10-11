"""A still's pure-black rows (a render's empty sky) are trimmed before it
is fitted to its card: a deployment's card showed a black band across
its top (design review, 2026-10-12)."""

from __future__ import annotations

import unittest

import numpy as np

from trainnr.project.previews import _trim_black_bands

try:
    from PIL import Image
except ImportError:  # the picture extra is not installed
    Image = None


@unittest.skipIf(Image is None, "Pillow is not installed")
class BlackBands(unittest.TestCase):
    def test_black_rows_above_and_below_the_scene_go(self) -> None:
        pixels = np.zeros((48, 64, 3), dtype=np.uint8)
        pixels[10:40] = 29  # the scene's dark ground
        trimmed = _trim_black_bands(Image.fromarray(pixels))
        self.assertEqual(trimmed.size, (64, 30))

    def test_a_still_with_no_band_or_all_black_is_kept(self) -> None:
        lit = Image.fromarray(np.full((20, 30, 3), 29, dtype=np.uint8))
        self.assertEqual(_trim_black_bands(lit).size, (30, 20))
        black = Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(_trim_black_bands(black).size, (30, 20))


if __name__ == "__main__":
    unittest.main()
