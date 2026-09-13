"""The render stream's camera, as MuJoCo sees it: a pan must reach
`MjvCamera.lookat` every frame, not only the status echo. Two "fixes"
were judged by the echo while the picture stood still (2026-09-12)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from tests._extras import needs_sim

REPO = Path(__file__).resolve().parents[2]


def render_stream():
    sys.path.insert(0, str(REPO / "tools"))  # the tool's own `_lab` helper
    spec = importlib.util.spec_from_file_location(
        "studio_render_stream", REPO / "tools" / "studio-render-stream.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@needs_sim
class PanReachesTheCamera(unittest.TestCase):
    def test_a_pan_moves_mujocos_lookat_on_the_next_apply(self) -> None:
        import mujoco  # noqa: PLC0415 - the sim extra

        srs = render_stream()
        orbit = srs.OrbitCamera(srs.RIG_CAMERAS[srs.DEFAULT_CAMERA])
        cam = mujoco.MjvCamera()
        orbit.apply_to(cam)
        before = list(cam.lookat)
        orbit.pan(1.0, 0.0, 0.0)  # one second of W
        orbit.apply_to(cam)
        moved = [b - a for a, b in zip(before, cam.lookat, strict=True)]
        expected = srs.PAN_RATE_PER_S * orbit.distance
        self.assertAlmostEqual(
            (moved[0] ** 2 + moved[1] ** 2) ** 0.5, expected, places=5
        )
        self.assertAlmostEqual(moved[2], 0.0)
        # And the status echo says the same thing the picture shows.
        self.assertEqual(list(cam.lookat), orbit.pose()[3:])

    def test_up_is_the_worlds_z_and_right_is_a_quarter_turn(self) -> None:
        import math  # noqa: PLC0415

        import mujoco  # noqa: PLC0415 - the sim extra

        srs = render_stream()
        orbit = srs.OrbitCamera(srs.RIG_CAMERAS[srs.DEFAULT_CAMERA])
        cam = mujoco.MjvCamera()
        orbit.pan(0.0, 0.0, 1.0)
        orbit.apply_to(cam)
        self.assertAlmostEqual(
            cam.lookat[2] - srs.RIG_CAMERAS[srs.DEFAULT_CAMERA]["lookat"][2],
            srs.PAN_RATE_PER_S * orbit.distance,
            places=5,
        )
        az = math.radians(orbit.azimuth)
        x0, y0 = cam.lookat[0], cam.lookat[1]
        orbit.pan(0.0, 1.0, 0.0)
        orbit.apply_to(cam)
        step = srs.PAN_RATE_PER_S * orbit.distance
        self.assertAlmostEqual(cam.lookat[0] - x0, step * math.sin(az), places=5)
        self.assertAlmostEqual(cam.lookat[1] - y0, -step * math.cos(az), places=5)


if __name__ == "__main__":
    unittest.main()
