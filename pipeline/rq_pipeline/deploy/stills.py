"""The deploy stage's pictures: one size, named cameras, one renderer
that is always closed. The attribution's fall at the cliff and the
pre-flight's handover (and the legged fit's stills, `tools/`) read these
instead of spelling their own numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STILL_SIZE = (640, 480)  # width, height in pixels


@dataclass(frozen=True)
class Camera:
    """A free camera around a point: metres back, degrees round and down."""

    distance: float
    azimuth: float
    elevation: float


# The fall, a little further back to hold a body on its side.
CLIFF_CAMERA = Camera(distance=1.6, azimuth=135.0, elevation=-18.0)
# The robot standing at the end of the ramp-in.
HANDOVER_CAMERA = Camera(distance=1.4, azimuth=135.0, elevation=-15.0)

Render = Callable[[Any, np.ndarray, Path], None]
"""Render `data` looking at `lookat` into the PNG at the path."""


@contextmanager
def renderer(model: Any, camera: Camera) -> Iterator[Render]:
    """A renderer for `model` through `camera`, closed on the way out
    whatever happens inside (an early return leaked it, the review of
    2026-09-24). Raises what MuJoCo raises when no GL context exists."""
    import mujoco  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    width, height = STILL_SIZE
    gl = mujoco.Renderer(model, height, width)
    view = mujoco.MjvCamera()
    view.distance = camera.distance
    view.azimuth = camera.azimuth
    view.elevation = camera.elevation

    def render(data: Any, lookat: np.ndarray, out: Path) -> None:
        view.lookat[:] = lookat
        gl.update_scene(data, view)
        Image.fromarray(gl.render()).save(out)

    try:
        yield render
    finally:
        gl.close()
