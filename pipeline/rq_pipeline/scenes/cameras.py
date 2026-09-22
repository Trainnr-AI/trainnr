"""Cameras that see the splat (docs/78 §4 E2): mujoco_warp's ray tracer
renders the model's cameras with the scene's gaussians composited in,
the robot occluding the scene and the scene the robot, depth from both
(docs/e2e-research/76 §2). Static splats, world frame, unlit, one BVH
built at open.

An instrument matter: the renderer arrived in mujoco_warp 3.13.0
(2026-08-19); the walk package is pinned at 3.11 until E0. So `open`
answers None with the reason whenever the renderer is not there, and
whatever holds a picture (the gate's mirror) says so instead of
pretending. Pixels come back packed as one uint32 per pixel with the
channels in the low three bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from rq_pipeline.scenes.splat import VISIBLE_OPACITY, Splats

RENDERER_SINCE = (3, 13)
NO_RENDERER = (
    "no mujoco_warp: the splat renderer needs mujoco_warp >= 3.13 (docs/78 E0)"
)
OLD_RENDERER = "mujoco_warp {found} has no splat renderer; 3.13 does (docs/78 E0)"
NO_GEOMS = (
    "the model has no geom in the drawn groups; the ray tracer needs at least one "
    "(it faults on an empty scene)"
)
DEFAULT_RESOLUTION = (160, 120)
# The groups the renderer draws: the robot's visual meshes, never the
# collision parts (group 3) the stage hides from the picture too.
VISUAL_GROUPS = (0, 1, 2)
CHANNEL_SHIFTS = (16, 8, 0)


def _version() -> tuple[int, ...] | None:
    try:
        text = version("mujoco-warp")
    except PackageNotFoundError:
        return None
    return tuple(int(p) for p in text.split(".")[:2] if p.isdigit())


def splat_arguments(visible: Splats) -> dict[str, np.ndarray]:
    """The gaussians as mujoco_warp's render context takes them (its
    `splat_*` keywords): one transcription, used by the gate's cameras
    and by the walk package's training cameras."""
    return {
        "splat_position": np.ascontiguousarray(visible.means, dtype=np.float32),
        "splat_rotation": np.ascontiguousarray(visible.quats, dtype=np.float32),
        "splat_scale": np.ascontiguousarray(visible.scales, dtype=np.float32),
        "splat_rgba": np.ascontiguousarray(
            np.concatenate([visible.colors, visible.opacities[:, None]], axis=1),
            dtype=np.float32,
        ),
    }


def unpack(packed: np.ndarray) -> np.ndarray:
    """Packed uint32 pixels to (..., 3) uint8."""
    p = np.asarray(packed).astype(np.uint32)
    return np.stack([(p >> s) & 255 for s in CHANNEL_SHIFTS], axis=-1).astype(np.uint8)


@dataclass
class SplatCameras:
    """The renderer over one model and one splat, for named cameras."""

    mjw: Any
    wp: Any
    model: Any  # mujoco_warp's Model
    context: Any
    names: tuple[str, ...]
    resolution: tuple[int, int]
    splats: int
    instrument: str

    def render(self, mjm: Any, mjd: Any) -> dict[str, np.ndarray]:
        """RGB (h, w, 3) uint8 per camera name, from the data's poses."""
        data = self.mjw.put_data(mjm, mjd)
        self.mjw.render(self.model, data, self.context)
        self.wp.synchronize()
        w, h = self.resolution
        flat = self.context.rgb_data.numpy().reshape(-1)
        pixels = w * h
        return {
            name: unpack(flat[i * pixels : (i + 1) * pixels]).reshape(h, w, 3)
            for i, name in enumerate(self.names)
        }


def open_cameras(
    mjm: Any,
    splats: Splats,
    *,
    names: tuple[str, ...],
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> tuple[SplatCameras | None, str]:
    """The cameras, or (None, why) when the instrument lacks the renderer."""
    found = _version()
    if found is None:
        return None, NO_RENDERER
    if found < RENDERER_SINCE:
        return None, OLD_RENDERER.format(found=".".join(map(str, found)))
    import mujoco_warp as mjw  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    drawn = [g for g in range(mjm.ngeom) if int(mjm.geom_group[g]) in VISUAL_GROUPS]
    if not drawn:
        # measured 2026-09-23: a render context over a model with no geom in
        # the drawn groups builds an empty BVH and the kernel faults
        return None, NO_GEOMS
    visible = splats.visible(VISIBLE_OPACITY)
    context = mjw.create_render_context(
        mjm,
        nworld=1,
        cam_res=resolution,
        render_rgb=True,
        render_depth=False,
        enabled_geom_groups=list(VISUAL_GROUPS),
        cam_active=list(names),
        **splat_arguments(visible),
    )
    return SplatCameras(
        mjw=mjw,
        wp=wp,
        model=mjw.put_model(mjm),
        context=context,
        names=tuple(names),
        resolution=resolution,
        splats=int(visible.means.shape[0]),
        instrument=f"mujoco_warp {version('mujoco-warp')} on {wp.get_device().alias}",
    ), ""
