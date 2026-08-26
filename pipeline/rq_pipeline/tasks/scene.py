"""Scene facts every builder shares — spelled once (docs/24's rule).

Until 2026-08-26 the offscreen framebuffer size, the shadow-map budget,
the contact options MuJoCo drops on `attach`, and the geom groups tools
hide were retyped per builder and per tool. Now: constants here, and
two helpers that apply them to an `MjSpec`.
"""

from __future__ import annotations

from typing import Any

# The largest camera any rig renders (the ALOHA D405s and the ArmnetBench
# wrist camera, 1280x720): the offscreen framebuffer must cover it or
# `mujoco.Renderer` refuses.
OFFSCREEN_WIDTH = 1280
OFFSCREEN_HEIGHT = 720
# Menagerie scenes ask for an 8192x8192 shadow map — a screenshot setting.
# Every harness rollout and demo frame renders through the scene, so it
# is a per-frame cost (measured 2026-08-26: 31 ms per 640x480 frame on
# the RTX at 8192). Physics is untouched by it.
SHADOWSIZE = 2048
# Contact options the arms declare that `MjSpec.attach` drops (measured:
# MuJoCo keeps the parent's defaults with a warning); every scene sets
# them back so "arm alone" and "arm in scene" share contact physics.
IMPRATIO = 10
FLOOR_GEOM = "floor"


class GeomGroup:
    """Geom groups the viewers and mirrors filter by."""

    COLLISION = 3
    HIDDEN = 4


def restore_contact_options(spec: Any) -> None:
    import mujoco  # noqa: PLC0415 - sim extra

    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO


def set_render_budget(spec: Any) -> None:
    spec.visual.global_.offwidth = OFFSCREEN_WIDTH
    spec.visual.global_.offheight = OFFSCREEN_HEIGHT
    spec.visual.quality.shadowsize = SHADOWSIZE
