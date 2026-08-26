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
# THE NOMINAL SOLVER CONDITION — every value spelled here, because the
# 2026-08-27 solver review (docs/e2e-research/47/48) found half of them
# were inherited MuJoCo defaults nobody ever chose: Euler at 500 Hz was
# the global default, not a decision. sysid fits are fits OF this
# discretization, so the block is part of the identified artifact; if
# upstream MuJoCo ever changes a default (they recommend implicitfast
# already), an unpinned scene's nominal condition would silently move.
# elliptic + impratio 10 + the Newton solver is MuJoCo's own documented
# anti-slip recipe for pinch grasps — and measured (tools/solver-study):
# every other solver/cone fails the kitting referee outright.
# (mjSOL_NEWTON is MuJoCo's constraint solver, unrelated to the NVIDIA
# Newton engine.)
TIMESTEP = 0.002  # 500 Hz physics; control at 50 Hz rides on it
IMPRATIO = 10
FLOOR_GEOM = "floor"


class GeomGroup:
    """Geom groups the viewers and mirrors filter by."""

    COLLISION = 3
    HIDDEN = 4


def pin_nominal_options(spec: Any) -> None:
    """Write the nominal solver condition into a scene spec.

    Called by every builder — both the attach-built SO-101 scenes
    (where `MjSpec.attach` drops the arm's declared cone/impratio with
    a warning, measured) and the bundle-loaded ALOHA scenes (where the
    XML sets cone/impratio but left solver, integrator, and timestep to
    the global defaults). One function, so "arm alone", "arm in scene"
    and both rigs share one contact physics and one discretization.
    """
    import mujoco  # noqa: PLC0415 - sim extra

    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    spec.option.timestep = TIMESTEP
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO


def set_render_budget(spec: Any) -> None:
    spec.visual.global_.offwidth = OFFSCREEN_WIDTH
    spec.visual.global_.offheight = OFFSCREEN_HEIGHT
    spec.visual.quality.shadowsize = SHADOWSIZE
