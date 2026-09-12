"""Scene facts every builder shares — spelled once (docs/24's rule).

Until 2026-08-26 the offscreen framebuffer size, the shadow-map budget,
the contact options MuJoCo drops on `attach`, and the geom groups tools
hide were retyped per builder and per tool. Now: the render budget and
the nominal solver condition as structs here, MuJoCo's option enums by
short name, and the helpers that write them onto an `MjSpec` or a
compiled model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RenderBudget:
    """What every scene's renderer is sized to. The offscreen framebuffer
    must cover the largest camera any rig renders (the ALOHA D405s and
    the ArmnetBench wrist camera, 1280x720) or `mujoco.Renderer` refuses;
    Menagerie scenes ask for an 8192x8192 shadow map — a screenshot
    setting, and every harness rollout and demo frame renders through
    the scene, so it is a per-frame cost (measured 2026-08-26: 31 ms per
    640x480 frame on the RTX at 8192). Physics is untouched by it."""

    OFFSCREEN_WIDTH = 1280
    OFFSCREEN_HEIGHT = 720
    SHADOWSIZE = 2048


class NominalOptions:
    """THE NOMINAL SOLVER CONDITION — every value spelled here, because the
    2026-08-27 solver review (docs/e2e-research/47/48) found half of them
    were inherited MuJoCo defaults nobody ever chose: Euler at 500 Hz was
    the global default, not a decision. sysid fits are fits OF this
    discretization, so the block is part of the identified artifact; if
    upstream MuJoCo ever changes a default (they recommend implicitfast
    already), an unpinned scene's nominal condition would silently move.
    elliptic + impratio 10 + the Newton solver is MuJoCo's own documented
    anti-slip recipe for pinch grasps — and measured (tools/solver-study,
    docs/e2e-research/47 §7): the only configuration that passes the
    kitting referee on BOTH the Mac (arm64) and the WSL box (x86_64); CG
    and the pyramidal cone fail on both, PGS passes only on x86_64, at
    28x the solver iterations. (mjSOL_NEWTON is MuJoCo's constraint
    solver, unrelated to the NVIDIA Newton engine.) Short names —
    `OptionFamilies` maps them to MuJoCo's enum members."""

    SOLVER = "newton"
    CONE = "elliptic"
    INTEGRATOR = "euler"
    TIMESTEP = 0.002  # 500 Hz physics; control at 50 Hz rides on it
    IMPRATIO = 10


class OptionFamilies:
    """MuJoCo's option enums by short name: the one place a tool or a
    test spells `mjSOL_NEWTON`. Resolved lazily by `option_enum`, so
    this module imports without the sim extra."""

    SOLVER = (
        "mjtSolver",
        {"newton": "mjSOL_NEWTON", "cg": "mjSOL_CG", "pgs": "mjSOL_PGS"},
    )
    CONE = ("mjtCone", {"elliptic": "mjCONE_ELLIPTIC", "pyramidal": "mjCONE_PYRAMIDAL"})
    INTEGRATOR = (
        "mjtIntegrator",
        {
            "euler": "mjINT_EULER",
            "implicitfast": "mjINT_IMPLICITFAST",
            "implicit": "mjINT_IMPLICIT",
            "rk4": "mjINT_RK4",
        },
    )


_FAMILIES = {
    "solver": OptionFamilies.SOLVER,
    "cone": OptionFamilies.CONE,
    "integrator": OptionFamilies.INTEGRATOR,
}


def option_enum(family: str, name: str) -> Any:
    """`option_enum("solver", "newton")` -> `mujoco.mjtSolver.mjSOL_NEWTON`;
    an unknown family or name is refused with the known ones listed."""
    import mujoco  # noqa: PLC0415 - sim extra

    if family not in _FAMILIES:
        raise ValueError(
            f"unknown option family {family!r}; known: {sorted(_FAMILIES)}"
        )
    enum_name, members = _FAMILIES[family]
    if name not in members:
        raise ValueError(f"unknown {family} {name!r}; known: {sorted(members)}")
    return getattr(getattr(mujoco, enum_name), members[name])


def apply_options(  # noqa: PLR0913 - the five knobs of one condition, each named
    opt: Any,
    *,
    solver: str,
    cone: str,
    integrator: str,
    timestep: float | None = None,
    impratio: float | None = None,
) -> None:
    """Write a solver condition onto `spec.option` (before compile) or
    `model.opt` (after) — the two carry the same attribute names."""
    opt.solver = option_enum("solver", solver)
    opt.cone = option_enum("cone", cone)
    opt.integrator = option_enum("integrator", integrator)
    if timestep is not None:
        opt.timestep = timestep
    if impratio is not None:
        opt.impratio = impratio


FLOOR_GEOM = "floor"
# The geom every task object starts on, in both rigs: the SO-101 scenes
# add it by this name; the ALOHA bundle's table compiles under it too
# (pinned by tests/test_placement.py).
TABLE_GEOM = "table"


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
    apply_options(
        spec.option,
        solver=NominalOptions.SOLVER,
        cone=NominalOptions.CONE,
        integrator=NominalOptions.INTEGRATOR,
        timestep=NominalOptions.TIMESTEP,
        impratio=NominalOptions.IMPRATIO,
    )


def set_render_budget(spec: Any) -> None:
    spec.visual.global_.offwidth = RenderBudget.OFFSCREEN_WIDTH
    spec.visual.global_.offheight = RenderBudget.OFFSCREEN_HEIGHT
    spec.visual.quality.shadowsize = RenderBudget.SHADOWSIZE


def add_free_box(
    scene: Any, name: str, pos: Any, half: Any, *, rgba: Any, **geom: Any
) -> Any:
    """A free-floating box: a body with a free joint and one box geom
    (`half` a scalar or a triple), any further geom attributes verbatim.
    Every task object on both rigs is one of these. Returns the body."""
    import mujoco  # noqa: PLC0415 - sim extra

    body = scene.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint()
    size = [half] * 3 if isinstance(half, int | float) else list(half)
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        rgba=list(rgba),
        **geom,
    )
    return body


def add_slot_walls(  # noqa: PLR0913 - a well's geometry, each dimension named
    scene: Any,
    prefix: str,
    centre: tuple[float, float],
    *,
    offset: tuple[float, float],
    long_half: tuple[float, float],
    wall: float,
    height: float,
    rgba: Any,
) -> None:
    """Four static wall boxes around `centre`: north/south at ±offset_y
    with half-length long_half_x, east/west at ±offset_x with half-length
    long_half_y, each `wall` thick and `height` tall (half-heights, as
    MuJoCo sizes boxes). The kitting tray's wells and the SO-101 insert
    pocket are this, with their own offsets."""
    import mujoco  # noqa: PLC0415 - sim extra

    cx, cy = centre
    (ox, oy), (lx, ly) = offset, long_half
    for label, dx, dy, sx, sy in (
        ("north", 0.0, oy, lx, wall),
        ("south", 0.0, -oy, lx, wall),
        ("east", ox, 0.0, wall, ly),
        ("west", -ox, 0.0, wall, ly),
    ):
        scene.worldbody.add_geom(
            name=f"{prefix}_{label}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[sx, sy, height],
            pos=[cx + dx, cy + dy, height],
            rgba=list(rgba),
        )


# Metres between world origins on a display grid (the ALOHA rig is
# 1.22 m wide). Declared twice in two tools until 2026-09-01 — the
# duplication docs/32 flagged and this function retires.
GRID_PITCH = 1.6


# The sky every MuJoCo scene wears (Menagerie's scene.xml values): a
# gradient skybox, so a display's background is not black. Added to a
# stage that brings none of its own.
@dataclass(frozen=True)
class Sky:
    rgb1: tuple[float, float, float] = (0.3, 0.5, 0.7)
    rgb2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: int = 512
    height: int = 3072


SKY = Sky()


def add_sky(scene: Any) -> None:
    """A gradient skybox on a spec without one."""
    import mujoco  # noqa: PLC0415 - sim extra

    if any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in scene.textures):
        return
    scene.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=list(SKY.rgb1),
        rgb2=list(SKY.rgb2),
        width=SKY.width,
        height=SKY.height,
    )


def grid_of(
    name: str,
    children: Any,
    *,
    pitch: float = GRID_PITCH,
    disable_floor_contacts: bool = False,
    stage: Any = None,
) -> tuple[Any, list[str]]:
    """One MjSpec holding every child spec on a centred √n grid.

    The shared skeleton of every many-worlds display model: one
    directional light, one dark ground plane, each child's own floor
    hidden (it would coincide with the ground) and the whole child
    attached under a `wNN/` prefix at its grid cell. Options, render
    budgets and shadow sizes are the CALLER's to set on the returned
    scene. Returns (scene spec, prefixes).

    `stage`: a spec to build ON instead of the plain light and plane -
    a task's own terrain and dressing as its framework composed them
    (the walk view hands the render process mjlab's, 2026-09-12). The
    children are attached into it; it gets a sky if it has none.
    """
    import math  # noqa: PLC0415

    import mujoco  # noqa: PLC0415 - sim extra

    children = list(children)
    scene = stage if stage is not None else mujoco.MjSpec()
    scene.modelname = name
    if stage is None:
        scene.worldbody.add_light(
            pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        )
        scene.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.1],
            pos=[0, 0, -0.75],
            rgba=[0.12, 0.12, 0.12, 1],
        )
    add_sky(scene)
    side = math.isqrt(len(children) - 1) + 1 if children else 0
    prefixes = []
    for n, child in enumerate(children):
        for geom in child.geoms:
            if geom.name == FLOOR_GEOM:
                geom.group = GeomGroup.HIDDEN
                if disable_floor_contacts:
                    geom.contype = 0
                    geom.conaffinity = 0
        row, col = divmod(n, side)
        prefix = f"w{n:02d}/"
        frame = scene.worldbody.add_frame(
            pos=[(col - (side - 1) / 2) * pitch, (row - (side - 1) / 2) * pitch, 0]
        )
        scene.attach(child, prefix=prefix, frame=frame)
        prefixes.append(prefix)
    return scene, prefixes
