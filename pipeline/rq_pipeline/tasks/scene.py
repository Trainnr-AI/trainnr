"""Scene facts every builder shares — spelled once (docs/24's rule).

Until 2026-08-26 the offscreen framebuffer size, the shadow-map budget,
the contact options MuJoCo drops on `attach`, and the geom groups tools
hide were retyped per builder and per tool. Now: the render budget and
the nominal solver condition as structs here, MuJoCo's option enums by
short name, and the helpers that write them onto an `MjSpec` or a
compiled model.
"""

from __future__ import annotations

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
