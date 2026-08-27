"""Start validation on the simulator's own geometry (docs/e2e-research/42
§3, adopted: the validators, not the solver).

A task declares where each free body must start (`protocol.Placement`);
this module checks a candidate start against the compiled model and says
which check failed, for which body, with the numbers. Cheap, geometric
and exact: the body's world AABB comes from MuJoCo's own per-geom AABB
(`geom_aabb`) rotated into the world, penetration from the contacts
`mj_forward` generates. Nothing here samples or optimises — the start is
the protocol's `perturb`; this is the gate it must pass. Arena stores the
best failed layout and runs anyway; a required failure here is refused
with the culprit named, before a trial is spent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rq_pipeline.protocol import Placement, PlacementChecks

Verdicts = Mapping[str, Mapping[str, bool]]  # body -> check -> passed

AABB_CENTRE = slice(0, 3)  # mjModel.geom_aabb rows: centre then half-sizes
AABB_HALF = slice(3, 6)
_VERTICAL = 2


class PlacementRefusedError(ValueError):
    """A trial's start failed a required check: trial, body, checks, numbers."""


def _world_aabb(model: Any, data: Any, geom: int) -> tuple[Any, Any]:
    """The geom's axis-aligned box in the world: centre and half-sizes.
    The local AABB rotated by |R| is the tight world box of the OBB."""
    import numpy as np  # noqa: PLC0415

    centre = model.geom_aabb[geom, AABB_CENTRE]
    half = model.geom_aabb[geom, AABB_HALF]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    world_centre = data.geom_xpos[geom] + rotation @ centre
    return world_centre, np.abs(rotation) @ half


def _body_geoms(model: Any, body: int) -> list[int]:
    start = int(model.body_geomadr[body])
    return list(range(start, start + int(model.body_geomnum[body])))


def _named(model: Any, kind: Any, name: str, what: str) -> int:
    import mujoco  # noqa: PLC0415 - sim extra

    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise KeyError(f"placement names {what} {name!r}, which the model lacks")
    return index


def start_verdicts(model: Any, state: Any, placements: Sequence[Placement]) -> Verdicts:
    """Every declared body's verdicts at `state` (a FULLPHYSICS row), on
    a scratch MjData: forward the pose, read the boxes and the contacts."""
    import mujoco  # noqa: PLC0415 - sim extra
    import numpy as np  # noqa: PLC0415

    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model, data, np.asarray(state, dtype=float), mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    mujoco.mj_forward(model, data)
    penetrating = [
        (int(contact.geom1), int(contact.geom2))
        for contact in (data.contact[i] for i in range(data.ncon))
        if contact.dist < 0
    ]
    verdicts: dict[str, dict[str, bool]] = {}
    for placement in placements:
        body = _named(model, mujoco.mjtObj.mjOBJ_BODY, placement.body, "body")
        support = _named(
            model, mujoco.mjtObj.mjOBJ_GEOM, placement.support, "support geom"
        )
        geoms = _body_geoms(model, body)
        if not geoms:
            raise KeyError(f"placement body {placement.body!r} has no geoms")
        boxes = [_world_aabb(model, data, geom) for geom in geoms]
        low = np.min([centre - half for centre, half in boxes], axis=0)
        high = np.max([centre + half for centre, half in boxes], axis=0)
        centre = (low + high) / 2
        support_centre, support_half = _world_aabb(model, data, support)
        support_low, support_high = (
            support_centre - support_half,
            support_centre + support_half,
        )
        in_limits = all(
            band is None or band[0] <= centre[axis] <= band[1]
            for axis, band in enumerate((placement.x, placement.y))
        )
        footprint_inside = bool(
            np.all(low[:_VERTICAL] >= support_low[:_VERTICAL])
            and np.all(high[:_VERTICAL] <= support_high[:_VERTICAL])
        )
        gap = float(low[_VERTICAL] - support_high[_VERTICAL])
        on_support = footprint_inside and abs(gap) <= placement.tolerance_m
        no_overlap = not any(
            (a in geoms or b in geoms) and support not in (a, b) for a, b in penetrating
        )
        verdicts[placement.body] = {
            PlacementChecks.IN_LIMITS: bool(in_limits),
            PlacementChecks.ON_SUPPORT: bool(on_support),
            PlacementChecks.NO_OVERLAP: bool(no_overlap),
        }
    return verdicts


def require_start(
    model: Any, state: Any, placements: Sequence[Placement], *, trial: int
) -> Verdicts:
    """The gate: verdicts for `state`, or `PlacementRefusedError` naming the
    trial, every failing body and its failing checks."""
    verdicts = start_verdicts(model, state, placements)
    failures = {
        body: [check for check, passed in checks.items() if not passed]
        for body, checks in verdicts.items()
        if not all(checks.values())
    }
    if failures:
        detail = "; ".join(
            f"{body} fails {', '.join(checks)}" for body, checks in failures.items()
        )
        raise PlacementRefusedError(
            f"trial {trial}: the declared start is not admissible - {detail}. "
            "Nothing was run; fix the protocol's perturb or its placements."
        )
    return verdicts
