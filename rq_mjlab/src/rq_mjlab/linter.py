"""The DR no-op linter: silent randomisation made a refusal.

Two of five no-op domain-randomisation terms in one production repo
randomised fields the BAM actuator overwrites every physics step — the
draw happened, the physics never changed, nothing raised
(docs/e2e-research/57 §5; the other three were no-ops by other
mechanisms, which this linter does not claim to catch). The rule belongs
at env-cfg time: any event term that writes a model field the actuator's
law owns is a no-op by construction, and a cfg carrying one is refused
with the term and the field named. The same pass refuses a cfg that
FORGOT the field-expansion event — the other half of the same trap.


`lint(events_cfg, actuator_cfgs)` walks the cfg's event terms the way
mjlab's own EventManager does (dataclass fields / mapping entries whose
value is an EventTermCfg) and reads the fields each term's function
declared through `@requires_model_fields` — the framework's own
marking, never a name heuristic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any

from mjlab.managers.event_manager import EventTermCfg

from rq_mjlab.actuator import BamActuatorCfg
from rq_mjlab.events import expand_bam_fields

# The model fields the BAM law writes every step: an event that
# randomises one of these changes nothing the solver ever sees.
OVERWRITTEN_FIELDS = frozenset({"dof_frictionloss"})


class SilentNoOp(ValueError):
    """A DR term whose write the actuator overwrites every step."""


def _event_terms(events_cfg: Any) -> dict[str, EventTermCfg]:
    if events_cfg is None:
        return {}
    items: Iterable[tuple[str, Any]]
    if isinstance(events_cfg, dict):
        items = events_cfg.items()
    elif is_dataclass(events_cfg):
        items = (
            (f.name, getattr(events_cfg, f.name)) for f in dataclass_fields(events_cfg)
        )
    else:
        items = vars(events_cfg).items()
    return {name: term for name, term in items if isinstance(term, EventTermCfg)}


def lint(events_cfg: Any, actuator_cfgs: tuple[Any, ...]) -> None:
    """Refuse the cfg's silent no-ops; pass silently when there are none.

    Raises `SilentNoOp` naming every offending (term, field) pair, and
    `RuntimeError` when a BAM actuator is configured but no event
    expands `dof_frictionloss` per world.
    """
    # A dict iterates its KEY STRINGS and a GENERATOR drains here, both
    # leaving the isinstance filter empty — lint would bless anything
    # (the entity seam's coercion family; reviews 2026-09-01, twice).
    from mjlab.actuator.actuator import ActuatorCfg  # noqa: PLC0415

    if not isinstance(actuator_cfgs, tuple):
        raise TypeError(
            f"lint takes an ordered tuple of actuator cfgs, got "
            f"{type(actuator_cfgs).__name__}"
        )
    bad = [
        type(cfg).__name__ for cfg in actuator_cfgs if not isinstance(cfg, ActuatorCfg)
    ]
    if bad:
        raise TypeError(
            f"lint takes an ordered tuple of actuator cfgs, got element(s) "
            f"{bad} — pass (cfg,) not a dict or names"
        )
    bam_groups = [cfg for cfg in actuator_cfgs if isinstance(cfg, BamActuatorCfg)]
    if not bam_groups:
        return
    terms = _event_terms(events_cfg)
    offences: list[str] = []
    expansion_present = False
    for name, term in terms.items():
        declared = tuple(getattr(term.func, "model_fields", ()))
        if term.func is expand_bam_fields:
            expansion_present = True
            continue
        for field_name in declared:
            if field_name in OVERWRITTEN_FIELDS:
                offences.append(
                    f"event {name!r} ({term.func.__module__}.{term.func.__name__}) "
                    f"randomises {field_name!r}, which the BAM actuator overwrites "
                    "every physics step - the draw would change nothing. Randomise "
                    "the bundle's parameters instead (dr_from_bundle) or drop the term."
                )
    if offences:
        raise SilentNoOp("\n".join(offences))
    if not expansion_present:
        raise RuntimeError(
            "a BAM actuator is configured but no event expands its fields: "
            + "add rq_mjlab.events.bam_expansion_event() to the env cfg's events"
        )
