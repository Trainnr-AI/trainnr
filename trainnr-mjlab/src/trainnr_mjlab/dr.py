"""Domain randomisation from the bundle, with its basis on the record.

The ranges come from ONE place — `declared_ranges` in the pipeline's A1
module, the same function the data press draws episodes from: the
bundle's identified `uncertainty` intervals when it has them, else a
caller-DECLARED span around the point estimates, and a basis string
saying exactly which. A point-estimate bundle with no declared span is
a refusal, never a fabricated ±10 % — the microduck lesson
(docs/e2e-research/57 §5: fitted-treated-as-exact beside hand-guessed
spans, with nothing marking which).

Two sinks for one set of ranges:

- the LAW's own constants (kt, R, the friction budget's terms) go to
  the actuator per world through `bam_param_dr_event` — an mjlab event
  term whose draws land in `BamActuator.set_param_draws`;
- the PASSIVES (armature → `dof_armature`, friction_viscous →
  `dof_damping`) are model fields: wire mjlab's native `joint_armature`
  / `joint_damping` events with these same ranges — this module hands
  you the numbers, the framework already owns those writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from mjlab.managers.event_manager import EventMode, EventTermCfg
from trainnr.robot.actuator_bundle import declared_ranges, declared_span_ranges

from trainnr_mjlab.actuator import BamActuator, BamActuatorCfg
from trainnr_mjlab.bundle import verified_bundle

Ranges = dict[str, tuple[float, float]]


def dr_from_bundle(
    bundle_path: str | Path, *, fallback_span: float | None = None
) -> tuple[Ranges, str]:
    """The bundle's sampling region and its basis, verified first."""
    bundle, _advisories = verified_bundle(bundle_path)
    return declared_ranges(bundle, fallback_span=fallback_span)


def bam_param_dr_event(  # noqa: PLR0913 - every knob of the draw, named
    actuator_cfg: BamActuatorCfg,
    *,
    entity_name: str = "robot",
    fallback_span: float | None = None,
    mode: EventMode = "reset",
    pin_scale: float | None = None,
    pin_only: tuple[str, ...] | None = None,
) -> tuple[EventTermCfg, str]:
    """An event term that redraws the actuator's law parameters per
    world from ITS bundle's declared region — and the BASIS string the
    caller must put on the run's record (a datasheet, an artefact),
    because a draw whose provenance is not written down is a guess with
    better manners.

    Takes the CFG, not an instance, and reads the bundle path from it —
    one truth: DR from a different bundle than the actuator's is not
    expressible. The live `BamActuator` does not exist at cfg-build
    time (mjlab builds it later), so the event LOOKS IT UP at fire time
    on `env.scene[entity_name]`, matched by the cfg's stamp — the first
    wiring passed the cfg straight into `set_param_draws` and crashed
    at the first reset (review, 2026-09-01).

    Only the parameters the law actually consumes are drawn
    (`BamActuator.SCALABLE`); the region's other entries — the passives,
    `q_offset`, `command_delay` — are for mjlab's native events and the
    cfg, and are ignored here by name.

    `pin_scale` replaces the region with a DEGENERATE one — every law
    parameter at exactly fit x pin_scale, no draw — the envelope probe
    (the walk's mismatch matrix): judge a policy at a known
    offset from the measured fit and read where it fails. The basis
    says "pinned", never "identified" or "declared". `pin_only` names the
    law parameters the pin moves; the rest sit at the fit exactly — one
    axis of the mismatch space instead of its diagonal (the matrix of
    2026-09-04 moved every parameter together; a real fit error does
    not). Names must be in `BamActuator.SCALABLE`.
    """
    if not isinstance(actuator_cfg, BamActuatorCfg):
        raise TypeError(
            f"bam_param_dr_event takes the BamActuatorCfg, got "
            f"{type(actuator_cfg).__name__} — the live actuator does not "
            "exist yet at cfg-build time; the event finds it by stamp"
        )
    stamp = actuator_cfg.stamp
    # The bundle's POINT is its params — never `declared_ranges`, which
    # prefers a bundle's interval over any span and so, on a bundle that
    # carries one, handed the interval's LOW bounds to the pin and the
    # interval itself to a "declared" span (found 2026-09-06 on the
    # refit bundle: every pinned and spanned certificate of its arms was
    # judged in the wrong world and re-run).
    bundle, _advisories = verified_bundle(actuator_cfg.bundle_path)
    point = {
        name: (float(value), float(value))
        for name, value in bundle["params"].items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if pin_scale is not None:
        if pin_scale <= 0:
            raise ValueError(f"pin_scale must be positive, got {pin_scale}")
        if pin_only is not None:
            unknown = [n for n in pin_only if n not in BamActuator.SCALABLE]
            if unknown:
                raise ValueError(
                    f"pin_only names {unknown}; the law's scalable parameters are "
                    f"{BamActuator.SCALABLE}"
                )
            moved = set(pin_only)
        else:
            moved = set(point)
        ranges = {
            name: (low * pin_scale, low * pin_scale) if name in moved else (low, low)
            for name, (low, _high) in point.items()
        }
        which = "every law parameter" if pin_only is None else ", ".join(pin_only)
        basis = f"pinned: {which} at fit x {pin_scale:g}, the rest at the fit (no draw)"
    elif fallback_span is not None:
        # An explicit span is the caller's declaration, around the point,
        # whatever interval the bundle carries - the pipeline's one
        # spelling of that region and its basis (rig and firmware
        # parameters are not in it; the law cannot draw them either).
        ranges, basis = declared_span_ranges(bundle, fallback_span)
    else:
        ranges, basis = dr_from_bundle(
            actuator_cfg.bundle_path, fallback_span=fallback_span
        )
    # The identified SET, when the bundle carries its bootstrap replicates
    # (docs/e2e-research/72): draw whole parameter vectors, one replicate
    # per world, so the friction terms keep the trade-offs the bench
    # produced — independent marginal draws would combine values the data
    # never did. Only without a declared span or a pin.
    table: dict[str, torch.Tensor] = {}
    if fallback_span is None and pin_scale is None:
        samples = bundle.get("samples") or []
        if samples:
            names = [n for n in BamActuator.SCALABLE if n in samples[0]]
            table = {
                name: torch.tensor([float(s[name]) for s in samples]) for name in names
            }
            basis = (
                f"identified-set: joint draw from {len(samples)} bootstrap "
                f"replicates of {bundle.get('stamp', '<unstamped>')} "
                f"(marginal 95 % box on the bundle as `uncertainty`)"
            )
    drawable = {
        name: bounds for name, bounds in ranges.items() if name in BamActuator.SCALABLE
    }
    if not drawable:
        raise ValueError(
            f"the bundle's region names no law parameter the actuator can "
            f"draw ({sorted(ranges)}); scalable: {BamActuator.SCALABLE}"
        )

    def redraw(env: Any, env_ids: torch.Tensor) -> None:
        live = [
            act
            for act in env.scene[entity_name].actuators
            if isinstance(act, BamActuator) and act.cfg.stamp == stamp
        ]
        if len(live) != 1:
            raise RuntimeError(
                f"bam_param_dr: entity {entity_name!r} carries "
                f"{len(live)} BamActuator(s) with stamp {stamp!r} — "
                "the event must resolve exactly one"
            )
        count = int(env_ids.shape[0]) if hasattr(env_ids, "shape") else len(env_ids)
        device = env.device if hasattr(env, "device") else None
        draws = {}
        if table:
            picks = torch.randint(len(next(iter(table.values()))), (count,))
            for name, column in table.items():
                draws[name] = column[picks].to(device)
        else:
            for name, (low, high) in drawable.items():
                u = torch.rand(count, device=device)
                draws[name] = low + (high - low) * u
        live[0].set_param_draws(env_ids, draws)

    redraw.__name__ = "redraw_bam_params"
    return EventTermCfg(func=redraw, mode=mode), basis
