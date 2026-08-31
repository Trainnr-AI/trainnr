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
from mjlab.managers.event_manager import EventTermCfg
from rq_pipeline.robot.actuator_bundle import declared_ranges

from rq_mjlab.actuator import BamActuator
from rq_mjlab.bundle import verified_bundle

Ranges = dict[str, tuple[float, float]]


def dr_from_bundle(
    bundle_path: str | Path, *, fallback_span: float | None = None
) -> tuple[Ranges, str]:
    """The bundle's sampling region and its basis, verified first."""
    bundle, _advisories = verified_bundle(bundle_path)
    return declared_ranges(bundle, fallback_span=fallback_span)


def bam_param_dr_event(
    actuator: BamActuator,
    bundle_path: str | Path,
    *,
    fallback_span: float | None = None,
    mode: str = "reset",
) -> tuple[EventTermCfg, str]:
    """An event term that redraws the actuator's law parameters per
    world from the bundle's declared region — and the BASIS string the
    caller must put on the run's record (a datasheet, an artefact),
    because a draw whose provenance is not written down is a guess with
    better manners.

    Only the parameters the law actually consumes are drawn
    (`BamActuator.SCALABLE`); the region's other entries — the passives,
    `q_offset`, `command_delay` — are for mjlab's native events and the
    cfg, and are ignored here by name.
    """
    ranges, basis = dr_from_bundle(bundle_path, fallback_span=fallback_span)
    drawable = {
        name: bounds for name, bounds in ranges.items() if name in BamActuator.SCALABLE
    }
    if not drawable:
        raise ValueError(
            f"the bundle's region names no law parameter the actuator can "
            f"draw ({sorted(ranges)}); scalable: {BamActuator.SCALABLE}"
        )

    def redraw(env: Any, env_ids: torch.Tensor) -> None:
        count = int(env_ids.shape[0]) if hasattr(env_ids, "shape") else len(env_ids)
        draws = {}
        for name, (low, high) in drawable.items():
            u = torch.rand(count, device=env.device if hasattr(env, "device") else None)
            draws[name] = low + (high - low) * u
        actuator.set_param_draws(env_ids, draws)

    redraw.__name__ = "redraw_bam_params"
    return EventTermCfg(func=redraw, mode=mode), basis
