"""A walk trained under a measured fit (docs/70 §11 item 1, 2026-09-25):
the robot's joints set at the fit's estimates and randomized over its
intervals, instead of declared constants and a declared span.

A fit record (`rq_pipeline.robot.fit_record`) holds, per hinge, the
armature, viscous damping and Coulomb friction a recording identified,
each with a bootstrap half-width and a pinned verdict. Here:

- the training MJCF's joints carry the estimates (`apply_to_spec`), so
  the nominal world IS the measured robot;
- reset-time randomization draws each PINNED parameter uniformly over
  its interval, estimate ± half-width (floored at zero: none of the
  three can be negative), and each NOT PINNED parameter over a declared
  fallback span around its estimate (`NOT_PINNED`), said in the basis
  text so nobody mistakes it for a measurement;
- the run's identity names the fit's own stamp and its basis
  (`Identity.FIT`, `Identity.FIT_BASIS`), which the certificate and the
  deployment manifest carry on.

One truth for every walk: the Go2 wires it today; a walk whose joints
no fit describes refuses `fit=` by name (`rq_mjlab.walks`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from rq_pipeline.robot.fit_record import FitRecord, find_fit, fit_stamp

# The three joint terms a legged fit identifies, as the record names
# them (`<joint>.<term>`), each with the MjSpec joint attribute that
# carries it and the mjlab event that randomizes its compiled field.
TERMS: dict[str, tuple[str, Any]] = {
    "armature": ("armature", envs_mdp.dr.joint_armature),
    "damping": ("damping", envs_mdp.dr.joint_damping),
    "frictionloss": ("frictionloss", envs_mdp.dr.joint_friction),
}
EVENT_PREFIX = "fit_"  # the events' names: fit_armature, fit_damping, ...
FLOOR = 0.0  # armature, damping and friction are never negative


@dataclass(frozen=True)
class NotPinned:
    """The span around an estimate the bench did NOT pin: declared, ours,
    wide on purpose - a relative span, so a parameter the fit could not
    resolve still trains the policy against being wrong about it."""

    relative_span: float = 0.5
    word: str = "not pinned"


NOT_PINNED = NotPinned()


@dataclass(frozen=True)
class JointTerm:
    joint: str
    term: str
    estimate: float
    low: float
    high: float
    pinned: bool


def terms_of(record: FitRecord) -> tuple[JointTerm, ...]:
    """Every (joint, term) the fit names, with the range training draws
    from; a parameter name the record spells another way is refused."""
    out = []
    for parameter in record.parameters:
        joint, _, term = parameter.name.rpartition(".")
        if not joint or term not in TERMS:
            raise ValueError(
                f"fit parameter {parameter.name!r} is not <joint>.<term> with a "
                f"term in {sorted(TERMS)}"
            )
        estimate = max(float(parameter.estimate), FLOOR)
        if parameter.pinned and parameter.half_width != float("inf"):
            low = max(estimate - float(parameter.half_width), FLOOR)
            high = estimate + float(parameter.half_width)
        else:
            low = max(estimate * (1.0 - NOT_PINNED.relative_span), FLOOR)
            high = estimate * (1.0 + NOT_PINNED.relative_span)
        out.append(JointTerm(joint, term, estimate, low, high, bool(parameter.pinned)))
    return tuple(out)


def apply_to_spec(spec: Any, record: FitRecord) -> Any:
    """The training model's joints at the fit's estimates; a joint the
    fit names that the model lacks is refused by name (a fit of another
    robot, or names the vendor spells differently)."""
    joints = {j.name: j for j in spec.joints}
    for jt in terms_of(record):
        if jt.joint not in joints:
            raise ValueError(
                f"the fit names joint {jt.joint!r}; the model has "
                f"{sorted(n for n in joints if n)}"
            )
        _set(joints[jt.joint], TERMS[jt.term][0], jt.estimate)
    return spec


def _set(joint: Any, attribute: str, value: float) -> None:
    """A joint attribute at `value`. MuJoCo 3.13's spec holds `damping` as
    a short array whose first entry is the linear (viscous) coefficient -
    the one a fit measures - and the rest higher orders, kept at zero."""
    current = getattr(joint, attribute)
    if hasattr(current, "__len__"):
        array = list(current)
        array[0] = value
        setattr(joint, attribute, array)
    else:
        setattr(joint, attribute, value)


def dr_events(record: FitRecord) -> dict[str, EventTermCfg]:
    """One reset-time event per term: every joint drawn uniformly over its
    own range (`terms_of`), absolute values, per world."""
    ranges: dict[str, dict[str, tuple[float, float]]] = {t: {} for t in TERMS}
    for jt in terms_of(record):
        ranges[jt.term][f"^{jt.joint}$"] = (jt.low, jt.high)
    return {
        f"{EVENT_PREFIX}{term}": EventTermCfg(
            func=TERMS[term][1],
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "abs",
                "ranges": by_joint,
            },
        )
        for term, by_joint in ranges.items()
        if by_joint
    }


def basis_text(record: FitRecord) -> str:
    """The run's randomization basis in words: which fit, whose robot,
    how many terms drawn over their intervals and how many over the
    declared not-pinned span."""
    terms = terms_of(record)
    pinned = sum(jt.pinned for jt in terms)
    return (
        f"fit {fit_stamp(record)} ({record.basis or 'basis unrecorded'}, "
        f"recording {record.recording}): {pinned}/{len(terms)} joint terms drawn "
        f"over their bootstrap intervals, {len(terms) - pinned} "
        f"{NOT_PINNED.word} over ±{NOT_PINNED.relative_span:g} of the estimate"
    )


def resolve(bundle_dir: Path, wanted: str) -> FitRecord:
    """The fit a run names, from the robot's own bundle."""
    return find_fit(bundle_dir, wanted)
