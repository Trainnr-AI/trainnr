"""Our deploy manifest as Unitree's `deploy.yaml`: the file their C++
controller (unitree_rl_mjlab `deploy/`) reads to run a policy against
their DDS simulator and the real robot.

Their file is our manifest reordered - the manifest was shaped on its
field set (docs/77 §6) - with their names for the observation terms.
Their runtime implements a fixed set of terms (`REGISTER_OBSERVATION`
in their `deploy/include`), so a manifest whose policy observes
something else is REFUSED here, by name: the writer is the
compatibility check (the first certified Go2 policy observed the base
linear velocity, a quantity the robot cannot measure, 2026-09-11).

Terms are matched on the manifest's declared `source` - what the
runtime computes - never on a term's name, so a term called `phase`
that is not the gait clock is refused, not shipped as one.
`UNITREE_TERMS` is also the one truth a deployable actor is built from
(`trainnr_mjlab.go2_walk.deployable_actor`): their order, their names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainnr.deploy.manifest import (
    SOURCE_COMMAND_TWIST,
    SOURCE_GAIT_PHASE,
    SOURCE_IMU_ANG_VEL,
    SOURCE_JOINT_POS_REL,
    SOURCE_JOINT_VEL_REL,
    SOURCE_LAST_ACTION,
    SOURCE_PROJECTED_GRAVITY,
    Manifest,
    Observation,
)

UNITREE_DEPLOY_FILE = "deploy.yaml"
COMMAND_NAME = "base_velocity"  # their name for the twist command
ACTION_TERM = "JointPositionAction"  # their one action term
STEP_DT_DIGITS = 6


@dataclass(frozen=True)
class UnitreeTerm:
    """One observation their runtime implements: the manifest source it
    answers, their registered name, the params their term takes from
    ours, and the term's name in mjlab's velocity task (what the
    exporter's `_source_of` resolves to this source)."""

    source: str
    theirs: str
    params: tuple[str, ...]
    mjlab_term: str


# Their deploy runtime's terms, in the order their shipped Go2
# `deploy.yaml` lists them (read 2026-09-11).
UNITREE_TERMS: tuple[UnitreeTerm, ...] = (
    UnitreeTerm(SOURCE_IMU_ANG_VEL, "base_ang_vel", (), "base_ang_vel"),
    UnitreeTerm(SOURCE_PROJECTED_GRAVITY, "projected_gravity", (), "projected_gravity"),
    UnitreeTerm(SOURCE_COMMAND_TWIST, "velocity_commands", (), "command"),
    UnitreeTerm(SOURCE_GAIT_PHASE, "gait_phase", ("period",), "phase"),
    UnitreeTerm(SOURCE_JOINT_POS_REL, "joint_pos_rel", (), "joint_pos"),
    UnitreeTerm(SOURCE_JOINT_VEL_REL, "joint_vel_rel", (), "joint_vel"),
    UnitreeTerm(SOURCE_LAST_ACTION, "last_action", (), "actions"),
)
TERM_BY_SOURCE = {t.source: t for t in UNITREE_TERMS}


def deployable_actor_terms() -> tuple[str, ...]:
    """The mjlab term names an actor must observe, in their order, to be
    deployable through their stack."""
    return tuple(t.mjlab_term for t in UNITREE_TERMS)


class NotDeployableError(ValueError):
    """The policy observes something their runtime cannot provide."""


def _their_term(term: Observation) -> UnitreeTerm:
    theirs = TERM_BY_SOURCE.get(term.source)
    if theirs is None:
        raise NotDeployableError(
            f"observation {term.name!r} (source {term.source!r}) has no term in "
            f"Unitree's deploy runtime (it implements "
            f"{', '.join(t.theirs for t in UNITREE_TERMS)}); train the actor on "
            "what the robot measures (go2_walk.deployable_actor)"
        )
    missing = [k for k in theirs.params if term.params.get(k) is None]
    if missing:
        raise NotDeployableError(
            f"observation {term.name!r} lacks the parameter(s) their "
            f"{theirs.theirs!r} takes: {missing}"
        )
    return theirs


def unitree_observations(manifest: Manifest) -> dict[str, dict[str, Any]]:
    """The manifest's observation list in their order-preserving form;
    refuses a term their runtime does not implement."""
    out: dict[str, dict[str, Any]] = {}
    for term in manifest.observations:
        theirs = _their_term(term)
        params: dict[str, Any] = {k: term.params[k] for k in theirs.params}
        if theirs.source == SOURCE_COMMAND_TWIST:
            params = {"command_name": COMMAND_NAME}
        scale = term.scale
        out[theirs.theirs] = {
            "params": params,
            "clip": list(term.clip) if term.clip is not None else None,
            "scale": (
                [float(s) for s in scale]
                if isinstance(scale, list)
                else [float(scale)] * term.width
            ),
            "history_length": their_history(term.history_length),
        }
    return out


def their_history(history_length: int) -> int:
    """mjlab counts PAST frames (0 = the current frame only); their
    observation manager counts frames KEPT, the current one included,
    and a term with 0 keeps nothing - its buffer empties, the policy
    reads an empty observation, and the robot froze in the fixed stand
    through 20 trials of the DDS gate (2026-09-12). Their reference
    file writes 1 for no history."""
    return max(1, int(history_length))


def unitree_deploy(manifest: Manifest) -> dict[str, Any]:
    """The whole file as a mapping, ready for YAML."""
    joints = manifest.joints
    action = manifest.action
    if joints.sdk_order_map is None:
        raise NotDeployableError(
            f"{manifest.root}: the manifest carries no SDK joint order for this "
            "robot; their controller needs joint_ids_map"
        )
    twist = manifest.commands.twist
    return {
        "joint_ids_map": list(joints.sdk_order_map),
        "step_dt": round(manifest.control.step_dt, STEP_DT_DIGITS),
        "stiffness": list(joints.stiffness),
        "damping": list(joints.damping),
        "default_joint_pos": list(joints.default_pos),
        "commands": {
            COMMAND_NAME: {
                "ranges": {
                    k: (list(v) if v is not None else None) for k, v in twist.items()
                }
            }
        },
        "actions": {
            ACTION_TERM: {
                "clip": list(action.clip) if action.clip is not None else None,
                "joint_names": [".*"],
                "scale": list(action.scale),
                "offset": list(action.offset),
                "joint_ids": None,
            }
        },
        "observations": unitree_observations(manifest),
    }


def write_unitree_deploy(manifest: Manifest, out: Path) -> Path:
    import yaml  # noqa: PLC0415

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(unitree_deploy(manifest), sort_keys=False), encoding="utf-8"
    )
    return out
