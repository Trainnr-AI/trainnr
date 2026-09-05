"""The walk C1 study on a robot Unitree ships: mjlab's own Go1 flat
velocity task, unchanged, with the study's one knob added — a span of
actuator randomization around mjlab's DERIVED gains.

mjlab models the Go1's motors as builtin PD actuators whose stiffness,
damping and armature are derived, not tuned: armature is the rotor
inertia reflected through the gear
(`mjlab.asset_zoo.robots.unitree_go1.go1_constants`, mjlab 1.6:
`reflected_inertia(ROTOR_INERTIA, gear)`), stiffness is
`armature * (2π·10 Hz)²` and damping `2·ζ·armature·2π·10 Hz` with
ζ = 2.0. That derivation is this study's "identified point" — the
analogue of the microduck's certified BAM fit (docs/e2e-research/65
§the convergence). Their velocity task randomizes pushes, foot
friction, encoder bias and base CoM, and NOTHING about the actuator
(`mjlab/tasks/velocity/velocity_env_cfg.py`, events; read 2026-09-04).

The knob: `dr_span` scales kp, kd and armature per reset by a draw
from [1 - span, 1 + span] through mjlab's own `pd_gains` and
`joint_armature` events — real draws, because the builtin PD actuator
writes its gains into the model at build and never rewrites them
(the class the linter refuses cannot occur here; `lint` is still
called so the fact is checked, not assumed). `pin_scale` collapses the
draw to one value: the envelope / mismatch probe. `None` or 0 means
NO actuator DR — the derived gains exactly, the world every arm is
judged in.

Identity, the three strings a run's record carries, mirrors the
microduck's: `robot` stamps the MJCF mjlab ships, `actuator` stamps
the derived constants (the numbers a Unitree engineer can recompute
by hand), `dr_basis` says which span or pin.
"""

from __future__ import annotations

import hashlib
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from rq_pipeline.bundles.hashing import fields_hash

from rq_mjlab.linter import lint

# The study's default span; the C1 arms are 0 / 0.10 / 0.30.
ACTUATOR_DR_SPAN = 0.10
# What the span scales: mjlab's builtin PD writes these into the model
# and never rewrites them (mjlab/actuator/builtin_actuator.py, 1.6).
SCALED = ("kp", "kd", "armature")
ROBOT = "unitree-go1"
ALL_ACTUATORS = SceneEntityCfg("robot")
ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))


def derived_actuator_constants() -> dict[str, Any]:
    """mjlab's Go1 actuator derivation, as numbers: the identified
    point this study randomizes around, and the fields its stamp hashes."""
    from mjlab.asset_zoo.robots.unitree_go1 import go1_constants as go1  # noqa: PLC0415

    return {
        "model": "mjlab BuiltinPositionActuator (PD baked into the MJCF at build)",
        "rotor_inertia_kgm2": go1.ROTOR_INERTIA,
        "gear_hip": go1.HIP_GEAR_RATIO,
        "gear_knee": go1.KNEE_GEAR_RATIO,
        "natural_frequency_rad_s": go1.NATURAL_FREQ,
        "damping_ratio": go1.DAMPING_RATIO,
        "hip": {
            "armature": go1.HIP_ACTUATOR.reflected_inertia,
            "stiffness": go1.STIFFNESS_HIP,
            "damping": go1.DAMPING_HIP,
            "effort_limit": go1.HIP_ACTUATOR.effort_limit,
        },
        "knee": {
            "armature": go1.KNEE_ACTUATOR.reflected_inertia,
            "stiffness": go1.STIFFNESS_KNEE,
            "damping": go1.DAMPING_KNEE,
            "effort_limit": go1.KNEE_ACTUATOR.effort_limit,
        },
        "source": "mjlab/asset_zoo/robots/unitree_go1/go1_constants.py (mjlab 1.6)",
    }


def robot_stamp() -> str:
    """The MJCF mjlab ships, content-hashed — a new mjlab that changes
    the model changes the stamp."""
    from mjlab.asset_zoo.robots.unitree_go1 import go1_constants as go1  # noqa: PLC0415

    digest = hashlib.sha256(go1.GO1_XML.read_bytes()).hexdigest()[:12]
    return f"{ROBOT}@{digest}"


def actuator_stamp() -> str:
    return f"{ROBOT}-derived-pd@{fields_hash(derived_actuator_constants())}"


PIN_AXES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "kp": ("kp",),
    "kd": ("kd",),
    "armature": ("armature",),
}


def actuator_dr_events(
    *,
    dr_span: float | None,
    pin_scale: float | None,
    pin_only: tuple[str, ...] | None = None,
) -> tuple[dict[str, EventTermCfg], str]:
    """The study's events and their basis string: a drawn span, a
    pinned scale (of every scaled parameter, or only `pin_only`), or
    nothing."""
    if pin_scale is not None:
        moved = set(SCALED if pin_only is None else pin_only)
        unknown = moved - set(SCALED)
        if unknown:
            raise ValueError(f"pin_only names {sorted(unknown)}; scaled: {SCALED}")
        lo = hi = float(pin_scale)
        which = ", ".join(n for n in SCALED if n in moved)
        basis = (
            f"pinned: {which} at derived x {pin_scale:g}, the rest at derived (no draw)"
        )
    elif dr_span:
        lo, hi = 1.0 - float(dr_span), 1.0 + float(dr_span)
        basis = f"declared ±{dr_span:g} scale on {', '.join(SCALED)} around derived"
    else:
        return {}, "none: mjlab's derived PD gains exactly (no actuator DR)"
    if pin_scale is None:
        moved = set(SCALED)
    unit = (1.0, 1.0)  # a parameter the pin leaves at derived
    events = {
        "actuator_gains": EventTermCfg(
            func=envs_mdp.dr.pd_gains,
            mode="reset",
            params={
                "asset_cfg": ALL_ACTUATORS,
                "kp_range": (lo, hi) if "kp" in moved else unit,
                "kd_range": (lo, hi) if "kd" in moved else unit,
                "operation": "scale",
            },
        ),
        "actuator_armature": EventTermCfg(
            func=envs_mdp.dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": ALL_JOINTS,
                "operation": "scale",
                "ranges": (lo, hi) if "armature" in moved else unit,
            },
        ),
    }
    return events, basis


def go1_walk_env_cfg(
    *,
    play: bool = False,
    dr_span: float | None = ACTUATOR_DR_SPAN,
    pin_scale: float | None = None,
    pin_only: tuple[str, ...] | None = None,
) -> tuple[ManagerBasedRlEnvCfg, dict[str, str]]:
    """mjlab's Go1 flat cfg, their events untouched, ours added; and
    the identity. `play` is their play mode (no corruption, gentler
    pushes) — judged runs use the training world, pushes on, as the
    microduck study does."""
    from mjlab.tasks.velocity.config.go1.env_cfgs import (  # noqa: PLC0415
        unitree_go1_flat_env_cfg,
    )

    cfg = unitree_go1_flat_env_cfg(play=play)
    events, dr_basis = actuator_dr_events(
        dr_span=dr_span, pin_scale=pin_scale, pin_only=pin_only
    )
    for name in events:
        if name in cfg.events:
            raise ValueError(f"mjlab's Go1 cfg already carries an event named {name!r}")
    cfg.events.update(events)
    # No BAM actuator here, so the linter has nothing to refuse — it is
    # still the gate, so that a future actuator swap is checked.
    lint(cfg.events, ())
    return cfg, {
        "robot": robot_stamp(),
        "actuator": actuator_stamp(),
        "dr_basis": dr_basis,
    }


def go1_agent(iterations: int) -> Any:
    """mjlab's own Go1 PPO runner cfg, iterations set by the study."""
    from dataclasses import replace  # noqa: PLC0415

    from mjlab.tasks.velocity.config.go1.rl_cfg import (  # noqa: PLC0415
        unitree_go1_ppo_runner_cfg,
    )

    # Their cfg logs to wandb (mjlab/rl/config.py default); the study's
    # runs log to tensorboard like the microduck's — the first Go1 smoke
    # died in rsl-rl's wandb writer (2026-09-04).
    return replace(
        unitree_go1_ppo_runner_cfg(), max_iterations=iterations, logger="tensorboard"
    )
