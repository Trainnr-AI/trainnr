"""BAM's M1-M6 friction budget: one function, not six.

MuJoCo's native friction (`dof_frictionloss` + `dof_damping`) is
BAM's M1 — Coulomb-Viscous — and every richer model is M1 plus one
more optional term (docs/e2e-research/53's theory, citing BAM/ICRA
2025): M2 adds Stribeck (presliding softness near zero velocity), M3
adds an UNDIRECTED load-dependent term, M4 layers Stribeck onto M3,
M5 replaces the undirected load term with a DIRECTIONAL motor/
external split (captures backdrivability asymmetry), M6 adds a
quadratic term on top of M5. The six published model files therefore
differ only in WHICH fields are present — this module reads that
directly off `FrictionParams`: `None` means the term is absent, so
one function computes M1 through M6 depending on what was identified.

The result is a torque BUDGET, not a torque: BAM's own framing
(docs/e2e-research/53 §1) is a maximum resistive torque the caller
clips a stopping torque into, `clip(tau_stop, -budget, budget)` — that
clip is the caller's job (it needs the timestep and the unclipped
dynamics), not this module's.

Pure numpy, lazily imported (the `sim` extra, not the core package):
every operation here (`abs`, `exp`, `where`) has an identical
`jax.numpy` spelling, so this reads unchanged from either physics
instrument — no engine import, no framework dependency beyond numpy
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrictionParams:
    """One actuator's identified friction budget, BAM's M1-M6 schema.

    `friction_viscous` (Kv) and `friction_base` (Kc) are the only
    required fields — every model has them. Everything else is
    `None` when that term was not fit, and `friction_budget` treats
    absence as exactly zero contribution, never a silent guess.
    """

    friction_viscous: float
    friction_base: float
    # Stribeck (M2, M4, M5, M6): present together or not at all.
    friction_stribeck: float | None = None
    dtheta_stribeck: float | None = None
    alpha: float | None = None
    # Undirected load-dependent (M3, M4) — mutually exclusive with the
    # directional pair below; BAM never fits both on one actuator.
    load_friction_base: float | None = None
    load_friction_stribeck: float | None = None
    # Directional load-dependent (M5, M6): motor-side vs external-side.
    load_friction_motor: float | None = None
    load_friction_external: float | None = None
    load_friction_motor_stribeck: float | None = None
    load_friction_external_stribeck: float | None = None
    # Quadratic (M6 only).
    load_friction_motor_quad: float | None = None
    load_friction_external_quad: float | None = None

    @classmethod
    def from_json(cls, fields: dict[str, Any]) -> FrictionParams:
        """Build from a BAM-schema dict, ignoring servo-model fields
        (`kt`, `R`, `armature`, ...) and bookkeeping (`model`,
        `actuator`) that belong to `actuator_library.ServoParams`."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in fields.items() if k in known})


def friction_torque_budget(
    params: FrictionParams, theta_dot: Any, tau_m: Any, tau_e: Any
) -> Any:
    """The maximum resistive torque BAM's identified model allows,
    at the given velocity and motor/external torque split.

    Broadcasts over numpy or jax arrays; `theta_dot`, `tau_m`, `tau_e`
    must already share a shape. Directly reproduces the M1-M6
    equations of docs/e2e-research/53 (BAM's own theory/models.rst):
    M1 is the first line alone; each later model is this function fed
    a `FrictionParams` with one more group of fields populated.
    """
    import numpy as np  # noqa: PLC0415 - lazy: numpy is not a core dependency here

    budget = params.friction_viscous * np.abs(theta_dot) + params.friction_base

    if params.load_friction_base is not None:  # M3, M4: undirected load
        budget = budget + params.load_friction_base * np.abs(tau_m - tau_e)
    elif params.load_friction_motor is not None:  # M5, M6: directional load
        budget = budget + np.abs(
            params.load_friction_motor * tau_m - params.load_friction_external * tau_e
        )

    if params.friction_stribeck is not None:  # M2, M4, M5, M6
        envelope = np.exp(-(np.abs(theta_dot / params.dtheta_stribeck) ** params.alpha))
        stribeck_term = params.friction_stribeck
        if params.load_friction_stribeck is not None:  # M4
            stribeck_term = stribeck_term + params.load_friction_stribeck * np.abs(
                tau_m - tau_e
            )
        elif params.load_friction_motor_stribeck is not None:  # M5, M6
            stribeck_term = stribeck_term + np.abs(
                params.load_friction_motor_stribeck * tau_m
                - params.load_friction_external_stribeck * tau_e
            )
        if params.load_friction_motor_quad is not None:  # M6
            stribeck_term = stribeck_term + np.where(
                np.abs(tau_m) > np.abs(tau_e),
                params.load_friction_external_quad * tau_e**2,
                params.load_friction_motor_quad * tau_m**2,
            )
        budget = budget + envelope * stribeck_term

    return budget
