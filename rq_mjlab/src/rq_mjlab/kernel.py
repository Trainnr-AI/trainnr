"""BAM's servo law as batched torch — the ~30-line kernel, ported once.

The equations are Rhoban's BAM (Duclusaud & Passault, ICRA 2025,
Apache-2.0), the same transcription discipline as the pipeline's CPU
law: the CPU reference (*bam/model.py*), never the mjlab kernel BAM
itself ships — the two disagree on the quadratic term's sign gate, and
the fits were made through the reference. Every symbol defined here
once, shapes `(num_envs, n)` throughout:

- `q`, `qd`: joint position and velocity; `target` the (possibly
  delayed) commanded position. The firmware P law:
  `duty = clip((target - q) * kp * error_gain * error_gain_ratio,
  +-max_pwm)`, `V = vin * duty`; the XL330 applies its current limiter
  as a duty window `[(kt*qd - R*I_max)/vin, (kt*qd + R*I_max)/vin]`
  before the pwm clip.
- The DC motor: `tau = kt*V/R - kt^2*qd/R` (back-EMF).
- The friction budget (M6; lower tiers are this with absent terms
  zero): `s = exp(-(|qd|/dtheta_stribeck)^alpha)`,
  `g = |tau_ext*lfe - tau_prev*lfm|`,
  `g_s = |tau_ext*lfes - tau_prev*lfms|`,
  `Q = [sign(tau_ext) != sign(tau_prev)] * ([|tau_ext| < |tau_prev|]
  * lfeq * tau_ext^2 + [|tau_ext| > |tau_prev|] * lfmq * tau_prev^2)`,
  `frictionloss = friction_base + g + s*(friction_stribeck + g_s + Q)`.
  `tau_prev` is the PREVIOUS step's actuator torque and `tau_ext` the
  external load with last step's solver friction stripped out, so the
  budget never feeds on itself.

Pure functions over tensors; no mjlab, no warp — the actuator layer
owns the plumbing, this module owns the physics, and the parity test
pins it against BAM's own model where the `bam` extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LawParams:
    """One actuator group's constants: the bundle's identified block plus
    the firmware register values the deployment declares. Absent friction
    terms are zero, which is how M1..M6 share one formula."""

    kt: float
    R: float
    friction_base: float
    friction_stribeck: float = 0.0
    load_friction_motor: float = 0.0
    load_friction_external: float = 0.0
    load_friction_motor_stribeck: float = 0.0
    load_friction_external_stribeck: float = 0.0
    load_friction_motor_quad: float = 0.0
    load_friction_external_quad: float = 0.0
    dtheta_stribeck: float = 1.0
    alpha: float = 1.0
    error_gain_ratio: float = 1.0
    vin: float = 7.5
    kp: float = 400.0
    error_gain: float = 1.0
    max_pwm: float = 1.0
    max_current: float = 0.0  # 0: no firmware current limiter

    @property
    def stall_torque(self) -> float:
        """`vin*kt/R`: the motor's force range, the <motor>'s clamp."""
        return self.vin * self.kt / self.R


def duty(law: LawParams, q: torch.Tensor, qd: torch.Tensor, target: torch.Tensor):
    """The firmware P law's duty cycle in [-max_pwm, max_pwm]."""
    d = (target - q) * law.kp * law.error_gain * law.error_gain_ratio
    if law.max_current > 0:
        centre = law.kt * qd / law.vin
        window = law.R * law.max_current / law.vin
        d = torch.clamp(d, centre - window, centre + window)
    return torch.clamp(d, -law.max_pwm, law.max_pwm)


def torque(law: LawParams, qd: torch.Tensor, duty_cycle: torch.Tensor):
    """The DC motor's torque from the applied voltage, back-EMF included."""
    return law.kt * law.vin * duty_cycle / law.R - law.kt**2 * qd / law.R


def friction_budget(
    law: LawParams,
    tau_prev: torch.Tensor,
    tau_ext: torch.Tensor,
    qd: torch.Tensor,
) -> torch.Tensor:
    """The whole `dof_frictionloss` the solver should apply this step:
    `friction_base` plus the state-dependent budget. The sign gate on the
    quadratic term is the CPU reference's (*bam/model.py* 172-192);
    BAM's own mjlab kernel drops it, and the fits ran through the
    reference — we follow the fits. Pinned against the pipeline's numpy
    transcription by `tests/test_kernel_parity.py` across the sign/tie
    grid (the two copies disagreed until 2026-09-01's review)."""
    stribeck = torch.exp(-((qd.abs() / law.dtheta_stribeck) ** law.alpha))
    gearbox = (
        tau_ext * law.load_friction_external - tau_prev * law.load_friction_motor
    ).abs()
    gearbox_stribeck = (
        tau_ext * law.load_friction_external_stribeck
        - tau_prev * law.load_friction_motor_stribeck
    ).abs()
    opposing = torch.sign(tau_ext) != torch.sign(tau_prev)
    quadratic = opposing * (
        (tau_ext.abs() < tau_prev.abs()) * law.load_friction_external_quad * tau_ext**2
        + (tau_ext.abs() > tau_prev.abs()) * law.load_friction_motor_quad * tau_prev**2
    )
    return (
        law.friction_base
        + gearbox
        + stribeck * (law.friction_stribeck + gearbox_stribeck + quadratic)
    )


def external_torque(
    qfrc_bias: torch.Tensor,
    qfrc_constraint: torch.Tensor,
    qfrc_friction: torch.Tensor,
) -> torch.Tensor:
    """`tau_ext`: gravity/Coriolis and constraints, minus the friction the
    solver itself applied last step (else the budget feeds on itself)."""
    return -qfrc_bias + qfrc_constraint - qfrc_friction
