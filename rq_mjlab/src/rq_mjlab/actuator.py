"""BAM's servo law as an mjlab actuator, built only from certified bundles.

The lifecycle is mjlab 1.6's own three doors (`edit_spec` pre-compile,
`initialize` post-compile, `compute` per step); the physics is
`rq_mjlab.kernel` (the CPU-reference transcription); the constants come
from a VERIFIED bundle plus the firmware table — an unverifiable bundle
or an unknown firmware family is refused by name before any spec is
edited. The command delay is mjlab's native one (`delay_min_lag`/
`delay_max_lag`), set from the bundle's identified `command_delay`;
nothing is re-implemented that the framework now carries.

What `compute` does each physics step, shapes `(num_envs, n)`:
read `q`/`qd` from the command's current state, `tau_prev` from the
previous step's `qfrc_actuator`, `tau_ext` from `qfrc_bias`/
`qfrc_constraint` with the solver's own DOF-friction rows scattered out
of `efc` (so the budget never feeds on itself); write the whole
friction budget into the per-world `dof_frictionloss`; return the DC
motor's torque for the `<motor>` actuators `edit_spec` created. The
`<motor>`'s forcerange is the stall torque `vin*kt/R`.

Not ported yet, refused loudly: firmware families with a rate-limited
internal target (the Feetech STS line) — the rate limit is stateful
across steps and needs a reset story before it can be honest here. The
XL330 — the validation target (docs/e2e-research/58 §6) — has none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import torch
import warp as wp
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

from rq_mjlab.bundle import verified_bundle
from rq_mjlab.events import MISSING_EXPANSION_MESSAGE
from rq_mjlab.firmware import firmware_for
from rq_mjlab.kernel import LawParams, duty, external_torque, friction_budget, torque

# mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF — as a number so the module
# states it once; pinned against the enum by test.
FRICTION_DOF = 1

# The friction terms a bundle may carry beyond the base; absent means zero.
_OPTIONAL_FRICTION = (
    "friction_stribeck",
    "load_friction_motor",
    "load_friction_external",
    "load_friction_motor_stribeck",
    "load_friction_external_stribeck",
    "load_friction_motor_quad",
    "load_friction_external_quad",
)
_OPTIONAL_SHAPE = ("dtheta_stribeck", "alpha", "error_gain_ratio")


@dataclass(kw_only=True)
class BamActuatorCfg(ActuatorCfg):
    """A BAM-law actuator group from one verified bundle.

    Build through `from_bundle`, which fills the passive fields
    (`armature`, `frictionloss`, `viscous_damping`) from the bundle and
    the delay from its identified `command_delay` — hand-constructing
    this cfg with numbers is exactly the guessing the store exists to
    end. `kp` and `vin` are the deployment's register choices; None
    takes the firmware table's measured default, and the value used is
    recorded on the cfg either way.
    """

    bundle_path: str = ""
    stamp: str = ""  # the bundle's `<slug>-<model>@<hash>`; identity, not input
    # What the bundle could not promise (verify's advisories), carried so
    # a consumer reads them where the numbers are used.
    advisories: tuple[str, ...] = ()
    kp: float | None = None
    vin: float | None = None
    law: LawParams = field(
        default_factory=lambda: LawParams(kt=1.0, R=1.0, friction_base=0.0)
    )

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        target_names_expr: tuple[str, ...],
        physics_dt: float,
        kp: float | None = None,
        vin: float | None = None,
        **overrides: Any,
    ) -> BamActuatorCfg:
        bundle, advisories = verified_bundle(bundle_path)
        params = bundle["params"]
        firmware = firmware_for(params["actuator"])
        if firmware.max_velocity is not None:
            raise NotImplementedError(
                f"{params['actuator']}: this firmware rate-limits its internal "
                "target; the stateful limiter is not ported yet (docs/e2e-research/58 "
                "§6 validates on the XL330, which has none)"
            )
        law = LawParams(
            kt=params["kt"],
            R=params["R"],
            friction_base=params["friction_base"],
            **{k: params[k] for k in _OPTIONAL_FRICTION if k in params},
            **{k: params[k] for k in _OPTIONAL_SHAPE if k in params},
            vin=firmware.vin if vin is None else vin,
            kp=firmware.kp if kp is None else kp,
            error_gain=firmware.error_gain,
            max_pwm=firmware.max_pwm,
            max_current=firmware.max_current or 0.0,
        )
        delay_steps = math.ceil(round(params.get("command_delay", 0.0) / physics_dt, 9))
        return cls(
            target_names_expr=target_names_expr,
            bundle_path=str(bundle_path),
            stamp=bundle["stamp"],
            advisories=advisories,
            kp=law.kp,
            vin=law.vin,
            law=law,
            # The bundle's passives ride the XML through mjlab's own
            # override path; the kernel writes only the live budget.
            armature=params["armature"],
            frictionloss=params["friction_base"],
            viscous_damping=params["friction_viscous"],
            delay_min_lag=delay_steps,
            delay_max_lag=delay_steps,
            **overrides,
        )

    def build(self, *args: Any, **kwargs: Any) -> BamActuator:
        return BamActuator(self, *args, **kwargs)


class BamActuator(Actuator[BamActuatorCfg]):
    """The law at run time; see the module docstring for the per-step story."""

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        for target_name in target_names:
            self._mjs_actuators.append(
                create_motor_actuator(
                    spec,
                    target_name,
                    effort_limit=self.cfg.law.stall_torque,
                    armature=self.cfg.armature,
                    frictionloss=self.cfg.frictionloss,
                    viscous_damping=self.cfg.viscous_damping,
                    transmission_type=self.cfg.transmission_type,
                )
            )

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: Any,
        data: Any,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        joint_ids = [int(i) for i in self._target_ids_list]
        self._dof_ids = torch.tensor(
            [int(mj_model.jnt_dofadr[j]) for j in joint_ids],
            dtype=torch.long,
            device=device,
        )
        self._nv = int(mj_model.nv)
        # Torch views over the live warp arrays — zero-copy, per world.
        self._qfrc_actuator = wp.to_torch(data.qfrc_actuator)
        self._qfrc_bias = wp.to_torch(data.qfrc_bias)
        self._qfrc_constraint = wp.to_torch(data.qfrc_constraint)
        self._efc_id = wp.to_torch(data.efc.id)
        self._efc_type = wp.to_torch(data.efc.type)
        self._efc_force = wp.to_torch(data.efc.force)
        self._dof_frictionloss = wp.to_torch(model.dof_frictionloss)
        num_envs = int(self._qfrc_actuator.shape[0])
        if self._dof_frictionloss.shape[0] != num_envs:
            raise RuntimeError(MISSING_EXPANSION_MESSAGE)

    def _solver_friction(self) -> torch.Tensor:
        """Last solve's DOF-friction rows scattered onto the DOFs,
        `(num_envs, nv)` — padded efc rows carry zero force, so the
        masked scatter is exact."""
        rows = self._efc_type == FRICTION_DOF
        ids = torch.where(rows, self._efc_id, 0).long()
        force = torch.where(rows, self._efc_force, torch.zeros_like(self._efc_force))
        out = torch.zeros(
            (force.shape[0], self._nv), dtype=force.dtype, device=force.device
        )
        return out.scatter_add_(1, ids, force)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        law = self.cfg.law
        q, qd = cmd.pos, cmd.vel
        dofs = self._dof_ids
        tau_prev = self._qfrc_actuator[:, dofs]
        tau_ext = external_torque(
            self._qfrc_bias[:, dofs],
            self._qfrc_constraint[:, dofs],
            self._solver_friction()[:, dofs],
        )
        self._dof_frictionloss[:, dofs] = friction_budget(law, tau_prev, tau_ext, qd)
        return torque(law, qd, duty(law, q, qd, cmd.position_target))
