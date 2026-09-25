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
import mujoco_warp as mjwarp
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


def seconds_to_steps(seconds: float, physics_dt: float) -> int:
    """A delay in whole physics steps, CEIL not round: a delay that is not a
    whole number of steps becomes the next step UP, so the sim's command is
    never fresher than the delay it models (21 ms at dt=5 ms -> 25 ms).
    Direction stated because it is a choice (review 2026-09-01)."""
    return math.ceil(round(seconds / physics_dt, 9))


def as_torch(array: Any) -> torch.Tensor:
    """A torch view of a raw `wp.array` — everything else passes through
    untouched. Three shapes reach these seams: raw warp arrays (from
    `mjwarp.put_data`, the tests' harness), real torch tensors, and
    mjlab's `TorchArray` proxy ("behaves like a torch.Tensor with shared
    memory", *mjlab/sim/sim_data.py*) — warp's own `to_torch` explodes
    on the latter two, so only the genuine article is converted. Found
    by the B4 demo, the first run inside mjlab's real manager stack."""
    if isinstance(array, wp.array):
        return wp.to_torch(array)
    return array


# Every key this law consumes from a BAM fit. `from_bundle` reconciles
# the bundle against this roster and REFUSES leftovers — the silent
# cherry-pick that let a typo'd field zero a friction term is gone
# (review, 2026-09-01). Rig-side keys (q_offset, command_delay) are
# consumed for the cfg, never DR-sampled (they belong to the bench and
# the bus, not the motor — BAM's own docs, 57 §7).
_CONSUMED_KEYS = (
    "model",
    "actuator",
    "kt",
    "R",
    "friction_base",
    "friction_viscous",
    "armature",
    "q_offset",
    "command_delay",
    "max_velocity",
    "error_gain_ratio",
    "friction_stribeck",
    "load_friction_base",
    "load_friction_stribeck",
    "load_friction_motor",
    "load_friction_external",
    "load_friction_motor_stribeck",
    "load_friction_external_stribeck",
    "load_friction_motor_quad",
    "load_friction_external_quad",
    "dtheta_stribeck",
    "alpha",
)

# The friction terms a bundle may carry beyond the base; absent means zero.
_OPTIONAL_FRICTION = (
    "friction_stribeck",
    "load_friction_base",
    "load_friction_stribeck",
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

    def __post_init__(self) -> None:
        super().__post_init__()
        # The docstring forbids hand construction; nothing enforced it,
        # so a default cfg RAN with the placeholder law above (kt=1,
        # R=1 — invented physics). Refused since 2026-09-01's review.
        if not self.bundle_path or not self.stamp:
            raise ValueError(
                "BamActuatorCfg must be built by from_bundle(...) — a "
                "hand-constructed cfg carries a placeholder law, which is "
                "exactly the guessing the certified store exists to end"
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
        consumed = set(_CONSUMED_KEYS)
        unknown = sorted(set(params) - consumed)
        if unknown:
            raise ValueError(
                f"{Path(bundle_path).name}: fit key(s) {unknown} are not "
                "consumed by this actuator's law — a typo'd or newer-schema "
                "field would silently change the physics; teach the law the "
                "key or fix the bundle"
            )
        if "command_delay" not in params:
            advisories = (
                *advisories,
                "no command_delay in the fit: zero bus latency assumed "
                "(some BAM motor families never record it — 57 §3)",
            )
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
        delay_steps = seconds_to_steps(params.get("command_delay", 0.0), physics_dt)
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


class _EffectiveLaw:
    """`LawParams` with per-world tensors substituted for drawn fields;
    everything else reads through to the cfg's scalars."""

    def __init__(self, law: LawParams, draws: dict[str, torch.Tensor]) -> None:
        self._law = law
        self._draws = draws

    def __getattr__(self, name: str) -> Any:
        draws = object.__getattribute__(self, "_draws")
        if name in draws:
            return draws[name]
        return getattr(object.__getattribute__(self, "_law"), name)


class BamActuator(Actuator[BamActuatorCfg]):
    """The law at run time; see the module docstring for the per-step story."""

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        # The law REPLACES whatever servo the model declares for its
        # joints - microduck's XML ships its own <position> actuator per
        # joint, and adding a motor beside it is a repeated-name refusal
        # from the compiler (caught by the first real Entity build,
        # 2026-09-01). Delete-then-add, the actuator-law rule: a partial
        # edit of the existing servo would keep its biasprm behind.
        doomed = [
            actuator
            for actuator in spec.actuators
            if actuator.target in target_names or actuator.name in target_names
        ]
        for actuator in doomed:
            spec.delete(actuator)
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
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        # target ids are LOCAL indices into the entity's non-free
        # joints (mjlab's contract: "Local target indices"); the global
        # dof addresses come from the entity's own indexing — the same
        # mapping mjlab's EntityData uses. Indexing them straight into
        # mj_model.jnt_dofadr put the friction budget on the FREEJOINT's
        # linear dof and dropped the last servo on any floating-base
        # robot (found 2026-09-01 by review, invisible to the
        # single-hinge test rigs).
        from mjlab.utils.spec import TransmissionType  # noqa: PLC0415

        if self.cfg.transmission_type != TransmissionType.JOINT:
            # target ids would be TENDON/SITE ids; indexing joint_v_adr
            # with them is nonsense (review 2026-09-01).
            raise NotImplementedError(
                f"{type(self).__name__} drives joints only; this cfg asks for "
                f"{self.cfg.transmission_type} transmission"
            )
        indexing = self.entity.indexing
        if len(indexing.joint_v_adr) != len(indexing.joints):
            # joint_v_adr is per-DOF; target ids index the non-free joint
            # LIST. Equal only while every non-free joint is 1-dof — a
            # ball joint would shift the mapping for everything after it.
            raise NotImplementedError(
                "this entity carries a multi-dof non-free joint (ball); the "
                "law's per-joint dof mapping assumes hinge/slide targets"
            )
        local = torch.tensor(self._target_ids_list, dtype=torch.long)
        joint_v_adr = indexing.joint_v_adr
        self._dof_ids = joint_v_adr.to("cpu")[local].to(dtype=torch.long, device=device)
        self._nv = int(mj_model.nv)
        # Torch views over the live warp arrays — zero-copy, per world.
        self._qfrc_actuator = as_torch(data.qfrc_actuator)
        self._qfrc_bias = as_torch(data.qfrc_bias)
        self._qfrc_constraint = as_torch(data.qfrc_constraint)
        self._efc_id = as_torch(data.efc.id)
        self._efc_type = as_torch(data.efc.type)
        self._efc_force = as_torch(data.efc.force)
        self._dof_frictionloss = as_torch(model.dof_frictionloss)
        num_envs = int(self._qfrc_actuator.shape[0])
        if self._dof_frictionloss.shape[0] != num_envs:
            raise RuntimeError(MISSING_EXPANSION_MESSAGE)
        self._draws: dict[str, torch.Tensor] = {}

    # LawParams fields a DR draw may replace per world — the law's own
    # constants. The passives (armature, friction_viscous) are MODEL
    # fields: randomise them with mjlab's native joint_armature /
    # joint_damping events, ranges from the same bundle (rq_mjlab.dr).
    SCALABLE = (
        "kt",
        "R",
        "friction_base",
        "friction_stribeck",
        "load_friction_motor",
        "load_friction_external",
        "load_friction_motor_stribeck",
        "load_friction_external_stribeck",
        "load_friction_motor_quad",
        "load_friction_external_quad",
        "dtheta_stribeck",
        "alpha",
        "error_gain_ratio",
    )

    def set_param_draws(
        self, env_ids: torch.Tensor, draws: dict[str, torch.Tensor]
    ) -> None:
        """Write per-world ABSOLUTE parameter values for `env_ids` —
        the DR event's sink (`rq_mjlab.dr.bam_param_dr_event`). A first
        touch of a parameter fills every world with the bundle's point
        estimate; an unknown or non-scalable name is refused."""
        if not hasattr(self, "_qfrc_actuator"):
            raise RuntimeError("set_param_draws before initialize()")
        num_envs = int(self._qfrc_actuator.shape[0])
        device = self._qfrc_actuator.device
        for name, values in draws.items():
            if name not in self.SCALABLE:
                raise ValueError(
                    f"{name!r} is not a per-world law parameter; scalable: "
                    f"{self.SCALABLE} (passives go through mjlab's own events)"
                )
            if name not in self._draws:
                self._draws[name] = torch.full(
                    (num_envs, 1),
                    float(getattr(self.cfg.law, name)),
                    device=device,
                )
            self._draws[name][env_ids] = torch.as_tensor(
                values, dtype=self._draws[name].dtype, device=device
            ).reshape(-1, 1)

    def effective_law(self) -> Any:
        """The law with any per-world draws substituted — broadcastable
        tensors where a draw exists, the cfg's scalars elsewhere. The
        kernel's functions take either."""
        if not self._draws:
            return self.cfg.law
        return _EffectiveLaw(self.cfg.law, self._draws)

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
        law = self.effective_law()
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
