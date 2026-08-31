"""The law on the device, pinned against a CPU reference.

A deliberately tiny harness — one hinge, the XL330 M6 bundle, two
worlds — driven WITHOUT mjlab's manager stack: `mjwarp.put_model`,
the field expanded per world by hand (what `bam_expansion_event`
arranges inside an env), the actuator's own `initialize`/`compute` in a
step loop, `mjwarp.step`. The reference is the same kernel stepped
against plain CPU MuJoCo with the per-step `dof_frictionloss` write —
the pipeline's CpuServo discipline in thirty local lines. Pins:

- trajectory: max |q_device - q_cpu| under DEVICE_BOUND over the run;
- the budget bites: the written friction exceeds `friction_base`
  somewhere (M6's load terms under opposing torques);
- per-world independence: two worlds with different targets write
  DIFFERENT friction rows — the collapse the expansion event exists to
  prevent would fail this.

Runs on CUDA when Warp sees the card (`LD_LIBRARY_PATH=/usr/lib/wsl/lib`
on this box, docs/07 2026-08-27) and on Warp's CPU device otherwise —
the pin is engine-vs-reference either way; the device name is printed
into the assertion message so a skipped card is visible in a failure.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
XL330_M6 = REPO_ROOT / "robots" / "actuator-bundles" / "xl330.m6.bundle.json"
DT = 0.002
STEPS = 200
WORLDS = 2
TARGETS = (0.4, -0.3)  # one per world: independence is the point
DEVICE_BOUND = 5e-3  # float32 device vs float64 CPU over 200 contactless steps

XML = f"""
<mujoco>
  <option timestep="{DT}"/>
  <worldbody>
    <body>
      <joint name="j" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.008" mass="0.08"/>
    </body>
  </worldbody>
</mujoco>
"""


class TheDevicePin(unittest.TestCase):
    def _cfg(self):
        from rq_mjlab.actuator import BamActuatorCfg  # noqa: PLC0415

        return BamActuatorCfg.from_bundle(
            XL330_M6, target_names_expr=("j",), physics_dt=DT
        )

    def _spec_with_law(self, cfg):
        import mujoco  # noqa: PLC0415
        import torch  # noqa: PLC0415

        spec = mujoco.MjSpec.from_string(XML)
        entity = _Cmd(indexing=_Cmd(ctrl_ids=torch.tensor([0])))
        actuator = cfg.build(entity, [0], ["j"])
        actuator.edit_spec(spec, ["j"])
        return spec, actuator

    def _cpu_reference(self, law) -> np.ndarray:
        """The kernel against plain CPU MuJoCo, float64, per world."""
        import mujoco  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from rq_mjlab.kernel import (  # noqa: PLC0415
            duty,
            external_torque,
            friction_budget,
            torque,
        )

        cfg = self._cfg()
        spec, _ = self._spec_with_law(cfg)
        model = spec.compile()
        trajectories = []
        friction_dof = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
        for target in TARGETS:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            rows = []
            tt = torch.tensor([[target]], dtype=torch.float64)
            for _ in range(STEPS):
                rows_mask = np.asarray(data.efc_type[: data.nefc]) == friction_dof
                fric = float(np.asarray(data.efc_force[: data.nefc])[rows_mask].sum())
                q = torch.tensor([[float(data.qpos[0])]], dtype=torch.float64)
                qd = torch.tensor([[float(data.qvel[0])]], dtype=torch.float64)
                tau_prev = torch.tensor(
                    [[float(data.qfrc_actuator[0])]], dtype=torch.float64
                )
                tau_ext = external_torque(
                    torch.tensor([[float(data.qfrc_bias[0])]], dtype=torch.float64),
                    torch.tensor(
                        [[float(data.qfrc_constraint[0])]], dtype=torch.float64
                    ),
                    torch.tensor([[fric]], dtype=torch.float64),
                )
                model.dof_frictionloss[0] = float(
                    friction_budget(law, tau_prev, tau_ext, qd)
                )
                data.ctrl[0] = float(torque(law, qd, duty(law, q, qd, tt)))
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                rows.append(float(data.qpos[0]))
            trajectories.append(rows)
        return np.asarray(trajectories)  # (WORLDS, STEPS)

    def test_the_device_tracks_the_cpu_and_worlds_stay_apart(self) -> None:
        import mujoco  # noqa: PLC0415
        import mujoco_warp as mjw  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        from rq_mjlab.actuator import BamActuator  # noqa: PLC0415
        from rq_mjlab.kernel import LawParams  # noqa: PLC0415

        wp.init()
        device = "cuda" if wp.is_cuda_available() else "cpu"
        cfg = self._cfg()
        spec, actuator = self._spec_with_law(cfg)
        mj_model = spec.compile()
        mj_data = mujoco.MjData(mj_model)
        with wp.ScopedDevice(device):
            # `batch_sizes` is what bam_expansion_event arranges inside an
            # env: real per-world memory for the field, at put time.
            model = mjw.put_model(mj_model, batch_sizes={"dof_frictionloss": WORLDS})
            data = mjw.put_data(mj_model, mj_data, nworld=WORLDS)
            self.assertIsInstance(actuator, BamActuator)
            self.assertIsInstance(cfg.law, LawParams)
            actuator._target_ids_list = [0]
            actuator.initialize(mj_model, model, data, device)
            targets = torch.tensor(
                [[t] for t in TARGETS], dtype=torch.float32, device=device
            )
            qpos = wp.to_torch(data.qpos)
            qvel = wp.to_torch(data.qvel)
            ctrl = wp.to_torch(data.ctrl)
            rows = []
            peak_budget = 0.0
            for _ in range(STEPS):
                cmd = _Cmd(
                    position_target=targets,
                    velocity_target=torch.zeros_like(targets),
                    effort_target=torch.zeros_like(targets),
                    pos=qpos[:, :1].clone(),
                    vel=qvel[:, :1].clone(),
                )
                ctrl[:, :1] = actuator.compute(cmd)
                peak_budget = max(peak_budget, float(actuator._dof_frictionloss.max()))
                mjw.step(model, data)
                rows.append(qpos[:, 0].detach().cpu().numpy().copy())
            written = actuator._dof_frictionloss.detach().cpu().numpy()
        device_rows = np.asarray(rows).T  # (WORLDS, STEPS)
        reference = self._cpu_reference(cfg.law)
        gap = float(np.abs(device_rows - reference).max())
        self.assertLess(
            gap, DEVICE_BOUND, f"device {device}: max |dq| {gap} vs the CPU reference"
        )
        # The budget bit somewhere beyond the base...
        self.assertGreater(peak_budget, cfg.law.friction_base * 1.01)
        # ...and the two worlds' rows are their own (the expansion works).
        self.assertGreater(float(np.abs(written[0] - written[1]).max()), 0.0)
        self.assertGreater(float(np.abs(device_rows[0] - device_rows[1]).max()), 0.05)


class _Cmd:
    """The five fields of mjlab's ActuatorCmd, duck-typed for the harness."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _bundle_delay_is_zeroed_for_this_pin() -> bool:  # pragma: no cover
    """The cfg carries the bundle's delay for real envs; this harness
    drives compute directly (no mjlab delay stage), so the pin compares
    like with like by construction — both sides undelayed."""
    return True


if __name__ == "__main__":
    unittest.main()
